#pragma once

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include "mxfp8.cuh"
#include "utils.cuh"

#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/zeros.h>

using namespace kittens;

enum class GluActivation {
    SWIGLU,
    SITUGLU,
};

template <int NUM_DEVICES,
          utils::RoutedPrecision ROUTED_PRECISION = utils::RoutedPrecision::MXFP8,
          GluActivation ACTIVATION = GluActivation::SWIGLU>
struct dispatch_mlp_swiglu_combiner {

static constexpr bool USE_MXFP8 = ROUTED_PRECISION == utils::RoutedPrecision::MXFP8;
static constexpr bool USE_SITUGLU = ACTIVATION == GluActivation::SITUGLU;

// beta1 and beta2 are the constants used by situ-glu in the kimi k3 technical report.
static constexpr float SITUGLU_GATE_BETA = 4.0f;
static constexpr float SITUGLU_UP_BETA = 25.0f;

struct situglu_backward_terms {
    float hidden;
    float d_gate;
    float d_up;
};

static __device__ __forceinline__ float situglu_sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

static __device__ __forceinline__ float situglu_forward(float gate, float up) {
    // h = [beta1 * tanh(g / beta1) * sigmoid(g)] * [beta2 * tanh(u / beta2)].
    const float gate_factor = SITUGLU_GATE_BETA * tanhf(gate / SITUGLU_GATE_BETA) * situglu_sigmoid(gate);
    const float up_factor = SITUGLU_UP_BETA * tanhf(up / SITUGLU_UP_BETA);
    return gate_factor * up_factor;
}

static __device__ __forceinline__ situglu_backward_terms situglu_backward(float gate, float up) {
    const float gate_tanh = tanhf(gate / SITUGLU_GATE_BETA);
    const float up_tanh = tanhf(up / SITUGLU_UP_BETA);
    const float sigmoid = situglu_sigmoid(gate);
    const float gate_factor = SITUGLU_GATE_BETA * gate_tanh * sigmoid;
    const float up_factor = SITUGLU_UP_BETA * up_tanh;

    // d gate_factor / dg = (1 - tanh(g / beta1)^2) * sigmoid(g)
    //                        + beta1 * tanh(g / beta1) * sigmoid(g) * (1 - sigmoid(g)).
    const float d_gate_factor = (1.0f - gate_tanh * gate_tanh) * sigmoid
                              + gate_factor * (1.0f - sigmoid);
    // d up_factor / du = 1 - tanh(u / beta2)^2.
    const float d_up_factor = 1.0f - up_tanh * up_tanh;
    return {gate_factor * up_factor, d_gate_factor * up_factor, gate_factor * d_up_factor};
}

struct config {
    // Grouped GEMM
    static constexpr int MLP_Mb = 256;
    static constexpr int MLP_Nb = 256;
    static constexpr int MLP_FP8_Kb = 128;
    static constexpr int MLP_BF16_Kb = 64;
    static constexpr int MLP_SUPERGROUP_SIZE = 8;
    static constexpr int MLP_LOAD_PIPE_DEPTH = 6;
    static constexpr int MLP_EPI_PIPE_DEPTH = 8;
    static constexpr int MLP_NUM_BF16_D_TILES = 3;
    static constexpr int MLP_NUM_FP8_D_TILES = 4;

    // MXFP8 quantize
    static constexpr int QUANT_Mb = 128;
    static constexpr int QUANT_Nb = 128;

    // Fused SwiGLU + MXFP8 quantize
    static constexpr int SWIGLU_Mb = 128;
    static constexpr int SWIGLU_Nb = 128;
    static constexpr int SWIGLU_FWD_PIPE_DEPTH = 3; // gate / up
    static constexpr int SWIGLU_BWD_PIPE_DEPTH = 2; // gate / up / d_hidden

    // Dispatch
    static constexpr int DISPATCH_Mb = 128;
    static constexpr int DISPATCH_Nb = 512;
    static constexpr int DISPATCH_OUT_TILES = 2;

    // Combine
    static constexpr int COMBINE_Mb = 16;
    static constexpr int COMBINE_Nb = 1024;
    static constexpr int COMBINE_PIPE_DEPTH = 7;
    
    // Kernel launch
    static constexpr int CLC_PIPE_DEPTH = 1;
    static constexpr int CLC_DRAIN_PIPE_DEPTH = 8; // roughly a good number, but variance is low
    static constexpr int CLUSTER_SIZE = 2;
    static constexpr int NUM_CONSUMERS = 1;
    static constexpr int NUM_PRODUCERS = 1;
    static constexpr int NUM_WARPS = (NUM_CONSUMERS + NUM_PRODUCERS) * WARPGROUP_WARPS; // 8
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS; // 256
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - 1024;
};

// Grouped GEMM tiles
using mlp_fp8_tile = st_fp8e4m3<config::MLP_Mb / 2, config::MLP_FP8_Kb>;
using mlp_bf16_tile = st_bf<config::MLP_Mb / 2, config::MLP_BF16_Kb>;
using mlp_bf16_t_tile = st_bf<config::MLP_BF16_Kb, config::MLP_Mb / 2>; // shared-expert BF16 K-major operand
using mlp_sc_tile = st_fp8e8m0<32, 16, false>;
using mlp_bf16_d_tile = st_bf<config::MLP_Mb / 2, config::MLP_Nb / config::MLP_EPI_PIPE_DEPTH>;
using mlp_fp8_d_tile = st_fp8e4m3<config::MLP_Mb / 2, 32>;

// MXFP8 quantize tiles
using quant_bf16_tile = mxfp8_quantize::globals::x_bf16_tile; // st_bf<128, 128, false>
using quant_fp8_tile = mxfp8_quantize::globals::x_fp8_tile;   // st_fp8e4m3<128, 128, false>
using quant_sc_tile = mxfp8_quantize::globals::x_sc_tile;     // st_fp8e8m0<32, 16, false>

// Fused SwiGLU tiles
using swiglu_tile = st_bf<config::SWIGLU_Mb, config::SWIGLU_Nb>;

// Global layouts
using mlp_bf16_gl = gl<bf16, 1, 1, -1, -1, mlp_bf16_tile, mlp_bf16_t_tile, swiglu_tile>;
using epi_bf16_gl = gl<bf16, 1, 1, -1, -1, mlp_bf16_d_tile, swiglu_tile, quant_bf16_tile>;
using swiglu_bf16_gl = gl<bf16, 1, 1, -1, -1, swiglu_tile>;
using wgrad_bf16_gl = gl<bf16, 1, 1, -1, -1, mlp_bf16_t_tile>;
using mlp_fp8_gl = gl<fp8e4m3, 1, 1, -1, -1, mlp_fp8_tile, quant_fp8_tile>;
using gate_up_fp8_gl = gl<fp8e4m3, 1, 1, -1, -1, mlp_fp8_d_tile, quant_fp8_tile>;

using router_weight_gl = gl<float, 1, 1, 1, -1>;
using d_router_weight_partials_gl = gl<float, 1, 1, -1, -1>;
using router_weight_pgl = std::array<float *, NUM_DEVICES>;  // lightweight pgl
using activation_bf16_pgl = std::array<bf16 *, NUM_DEVICES>; // lightweight pgl
using weight_bf16_gl = gl<bf16, 1, -1, -1, -1, mlp_bf16_tile, mlp_bf16_t_tile>;
using d_weight_bf16_gl = gl<bf16, 1, -1, -1, -1, mlp_bf16_d_tile>;
using weight_fp8_gl = gl<fp8e4m3, 1, -1, -1, -1, mlp_fp8_tile>;
using sc_gl = gl<fp8e8m0, -1, -1, 32, 16, mlp_sc_tile>;
using index_gl = gl<int, 1, 1, 1, -1>;
using routed_bf16_gl = gl<bf16, 1, 1, -1, -1, mlp_bf16_tile, mlp_bf16_t_tile, mlp_bf16_d_tile, swiglu_tile, quant_bf16_tile>;
using routed_activation_gl = std::conditional_t<USE_MXFP8, mlp_fp8_gl, routed_bf16_gl>;
using routed_gate_up_gl = std::conditional_t<USE_MXFP8, gate_up_fp8_gl, routed_bf16_gl>;
using routed_weight_gl = std::conditional_t<USE_MXFP8, weight_fp8_gl, weight_bf16_gl>;

struct unused_gl {};
using routed_sc_gl = std::conditional_t<USE_MXFP8, sc_gl, unused_gl>;
using routed_transposed_gl = std::conditional_t<USE_MXFP8, mlp_fp8_gl, routed_bf16_gl>;

struct globals_fwd {
    mlp_bf16_gl x_shared;                     // (num_local_tokens, H) gate/up GEMM A
    routed_activation_gl x_fp8_routed;        // (macrobatch_size, H) gate/up GEMM A + dispatch out
    routed_sc_gl x_sc_routed;                 // MXFP8 only: (macrobatch_size / 128, H / 128, 32, 16)
    routed_transposed_gl x_fp8_t_routed;      // MXFP8 only: (H, macrobatch_size) dispatch out
    routed_sc_gl x_sc_t_routed;               // MXFP8 only: (H / 128, macrobatch_size / 128, 32, 16)
    epi_bf16_gl gate_shared;                  // (num_local_tokens, I) gate GEMM D + swiglu in
    epi_bf16_gl gate_routed;                  // (macrobatch_size, I) gate GEMM D + swiglu in
    routed_gate_up_gl gate_fp8_routed;        // (macrobatch_size, I) routed gate saved for backward
    routed_sc_gl gate_sc_routed;              // MXFP8 only: (macrobatch_size / 128, I / 128, 32, 16)
    epi_bf16_gl up_shared;                    // (num_local_tokens, I) up GEMM D + swiglu in
    epi_bf16_gl up_routed;                    // (macrobatch_size, I) up GEMM D + swiglu in
    routed_gate_up_gl up_fp8_routed;          // (macrobatch_size, I) routed up saved for backward
    routed_sc_gl up_sc_routed;                // MXFP8 only: (macrobatch_size / 128, I / 128, 32, 16)
    mlp_bf16_gl hidden_shared;                // (num_local_tokens, I) swiglu out + down GEMM A
    routed_activation_gl hidden_fp8_routed;   // (macrobatch_size, I) swiglu out + down GEMM A
    routed_sc_gl hidden_sc_routed;            // MXFP8 only: (macrobatch_size / 128, I / 128, 32, 16)
    routed_transposed_gl hidden_fp8_t_routed; // MXFP8 only: (I, macrobatch_size) swiglu out
    routed_sc_gl hidden_sc_t_routed;          // MXFP8 only: (I / 128, macrobatch_size / 128, 32, 16)
    epi_bf16_gl y_shared;                     // (num_local_tokens, H) down GEMM D
    epi_bf16_gl y_routed;                     // (macrobatch_size, H) down GEMM D + combine

    activation_bf16_pgl x_routed_send_buffer; // (num_local_tokens, H)
    activation_bf16_pgl y_routed_recv_buffer; // (num_local_tokens * topk, H)

    weight_bf16_gl w_shared_gate;             // (I, H)
    routed_weight_gl w_routed_gate;           // (num_local_experts, I, H)
    routed_sc_gl w_routed_gate_sc;            // MXFP8 only: (num_local_experts * I / 128, H / 128, 32, 16)
    weight_bf16_gl w_shared_up;               // (I, H)
    routed_weight_gl w_routed_up;             // (num_local_experts, I, H)
    routed_sc_gl w_routed_up_sc;              // MXFP8 only: (num_local_experts * I / 128, H / 128, 32, 16)
    weight_bf16_gl w_shared_down;             // (H, I)
    routed_weight_gl w_routed_down;           // (num_local_experts, H, I)
    routed_sc_gl w_routed_down_sc;            // MXFP8 only: (num_local_experts * H / 128, I / 128, 32, 16)

    index_gl schedule_peer_rank;              // (schedule_capacity,)
    index_gl schedule_peer_token_idx;         // (schedule_capacity,)
    index_gl num_tokens;                      // (1,)
    index_gl tokens_per_expert;               // (num_local_experts,)

    index_gl gate_up_tile_ready;              // (shared_gate_up_tasks + routed_gate_up_tasks,)
    index_gl hidden_row_block_ready;          // (shared_row_blocks + routed_row_blocks,)
    index_gl x_routed_ready;                  // (num_minibatches,)
    index_gl y_routed_ready;                  // (num_minibatches,) routed down -> dispatch/combine
    index_gl y_routed_done;                   // (num Down CTA output tiles,) combine -> next routed down

    const int topk;
    const int num_comm_sms;
    const int macrobatch_size;
    const int minibatch_size;

    __host__ inline dim3 grid() const {
        const int num_minibatches = (schedule_peer_rank.cols() + minibatch_size - 1) / minibatch_size; // across all macrobatches
        const int shared_row_blocks = x_shared.rows() / config::MLP_Mb;
        const int minibatch_routed_row_blocks = minibatch_size / config::MLP_Mb;
        const int shared_gate_up_tasks = shared_row_blocks * (w_shared_gate.rows() / config::MLP_Nb);
        const int minibatch_routed_gate_up_tasks = minibatch_routed_row_blocks * (w_routed_gate.rows() / config::MLP_Nb);
        const int shared_swiglu_tiles = (hidden_shared.rows() / config::SWIGLU_Mb) * (hidden_shared.cols() / config::SWIGLU_Nb);
        const int minibatch_routed_swiglu_tiles = (minibatch_size / config::SWIGLU_Mb) * (hidden_fp8_routed.cols() / config::SWIGLU_Nb);
        const int shared_swiglu_tasks = (shared_swiglu_tiles + config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH);
        const int minibatch_routed_swiglu_tasks = (minibatch_routed_swiglu_tiles + config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH);
        const int shared_down_tasks = shared_row_blocks * (w_shared_down.rows() / config::MLP_Nb);
        const int minibatch_routed_down_tasks = minibatch_routed_row_blocks * (w_routed_down.rows() / config::MLP_Nb);
        const int shared_tasks = 2 * shared_gate_up_tasks + shared_swiglu_tasks + shared_down_tasks;
        const int minibatch_tasks = 2 * minibatch_routed_gate_up_tasks + minibatch_routed_swiglu_tasks + minibatch_routed_down_tasks;
        return dim3(config::CLUSTER_SIZE * (shared_tasks + num_minibatches * minibatch_tasks) + num_comm_sms);
    }
};

struct globals_bwd {
    // Saved/replayed forward activations
    wgrad_bf16_gl x_shared;                               // (num_local_tokens, H) wgrad gate/up B
    routed_activation_gl x_fp8_routed;                    // (macrobatch_size, H) replay gate/up A + replay-dispatch out
    routed_sc_gl x_sc_routed;                             // MXFP8 only: (macrobatch_size / 128, H / 128, 32, 16)
    routed_transposed_gl x_fp8_t_routed;                  // MXFP8 only: (H, macrobatch_size) wgrad gate/up B
    routed_sc_gl x_sc_t_routed;                           // MXFP8 only: (H / 128, macrobatch_size / 128, 32, 16)
    swiglu_bf16_gl gate_shared;                           // (num_local_tokens, I) swiglu bwd in
    epi_bf16_gl gate_routed;                              // (macrobatch_size, I) replay gate D + swiglu fwd in
    routed_gate_up_gl gate_fp8_routed;                    // (macrobatch_size, I) replay gate D + swiglu (fwd/bwd) in
    routed_sc_gl gate_sc_routed;                          // MXFP8 only: (macrobatch_size / 128, I / 128, 32, 16)
    swiglu_bf16_gl up_shared;                             // (num_local_tokens, I) swiglu bwd in
    epi_bf16_gl up_routed;                                // (macrobatch_size, I) replay up D + swiglu fwd in
    routed_gate_up_gl up_fp8_routed;                      // (macrobatch_size, I) replay up D + swiglu (fwd/bwd) in
    routed_sc_gl up_sc_routed;                            // MXFP8 only: (macrobatch_size / 128, I / 128, 32, 16)
    wgrad_bf16_gl hidden_shared;                          // (num_local_tokens, I) wgrad down B
    routed_activation_gl hidden_fp8_routed;               // (macrobatch_size, I) replay swiglu out
    routed_sc_gl hidden_sc_routed;                        // MXFP8 only: (macrobatch_size / 128, I / 128, 32, 16)
    routed_transposed_gl hidden_fp8_t_routed;             // MXFP8 only: (I, macrobatch_size) wgrad down B
    routed_sc_gl hidden_sc_t_routed;                      // MXFP8 only: (I / 128, macrobatch_size / 128, 32, 16)

    // Activation gradients
    mlp_bf16_gl d_y_shared;                               // (num_local_tokens, H) dgrad down A + wgrad down A
    routed_activation_gl d_y_fp8_routed;                  // (macrobatch_size, H) dgrad/wgrad down A + reverse-combine out
    routed_sc_gl d_y_sc_routed;                           // MXFP8 only: (macrobatch_size / 128, H / 128, 32, 16)
    routed_transposed_gl d_y_fp8_t_routed;                // MXFP8 only: (H, macrobatch_size) wgrad down A
    routed_sc_gl d_y_sc_t_routed;                         // MXFP8 only: (H / 128, macrobatch_size / 128, 32, 16)
    epi_bf16_gl d_hidden_shared;                          // (num_local_tokens, I) dgrad down out + swiglu bwd in
    epi_bf16_gl d_hidden_routed;                          // (macrobatch_size, I) dgrad down out + swiglu bwd in
    mlp_bf16_gl d_gate_shared;                            // (num_local_tokens, I) swiglu bwd out + dgrad/wgrad gate A
    routed_activation_gl d_gate_fp8_routed;               // (macrobatch_size, I) swiglu bwd out + dgrad/wgrad gate A
    routed_sc_gl d_gate_sc_routed;                        // MXFP8 only: (macrobatch_size / 128, I / 128, 32, 16)
    routed_transposed_gl d_gate_fp8_t_routed;             // MXFP8 only: (I, macrobatch_size) wgrad gate A
    routed_sc_gl d_gate_sc_t_routed;                      // MXFP8 only: (I / 128, macrobatch_size / 128, 32, 16)
    mlp_bf16_gl d_up_shared;                              // (num_local_tokens, I) swiglu bwd out + dgrad/wgrad up A
    routed_activation_gl d_up_fp8_routed;                 // (macrobatch_size, I) swiglu bwd out + dgrad/wgrad up A
    routed_sc_gl d_up_sc_routed;                          // MXFP8 only: (macrobatch_size / 128, I / 128, 32, 16)
    routed_transposed_gl d_up_fp8_t_routed;               // MXFP8 only: (I, macrobatch_size) wgrad up A
    routed_sc_gl d_up_sc_t_routed;                        // MXFP8 only: (I / 128, macrobatch_size / 128, 32, 16)
    epi_bf16_gl d_x_shared;                               // (num_local_tokens, H) dgrad gate/up out
    epi_bf16_gl d_x_routed;                               // (macrobatch_size, H) dgrad gate/up out + reverse-dispatch in

    // Symmetric buffers
    activation_bf16_pgl x_routed_send_buffer;             // (num_local_tokens, H)
    activation_bf16_pgl d_y_buffer;                       // (num_local_tokens, H)
    activation_bf16_pgl d_x_routed_buffer;                // (num_local_tokens * topk, H)
    router_weight_pgl router_weight_buffer;               // (num_local_tokens, topk)
    router_weight_pgl d_router_weight_buffer;             // (num_local_tokens, topk)

    // Router (locally staged)
    router_weight_gl router_weights;                      // (macrobatch_size,)
    d_router_weight_partials_gl d_router_weight_partials; // (macrobatch_size, I / SWIGLU_Nb)

    // Weights
    routed_weight_gl w_routed_gate;                       // (num_local_experts, I, H) replay
    routed_sc_gl w_routed_gate_sc;                        // MXFP8 only: (num_local_experts * I / 128, H / 128, 32, 16)
    routed_weight_gl w_routed_up;                         // (num_local_experts, I, H) replay
    routed_sc_gl w_routed_up_sc;                          // MXFP8 only: (num_local_experts * I / 128, H / 128, 32, 16)
    weight_bf16_gl w_shared_gate;                         // (I, H)
    routed_weight_gl w_routed_gate_T;                     // MXFP8: transposed; BF16: normal (I, H)
    routed_sc_gl w_routed_gate_T_sc;                      // MXFP8 only: (num_local_experts * H / 128, I / 128, 32, 16)
    weight_bf16_gl w_shared_up;                           // (I, H)
    routed_weight_gl w_routed_up_T;                       // MXFP8: transposed; BF16: normal (I, H)
    routed_sc_gl w_routed_up_T_sc;                        // MXFP8 only: (num_local_experts * H / 128, I / 128, 32, 16)
    weight_bf16_gl w_shared_down;                         // (H, I)
    routed_weight_gl w_routed_down_T;                     // MXFP8: transposed; BF16: normal (H, I)
    routed_sc_gl w_routed_down_T_sc;                      // MXFP8 only: (num_local_experts * I / 128, H / 128, 32, 16)

    // Weight gradients
    d_weight_bf16_gl d_w_shared_gate;                     // (I, H)
    d_weight_bf16_gl d_w_routed_gate;                     // (num_local_experts, I, H)
    d_weight_bf16_gl d_w_shared_up;                       // (I, H)
    d_weight_bf16_gl d_w_routed_up;                       // (num_local_experts, I, H)
    d_weight_bf16_gl d_w_shared_down;                     // (H, I)
    d_weight_bf16_gl d_w_routed_down;                     // (num_local_experts, H, I)

    // Schedules
    index_gl schedule_peer_rank;                          // (schedule_capacity,)
    index_gl schedule_peer_token_idx;                     // (schedule_capacity,)
    index_gl num_tokens;                                  // (1,)
    index_gl tokens_per_expert;                           // (num_local_experts,)

    // Barrier
    index_gl router_weights_ready;                        // (num_macrobatches,) router preload -> reverse-combine
    index_gl d_y_routed_ready;                            // (num_minibatches,) reverse-combine -> dgrad/wgrad down
    index_gl d_hidden_ready;                              // (shared + routed dgrad-down tasks,) dgrad down -> swiglu bwd
    index_gl d_gate_up_ready;                             // (shared + routed row blocks,) swiglu bwd -> dgrad gate/up and wgrad gate/up
    index_gl d_x_routed_ready;                            // (num_minibatches,) dgrad gate/up -> reverse-dispatch
    index_gl replayed_x_routed_ready;                     // (num_minibatches,) replayed dispatch -> replayed gate/up and wgrad gate/up
    index_gl replayed_gate_up_ready;                      // (routed gate/up tasks,) replayed gate/up -> replayed swiglu
    index_gl replayed_hidden_ready;                       // (routed row blocks,) replayed swiglu -> wgrad down
    index_gl routed_buffers_done;                         // (num_macrobatches,) current macrobatch -> next macrobatch

    const int topk;
    const int num_comm_sms;
    const int macrobatch_size;
    const int minibatch_size;

    __host__ inline dim3 grid() const {
        const int capacity = schedule_peer_rank.cols();
        const int num_minibatches = (capacity + minibatch_size - 1) / minibatch_size; // across all macrobatches
        const int num_macrobatches = (capacity + macrobatch_size - 1) / macrobatch_size;
        const int shared_row_blocks = d_y_shared.rows() / config::MLP_Mb;
        const int minibatch_row_blocks = minibatch_size / config::MLP_Mb;
        const int intermediate_dim_col_blocks = hidden_shared.cols() / config::MLP_Nb;
        const int hidden_dim_col_blocks = d_y_shared.cols() / config::MLP_Nb;
        const int shared_swiglu_bwd_tiles = (hidden_shared.rows() / config::SWIGLU_Mb) * (hidden_shared.cols() / config::SWIGLU_Nb);
        const int minibatch_swiglu_tiles = (minibatch_size / config::SWIGLU_Mb) * (hidden_fp8_routed.cols() / config::SWIGLU_Nb);
        const int shared_swiglu_bwd_tasks = (shared_swiglu_bwd_tiles + config::CLUSTER_SIZE * config::SWIGLU_BWD_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_BWD_PIPE_DEPTH);
        const int minibatch_swiglu_bwd_tasks = (minibatch_swiglu_tiles + config::CLUSTER_SIZE * config::SWIGLU_BWD_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_BWD_PIPE_DEPTH);
        const int minibatch_swiglu_fwd_tasks = (minibatch_swiglu_tiles + config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH);
        const int shared_tasks = shared_row_blocks * intermediate_dim_col_blocks + shared_swiglu_bwd_tasks + shared_row_blocks * hidden_dim_col_blocks + 3 * intermediate_dim_col_blocks * hidden_dim_col_blocks;
        const int minibatch_bwd_tasks = minibatch_row_blocks * intermediate_dim_col_blocks + minibatch_swiglu_bwd_tasks + minibatch_row_blocks * hidden_dim_col_blocks;
        const int minibatch_replay_tasks = 2 * minibatch_row_blocks * intermediate_dim_col_blocks + minibatch_swiglu_fwd_tasks;
        const int num_replay_minibatches = (num_macrobatches - 1) * (macrobatch_size / minibatch_size);
        const int wgrad_tasks = 3 * w_routed_gate.depth() * intermediate_dim_col_blocks * hidden_dim_col_blocks;
        return dim3(config::CLUSTER_SIZE * (shared_tasks + num_minibatches * minibatch_bwd_tasks + num_replay_minibatches * minibatch_replay_tasks + num_macrobatches * wgrad_tasks) + num_comm_sms);
    }
};

static __device__ __forceinline__ void barrier_wait(const index_gl &counter, int index, int required_count) {
    int value;
    while (true) {
        asm volatile("{ld.relaxed.gpu.global.s32 %0, [%1];}" : "=r"(value) : "l"(&counter[{index}]) : "memory");
        if (value >= required_count) break;
        __nanosleep(16);
    }
    asm volatile("{fence.acquire.gpu;}" ::: "memory");
}

static __device__ __forceinline__ void barrier_arrive(const index_gl &counter, int index, int increment = 1) {
    asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}" :: "l"(&counter[{index}]), "r"(increment) : "memory");
}

static __device__ __forceinline__ void preload_router_weights_kernel(
    const router_weight_pgl &peer_buf,
    const router_weight_gl &router_weights,
    const index_gl &schedule_peer_rank,
    const index_gl &schedule_peer_token_idx,
    const index_gl *buffer_ready,
    const index_gl &transfer_done,
    const int num_tokens,
    const int macrobatch_size,
    const int macrobatch_idx,
    const int task_idx,
    const int num_workers,
    const int previous_macrobatch_idx,
    const int buffer_ready_required_count
) {
    if (threadIdx.x == 0 && buffer_ready != nullptr)
        barrier_wait(*buffer_ready, previous_macrobatch_idx, buffer_ready_required_count);
    __syncthreads();

    const int macrobatch_offset = macrobatch_idx * macrobatch_size;
    const int macrobatch_tokens = min(macrobatch_size, num_tokens - macrobatch_offset);
    for (int row = task_idx * config::NUM_THREADS + threadIdx.x; row < macrobatch_tokens; row += num_workers * config::NUM_THREADS) {
        const int peer_rank = schedule_peer_rank[{macrobatch_offset + row}];
        const int peer_token_idx = schedule_peer_token_idx[{macrobatch_offset + row}];
        router_weights.raw_ptr[row] = peer_rank >= 0 ? peer_buf[peer_rank][peer_token_idx] : 0.0f;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        barrier_arrive(transfer_done, macrobatch_idx);
        barrier_wait(transfer_done, macrobatch_idx, num_workers); // all comm SM barrier
    }
    __syncthreads();
}

template <bool SCALE_ROWS = false>
static __device__ __forceinline__ void dispatch_kernel(
    const activation_bf16_pgl &peer_buf,
    const routed_activation_gl &x_gmem,
    const routed_sc_gl *x_sc_gmem,
    const routed_transposed_gl *x_t_gmem,
    const routed_sc_gl *x_sc_t_gmem,
    const router_weight_gl *router_weights,
    const index_gl &schedule_peer_rank,
    const index_gl &schedule_peer_token_idx,
    const index_gl *transfer_ready,
    const index_gl *buffer_ready,
    const index_gl &transfer_done,
    semaphore &inputs_arrived,
    uint32_t &bitfield,
    const int num_tokens,
    const int macrobatch_size,
    const int minibatch_size,
    const int macrobatch_idx,
    const int task_idx,
    const int row_divisor,
    const int previous_macrobatch_idx,
    const int buffer_ready_required_count,
    const uint64_t smem_base_addr
) {
    auto &token_chunks        = *reinterpret_cast<bf16 (*)[config::DISPATCH_Mb][config::DISPATCH_Nb]>(smem_base_addr);
    auto &x_fp8_tiles         = *reinterpret_cast<quant_fp8_tile (*)[config::DISPATCH_OUT_TILES]>(smem_base_addr + sizeof(token_chunks));
    auto &x_sc_tiles          = *reinterpret_cast<quant_sc_tile (*)[config::DISPATCH_OUT_TILES]>(smem_base_addr + sizeof(token_chunks) + sizeof(x_fp8_tiles));
    auto &x_fp8_t_tiles       = *reinterpret_cast<quant_fp8_tile (*)[config::DISPATCH_OUT_TILES]>(smem_base_addr + sizeof(token_chunks) + sizeof(x_fp8_tiles) + sizeof(x_sc_tiles));
    auto &x_sc_t_tiles        = *reinterpret_cast<quant_sc_tile (*)[config::DISPATCH_OUT_TILES]>(smem_base_addr + sizeof(token_chunks) + sizeof(x_fp8_tiles) + sizeof(x_sc_tiles) + sizeof(x_fp8_t_tiles));
    auto &router_weights_smem = *reinterpret_cast<float (*)[config::DISPATCH_Mb]>(smem_base_addr + sizeof(token_chunks) + sizeof(x_fp8_tiles) + sizeof(x_sc_tiles) + sizeof(x_fp8_t_tiles) + sizeof(x_sc_t_tiles));

    const int tid = threadIdx.x;
    const bool is_worker = tid < config::DISPATCH_Mb; // only these threads move tokens, but all threads join the barriers and waits

    const int cols = x_gmem.cols();
    const int col_blocks = (cols + config::DISPATCH_Nb - 1) / config::DISPATCH_Nb;

    const int macrobatch_offset = macrobatch_idx * macrobatch_size;
    const int num_macrobatch_tokens = min(macrobatch_size, num_tokens - macrobatch_offset);
    if (task_idx >= num_macrobatch_tokens / config::DISPATCH_Mb * col_blocks) return;

    const int row_idx = task_idx / col_blocks * config::DISPATCH_Mb;
    const int col_block_idx = task_idx % col_blocks;
    const int chunk_cols = min(config::DISPATCH_Nb, cols - col_block_idx * config::DISPATCH_Nb);
    const uint32_t chunk_bytes = chunk_cols * sizeof(bf16);

    const int peer_rank = is_worker ? schedule_peer_rank[{macrobatch_offset + row_idx + tid}] : -1;
    const int peer_token_idx = is_worker ? schedule_peer_token_idx[{macrobatch_offset + row_idx + tid}] : -1;
    const int num_valid = __syncthreads_count(peer_rank >= 0);

    if (tid == 0) {
        if (transfer_ready != nullptr) {
            // Dispatch can overwrite rows still read by the previously processed macrobatch's GEMMs.
            const int previous_macrobatch_offset = previous_macrobatch_idx * macrobatch_size;
            const int previous_macrobatch_tokens = min(macrobatch_size, num_tokens - previous_macrobatch_offset);
            if (row_idx < previous_macrobatch_tokens) { // otherwise, previous macrobatch was partial & no need to wait
                const int global_minibatch_idx = (previous_macrobatch_offset + row_idx) / minibatch_size;
                const int minibatch_rows = min(minibatch_size, num_tokens - global_minibatch_idx * minibatch_size);
                const int required_count = ((minibatch_rows + config::MLP_Mb - 1) / config::MLP_Mb) * (cols / config::MLP_Nb) * config::CLUSTER_SIZE;
                barrier_wait(*transfer_ready, global_minibatch_idx, required_count);
            }
        }
        if (buffer_ready != nullptr)
            barrier_wait(*buffer_ready, previous_macrobatch_idx, buffer_ready_required_count);
        tma::expect_bytes(inputs_arrived, num_valid * chunk_bytes);
    }
    __syncthreads();

    if constexpr (SCALE_ROWS) {
        if (is_worker) router_weights_smem[tid] = peer_rank >= 0 ? router_weights->raw_ptr[row_idx + tid] : 0.0f;
    }

    if (peer_rank >= 0) {
        tma::load_async(token_chunks[tid],
                        &peer_buf[peer_rank][static_cast<size_t>(peer_token_idx / row_divisor) * cols + col_block_idx * config::DISPATCH_Nb],
                        chunk_bytes, inputs_arrived);
    } else if (is_worker) { // zero-fill padding rows for correct transpose-quantize
        auto *chunk = reinterpret_cast<float4 *>(token_chunks[tid]);
        #pragma unroll
        for (int i = 0; i < config::DISPATCH_Nb * static_cast<int>(sizeof(bf16)) / static_cast<int>(sizeof(float4)); ++i)
            chunk[i] = float4{0.0f, 0.0f, 0.0f, 0.0f};
    }

    wait(inputs_arrived, get_phasebit<0>(bitfield, 0));
    update_phasebit<0>(bitfield, 0);

    if constexpr (USE_MXFP8) {
        // Quantize each 128x128 subtile of the staging buffer
        const int row_block = row_idx / config::QUANT_Mb;
        const int num_subtiles = chunk_cols / config::QUANT_Nb;
        for (int subtile = 0; subtile < config::DISPATCH_Nb / config::QUANT_Nb; ++subtile) {
            if (subtile < num_subtiles) {
                const auto &x_bf16_subtile = *reinterpret_cast<const quant_bf16_tile *>(
                    smem_base_addr + subtile * config::QUANT_Nb * sizeof(bf16));
                const int out = subtile % config::DISPATCH_OUT_TILES;

                if (tid == 0) tma::store_async_read_wait<4 * (config::DISPATCH_OUT_TILES - 1)>();
                __syncthreads(); // also makes zero-filled rows visible before the first transpose-quantize
                if constexpr (!SCALE_ROWS)
                    mxfp8_quantize::mxfp8_quantize_tile<true, true, config::DISPATCH_Nb, true, false>(x_bf16_subtile, x_fp8_tiles[out], x_sc_tiles[out], x_fp8_t_tiles[out], x_sc_t_tiles[out], nullptr, tid, 1);
                else
                    mxfp8_quantize::mxfp8_quantize_tile<true, true, config::DISPATCH_Nb, true, true> (x_bf16_subtile, x_fp8_tiles[out], x_sc_tiles[out], x_fp8_t_tiles[out], x_sc_t_tiles[out], router_weights_smem, tid, 1);
                __syncthreads(); // quantized tiles must be complete before TMA reads them

                if (tid == 0) {
                    const int col_block = col_block_idx * (config::DISPATCH_Nb / config::QUANT_Nb) + subtile;
                    tma::store_async(x_gmem, x_fp8_tiles[out], {row_block, col_block});
                    tma::store_async(*x_sc_gmem, x_sc_tiles[out], {row_block, col_block, 0, 0});
                    tma::store_async(*x_t_gmem, x_fp8_t_tiles[out], {col_block, row_block});
                    tma::store_async(*x_sc_t_gmem, x_sc_t_tiles[out], {col_block, row_block, 0, 0});
                }
            }
        }
    } else {
        if constexpr (SCALE_ROWS) {
            if (is_worker && peer_rank >= 0) {
                auto *pairs = reinterpret_cast<bf16_2 *>(token_chunks[tid]);
                const float weight = router_weights_smem[tid];
                for (int col = 0; col < chunk_cols / 2; ++col) {
                    float2 value = __bfloat1622float2(pairs[col]);
                    pairs[col] = __floats2bfloat162_rn(value.x * weight, value.y * weight);
                }
            }
            __syncthreads();
        }
        if (is_worker)
            tma::store_async(&x_gmem.raw_ptr[static_cast<size_t>(row_idx + tid) * cols + col_block_idx * config::DISPATCH_Nb],
                             token_chunks[tid], chunk_bytes);
    }

    // Bulk groups are per-thread: every BF16 worker issues a store, while only
    // thread 0 issues the MXFP8 stores.
    if constexpr (USE_MXFP8) {
        if (tid == 0) {
            tma::store_async_wait();
            const int global_minibatch_idx = (macrobatch_offset + row_idx) / minibatch_size;
            barrier_arrive(transfer_done, global_minibatch_idx);
        }
        __syncthreads(); // the next task on this CTA reuses the staging buffer
    } else {
        if (is_worker)
            tma::store_async_wait();
        __syncthreads(); // stores are visible and the staging buffer is reusable
        if (tid == 0) {
            const int global_minibatch_idx = (macrobatch_offset + row_idx) / minibatch_size;
            barrier_arrive(transfer_done, global_minibatch_idx);
        }
    }
}

template <bool ARRIVE_BY_ROW = false>
static __device__ __forceinline__ void combine_kernel(
    const activation_bf16_pgl &peer_buf,
    const epi_bf16_gl &local_buf,
    const router_weight_pgl *d_router_weight_buffer,
    const d_router_weight_partials_gl *d_router_weight_partials,
    const index_gl &schedule_peer_rank,
    const index_gl &schedule_peer_token_idx,
    const index_gl &transfer_ready,
    const index_gl *transfer_done,
    semaphore (&inputs_arrived)[config::COMBINE_PIPE_DEPTH],
    uint32_t &bitfield,
    const int num_tokens,
    const int macrobatch_size,
    const int minibatch_size,
    const int macrobatch_idx,
    const int task_idx,
    const uint64_t smem_base_addr
) {
    auto &token_chunks = *reinterpret_cast<bf16 (*)[config::COMBINE_PIPE_DEPTH][config::COMBINE_Mb][config::COMBINE_Nb]>(smem_base_addr);

    const int tid = threadIdx.x;
    const bool is_worker = tid < config::COMBINE_Mb; // only these threads move tokens, but all threads join the barriers and waits

    const int cols = local_buf.cols();
    const int col_blocks = (cols + config::COMBINE_Nb - 1) / config::COMBINE_Nb;
    const int first_tile_idx = task_idx * config::COMBINE_PIPE_DEPTH;

    const int macrobatch_offset = macrobatch_idx * macrobatch_size;
    const int num_macrobatch_tokens = min(macrobatch_size, num_tokens - macrobatch_offset);
    const int num_valid_tiles = min(config::COMBINE_PIPE_DEPTH, num_macrobatch_tokens / config::COMBINE_Mb * col_blocks - first_tile_idx); // because we pad to 256
    if (num_valid_tiles <= 0) return;

    const int first_row_idx = first_tile_idx / col_blocks * config::COMBINE_Mb + tid;
    const int first_col_block_idx = first_tile_idx % col_blocks;

    int row_idx[config::COMBINE_PIPE_DEPTH], col_block_idx[config::COMBINE_PIPE_DEPTH], peer_rank[config::COMBINE_PIPE_DEPTH], 
        peer_token_idx[config::COMBINE_PIPE_DEPTH], num_valid[config::COMBINE_PIPE_DEPTH];
    #pragma unroll
    for (int stage = 0, row = first_row_idx, col = first_col_block_idx; stage < config::COMBINE_PIPE_DEPTH; ++stage) {
        const bool is_valid_tile = stage < num_valid_tiles;
        row_idx[stage] = row;
        col_block_idx[stage] = col;
        peer_rank[stage] = is_valid_tile && is_worker ? schedule_peer_rank[{macrobatch_offset + row}] : -1;
        peer_token_idx[stage] = is_valid_tile && is_worker ? schedule_peer_token_idx[{macrobatch_offset + row}] : -1;
        num_valid[stage] = !is_valid_tile ? 0
                         : (stage == 0 || col == 0) ? __syncthreads_count(peer_rank[stage] >= 0)
                         : num_valid[stage - 1];
        if (++col == col_blocks) { col = 0; row += config::COMBINE_Mb; }
    }

    auto chunk_bytes = [&](int col_block) {
        return static_cast<uint32_t>(min(config::COMBINE_Nb, cols - col_block * config::COMBINE_Nb) * sizeof(bf16));
    };

    if (tid == 0) {
        // Wait until the GEMMs have fully written every minibatch this task reads
        const int first_global_minibatch_idx = (macrobatch_offset + first_row_idx) / minibatch_size;
        const int last_global_minibatch_idx = (macrobatch_offset + (first_tile_idx + num_valid_tiles - 1) / col_blocks * config::COMBINE_Mb) / minibatch_size;
        for (int global_minibatch_idx = first_global_minibatch_idx; global_minibatch_idx <= last_global_minibatch_idx; ++global_minibatch_idx) {
            const int minibatch_rows = min(minibatch_size, num_tokens - global_minibatch_idx * minibatch_size);
            const int required_count = ((minibatch_rows + config::MLP_Mb - 1) / config::MLP_Mb) * (cols / config::MLP_Nb) * config::CLUSTER_SIZE;
            barrier_wait(transfer_ready, global_minibatch_idx, required_count);
        }
        #pragma unroll
        for (int stage = 0; stage < config::COMBINE_PIPE_DEPTH; ++stage)
            if (stage < num_valid_tiles)
                tma::expect_bytes(inputs_arrived[stage], num_valid[stage] * chunk_bytes(col_block_idx[stage]));
    }
    __syncthreads();

    #pragma unroll
    for (int stage = 0; stage < config::COMBINE_PIPE_DEPTH; ++stage)
        if (peer_rank[stage] >= 0)
            tma::load_async(token_chunks[stage][tid],
                            &local_buf.raw_ptr[static_cast<size_t>(row_idx[stage]) * cols + col_block_idx[stage] * config::COMBINE_Nb],
                            chunk_bytes(col_block_idx[stage]), inputs_arrived[stage]);

    // Store each tile out as its loads arrive
    #pragma unroll
    for (int stage = 0; stage < config::COMBINE_PIPE_DEPTH; ++stage) {
        if (stage < num_valid_tiles) {
            wait(inputs_arrived[stage], get_phasebit<0>(bitfield, stage)); // semaphores are reused across tasks
            update_phasebit<0>(bitfield, stage);
            if (peer_rank[stage] >= 0) {
                tma::store_async(&peer_buf[peer_rank[stage]][static_cast<size_t>(peer_token_idx[stage]) * cols + col_block_idx[stage] * config::COMBINE_Nb],
                                 token_chunks[stage][tid], chunk_bytes(col_block_idx[stage]));
                if (d_router_weight_buffer != nullptr && col_block_idx[stage] == 0) {
                    float d_router_weight = 0.0f;
                    for (int col = 0; col < d_router_weight_partials->cols(); ++col)
                        d_router_weight += (*d_router_weight_partials)[{row_idx[stage], col}];
                    (*d_router_weight_buffer)[peer_rank[stage]][peer_token_idx[stage]] = d_router_weight;
                }
            }
        }
    }

    if constexpr (ARRIVE_BY_ROW) {
        if (transfer_done != nullptr) {
            const int stage = warpid();
            if (warp::laneid() == 0 && stage < num_valid_tiles) {
                const int row = (first_tile_idx + stage) / col_blocks * config::COMBINE_Mb;
                barrier_arrive(*transfer_done, (macrobatch_offset + row) / (config::MLP_Mb / config::CLUSTER_SIZE));
            }
        }
    }

    // The next task on this CTA reuses token_chunks; make sure outgoing stores are done reading shared memory
    tma::store_async_read_wait();
    __syncthreads();
    if constexpr (!ARRIVE_BY_ROW) {
        if (tid == 0 && transfer_done != nullptr)
            barrier_arrive(*transfer_done, macrobatch_idx);
    }
}

template <bool IS_SHARED>
static __device__ __forceinline__ void swiglu_fwd_kernel(
    const epi_bf16_gl &gate_gmem,
    const epi_bf16_gl &up_gmem,
    const std::conditional_t<IS_SHARED, mlp_bf16_gl, routed_activation_gl> &hidden_gmem,
    const routed_sc_gl *hidden_sc_gmem,          // routed MXFP8 only
    const routed_transposed_gl *hidden_t_gmem,   // routed MXFP8 only
    const routed_sc_gl *hidden_sc_t_gmem,        // routed MXFP8 only
    const index_gl &gate_up_tile_ready,
    const index_gl &hidden_row_block_ready,
    semaphore (&swiglu_inputs_arrived)[config::SWIGLU_FWD_PIPE_DEPTH],
    uint32_t &swiglu_bitfield,
    const int num_tokens,
    const int macrobatch_size,
    const int minibatch_size,
    const int macrobatch_idx,
    const int minibatch_idx,
    const int task_idx,
    const int cta_rank,
    const int gate_up_tile_ready_base_index,
    const int hidden_row_block_ready_base_index,
    const uint64_t smem_base_addr
) {
    static constexpr bool USE_ROUTED_MXFP8 = !IS_SHARED && USE_MXFP8;
    using input_bf16_tile = std::conditional_t<USE_ROUTED_MXFP8, quant_bf16_tile, swiglu_tile>;
    using output_bf16_tile = std::conditional_t<USE_ROUTED_MXFP8, quant_bf16_tile, swiglu_tile>;
    auto (&gate_smem)[config::SWIGLU_FWD_PIPE_DEPTH] = *reinterpret_cast<input_bf16_tile (*)[config::SWIGLU_FWD_PIPE_DEPTH]>(smem_base_addr);
    auto (&up_smem)[config::SWIGLU_FWD_PIPE_DEPTH]   = *reinterpret_cast<input_bf16_tile (*)[config::SWIGLU_FWD_PIPE_DEPTH]>(reinterpret_cast<uint64_t>(&gate_smem) + sizeof(gate_smem));
    auto &hidden_smem                                = *reinterpret_cast<output_bf16_tile *>(reinterpret_cast<uint64_t>(&up_smem) + sizeof(up_smem));

    const int intermediate_dim_col_blocks = hidden_gmem.cols() / config::MLP_Nb;
    const int global_minibatch_idx = macrobatch_idx * (macrobatch_size / minibatch_size) + minibatch_idx;
    const int macrobatch_row_block_offset = macrobatch_idx * (macrobatch_size / config::SWIGLU_Mb);

    const int row_blocks = num_tokens / config::SWIGLU_Mb;
    const int col_blocks = hidden_gmem.cols() / config::SWIGLU_Nb;
    const int num_tiles = row_blocks * col_blocks;
    int first_tile_idx, tile_end;
    if constexpr (IS_SHARED) {
        first_tile_idx = task_idx * config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH + cta_rank * config::SWIGLU_FWD_PIPE_DEPTH;
        tile_end = num_tiles;
    } else {
        const int num_tiles_per_minibatch = (minibatch_size / config::SWIGLU_Mb) * col_blocks;
        const int minibatch_first_tile_idx = global_minibatch_idx * num_tiles_per_minibatch;
        first_tile_idx = minibatch_first_tile_idx + task_idx * config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH + cta_rank * config::SWIGLU_FWD_PIPE_DEPTH;
        tile_end = min(num_tiles, minibatch_first_tile_idx + num_tiles_per_minibatch);
    }
    if (first_tile_idx >= tile_end)
        return;

    const int first_row = first_tile_idx / col_blocks;
    const int first_col = first_tile_idx % col_blocks;

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int stage = 0; stage < config::SWIGLU_FWD_PIPE_DEPTH; ++stage) {
            const int tile_idx = first_tile_idx + stage;
            if (tile_idx < tile_end) {
                // This improves throughput
                int row = first_row;
                int col = first_col + stage;
                if (col >= col_blocks) {
                    ++row;
                    col -= col_blocks;
                }
                tma::expect_bytes(swiglu_inputs_arrived[stage], sizeof(gate_smem[stage]) + sizeof(up_smem[stage]));

                const int parent_task_idx = (row / (config::MLP_Mb / config::SWIGLU_Mb)) * intermediate_dim_col_blocks + col / (config::MLP_Nb / config::SWIGLU_Nb);
                barrier_wait(gate_up_tile_ready, gate_up_tile_ready_base_index + parent_task_idx, 2 * config::CLUSTER_SIZE);

                tma::load_async(gate_smem[stage], gate_gmem, {row - macrobatch_row_block_offset, col}, swiglu_inputs_arrived[stage]);
                tma::load_async(up_smem[stage],   up_gmem,   {row - macrobatch_row_block_offset, col}, swiglu_inputs_arrived[stage]);
            }
        }
    }

    using compute_group = group<config::NUM_WARPS>;
    #pragma unroll (IS_SHARED ? config::SWIGLU_FWD_PIPE_DEPTH : 1)
    for (int stage = 0; stage < config::SWIGLU_FWD_PIPE_DEPTH; ++stage) {
        const int tile_idx = first_tile_idx + stage;
        if (tile_idx < tile_end) {
            wait(swiglu_inputs_arrived[stage], get_phasebit<0>(swiglu_bitfield, stage));
            update_phasebit<0>(swiglu_bitfield, stage);

            // This improves throughput
            int row = first_row;
            int col = first_col + stage;
            if (col >= col_blocks) {
                ++row;
                col -= col_blocks;
            }

            if constexpr (!USE_ROUTED_MXFP8) {
                if constexpr (USE_SITUGLU) {
                    if (threadIdx.x == 0) tma::store_async_read_wait();
                    __syncthreads();
                    const auto *gate_pairs = reinterpret_cast<const bf16_2 *>(gate_smem[stage].data);
                    const auto *up_pairs = reinterpret_cast<const bf16_2 *>(up_smem[stage].data);
                    auto *hidden_pairs = reinterpret_cast<bf16_2 *>(hidden_smem.data);
                    #pragma unroll
                    for (int i = threadIdx.x; i < config::SWIGLU_Mb * config::SWIGLU_Nb / 2; i += config::NUM_THREADS) {
                        const float2 gate = __bfloat1622float2(gate_pairs[i]);
                        const float2 up = __bfloat1622float2(up_pairs[i]);
                        hidden_pairs[i] = __floats2bfloat162_rn(
                            situglu_forward(gate.x, up.x),
                            situglu_forward(gate.y, up.y));
                    }
                } else {
                    rt_fl<config::SWIGLU_Mb / config::NUM_WARPS, config::SWIGLU_Nb> gate, up, denominator;
                    compute_group::load(gate, gate_smem[stage]);
                    compute_group::load(up, up_smem[stage]);
                    compute_group::mul(denominator, gate, -1.0f);
                    compute_group::exp(denominator, denominator);
                    compute_group::add(denominator, denominator, 1.0f);
                    compute_group::div(gate, gate, denominator);
                    compute_group::mul(gate, gate, up);
                    if (threadIdx.x == 0) tma::store_async_read_wait();
                    __syncthreads();
                    compute_group::store(hidden_smem, gate);
                }
            } else {
                const auto *gate_pairs = reinterpret_cast<const bf16_2 *>(gate_smem[stage].data);
                const auto *up_pairs = reinterpret_cast<const bf16_2 *>(up_smem[stage].data);
                auto *hidden_pairs = reinterpret_cast<bf16_2 *>(hidden_smem.data);
                #pragma unroll
                for (int i = threadIdx.x; i < config::SWIGLU_Mb * config::SWIGLU_Nb / 2; i += config::NUM_THREADS) {
                    float2 gate = __bfloat1622float2(gate_pairs[i]);
                    const float2 up = __bfloat1622float2(up_pairs[i]);
                    if constexpr (USE_SITUGLU) {
                        gate = float2{situglu_forward(gate.x, up.x), situglu_forward(gate.y, up.y)};
                    } else {
                        float2 denominator = base_ops::mul::op<float2>(gate, float2{-1.0f, -1.0f});
                        denominator = base_ops::exp::op<float2>(denominator);
                        denominator = base_ops::sum::op<float2>(denominator, float2{1.0f, 1.0f});
                        gate = base_ops::div::op<float2>(gate, denominator);
                        gate = base_ops::mul::op<float2>(gate, up);
                    }
                    hidden_pairs[i] = __floats2bfloat162_rn(gate.x, gate.y);
                }
            }
            __syncthreads();

            if constexpr (USE_ROUTED_MXFP8) {
                auto &hidden_quant_smem = reinterpret_cast<quant_bf16_tile &>(hidden_smem);
                auto &hidden_fp8_smem = reinterpret_cast<quant_fp8_tile &>(gate_smem[stage]);
                auto &hidden_sc_smem = *reinterpret_cast<quant_sc_tile *>(reinterpret_cast<uint64_t>(&gate_smem[stage]) + sizeof(hidden_fp8_smem));
                auto &hidden_fp8_t_smem = reinterpret_cast<quant_fp8_tile &>(up_smem[stage]);
                auto &hidden_sc_t_smem = *reinterpret_cast<quant_sc_tile *>(reinterpret_cast<uint64_t>(&up_smem[stage]) + sizeof(hidden_fp8_t_smem));

                mxfp8_quantize::mxfp8_quantize_tile<true, true>(hidden_quant_smem, hidden_fp8_smem, hidden_sc_smem, hidden_fp8_t_smem, hidden_sc_t_smem, nullptr, threadIdx.x, 1);
                __syncthreads();

                if (threadIdx.x == 0) {
                    tma::store_async(hidden_gmem, hidden_fp8_smem, {row - macrobatch_row_block_offset, col});
                    tma::store_async(*hidden_sc_gmem, hidden_sc_smem, {row - macrobatch_row_block_offset, col, 0, 0});
                    tma::store_async(*hidden_t_gmem, hidden_fp8_t_smem, {col, row - macrobatch_row_block_offset});
                    tma::store_async(*hidden_sc_t_gmem, hidden_sc_t_smem, {col, row - macrobatch_row_block_offset, 0, 0});
                }
            } else if (threadIdx.x == 0) {
                tma::store_async(hidden_gmem, hidden_smem, {row - macrobatch_row_block_offset, col});
            }
        }
    }

    if (threadIdx.x == 0) {
        tma::store_async_wait();
        #pragma unroll
        for (int stage = 0; stage < config::SWIGLU_FWD_PIPE_DEPTH; ++stage) {
            const int tile_idx = first_tile_idx + stage;
            if (tile_idx < tile_end) {
                // This improves throughput
                int row = first_row;
                int col = first_col + stage;
                if (col >= col_blocks)
                    ++row;
                barrier_arrive(hidden_row_block_ready, hidden_row_block_ready_base_index + row / (config::MLP_Mb / config::SWIGLU_Mb));
            }
        }
    }
}

template <bool IS_SHARED>
static __device__ __forceinline__ void swiglu_bwd_kernel(
    const epi_bf16_gl &d_hidden_gmem,
    const std::conditional_t<IS_SHARED, swiglu_bf16_gl, routed_gate_up_gl> &gate_gmem,
    const std::conditional_t<IS_SHARED, swiglu_bf16_gl, routed_gate_up_gl> &up_gmem,
    const std::conditional_t<IS_SHARED, mlp_bf16_gl, routed_activation_gl> &d_gate_gmem,
    const std::conditional_t<IS_SHARED, mlp_bf16_gl, routed_activation_gl> &d_up_gmem,
    const routed_sc_gl *gate_sc_gmem,            // routed MXFP8 only
    const routed_sc_gl *up_sc_gmem,              // routed MXFP8 only
    const routed_sc_gl *d_gate_sc_gmem,          // routed MXFP8 only
    const routed_sc_gl *d_up_sc_gmem,            // routed MXFP8 only
    const routed_transposed_gl *d_gate_t_gmem,   // routed MXFP8 only
    const routed_sc_gl *d_gate_sc_t_gmem,        // routed MXFP8 only
    const routed_transposed_gl *d_up_t_gmem,     // routed MXFP8 only
    const routed_sc_gl *d_up_sc_t_gmem,          // routed MXFP8 only
    const router_weight_gl *router_weights,
    const d_router_weight_partials_gl *d_router_weight_partials,
    const index_gl *schedule_peer_rank,
    const index_gl &d_hidden_tile_ready,
    const index_gl *replayed_gate_up_tile_ready,
    const index_gl &d_gate_up_row_block_ready,
    const index_gl *buffer_done,
    semaphore (&swiglu_inputs_arrived)[config::SWIGLU_BWD_PIPE_DEPTH],
    uint32_t &swiglu_bitfield,
    const int num_tokens,
    const int macrobatch_size,
    const int minibatch_size,
    const int macrobatch_idx,
    const int minibatch_idx,
    const int task_idx,
    const int cta_rank,
    const int d_hidden_tile_ready_base_index,
    const int replayed_gate_up_tile_ready_base_index,
    const int d_gate_up_row_block_ready_base_index,
    const int buffer_done_index,
    const uint64_t smem_base_addr
) {
    static constexpr bool USE_ROUTED_MXFP8 = !IS_SHARED && USE_MXFP8;
    using gate_up_tile = std::conditional_t<USE_ROUTED_MXFP8, quant_fp8_tile, swiglu_tile>;
    using d_hidden_tile = std::conditional_t<USE_ROUTED_MXFP8, quant_bf16_tile, swiglu_tile>;

    auto &d_hidden_smem  = *reinterpret_cast<d_hidden_tile (*)[config::SWIGLU_BWD_PIPE_DEPTH]>(smem_base_addr);
    auto &gate_smem      = *reinterpret_cast<gate_up_tile (*)[config::SWIGLU_BWD_PIPE_DEPTH]>(reinterpret_cast<uint64_t>(&d_hidden_smem) + sizeof(d_hidden_smem));
    auto &up_smem        = *reinterpret_cast<gate_up_tile (*)[config::SWIGLU_BWD_PIPE_DEPTH]>(reinterpret_cast<uint64_t>(&gate_smem) + sizeof(gate_smem));
    auto &gate_sc_smem   = *reinterpret_cast<quant_sc_tile (*)[config::SWIGLU_BWD_PIPE_DEPTH]>(reinterpret_cast<uint64_t>(&up_smem) + sizeof(up_smem));
    auto &up_sc_smem     = *reinterpret_cast<quant_sc_tile (*)[config::SWIGLU_BWD_PIPE_DEPTH]>(reinterpret_cast<uint64_t>(&gate_sc_smem) + sizeof(gate_sc_smem));
    auto &d_up_bf16_smem = *reinterpret_cast<quant_bf16_tile *>(reinterpret_cast<uint64_t>(&up_sc_smem) + sizeof(up_sc_smem)); // staging shared across stages
    auto &d_t_fp8_smem   = *reinterpret_cast<quant_fp8_tile (*)[2]>(reinterpret_cast<uint64_t>(&d_up_bf16_smem) + sizeof(d_up_bf16_smem));
    auto &d_t_sc_smem    = *reinterpret_cast<quant_sc_tile (*)[2]>(reinterpret_cast<uint64_t>(&d_t_fp8_smem) + sizeof(d_t_fp8_smem));
    const uint64_t router_smem_addr = USE_ROUTED_MXFP8
        ? reinterpret_cast<uint64_t>(&d_t_sc_smem) + sizeof(d_t_sc_smem)
        : reinterpret_cast<uint64_t>(&up_smem) + sizeof(up_smem);
    auto &router_smem    = *reinterpret_cast<float (*)[config::SWIGLU_BWD_PIPE_DEPTH][config::SWIGLU_Mb]>(router_smem_addr);

    const int intermediate_dim_col_blocks = gate_gmem.cols() / config::MLP_Nb;
    const int global_minibatch_idx = macrobatch_idx * (macrobatch_size / minibatch_size) + minibatch_idx;
    const int macrobatch_row_block_offset = macrobatch_idx * (macrobatch_size / config::SWIGLU_Mb);

    const int row_blocks = num_tokens / config::SWIGLU_Mb;
    const int col_blocks = gate_gmem.cols() / config::SWIGLU_Nb;
    const int num_tiles = row_blocks * col_blocks;
    int first_tile_idx, tile_end;
    if constexpr (IS_SHARED) {
        first_tile_idx = task_idx * config::CLUSTER_SIZE * config::SWIGLU_BWD_PIPE_DEPTH + cta_rank * config::SWIGLU_BWD_PIPE_DEPTH;
        tile_end = num_tiles;
    } else {
        const int num_tiles_per_minibatch = (minibatch_size / config::SWIGLU_Mb) * col_blocks;
        const int minibatch_first_tile_idx = global_minibatch_idx * num_tiles_per_minibatch;
        first_tile_idx = minibatch_first_tile_idx + task_idx * config::CLUSTER_SIZE * config::SWIGLU_BWD_PIPE_DEPTH + cta_rank * config::SWIGLU_BWD_PIPE_DEPTH;
        tile_end = min(num_tiles, minibatch_first_tile_idx + num_tiles_per_minibatch);
    }
    if (first_tile_idx >= tile_end) {
        if (buffer_done != nullptr && threadIdx.x == 0)
            barrier_arrive(*buffer_done, buffer_done_index);
        return;
    }

    const int first_row = first_tile_idx / col_blocks;
    const int first_col = first_tile_idx % col_blocks;

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int stage = 0; stage < config::SWIGLU_BWD_PIPE_DEPTH; ++stage) {
            const int tile_idx = first_tile_idx + stage;
            if (tile_idx < tile_end) {
                int row = first_row;
                int col = first_col + stage;
                if (col >= col_blocks) {
                    ++row;
                    col -= col_blocks;
                }
                if constexpr (USE_ROUTED_MXFP8)
                    tma::expect_bytes(swiglu_inputs_arrived[stage], sizeof(d_hidden_smem[stage]) + sizeof(gate_smem[stage]) + sizeof(up_smem[stage]) + sizeof(gate_sc_smem[stage]) + sizeof(up_sc_smem[stage]));
                else
                    tma::expect_bytes(swiglu_inputs_arrived[stage], sizeof(d_hidden_smem[stage]) + sizeof(gate_smem[stage]) + sizeof(up_smem[stage]));

                const int parent_task_idx = (row / (config::MLP_Mb / config::SWIGLU_Mb)) * intermediate_dim_col_blocks + col / (config::MLP_Nb / config::SWIGLU_Nb);
                barrier_wait(d_hidden_tile_ready, d_hidden_tile_ready_base_index + parent_task_idx, config::CLUSTER_SIZE);
                if (replayed_gate_up_tile_ready != nullptr) // replayed macrobatch: wait for the replayed gate/up GEMMs
                    barrier_wait(*replayed_gate_up_tile_ready, replayed_gate_up_tile_ready_base_index + parent_task_idx, 2 * config::CLUSTER_SIZE);

                tma::load_async(d_hidden_smem[stage], d_hidden_gmem, {row - macrobatch_row_block_offset, col}, swiglu_inputs_arrived[stage]);
                tma::load_async(gate_smem[stage],     gate_gmem,     {row - macrobatch_row_block_offset, col}, swiglu_inputs_arrived[stage]);
                tma::load_async(up_smem[stage],       up_gmem,       {row - macrobatch_row_block_offset, col}, swiglu_inputs_arrived[stage]);
                if constexpr (USE_ROUTED_MXFP8) {
                    tma::load_async(gate_sc_smem[stage], *gate_sc_gmem, {row - macrobatch_row_block_offset, col, 0, 0}, swiglu_inputs_arrived[stage]);
                    tma::load_async(up_sc_smem[stage],   *up_sc_gmem,   {row - macrobatch_row_block_offset, col, 0, 0}, swiglu_inputs_arrived[stage]);
                }
            }
        }
    }

    using compute_group = group<config::NUM_WARPS>;
    #pragma unroll
    for (int stage = 0; stage < config::SWIGLU_BWD_PIPE_DEPTH; ++stage) {
        const int tile_idx = first_tile_idx + stage;
        if (tile_idx < tile_end) {
            wait(swiglu_inputs_arrived[stage], get_phasebit<0>(swiglu_bitfield, stage));
            update_phasebit<0>(swiglu_bitfield, stage);

            int row = first_row;
            int col = first_col + stage;
            if (col >= col_blocks) {
                ++row;
                col -= col_blocks;
            }

            if constexpr (USE_ROUTED_MXFP8) {
                constexpr int MXFP8_Kb = 32;
                const int tile_row = threadIdx.x % config::SWIGLU_Mb; // 2 threads per row
                const int tile_col_half = threadIdx.x / config::SWIGLU_Mb; // first half vs second half

                // For router-weight gradient: each row's two threads read the same weight
                const int global_token_idx = row * config::SWIGLU_Mb + tile_row;
                const int local_token_idx = (row - macrobatch_row_block_offset) * config::SWIGLU_Mb + tile_row;
                const int peer_rank = (*schedule_peer_rank)[{global_token_idx}];
                const float router_weight = router_weights->raw_ptr[local_token_idx];
                const float inv_router_weight = router_weight > 0.0f ? 1.0f / router_weight : 0.0f;
                float router_grad_partial = 0.0f;

                const uint32_t gate_fp8_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&gate_smem[stage]));
                const uint32_t up_fp8_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&up_smem[stage]));
                const uint32_t d_hidden_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&d_hidden_smem[stage]));
                const uint32_t d_up_bf16_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&d_up_bf16_smem));

                int gate_scale_word, up_scale_word;
                const int scale_word_offset = (tile_row % 32) * 16 + (tile_row / 32) * 4;
                move<int>::lds(gate_scale_word, static_cast<uint32_t>(__cvta_generic_to_shared(&gate_sc_smem[stage])) + scale_word_offset);
                move<int>::lds(up_scale_word, static_cast<uint32_t>(__cvta_generic_to_shared(&up_sc_smem[stage])) + scale_word_offset);

                const int k_block_pair = ((tile_row >> 1) & 1) ^ tile_col_half;
                uint32_t d_gate_scale_byte[2], d_up_scale_byte[2];
                #pragma unroll 1
                for (int j = 0; j < 2; ++j) { // 64 elements per thread = 2 32-element blocks per thread
                    const int k_block_idx = k_block_pair * 2 + ((tile_row + j) & 1);

                    // Load this block's gate/up values and the incoming d_hidden gradient
                    uint32_t gate_fp8[8], up_fp8[8];
                    float2 d_hidden_fp32[16];
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        const int col_idx = k_block_idx * MXFP8_Kb + ((tile_row / 4 + k) % 8) * 4;
                        move<int>::lds(reinterpret_cast<int &>(gate_fp8[k]), gate_fp8_addr + tile_row * config::SWIGLU_Nb + col_idx);
                        move<int>::lds(reinterpret_cast<int &>(up_fp8[k]), up_fp8_addr + tile_row * config::SWIGLU_Nb + col_idx);
                        float2 d_hidden_packed;
                        move<float2>::lds(d_hidden_packed, d_hidden_addr + (tile_row * config::SWIGLU_Nb + col_idx) * static_cast<int>(sizeof(bf16)));
                        const bf16_2 *d_hidden_pairs = reinterpret_cast<const bf16_2 *>(&d_hidden_packed);
                        d_hidden_fp32[k * 2] = __bfloat1622float2(d_hidden_pairs[0]);
                        d_hidden_fp32[k * 2 + 1] = __bfloat1622float2(d_hidden_pairs[1]);
                    }

                    // Dequantize
                    float2 gate_fp32[16], up_fp32[16];
                    mxfp8_quantize::mxfp8_dequantize_single_block(gate_fp8, (static_cast<uint32_t>(gate_scale_word) >> (k_block_idx * 8)) & 0xFF, gate_fp32);
                    mxfp8_quantize::mxfp8_dequantize_single_block(up_fp8, (static_cast<uint32_t>(up_scale_word) >> (k_block_idx * 8)) & 0xFF, up_fp32);

                    // apply the selected glu derivative in fp32 before requantizing the gradients.
                    bf16_2 d_gate_bf16[16], d_up_bf16[16];
                    #pragma unroll
                    for (int k = 0; k < 16; ++k) {
                        const float d_hidden_x = d_hidden_fp32[k].x;
                        const float d_hidden_y = d_hidden_fp32[k].y;
                        if constexpr (USE_SITUGLU) {
                            const situglu_backward_terms terms_x = situglu_backward(gate_fp32[k].x, up_fp32[k].x);
                            const situglu_backward_terms terms_y = situglu_backward(gate_fp32[k].y, up_fp32[k].y);
                            router_grad_partial += d_hidden_x * inv_router_weight * terms_x.hidden
                                                 + d_hidden_y * inv_router_weight * terms_y.hidden;
                            d_gate_bf16[k] = __float22bfloat162_rn(float2{
                                terms_x.d_gate * d_hidden_x,
                                terms_y.d_gate * d_hidden_y});
                            d_up_bf16[k] = __float22bfloat162_rn(float2{
                                terms_x.d_up * d_hidden_x,
                                terms_y.d_up * d_hidden_y});
                        } else {
                            const float sigmoid_x = 1.0f / (1.0f + __expf(-gate_fp32[k].x));
                            const float sigmoid_y = 1.0f / (1.0f + __expf(-gate_fp32[k].y));
                            const float silu_x = gate_fp32[k].x * sigmoid_x;
                            const float silu_y = gate_fp32[k].y * sigmoid_y;
                            const float dsilu_x = (1.0f - silu_x) * sigmoid_x + silu_x;
                            const float dsilu_y = (1.0f - silu_y) * sigmoid_y + silu_y;
                            router_grad_partial += d_hidden_x * inv_router_weight * silu_x * up_fp32[k].x
                                                 + d_hidden_y * inv_router_weight * silu_y * up_fp32[k].y;
                            d_gate_bf16[k] = __float22bfloat162_rn(float2{
                                dsilu_x * up_fp32[k].x * d_hidden_x,
                                dsilu_y * up_fp32[k].y * d_hidden_y});
                            d_up_bf16[k] = __float22bfloat162_rn(float2{
                                silu_x * d_hidden_x,
                                silu_y * d_hidden_y});
                        }
                    }

                    // Quantize both gradients; also stage them in BF16 for the transpose-quantize below
                    uint32_t d_gate_fp8[8], d_up_fp8[8];
                    mxfp8_quantize::mxfp8_quantize_single_block(d_gate_bf16, d_gate_fp8, d_gate_scale_byte[j]);
                    mxfp8_quantize::mxfp8_quantize_single_block(d_up_bf16, d_up_fp8, d_up_scale_byte[j]);
                    const uint32_t *d_gate_bf16_words = reinterpret_cast<const uint32_t *>(d_gate_bf16);
                    const uint32_t *d_up_bf16_words = reinterpret_cast<const uint32_t *>(d_up_bf16);
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        const int col_idx = k_block_idx * MXFP8_Kb + ((tile_row / 4 + k) % 8) * 4;
                        move<int>::sts(gate_fp8_addr + tile_row * config::SWIGLU_Nb + col_idx, std::bit_cast<int>(d_gate_fp8[k]));
                        move<int>::sts(up_fp8_addr + tile_row * config::SWIGLU_Nb + col_idx, std::bit_cast<int>(d_up_fp8[k]));
                        move<float2>::sts(d_hidden_addr + (tile_row * config::SWIGLU_Nb + col_idx) * static_cast<int>(sizeof(bf16)),
                                          float2{__uint_as_float(d_gate_bf16_words[k * 2]), __uint_as_float(d_gate_bf16_words[k * 2 + 1])});
                        move<float2>::sts(d_up_bf16_addr + (tile_row * config::SWIGLU_Nb + col_idx) * static_cast<int>(sizeof(bf16)),
                                          float2{__uint_as_float(d_up_bf16_words[k * 2]), __uint_as_float(d_up_bf16_words[k * 2 + 1])});
                    }
                }

                // Store this thread's two scale bytes per gradient (overwrites the input scale tiles)
                const uint16_t d_gate_scale_pair = static_cast<uint16_t>(d_gate_scale_byte[tile_row & 1] | (d_gate_scale_byte[(tile_row + 1) & 1] << 8));
                const uint16_t d_up_scale_pair = static_cast<uint16_t>(d_up_scale_byte[tile_row & 1] | (d_up_scale_byte[(tile_row + 1) & 1] << 8));
                move<bf16>::sts(static_cast<uint32_t>(__cvta_generic_to_shared(&gate_sc_smem[stage])) + scale_word_offset + k_block_pair * 2, std::bit_cast<bf16>(d_gate_scale_pair));
                move<bf16>::sts(static_cast<uint32_t>(__cvta_generic_to_shared(&up_sc_smem[stage])) + scale_word_offset + k_block_pair * 2, std::bit_cast<bf16>(d_up_scale_pair));
                __syncthreads(); // all normal-orientation outputs and BF16 stagings must be visible

                if (threadIdx.x == 0) {
                    tma::store_async(d_gate_gmem, gate_smem[stage], {row - macrobatch_row_block_offset, col});
                    tma::store_async(*d_gate_sc_gmem, gate_sc_smem[stage], {row - macrobatch_row_block_offset, col, 0, 0});
                    tma::store_async(d_up_gmem, up_smem[stage], {row - macrobatch_row_block_offset, col});
                    tma::store_async(*d_up_sc_gmem, up_sc_smem[stage], {row - macrobatch_row_block_offset, col, 0, 0});
                }

                // Transpose-quantize both gradients, one per warpgroup; d_gate was staged in the d_hidden tile, d_up in the shared staging buffer
                if (threadIdx.x < config::QUANT_Mb)
                    mxfp8_quantize::mxfp8_quantize_tile<false, true>(d_hidden_smem[stage], d_t_fp8_smem[0], d_t_sc_smem[0], d_t_fp8_smem[0], d_t_sc_smem[0], nullptr, threadIdx.x, 1);
                else
                    mxfp8_quantize::mxfp8_quantize_tile<false, true>(d_up_bf16_smem, d_t_fp8_smem[1], d_t_sc_smem[1], d_t_fp8_smem[1], d_t_sc_smem[1], nullptr, threadIdx.x - config::QUANT_Mb, 2);
                
                // Store partial router gradient to smem for the first col half to handle later
                if (tile_col_half != 0) 
                    router_smem[stage][tile_row] = router_grad_partial;
                __syncthreads(); // quantized tiles must be complete before TMA reads them & partial router gradient must be fully stored

                if (threadIdx.x == 0) {
                    tma::store_async(*d_gate_t_gmem, d_t_fp8_smem[0], {col, row - macrobatch_row_block_offset});
                    tma::store_async(*d_gate_sc_t_gmem, d_t_sc_smem[0], {col, row - macrobatch_row_block_offset, 0, 0});
                    tma::store_async(*d_up_t_gmem, d_t_fp8_smem[1], {col, row - macrobatch_row_block_offset});
                    tma::store_async(*d_up_sc_t_gmem, d_t_sc_smem[1], {col, row - macrobatch_row_block_offset, 0, 0});
                    tma::store_async_read_wait(); // the next stage reuses the staging and transposed-output buffers
                }
                if (tile_col_half == 0 && peer_rank >= 0) {
                    router_grad_partial += router_smem[stage][tile_row];
                    (*d_router_weight_partials)[{local_token_idx, col}] = router_grad_partial;
                }
                __syncthreads();
            } else if constexpr (IS_SHARED) {
                if constexpr (USE_SITUGLU) {
                    const auto *d_hidden_pairs = reinterpret_cast<const bf16_2 *>(d_hidden_smem[stage].data);
                    auto *gate_pairs = reinterpret_cast<bf16_2 *>(gate_smem[stage].data);
                    auto *up_pairs = reinterpret_cast<bf16_2 *>(up_smem[stage].data);
                    #pragma unroll
                    for (int i = threadIdx.x; i < config::SWIGLU_Mb * config::SWIGLU_Nb / 2; i += config::NUM_THREADS) {
                        const float2 gate = __bfloat1622float2(gate_pairs[i]);
                        const float2 up = __bfloat1622float2(up_pairs[i]);
                        const float2 d_hidden = __bfloat1622float2(d_hidden_pairs[i]);
                        const situglu_backward_terms terms_x = situglu_backward(gate.x, up.x);
                        const situglu_backward_terms terms_y = situglu_backward(gate.y, up.y);
                        gate_pairs[i] = __floats2bfloat162_rn(
                            terms_x.d_gate * d_hidden.x,
                            terms_y.d_gate * d_hidden.y);
                        up_pairs[i] = __floats2bfloat162_rn(
                            terms_x.d_up * d_hidden.x,
                            terms_y.d_up * d_hidden.y);
                    }
                } else {
                    rt_fl<config::SWIGLU_Mb / config::NUM_WARPS, config::SWIGLU_Nb> gate, up, d_hidden;
                    compute_group::load(gate, gate_smem[stage]);
                    compute_group::mul(d_hidden, gate, -1.0f);
                    compute_group::exp(d_hidden, d_hidden);
                    compute_group::add(d_hidden, d_hidden, 1.0f);         // d_hidden := 1 / sigmoid(gate)
                    compute_group::div(gate, gate, d_hidden);             // gate := silu(gate)
                    compute_group::mul(up, gate, -1.0f);
                    compute_group::add(up, up, 1.0f);
                    compute_group::div(up, up, d_hidden);
                    compute_group::add(up, up, gate);                     // up := dsilu(gate)
                    compute_group::load(d_hidden, d_hidden_smem[stage]);
                    compute_group::mul(gate, gate, d_hidden);             // gate := d_up
                    compute_group::mul(d_hidden, d_hidden, up);
                    compute_group::load(up, up_smem[stage]);
                    compute_group::mul(d_hidden, d_hidden, up);           // d_hidden := d_gate
                    compute_group::store(gate_smem[stage], d_hidden);     // d_gate overwrites the gate tile in place
                    compute_group::store(up_smem[stage], gate);           // d_up overwrites the up tile in place
                }
                __syncthreads();
                if (threadIdx.x == 0) {
                    tma::store_async(d_gate_gmem, gate_smem[stage], {row - macrobatch_row_block_offset, col});
                    tma::store_async(d_up_gmem, up_smem[stage], {row - macrobatch_row_block_offset, col});
                }
            } else {
                const int tile_row = threadIdx.x % config::SWIGLU_Mb;
                const int tile_col_half = threadIdx.x / config::SWIGLU_Mb;
                const int global_token_idx = row * config::SWIGLU_Mb + tile_row;
                const int local_token_idx = (row - macrobatch_row_block_offset) * config::SWIGLU_Mb + tile_row;
                const int peer_rank = (*schedule_peer_rank)[{global_token_idx}];
                const float router_weight = router_weights->raw_ptr[local_token_idx];
                const float inv_router_weight = router_weight > 0.0f ? 1.0f / router_weight : 0.0f;
                float router_grad_partial = 0.0f;

                const uint32_t gate_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&gate_smem[stage]));
                const uint32_t up_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&up_smem[stage]));
                const uint32_t d_hidden_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&d_hidden_smem[stage]));
                #pragma unroll
                for (int i = 0; i < config::SWIGLU_Nb / 8; ++i) {
                    const int tile_col = tile_col_half * (config::SWIGLU_Nb / 2) + i * 4;
                    float2 gate_packed, up_packed, d_hidden_packed;
                    move<float2>::lds(gate_packed, swiglu_tile::idx(gate_addr, {tile_row, tile_col}));
                    move<float2>::lds(up_packed, swiglu_tile::idx(up_addr, {tile_row, tile_col}));
                    move<float2>::lds(d_hidden_packed, swiglu_tile::idx(d_hidden_addr, {tile_row, tile_col}));
                    const auto *gate_pairs = reinterpret_cast<const bf16_2 *>(&gate_packed);
                    const auto *up_pairs = reinterpret_cast<const bf16_2 *>(&up_packed);
                    const auto *d_hidden_pairs = reinterpret_cast<const bf16_2 *>(&d_hidden_packed);
                    bf16_2 d_gate_pairs[2], d_up_pairs[2];
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        const float2 gate = __bfloat1622float2(gate_pairs[j]);
                        const float2 up = __bfloat1622float2(up_pairs[j]);
                        const float2 d_hidden = __bfloat1622float2(d_hidden_pairs[j]);
                        if constexpr (USE_SITUGLU) {
                            const situglu_backward_terms terms_x = situglu_backward(gate.x, up.x);
                            const situglu_backward_terms terms_y = situglu_backward(gate.y, up.y);
                            router_grad_partial += d_hidden.x * inv_router_weight * terms_x.hidden
                                                 + d_hidden.y * inv_router_weight * terms_y.hidden;
                            d_gate_pairs[j] = __floats2bfloat162_rn(
                                terms_x.d_gate * d_hidden.x,
                                terms_y.d_gate * d_hidden.y);
                            d_up_pairs[j] = __floats2bfloat162_rn(
                                terms_x.d_up * d_hidden.x,
                                terms_y.d_up * d_hidden.y);
                        } else {
                            const float sigmoid_x = 1.0f / (1.0f + __expf(-gate.x));
                            const float sigmoid_y = 1.0f / (1.0f + __expf(-gate.y));
                            const float silu_x = gate.x * sigmoid_x;
                            const float silu_y = gate.y * sigmoid_y;
                            const float dsilu_x = (1.0f - silu_x) * sigmoid_x + silu_x;
                            const float dsilu_y = (1.0f - silu_y) * sigmoid_y + silu_y;
                            router_grad_partial += d_hidden.x * inv_router_weight * silu_x * up.x
                                                 + d_hidden.y * inv_router_weight * silu_y * up.y;
                            d_gate_pairs[j] = __floats2bfloat162_rn(
                                dsilu_x * up.x * d_hidden.x,
                                dsilu_y * up.y * d_hidden.y);
                            d_up_pairs[j] = __floats2bfloat162_rn(
                                silu_x * d_hidden.x,
                                silu_y * d_hidden.y);
                        }
                    }
                    const auto *d_gate_words = reinterpret_cast<const uint32_t *>(d_gate_pairs);
                    const auto *d_up_words = reinterpret_cast<const uint32_t *>(d_up_pairs);
                    move<float2>::sts(swiglu_tile::idx(gate_addr, {tile_row, tile_col}),
                                      float2{__uint_as_float(d_gate_words[0]), __uint_as_float(d_gate_words[1])});
                    move<float2>::sts(swiglu_tile::idx(up_addr, {tile_row, tile_col}),
                                      float2{__uint_as_float(d_up_words[0]), __uint_as_float(d_up_words[1])});
                }

                if (tile_col_half != 0)
                    router_smem[stage][tile_row] = router_grad_partial;
                __syncthreads();
                if (tile_col_half == 0 && peer_rank >= 0) {
                    router_grad_partial += router_smem[stage][tile_row];
                    (*d_router_weight_partials)[{local_token_idx, col}] = router_grad_partial;
                }
                __syncthreads();
                if (threadIdx.x == 0) {
                    tma::store_async(d_gate_gmem, gate_smem[stage], {row - macrobatch_row_block_offset, col});
                    tma::store_async(d_up_gmem, up_smem[stage], {row - macrobatch_row_block_offset, col});
                }
            }
        }
    }

    if (threadIdx.x == 0) {
        tma::store_async_wait();
        #pragma unroll
        for (int stage = 0; stage < config::SWIGLU_BWD_PIPE_DEPTH; ++stage) {
            const int tile_idx = first_tile_idx + stage;
            if (tile_idx < tile_end) {
                int row = first_row;
                int col = first_col + stage;
                if (col >= col_blocks)
                    ++row;
                barrier_arrive(d_gate_up_row_block_ready, d_gate_up_row_block_ready_base_index + row / (config::MLP_Mb / config::SWIGLU_Mb));
            }
        }
        if (buffer_done != nullptr)
            barrier_arrive(*buffer_done, buffer_done_index);
    }
}

template <bool IS_SHARED, bool IS_WGRAD = false, bool IS_AB = false>
static __device__ __forceinline__ void expert_grouped_gemm_kernel(
    const std::conditional_t<IS_SHARED, mlp_bf16_gl, routed_activation_gl> &a_gmem,
    const std::conditional_t<IS_WGRAD, std::conditional_t<IS_SHARED, wgrad_bf16_gl, routed_activation_gl>,
                                       std::conditional_t<IS_SHARED, weight_bf16_gl, routed_weight_gl>> &b_gmem,
    const routed_sc_gl *a_sc_gmem,          // routed MXFP8 only
    const routed_sc_gl *b_sc_gmem,          // routed MXFP8 only
    const std::conditional_t<IS_SHARED, mlp_bf16_gl, routed_activation_gl> *a2_gmem, // accumulated second GEMM
    const std::conditional_t<IS_SHARED, weight_bf16_gl, routed_weight_gl> *b2_gmem,
    const routed_sc_gl *a2_sc_gmem,         // routed MXFP8 only
    const routed_sc_gl *b2_sc_gmem,         // routed MXFP8 only
    const std::conditional_t<IS_WGRAD, d_weight_bf16_gl, epi_bf16_gl> &d_gmem,
    const routed_gate_up_gl *d_routed_gmem, // routed gate/up saved activation
    const routed_sc_gl *d_sc_gmem,          // routed MXFP8 only
    const index_gl &tokens_per_expert,
    const index_gl *input_minibatch_ready,  // comms -> GEMM
    const index_gl *input_row_block_ready,  // SwiGLU -> GEMM
    const index_gl *output_row_ready,       // previous combine -> GEMM epilogue
    const index_gl *output_tile_ready,      // GEMM -> SwiGLU
    const index_gl *output_minibatch_ready, // GEMM -> comms
    const index_gl *buffer_done,
    tt<float, config::MLP_Mb / 2, config::MLP_Nb> &d_tt,
    const full_tt_fp8e8m0<16 * config::MLP_LOAD_PIPE_DEPTH> &a_sc_tt,
    const full_tt_fp8e8m0<32 * config::MLP_LOAD_PIPE_DEPTH> &b_sc_tt,
    semaphore (&gemm_inputs_arrived)[config::MLP_LOAD_PIPE_DEPTH],
    semaphore (&gemm_scales_arrived)[config::MLP_LOAD_PIPE_DEPTH],
    semaphore (&gemm_inputs_finished)[config::MLP_LOAD_PIPE_DEPTH],
    semaphore (&gemm_scales_finished)[config::MLP_LOAD_PIPE_DEPTH],
    semaphore &gemm_outputs_arrived,
    semaphore &gemm_outputs_finished,
    uint32_t &gemm_bitfield,
    const int num_tokens,
    const int macrobatch_size,
    const int minibatch_size,
    const int macrobatch_idx,
    const int minibatch_idx,
    int task_idx,
    const int cta_rank,
    const int input_minibatch_ready_num_cols,
    const int input_row_block_ready_base_index,
    const int input_row_block_ready_required_count,
    const int output_tile_ready_base_index,
    const int buffer_done_index,
    const uint64_t smem_base_addr
) {
    static constexpr bool USE_ROUTED_MXFP8 = !IS_SHARED && USE_MXFP8;
    using a_tile = std::conditional_t<USE_ROUTED_MXFP8, mlp_fp8_tile,
                                      std::conditional_t<IS_WGRAD, mlp_bf16_t_tile, mlp_bf16_tile>>;
    using b_tile = std::conditional_t<USE_ROUTED_MXFP8, mlp_fp8_tile,
                                      std::conditional_t<IS_WGRAD || IS_AB, mlp_bf16_t_tile, mlp_bf16_tile>>;
    constexpr int MLP_Kb = USE_ROUTED_MXFP8 ? config::MLP_FP8_Kb : config::MLP_BF16_Kb;

    auto (&a_smem)[config::MLP_LOAD_PIPE_DEPTH]       = *reinterpret_cast<a_tile (*)[config::MLP_LOAD_PIPE_DEPTH]>(smem_base_addr);
    auto (&b_smem)[config::MLP_LOAD_PIPE_DEPTH]       = *reinterpret_cast<b_tile (*)[config::MLP_LOAD_PIPE_DEPTH]>(smem_base_addr + sizeof(a_smem));
    auto (&a_sc_smem)[config::MLP_LOAD_PIPE_DEPTH]    = *reinterpret_cast<mlp_sc_tile (*)[config::MLP_LOAD_PIPE_DEPTH]>(smem_base_addr + sizeof(a_smem) + sizeof(b_smem));
    auto (&b_sc_smem)[config::MLP_LOAD_PIPE_DEPTH][2] = *reinterpret_cast<mlp_sc_tile (*)[config::MLP_LOAD_PIPE_DEPTH][2]>(smem_base_addr + sizeof(a_smem) + sizeof(b_smem) + sizeof(a_sc_smem));
    auto (&d_bf16_smem)[config::MLP_NUM_BF16_D_TILES] = *reinterpret_cast<mlp_bf16_d_tile (*)[config::MLP_NUM_BF16_D_TILES]>((smem_base_addr + sizeof(a_smem) + sizeof(b_smem) + sizeof(a_sc_smem) + sizeof(b_sc_smem) + 1023) & ~uint64_t(1023));
    auto &d_fp8_smem                                  = *reinterpret_cast<mlp_fp8_d_tile *>(&d_bf16_smem[2]);
    auto (&d_sc_smem)[2]                              = *reinterpret_cast<mlp_sc_tile (*)[2]>(reinterpret_cast<uint64_t>(&d_fp8_smem) + sizeof(d_fp8_smem));
    static_assert(config::MLP_NUM_BF16_D_TILES >= 3);
    static_assert(sizeof(mlp_fp8_d_tile) + 2 * sizeof(mlp_sc_tile) <= sizeof(mlp_bf16_d_tile));

    const int col_blocks = ((IS_WGRAD && !USE_ROUTED_MXFP8) || IS_AB) ? b_gmem.cols() / config::MLP_Nb : b_gmem.rows() / config::MLP_Nb;
    const int global_minibatch_idx = macrobatch_idx * (macrobatch_size / minibatch_size) + minibatch_idx;
    const int macrobatch_row_block_offset = macrobatch_idx * (macrobatch_size / config::MLP_Mb);

    // Output tile (and for wgrad, K range in token rows) of this task
    int3 tile_coord = {-1, -1, -1};
    int k_start = 0, k_end = 0;
    bool is_first_wgrad_contribution = IS_SHARED;
    if constexpr (IS_WGRAD) {
        const int row_blocks = USE_ROUTED_MXFP8 ? a_gmem.rows() / config::MLP_Mb : a_gmem.cols() / config::MLP_Mb;
        const int expert_idx = IS_SHARED ? 0 : task_idx / (row_blocks * col_blocks);
        if constexpr (IS_SHARED) {
            k_end = a_gmem.rows();
        } else {
            int expert_row_offset = 0;
            for (int i = 0; i < expert_idx; ++i)
                expert_row_offset += tokens_per_expert[{i}];
            k_start = max(expert_row_offset, macrobatch_idx * macrobatch_size);
            k_end = min(expert_row_offset + tokens_per_expert[{expert_idx}], min((macrobatch_idx + 1) * macrobatch_size, num_tokens));
            is_first_wgrad_contribution = k_start == expert_row_offset;
        }
        if (k_start < k_end) {
            const int2 swizzled = get_swizzled_2d_idx<config::MLP_SUPERGROUP_SIZE>(row_blocks, col_blocks, IS_SHARED ? task_idx : task_idx % (row_blocks * col_blocks));
            tile_coord = {swizzled.x, swizzled.y, expert_idx};
        }
    } else if constexpr (IS_SHARED) {
        const int row_blocks = a_gmem.rows() / config::MLP_Mb;
        const int num_tasks = row_blocks * col_blocks;
        if (task_idx < num_tasks) {
            const int2 swizzled = get_swizzled_2d_idx<config::MLP_SUPERGROUP_SIZE>(row_blocks, col_blocks, task_idx);
            tile_coord = {swizzled.x, swizzled.y, 0};
        }
    } else {
        const int minibatch_routed_row_blocks = minibatch_size / config::MLP_Mb;
        const int global_minibatch_routed_first_row_block = global_minibatch_idx * minibatch_routed_row_blocks;
        int global_row_block_offset = 0;
        for (int expert_idx = 0; expert_idx < b_gmem.depth(); ++expert_idx) {
            const int expert_row_blocks = tokens_per_expert[{expert_idx}] / config::MLP_Mb;
            const int global_first_row_block = max(global_minibatch_routed_first_row_block, global_row_block_offset);
            const int row_blocks = max(0, min(global_minibatch_routed_first_row_block + minibatch_routed_row_blocks, global_row_block_offset + expert_row_blocks) - global_first_row_block);
            const int num_tasks = row_blocks * col_blocks;
            if (task_idx < num_tasks) {
                const int2 swizzled = get_swizzled_2d_idx<config::MLP_SUPERGROUP_SIZE>(row_blocks, col_blocks, task_idx);
                tile_coord = {global_first_row_block + swizzled.x - macrobatch_row_block_offset, swizzled.y, expert_idx};
                break;
            }
            task_idx -= num_tasks;
            global_row_block_offset += expert_row_blocks;
        }
    }
    if (tile_coord.z < 0) {
        if (buffer_done != nullptr && threadIdx.x == 0)
            barrier_arrive(*buffer_done, buffer_done_index);
        return;
    }

    const int first_gemm_iters = IS_WGRAD ? 0 : a_gmem.cols() / MLP_Kb;
    const int iters_per_task = IS_WGRAD ? (k_end - k_start) / MLP_Kb
                                        : first_gemm_iters + (a2_gmem != nullptr ? a2_gmem->cols() / MLP_Kb : 0);

    auto wait_for_a_operand = [&]() {
        if (input_row_block_ready != nullptr) {
            barrier_wait(*input_row_block_ready, input_row_block_ready_base_index + macrobatch_row_block_offset + tile_coord.x, input_row_block_ready_required_count);
        }
        if (input_minibatch_ready != nullptr) {
            const int minibatch_first_row = global_minibatch_idx * minibatch_size;
            const int minibatch_rows = max(0, min(minibatch_size, num_tokens - minibatch_first_row));
            const int required_count = ((minibatch_rows + config::DISPATCH_Mb - 1) / config::DISPATCH_Mb) * ((a_gmem.cols() + config::DISPATCH_Nb - 1) / config::DISPATCH_Nb);
            barrier_wait(*input_minibatch_ready, global_minibatch_idx, required_count);
        }
    };
    // Wgrad iterates K over token rows: wait as the loads cross into each producing row block / minibatch
    auto wait_for_wgrad_operands = [&](int idx, int row) {
        if (idx == 0 || row % config::MLP_Mb == 0) {
            if (input_row_block_ready != nullptr)
                barrier_wait(*input_row_block_ready, input_row_block_ready_base_index + row / config::MLP_Mb, input_row_block_ready_required_count);
        }
        if (idx == 0 || row % minibatch_size == 0) {
            if (input_minibatch_ready != nullptr) {
                const int row_minibatch_idx = row / minibatch_size;
                const int minibatch_rows = min(minibatch_size, num_tokens - row_minibatch_idx * minibatch_size);
                const int required_count = ((minibatch_rows + config::DISPATCH_Mb - 1) / config::DISPATCH_Mb) * ((input_minibatch_ready_num_cols + config::DISPATCH_Nb - 1) / config::DISPATCH_Nb);
                barrier_wait(*input_minibatch_ready, row_minibatch_idx, required_count);
            }
        }
    };

    if (warpgroup::groupid() == config::NUM_CONSUMERS) {
        if (warpgroup::warpid() == 3 && warp::elect_leader()) {
            int input_ring = 0;
            if constexpr (IS_WGRAD) {
                const int macrobatch_k_offset = macrobatch_idx * (macrobatch_size / MLP_Kb);
                for (int idx = 0, k_block = k_start / MLP_Kb; idx < iters_per_task; ++idx, ++k_block) {
                    wait_for_wgrad_operands(idx, k_block * MLP_Kb);
                    wait(gemm_inputs_finished[input_ring], get_phasebit<1>(gemm_bitfield, input_ring));
                    if constexpr (!USE_ROUTED_MXFP8) { // BF16 AtB: A/B tiles are K-major slices of normal token-major activations
                        tma::cluster::load_async(a_smem[input_ring], a_gmem, {k_block - macrobatch_k_offset, tile_coord.x * 2 + cta_rank}, gemm_inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                        tma::cluster::load_async(b_smem[input_ring], b_gmem, {k_block - macrobatch_k_offset, tile_coord.y * 2 + cta_rank}, gemm_inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                    } else { // MXFP8 ABt: A/B are transpose-quantized activations with K = tokens
                        tma::cluster::load_async(a_smem[input_ring], a_gmem, {tile_coord.x * 2 + cta_rank, k_block - macrobatch_k_offset}, gemm_inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                        tma::cluster::load_async(b_smem[input_ring], b_gmem, {tile_coord.y * 2 + cta_rank, k_block - macrobatch_k_offset}, gemm_inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                    }
                    update_phasebit<1>(gemm_bitfield, input_ring);
                    input_ring = ring_advance<config::MLP_LOAD_PIPE_DEPTH>(input_ring);
                }
            } else {
                wait_for_a_operand();
                for (int idx = 0; idx < iters_per_task; ++idx) {
                    const auto &a_gmem_curr = idx < first_gemm_iters ? a_gmem : *a2_gmem;
                    const auto &b_gmem_curr = idx < first_gemm_iters ? b_gmem : *b2_gmem;
                    const int k_block = idx < first_gemm_iters ? idx : idx - first_gemm_iters;
                    wait(gemm_inputs_finished[input_ring], get_phasebit<1>(gemm_bitfield, input_ring));
                    tma::cluster::load_async(a_smem[input_ring], a_gmem_curr, {tile_coord.x * 2 + cta_rank, k_block},               gemm_inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                    if constexpr (IS_AB)
                        tma::cluster::load_async(b_smem[input_ring], b_gmem_curr, {tile_coord.z, k_block, tile_coord.y * 2 + cta_rank}, gemm_inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                    else
                        tma::cluster::load_async(b_smem[input_ring], b_gmem_curr, {tile_coord.z, tile_coord.y * 2 + cta_rank, k_block}, gemm_inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                    update_phasebit<1>(gemm_bitfield, input_ring);
                    input_ring = ring_advance<config::MLP_LOAD_PIPE_DEPTH>(input_ring);
                }
            }
        } else if (warpgroup::warpid() == 2 && warp::elect_leader()) {
            if constexpr (USE_ROUTED_MXFP8) {
                int input_ring = 0;
                if constexpr (IS_WGRAD) { // MXFP8 wgrad: both operands' scales follow the token-major layout
                    const int macrobatch_k_offset = macrobatch_idx * (macrobatch_size / MLP_Kb);
                    for (int idx = 0, k_block = k_start / MLP_Kb; idx < iters_per_task; ++idx, ++k_block) {
                        wait_for_wgrad_operands(idx, k_block * MLP_Kb);
                        wait(gemm_scales_finished[input_ring], get_phasebit<1>(gemm_bitfield, input_ring));
                        tma::cluster::load_async(a_sc_smem[input_ring], *a_sc_gmem, {tile_coord.x * 2 + cta_rank, k_block - macrobatch_k_offset, 0, 0}, gemm_scales_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                        tma::cluster::load_async(b_sc_smem[input_ring][cta_rank], *b_sc_gmem, {tile_coord.y * 2 + cta_rank, k_block - macrobatch_k_offset, 0, 0}, gemm_scales_arrived[input_ring], (uint16_t)(0b11), 0);
                        update_phasebit<1>(gemm_bitfield, input_ring);
                        input_ring = ring_advance<config::MLP_LOAD_PIPE_DEPTH>(input_ring);
                    }
                } else {
                    wait_for_a_operand();
                    for (int idx = 0; idx < iters_per_task; ++idx) {
                        wait(gemm_scales_finished[input_ring], get_phasebit<1>(gemm_bitfield, input_ring));
                        const sc_gl &a_sc_curr = idx < first_gemm_iters ? *a_sc_gmem : *a2_sc_gmem;
                        const sc_gl &b_sc_curr = idx < first_gemm_iters ? *b_sc_gmem : *b2_sc_gmem;
                        const auto &b_gmem_curr = idx < first_gemm_iters ? b_gmem : *b2_gmem;
                        const int k_block = idx < first_gemm_iters ? idx : idx - first_gemm_iters;
                        tma::cluster::load_async(a_sc_smem[input_ring], a_sc_curr, {tile_coord.x * 2 + cta_rank, k_block, 0, 0}, gemm_scales_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);
                        tma::cluster::load_async(b_sc_smem[input_ring][cta_rank], b_sc_curr, {tile_coord.z * (b_gmem_curr.rows() / config::QUANT_Mb) + tile_coord.y * 2 + cta_rank, k_block, 0, 0}, gemm_scales_arrived[input_ring], (uint16_t)(0b11), 0);
                        update_phasebit<1>(gemm_bitfield, input_ring);
                        input_ring = ring_advance<config::MLP_LOAD_PIPE_DEPTH>(input_ring);
                    }
                }
            }
        } else if (cta_rank == 0 && warpgroup::warpid() == 0 && warp::elect_leader()) {
            int input_ring = 0;
            wait(gemm_outputs_finished, get_phasebit<1>(gemm_bitfield, config::MLP_LOAD_PIPE_DEPTH));
            update_phasebit<1>(gemm_bitfield, config::MLP_LOAD_PIPE_DEPTH);
            tensor_after_thread_sync();
            for (int idx = 0; idx < iters_per_task; ++idx) {
                if constexpr (USE_ROUTED_MXFP8) {
                    tma::expect_bytes(gemm_scales_arrived[input_ring], config::CLUSTER_SIZE * 3 * sizeof(mlp_sc_tile));
                    wait(gemm_scales_arrived[input_ring], get_phasebit<0>(gemm_bitfield, config::MLP_LOAD_PIPE_DEPTH + 1 + input_ring));
                    update_phasebit<0>(gemm_bitfield, config::MLP_LOAD_PIPE_DEPTH + 1 + input_ring);
                    auto a_sc_tt_subtile = a_sc_tt.template subtile<full_tt_fp8e8m0<16>>(input_ring * 16);
                    auto b_sc_tt_subtile_0 = b_sc_tt.template subtile<full_tt_fp8e8m0<16>>(input_ring * 32);
                    auto b_sc_tt_subtile_1 = b_sc_tt.template subtile<full_tt_fp8e8m0<16>>(input_ring * 32 + 16);
                    load_mxnv_scale_async2(a_sc_tt_subtile, a_sc_smem[input_ring]);
                    load_mxnv_scale_async2(b_sc_tt_subtile_0, b_sc_smem[input_ring][0]);
                    load_mxnv_scale_async2(b_sc_tt_subtile_1, b_sc_smem[input_ring][1], gemm_scales_finished[input_ring]);
                    tma::expect_bytes(gemm_inputs_arrived[input_ring], config::CLUSTER_SIZE * (sizeof(a_tile) + sizeof(b_tile)));
                    wait(gemm_inputs_arrived[input_ring], get_phasebit<0>(gemm_bitfield, input_ring));
                    if (idx == 0) mm2_ABt (d_tt, a_smem[input_ring], b_smem[input_ring],
                                           a_sc_tt.template subtile<full_tt_fp8e8m0<16>>(input_ring * 16),
                                           b_sc_tt.template subtile<full_tt_fp8e8m0<32>>(input_ring * 32),
                                           gemm_inputs_finished[input_ring]);
                    else          mma2_ABt(d_tt, a_smem[input_ring], b_smem[input_ring],
                                           a_sc_tt.template subtile<full_tt_fp8e8m0<16>>(input_ring * 16),
                                           b_sc_tt.template subtile<full_tt_fp8e8m0<32>>(input_ring * 32),
                                           gemm_inputs_finished[input_ring]);
                } else {
                    tma::expect_bytes(gemm_inputs_arrived[input_ring], config::CLUSTER_SIZE * (sizeof(a_tile) + sizeof(b_tile)));
                    wait(gemm_inputs_arrived[input_ring], get_phasebit<0>(gemm_bitfield, input_ring));
                    if constexpr (IS_WGRAD) {
                        if (idx == 0) mm2_AtB (d_tt, a_smem[input_ring], b_smem[input_ring], gemm_inputs_finished[input_ring]);
                        else          mma2_AtB(d_tt, a_smem[input_ring], b_smem[input_ring], gemm_inputs_finished[input_ring]);
                    } else if constexpr (IS_AB) {
                        if (idx == 0) mm2_AB (d_tt, a_smem[input_ring], b_smem[input_ring], gemm_inputs_finished[input_ring]);
                        else          mma2_AB(d_tt, a_smem[input_ring], b_smem[input_ring], gemm_inputs_finished[input_ring]);
                    } else {
                        if (idx == 0) mm2_ABt (d_tt, a_smem[input_ring], b_smem[input_ring], gemm_inputs_finished[input_ring]);
                        else          mma2_ABt(d_tt, a_smem[input_ring], b_smem[input_ring], gemm_inputs_finished[input_ring]);
                    }
                }
                update_phasebit<0>(gemm_bitfield, input_ring);
                input_ring = ring_advance<config::MLP_LOAD_PIPE_DEPTH>(input_ring);
            }
            detail::tcgen05::commit<config::CLUSTER_SIZE>(gemm_outputs_arrived);
        }
    } else {
        using epilogue_group = group<WARPGROUP_WARPS>;
        wait(gemm_outputs_arrived, get_phasebit<0>(gemm_bitfield, config::MLP_LOAD_PIPE_DEPTH));
        update_phasebit<0>(gemm_bitfield, config::MLP_LOAD_PIPE_DEPTH);
        auto store_bf16 = [&]() {
            rt_bf<config::MLP_Mb / 8, config::MLP_Nb / config::MLP_EPI_PIPE_DEPTH> d_reg[config::MLP_EPI_PIPE_DEPTH];
            #pragma unroll
            for (int i = 0; i < config::MLP_EPI_PIPE_DEPTH; ++i)
                warpgroup::load_async(d_reg[i], d_tt.template subtile<tt<float, config::MLP_Mb / 2, config::MLP_Nb / config::MLP_EPI_PIPE_DEPTH>>(0, config::MLP_Nb / config::MLP_EPI_PIPE_DEPTH * i));
            tensor_load_wait();
            warpgroup::sync(1);
            warpgroup::tma::cluster::arrive(gemm_outputs_finished, 0);
            if (output_row_ready != nullptr && epilogue_group::laneid() == 0) {
                const int previous_macrobatch_offset = (macrobatch_idx + 1) * macrobatch_size;
                const int row_idx = tile_coord.x * config::MLP_Mb + cta_rank * (config::MLP_Mb / config::CLUSTER_SIZE);
                const int previous_macrobatch_tokens = min(macrobatch_size, num_tokens - previous_macrobatch_offset);
                const int required_count = (config::MLP_Mb / config::CLUSTER_SIZE / config::COMBINE_Mb) * ((d_gmem.cols() + config::COMBINE_Nb - 1) / config::COMBINE_Nb);
                if (row_idx < previous_macrobatch_tokens)
                    barrier_wait(*output_row_ready, (previous_macrobatch_offset + row_idx) / (config::MLP_Mb / config::CLUSTER_SIZE), required_count);
            }
            #pragma unroll
            for (int i = 0; i < config::MLP_EPI_PIPE_DEPTH; ++i) {
                warpgroup::tma::store_async_read_wait<config::MLP_NUM_BF16_D_TILES - 1>();
                warpgroup::sync(1);
                warpgroup::store(d_bf16_smem[i % config::MLP_NUM_BF16_D_TILES], d_reg[i]);
                warpgroup::sync(1);
                if constexpr (IS_WGRAD) {
                    if (is_first_wgrad_contribution)
                        warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(d_gmem, d_bf16_smem[i % config::MLP_NUM_BF16_D_TILES], {tile_coord.z, 2 * tile_coord.x + cta_rank, config::MLP_EPI_PIPE_DEPTH * tile_coord.y + i});
                    else
                        // Macrobatches are serialized by routed_buffers_done, so additions occur in a fixed order, preserving determinism
                        warpgroup::tma::store_add_async<dim::ROW, cache_policy::EVICT_FIRST>(d_gmem, d_bf16_smem[i % config::MLP_NUM_BF16_D_TILES], {tile_coord.z, 2 * tile_coord.x + cta_rank, config::MLP_EPI_PIPE_DEPTH * tile_coord.y + i});
                } else {
                    warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(d_gmem, d_bf16_smem[i % config::MLP_NUM_BF16_D_TILES], {2 * tile_coord.x + cta_rank, config::MLP_EPI_PIPE_DEPTH * tile_coord.y + i});
                }
            }
            warpgroup::tma::store_async_read_wait();
        };
        if constexpr (USE_ROUTED_MXFP8) {
          if (d_routed_gmem != nullptr) {
            constexpr int NUM_MXFP8_BLOCKS = config::MLP_Nb / 32;
            const int tile_row = warpgroup::laneid();
            uint32_t scale_word = 0;
            #pragma unroll 1
            for (int i = 0; i < NUM_MXFP8_BLOCKS; ++i) {
                float2 tmp[16];
                asm volatile(R"(
                    tcgen05.ld.sync.aligned.32x32b.x32.b32
                    {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,
                     %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];
                    )"
                    : "=f"(tmp[0].x), "=f"(tmp[0].y), "=f"(tmp[1].x), "=f"(tmp[1].y),
                      "=f"(tmp[2].x), "=f"(tmp[2].y), "=f"(tmp[3].x), "=f"(tmp[3].y),
                      "=f"(tmp[4].x), "=f"(tmp[4].y), "=f"(tmp[5].x), "=f"(tmp[5].y),
                      "=f"(tmp[6].x), "=f"(tmp[6].y), "=f"(tmp[7].x), "=f"(tmp[7].y),
                      "=f"(tmp[8].x), "=f"(tmp[8].y), "=f"(tmp[9].x), "=f"(tmp[9].y),
                      "=f"(tmp[10].x), "=f"(tmp[10].y), "=f"(tmp[11].x), "=f"(tmp[11].y),
                      "=f"(tmp[12].x), "=f"(tmp[12].y), "=f"(tmp[13].x), "=f"(tmp[13].y),
                      "=f"(tmp[14].x), "=f"(tmp[14].y), "=f"(tmp[15].x), "=f"(tmp[15].y)
                    : "r"(d_tt.addr + ((warpgroup::warpid() * 32) << 16) + i * 32));
                tensor_load_wait();
                bf16_2 d_reg[16];
                #pragma unroll
                for (int j = 0; j < 16; ++j)
                    d_reg[j] = __float22bfloat162_rn(tmp[j]);
                warpgroup::sync(1);
                const uint32_t d_bf16_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&d_bf16_smem[i % 2]));
                const uint32_t *d_bf16_words = reinterpret_cast<const uint32_t *>(d_reg);
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    move<float4>::sts(mlp_bf16_d_tile::idx(d_bf16_addr, {tile_row, j * 8}),
                        float4{__uint_as_float(d_bf16_words[j * 4]), __uint_as_float(d_bf16_words[j * 4 + 1]),
                               __uint_as_float(d_bf16_words[j * 4 + 2]), __uint_as_float(d_bf16_words[j * 4 + 3])});
                warpgroup::sync(1);
                warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(d_gmem, d_bf16_smem[i % 2], {2 * tile_coord.x + cta_rank, NUM_MXFP8_BLOCKS * tile_coord.y + i});

                // Wait for previous iteration's FP8 store to complete
                warpgroup::tma::store_async_read_wait<1>();
                warpgroup::sync(1);
                uint32_t d_fp8[8];
                uint32_t d_scale_byte;
                mxfp8_quantize::mxfp8_quantize_single_block(d_reg, d_fp8, d_scale_byte);
                scale_word |= d_scale_byte << ((i % 4) * 8);
                const uint32_t d_fp8_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&d_fp8_smem));
                #pragma unroll
                for (int m = 0; m < 2; ++m) {
                    move<float4>::sts(mlp_fp8_d_tile::idx(d_fp8_addr, {tile_row, m * 16}),
                        float4{__uint_as_float(d_fp8[m * 4]), __uint_as_float(d_fp8[m * 4 + 1]), __uint_as_float(d_fp8[m * 4 + 2]), __uint_as_float(d_fp8[m * 4 + 3])});
                }
                if (i % 4 == 3) {
                    move<int>::sts(static_cast<uint32_t>(__cvta_generic_to_shared(&d_sc_smem[i / 4])) + (tile_row % 32) * 16 + (tile_row / 32) * 4, std::bit_cast<int>(scale_word));
                    scale_word = 0;
                }
                warpgroup::sync(1);
                if (warpgroup::laneid() == 0) {
                    tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(*d_routed_gmem, d_fp8_smem, {2 * tile_coord.x + cta_rank, NUM_MXFP8_BLOCKS * tile_coord.y + i});
                    if (i % 4 == 3)
                        tma::store_async(*d_sc_gmem, d_sc_smem[i / 4], {2 * tile_coord.x + cta_rank, 2 * tile_coord.y + i / 4, 0, 0});
                }
            }
            tensor_before_thread_sync();
            warpgroup::sync(1);
            warpgroup::tma::cluster::arrive(gemm_outputs_finished, 0);
          } else {
            store_bf16();
          }
        } else {
            store_bf16();
        }
        epilogue_group::sync(4);
        if (epilogue_group::warpid() == 0 && warp::elect_leader()) {
            if (output_tile_ready != nullptr) {
                tma::store_async_wait();
                barrier_arrive(*output_tile_ready, output_tile_ready_base_index + (macrobatch_row_block_offset + tile_coord.x) * col_blocks + tile_coord.y);
            } else if (output_minibatch_ready != nullptr) {
                tma::store_async_wait();
                barrier_arrive(*output_minibatch_ready, global_minibatch_idx);
            }
            if (buffer_done != nullptr) {
                tma::store_async_wait();
                barrier_arrive(*buffer_done, buffer_done_index);
            }
        }
    }
}

static __device__ __forceinline__ void dispatch_mlp_swiglu_combine_fwd_kernel(const globals_fwd &g) {
    int cluster_idx = clusterIdx().x;
    const int cta_rank = cluster_ctarank();
    const int shared_row_blocks = g.x_shared.rows() / config::MLP_Mb;
    const int minibatch_routed_row_blocks = g.minibatch_size / config::MLP_Mb;
    const int shared_gate_up_tasks = shared_row_blocks * (g.w_shared_gate.rows() / config::MLP_Nb);
    const int minibatch_routed_gate_up_tasks = minibatch_routed_row_blocks * (g.w_routed_gate.rows() / config::MLP_Nb);
    const int shared_swiglu_tiles = (g.hidden_shared.rows() / config::SWIGLU_Mb) * (g.hidden_shared.cols() / config::SWIGLU_Nb);
    const int minibatch_routed_swiglu_tiles = (g.minibatch_size / config::SWIGLU_Mb) * (g.hidden_fp8_routed.cols() / config::SWIGLU_Nb);
    const int shared_swiglu_tasks = (shared_swiglu_tiles + config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH);
    const int minibatch_routed_swiglu_tasks = (minibatch_routed_swiglu_tiles + config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH);
    const int shared_down_tasks = shared_row_blocks * (g.w_shared_down.rows() / config::MLP_Nb);
    const int minibatch_routed_down_tasks = minibatch_routed_row_blocks * (g.w_routed_down.rows() / config::MLP_Nb);
    const int shared_tasks = 2 * shared_gate_up_tasks + shared_swiglu_tasks + shared_down_tasks;
    const int minibatch_tasks = 2 * minibatch_routed_gate_up_tasks + minibatch_routed_swiglu_tasks + minibatch_routed_down_tasks;
    const int comm_clusters = g.num_comm_sms / config::CLUSTER_SIZE;
    const int macrobatch_size = g.macrobatch_size;

    const int num_tokens = g.num_tokens[{0}];
    const int num_macrobatches = (num_tokens + macrobatch_size - 1) / macrobatch_size;
    const int minibatches_per_macrobatch = macrobatch_size / g.minibatch_size;
    const int true_num_global_minibatches = (num_tokens + g.minibatch_size - 1) / g.minibatch_size;
    const int last_macrobatch_num_minibatches = true_num_global_minibatches - (num_macrobatches - 1) * minibatches_per_macrobatch;
    const int true_num_clusters = comm_clusters + shared_tasks + true_num_global_minibatches * minibatch_tasks;
    if (cluster_idx >= true_num_clusters) return;

    warpgroup::increase_registers<256>();

    extern __shared__ int __shm[];
    const uint64_t smem_base_addr = (reinterpret_cast<uint64_t>(&__shm[0]) + 1023) & ~uint64_t(1023);

    uint32_t gemm_bitfield = 0xFFFF0000;
    uint32_t swiglu_bitfield = 0xFFFF0000;
    uint32_t dispatch_bitfield = 0xFFFF0000;
    uint32_t combine_bitfield = 0xFFFF0000;

    __shared__ clc::handle clc_handle[config::CLC_PIPE_DEPTH];
    __shared__ clc::handle clc_drain_handle[config::CLC_DRAIN_PIPE_DEPTH];
    __shared__ semaphore schedule_arrived[config::CLC_PIPE_DEPTH], schedule_finished[config::CLC_PIPE_DEPTH];
    __shared__ semaphore drain_schedule_arrived[config::CLC_DRAIN_PIPE_DEPTH];
    __shared__ semaphore drain_schedule_finished[config::CLC_DRAIN_PIPE_DEPTH];
    __shared__ semaphore swiglu_inputs_arrived[config::SWIGLU_FWD_PIPE_DEPTH];
    __shared__ semaphore gemm_inputs_arrived[config::MLP_LOAD_PIPE_DEPTH];
    __shared__ semaphore gemm_scales_arrived[config::MLP_LOAD_PIPE_DEPTH];
    __shared__ semaphore gemm_inputs_finished[config::MLP_LOAD_PIPE_DEPTH];
    __shared__ semaphore gemm_scales_finished[config::MLP_LOAD_PIPE_DEPTH];
    __shared__ semaphore gemm_outputs_arrived, gemm_outputs_finished;
    __shared__ semaphore dispatch_inputs_arrived;
    __shared__ semaphore combine_inputs_arrived[config::COMBINE_PIPE_DEPTH];

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < config::SWIGLU_FWD_PIPE_DEPTH; ++i) {
            init_semaphore(swiglu_inputs_arrived[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < config::MLP_LOAD_PIPE_DEPTH; ++i) {
            init_semaphore(gemm_inputs_arrived[i], 0, 1);
            init_semaphore(gemm_scales_arrived[i], 0, 1);
            init_semaphore(gemm_inputs_finished[i], 0, 1);
            init_semaphore(gemm_scales_finished[i], 0, 1);
        }
        init_semaphore(gemm_outputs_arrived, 0, 1);
        init_semaphore(gemm_outputs_finished, 0, config::CLUSTER_SIZE);
        #pragma unroll
        for (int i = 0; i < config::CLC_PIPE_DEPTH; ++i) {
            init_semaphore(schedule_arrived[i], 0, 1);
            init_semaphore(schedule_finished[i], 0, config::CLUSTER_SIZE * config::NUM_WARPS);
        }
        #pragma unroll
        for (int i = 0; i < config::CLC_DRAIN_PIPE_DEPTH; ++i) {
            init_semaphore(drain_schedule_arrived[i], 0, 1);
            init_semaphore(drain_schedule_finished[i], 0, config::CLUSTER_SIZE);
        }
        init_semaphore(dispatch_inputs_arrived, 0, 1);
        #pragma unroll
        for (int i = 0; i < config::COMBINE_PIPE_DEPTH; ++i) {
            init_semaphore(combine_inputs_arrived[i], 0, 1);
        }
    }

    tensor_allocator<1, config::CLUSTER_SIZE> tm_alloc{};
    tt<float, config::MLP_Mb / 2, config::MLP_Nb> d_tt = tm_alloc.template allocate<tt<float, config::MLP_Mb / 2, config::MLP_Nb>>(0);
    full_tt_fp8e8m0<16 * config::MLP_LOAD_PIPE_DEPTH> a_sc_tt = tm_alloc.template allocate<full_tt_fp8e8m0<16 * config::MLP_LOAD_PIPE_DEPTH>>(256);
    full_tt_fp8e8m0<32 * config::MLP_LOAD_PIPE_DEPTH> b_sc_tt = tm_alloc.template allocate<full_tt_fp8e8m0<32 * config::MLP_LOAD_PIPE_DEPTH>>(384);
    everyone::tma::cluster::sync();

    if (cluster_idx < comm_clusters) {
        const int comm_cta_idx = cluster_idx * config::CLUSTER_SIZE + cta_rank;
        auto num_dispatch_tasks = [&](int macrobatch_idx) {
            const int macrobatch_tokens = min(macrobatch_size, num_tokens - macrobatch_idx * macrobatch_size);
            const int dispatch_col_blocks = (g.x_fp8_routed.cols() + config::DISPATCH_Nb - 1) / config::DISPATCH_Nb;
            return (macrobatch_tokens / config::DISPATCH_Mb) * dispatch_col_blocks;
        };
        auto num_combine_tasks = [&](int macrobatch_idx) {
            const int macrobatch_tokens = min(macrobatch_size, num_tokens - macrobatch_idx * macrobatch_size);
            const int combine_col_blocks = (g.y_routed.cols() + config::COMBINE_Nb - 1) / config::COMBINE_Nb;
            const int combine_tiles = (macrobatch_tokens / config::COMBINE_Mb) * combine_col_blocks;
            return (combine_tiles + config::COMBINE_PIPE_DEPTH - 1) / config::COMBINE_PIPE_DEPTH;
        };
        auto dispatch = [&](int macrobatch_idx, int task_idx) {
            dispatch_kernel<false>(g.x_routed_send_buffer, g.x_fp8_routed, &g.x_sc_routed, &g.x_fp8_t_routed, &g.x_sc_t_routed,
                                     nullptr, g.schedule_peer_rank, g.schedule_peer_token_idx,
                                     macrobatch_idx + 1 < num_macrobatches ? &g.y_routed_ready : nullptr, nullptr, g.x_routed_ready,
                                     dispatch_inputs_arrived, dispatch_bitfield,
                                     num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, task_idx, g.topk,
                                     macrobatch_idx + 1, 0, smem_base_addr);
        };
        auto combine = [&](int macrobatch_idx, int task_idx) {
            combine_kernel<true>(g.y_routed_recv_buffer, g.y_routed, nullptr, nullptr,
                                 g.schedule_peer_rank, g.schedule_peer_token_idx,
                                 g.y_routed_ready, macrobatch_idx > 0 ? &g.y_routed_done : nullptr,
                                 combine_inputs_arrived, combine_bitfield,
                                 num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, task_idx, smem_base_addr);
        };
        if (num_macrobatches == 0) return;
        for (int task_idx = comm_cta_idx; task_idx < num_dispatch_tasks(num_macrobatches - 1); task_idx += g.num_comm_sms)
            dispatch(num_macrobatches - 1, task_idx);
        for (int macrobatch_idx = num_macrobatches - 1; macrobatch_idx >= 0; --macrobatch_idx) {
            const int combine_tasks = num_combine_tasks(macrobatch_idx);
            const int dispatch_tasks = macrobatch_idx > 0 ? num_dispatch_tasks(macrobatch_idx - 1) : 0;
            for (int task_idx = comm_cta_idx; task_idx < max(combine_tasks, dispatch_tasks); task_idx += g.num_comm_sms) {
                if (task_idx < combine_tasks)
                    combine(macrobatch_idx, task_idx);
                if (task_idx < dispatch_tasks)
                    dispatch(macrobatch_idx - 1, task_idx);
            }
        }
        return;
    }

    // Swiglu tasks are CTA-local, GEMM is not
    auto is_cta_local_task = [&](int compute_cluster_idx) {
        const int minibatch_task_idx = (compute_cluster_idx - shared_tasks) % minibatch_tasks;
        if (compute_cluster_idx < 0) return false;
        else if (compute_cluster_idx < 2 * shared_gate_up_tasks) return false; // shared gate/up
        else if (compute_cluster_idx < 2 * shared_gate_up_tasks + shared_swiglu_tasks) return true; // shared swiglu
        else if (compute_cluster_idx < shared_tasks) return false; // shared down
        else if (minibatch_task_idx < 2 * minibatch_routed_gate_up_tasks) return false; // routed gate/up
        else if (minibatch_task_idx < 2 * minibatch_routed_gate_up_tasks + minibatch_routed_swiglu_tasks) return true; // routed swiglu
        else return false; // routed down
    };
    const int hidden_row_block_ready_required_count = (config::MLP_Mb / config::SWIGLU_Mb) * (g.hidden_shared.cols() / config::SWIGLU_Nb);

    for (int task_iter = 0; cluster_idx >= 0 && cluster_idx < true_num_clusters; ++task_iter) {
        const int clc_stage = task_iter % config::CLC_PIPE_DEPTH;
        if (warpgroup::groupid() == config::NUM_CONSUMERS && warpgroup::warpid() == 1 && warp::elect_leader()) { // warp not used by the gemms
            if (cta_rank == 0) {
                wait(schedule_finished[clc_stage], ((task_iter + config::CLC_PIPE_DEPTH) / config::CLC_PIPE_DEPTH) % 2);
                clc::schedule(clc_handle[clc_stage], schedule_arrived[clc_stage]);
            }
            tma::expect_bytes(schedule_arrived[clc_stage], sizeof(clc_handle[clc_stage]));
        }

        const int compute_cluster_idx = cluster_idx - comm_clusters;
        const bool current_is_cta_local = is_cta_local_task(compute_cluster_idx);

        if (compute_cluster_idx < shared_gate_up_tasks) {
            // Shared gate (BF16)
            const int task_idx = compute_cluster_idx;
            expert_grouped_gemm_kernel<true>(g.x_shared, g.w_shared_gate, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                      g.gate_shared, nullptr, nullptr,
                                      g.tokens_per_expert, nullptr, nullptr, nullptr, &g.gate_up_tile_ready, nullptr, nullptr,
                                      d_tt, a_sc_tt, b_sc_tt,
                                      gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                      num_tokens, macrobatch_size, g.minibatch_size, 0, 0, task_idx, cta_rank,
                                      0, 0, 0, 0, 0, smem_base_addr);
        } else if (compute_cluster_idx < shared_gate_up_tasks * 2) {
            // Shared up (BF16)
            const int task_idx = compute_cluster_idx - shared_gate_up_tasks;
            expert_grouped_gemm_kernel<true>(g.x_shared, g.w_shared_up, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                      g.up_shared, nullptr, nullptr,
                                      g.tokens_per_expert, nullptr, nullptr, nullptr, &g.gate_up_tile_ready, nullptr, nullptr,
                                      d_tt, a_sc_tt, b_sc_tt,
                                      gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                      num_tokens, macrobatch_size, g.minibatch_size, 0, 0, task_idx, cta_rank,
                                      0, 0, 0, 0, 0, smem_base_addr);
        } else if (compute_cluster_idx < shared_gate_up_tasks * 2 + shared_swiglu_tasks) {
            // Shared Swiglu (BF16)
            const int task_idx = compute_cluster_idx - shared_gate_up_tasks * 2;
            swiglu_fwd_kernel<true>(g.gate_shared, g.up_shared, g.hidden_shared, nullptr, nullptr, nullptr,
                             g.gate_up_tile_ready, g.hidden_row_block_ready,
                             swiglu_inputs_arrived, swiglu_bitfield,
                             g.x_shared.rows(), macrobatch_size, g.minibatch_size, 0, 0, task_idx, cta_rank, 0, 0, smem_base_addr);
        } else if (compute_cluster_idx < shared_tasks) {
            // Shared down (BF16)
            const int task_idx = compute_cluster_idx - shared_gate_up_tasks * 2 - shared_swiglu_tasks;
            expert_grouped_gemm_kernel<true>(g.hidden_shared, g.w_shared_down, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                      g.y_shared, nullptr, nullptr,
                                      g.tokens_per_expert, nullptr, &g.hidden_row_block_ready, nullptr, nullptr, nullptr, nullptr,
                                      d_tt, a_sc_tt, b_sc_tt,
                                      gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                      num_tokens, macrobatch_size, g.minibatch_size, 0, 0, task_idx, cta_rank,
                                      0, 0, hidden_row_block_ready_required_count, 0, 0, smem_base_addr);
        } else {
            // Routed expert with macro/minibatching
            const int task_ordered_global_minibatch_idx = (compute_cluster_idx - shared_tasks) / minibatch_tasks;
            const int minibatch_task_idx = (compute_cluster_idx - shared_tasks) - task_ordered_global_minibatch_idx * minibatch_tasks;
            int macrobatch_idx, minibatch_idx;
            if (task_ordered_global_minibatch_idx < last_macrobatch_num_minibatches) {
                macrobatch_idx = num_macrobatches - 1;
                minibatch_idx = task_ordered_global_minibatch_idx;
            } else {
                const int idx = task_ordered_global_minibatch_idx - last_macrobatch_num_minibatches;
                macrobatch_idx = num_macrobatches - 2 - idx / minibatches_per_macrobatch;
                minibatch_idx = idx % minibatches_per_macrobatch;
            }

            if (minibatch_task_idx < minibatch_routed_gate_up_tasks) {
                // Routed gate
                const int task_idx = minibatch_task_idx;
                expert_grouped_gemm_kernel<false>(g.x_fp8_routed, g.w_routed_gate, &g.x_sc_routed, &g.w_routed_gate_sc, nullptr, nullptr, nullptr, nullptr,
                                           g.gate_routed, &g.gate_fp8_routed, &g.gate_sc_routed,
                                           g.tokens_per_expert, &g.x_routed_ready, nullptr, nullptr, &g.gate_up_tile_ready, nullptr, nullptr,
                                           d_tt, a_sc_tt, b_sc_tt,
                                           gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                           num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, minibatch_idx, task_idx, cta_rank,
                                           0, 0, 0, shared_gate_up_tasks, 0, smem_base_addr);
            } else if (minibatch_task_idx < minibatch_routed_gate_up_tasks * 2) {
                // Routed up
                const int task_idx = minibatch_task_idx - minibatch_routed_gate_up_tasks;
                expert_grouped_gemm_kernel<false>(g.x_fp8_routed, g.w_routed_up, &g.x_sc_routed, &g.w_routed_up_sc, nullptr, nullptr, nullptr, nullptr,
                                           g.up_routed, &g.up_fp8_routed, &g.up_sc_routed,
                                           g.tokens_per_expert, &g.x_routed_ready, nullptr, nullptr, &g.gate_up_tile_ready, nullptr, nullptr,
                                           d_tt, a_sc_tt, b_sc_tt,
                                           gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                           num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, minibatch_idx, task_idx, cta_rank,
                                           0, 0, 0, shared_gate_up_tasks, 0, smem_base_addr);
            } else if (minibatch_task_idx < minibatch_routed_gate_up_tasks * 2 + minibatch_routed_swiglu_tasks) {
                // Routed Swiglu
                const int task_idx = minibatch_task_idx - minibatch_routed_gate_up_tasks * 2;
                swiglu_fwd_kernel<false>(g.gate_routed, g.up_routed, g.hidden_fp8_routed,
                                  &g.hidden_sc_routed, &g.hidden_fp8_t_routed, &g.hidden_sc_t_routed,
                                  g.gate_up_tile_ready, g.hidden_row_block_ready,
                                  swiglu_inputs_arrived, swiglu_bitfield,
                                  num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, minibatch_idx, task_idx, cta_rank,
                                  shared_gate_up_tasks, shared_row_blocks, smem_base_addr);
            } else {
                // Routed down
                const int task_idx = minibatch_task_idx - minibatch_routed_gate_up_tasks * 2 - minibatch_routed_swiglu_tasks;
                expert_grouped_gemm_kernel<false>(g.hidden_fp8_routed, g.w_routed_down, &g.hidden_sc_routed, &g.w_routed_down_sc, nullptr, nullptr, nullptr, nullptr,
                                           g.y_routed, nullptr, nullptr,
                                           g.tokens_per_expert, nullptr, &g.hidden_row_block_ready,
                                           macrobatch_idx + 1 < num_macrobatches ? &g.y_routed_done : nullptr,
                                           nullptr, &g.y_routed_ready, nullptr,
                                           d_tt, a_sc_tt, b_sc_tt,
                                           gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                           num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, minibatch_idx, task_idx, cta_rank,
                                           0, shared_row_blocks, hidden_row_block_ready_required_count, 0, 0, smem_base_addr);
            }
        }

        wait(schedule_arrived[clc_stage], (task_iter / config::CLC_PIPE_DEPTH) % 2);
        const auto schedule = clc::query(clc_handle[clc_stage]);
        cluster_idx = schedule.success ? static_cast<int>(schedule.x / config::CLUSTER_SIZE) : -1;
        __syncwarp();
        warp::tma::cluster::arrive(schedule_finished[clc_stage], 0);

        // SWIGLU -> GEMM requires a cluster-wide sync
        const int next_compute_cluster_idx = cluster_idx - comm_clusters;
        if (current_is_cta_local && cluster_idx >= 0 && !is_cta_local_task(next_compute_cluster_idx))
            everyone::tma::cluster::sync();
    }

    everyone::tma::cluster::sync();

    // CLC drain for no-op threadblocks
    if (cluster_idx >= 0 && warp::laneid() == 0) {
        const int stage = warpid();
        int iter = 0;
        if (cta_rank == 0)
            clc::schedule(clc_drain_handle[stage], drain_schedule_arrived[stage]);
        tma::expect_bytes(drain_schedule_arrived[stage], sizeof(clc::handle));
        while (true) {
            wait(drain_schedule_arrived[stage], iter % 2);
            const auto schedule = clc::query(clc_drain_handle[stage]);
            warp::tma::cluster::arrive(drain_schedule_finished[stage], 0);
            if (cta_rank == 0)
                wait(drain_schedule_finished[stage], iter % 2);
            if (!schedule.success)
                break;
            if (cta_rank == 0)
                clc::schedule(clc_drain_handle[stage], drain_schedule_arrived[stage]);
            tma::expect_bytes(drain_schedule_arrived[stage], sizeof(clc::handle));
            ++iter;
        }
    }
}

static __host__ __forceinline__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                           at::Tensor, at::Tensor, at::Tensor>
dispatch_mlp_swiglu_combine_fwd_mxfp8(
    // Inputs and communication buffers
    const at::Tensor &x,
    const std::vector<int64_t> &x_ptrs,
    const at::Tensor &combine_buffer,
    const std::vector<int64_t> &combine_buffer_ptrs,

    // Weights
    const at::Tensor &w_shared_gate,
    const at::Tensor &w_routed_gate,
    const at::Tensor &w_routed_gate_sc,
    const at::Tensor &w_shared_up,
    const at::Tensor &w_routed_up,
    const at::Tensor &w_routed_up_sc,
    const at::Tensor &w_shared_down,
    const at::Tensor &w_routed_down,
    const at::Tensor &w_routed_down_sc,

    // Dispatch/combine schedule
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens,
    const at::Tensor &tokens_per_expert,

    // Metadata
    int topk,
    int num_comm_sms,
    int macrobatch_size,
    int minibatch_size
) {
    const int num_local_tokens = x.size(0);
    const int schedule_capacity = schedule_peer_rank.size(0);
    const int hidden_dim = x.size(1);
    const int intermediate_dim = w_shared_gate.size(0);
    const int num_global_minibatches = (schedule_capacity + minibatch_size - 1) / minibatch_size;
    const int num_global_row_blocks = schedule_capacity / (config::MLP_Mb / config::CLUSTER_SIZE);
    const int shared_row_blocks = num_local_tokens / config::MLP_Mb;
    const int routed_row_blocks = schedule_capacity / config::MLP_Mb;
    const int shared_gate_up_tasks = shared_row_blocks * (w_shared_gate.size(0) / config::MLP_Nb);
    const int routed_gate_up_tasks = routed_row_blocks * (w_routed_gate.size(1) / config::MLP_Nb);

    activation_bf16_pgl x_routed_send_buffer_data;
    activation_bf16_pgl y_routed_recv_buffer_data;
    for (int i = 0; i < NUM_DEVICES; ++i) {
        x_routed_send_buffer_data[i] = reinterpret_cast<bf16*>(x_ptrs[i]);
        y_routed_recv_buffer_data[i] = reinterpret_cast<bf16*>(combine_buffer_ptrs[i]);
    }

    at::Tensor x_fp8_routed = at::empty({macrobatch_size, hidden_dim}, x.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor x_sc_routed = at::empty({macrobatch_size / 128, hidden_dim / 128, 32, 16}, x.options().dtype(at::kByte));
    at::Tensor x_fp8_t_routed = at::empty({hidden_dim, macrobatch_size}, x.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor x_sc_t_routed = at::empty({hidden_dim / 128, macrobatch_size / 128, 32, 16}, x.options().dtype(at::kByte));
    at::Tensor gate_shared = at::empty({num_local_tokens, intermediate_dim}, x.options());
    at::Tensor gate_routed = at::empty({macrobatch_size, intermediate_dim}, x.options());
    at::Tensor gate_fp8_routed = at::empty({macrobatch_size, intermediate_dim}, x.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor gate_sc_routed = at::empty({macrobatch_size / 128, intermediate_dim / 128, 32, 16}, x.options().dtype(at::kByte));
    at::Tensor up_shared = at::empty({num_local_tokens, intermediate_dim}, x.options());
    at::Tensor up_routed = at::empty({macrobatch_size, intermediate_dim}, x.options());
    at::Tensor up_fp8_routed = at::empty({macrobatch_size, intermediate_dim}, x.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor up_sc_routed = at::empty({macrobatch_size / 128, intermediate_dim / 128, 32, 16}, x.options().dtype(at::kByte));
    at::Tensor hidden_shared = at::empty({num_local_tokens, intermediate_dim}, x.options());
    at::Tensor hidden_fp8_routed = at::empty({macrobatch_size, intermediate_dim}, x.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor hidden_sc_routed = at::empty({macrobatch_size / 128, intermediate_dim / 128, 32, 16}, x.options().dtype(at::kByte));
    at::Tensor hidden_fp8_t_routed = at::empty({intermediate_dim, macrobatch_size}, x.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor hidden_sc_t_routed = at::empty({intermediate_dim / 128, macrobatch_size / 128, 32, 16}, x.options().dtype(at::kByte));
    at::Tensor y_shared = at::empty_like(x);
    at::Tensor y_routed = at::empty({macrobatch_size, hidden_dim}, x.options());
    at::Tensor x_routed_ready = at::zeros({num_global_minibatches}, tokens_per_expert.options());
    at::Tensor gate_up_tile_ready = at::zeros({shared_gate_up_tasks + routed_gate_up_tasks}, tokens_per_expert.options());
    at::Tensor hidden_row_block_ready = at::zeros({shared_row_blocks + routed_row_blocks}, tokens_per_expert.options());
    at::Tensor y_routed_ready = at::zeros({num_global_minibatches}, tokens_per_expert.options());
    at::Tensor y_routed_done = at::zeros({num_global_row_blocks}, tokens_per_expert.options());

    globals_fwd g {
        .x_shared = kittens::py::tensor_to_gl<mlp_bf16_gl>(x),
        .x_fp8_routed = kittens::py::tensor_to_gl<mlp_fp8_gl>(x_fp8_routed),
        .x_sc_routed = kittens::py::tensor_to_gl<sc_gl>(x_sc_routed),
        .x_fp8_t_routed = kittens::py::tensor_to_gl<mlp_fp8_gl>(x_fp8_t_routed),
        .x_sc_t_routed = kittens::py::tensor_to_gl<sc_gl>(x_sc_t_routed),
        .gate_shared = kittens::py::tensor_to_gl<epi_bf16_gl>(gate_shared),
        .gate_routed = kittens::py::tensor_to_gl<epi_bf16_gl>(gate_routed),
        .gate_fp8_routed = kittens::py::tensor_to_gl<gate_up_fp8_gl>(gate_fp8_routed),
        .gate_sc_routed = kittens::py::tensor_to_gl<sc_gl>(gate_sc_routed),
        .up_shared = kittens::py::tensor_to_gl<epi_bf16_gl>(up_shared),
        .up_routed = kittens::py::tensor_to_gl<epi_bf16_gl>(up_routed),
        .up_fp8_routed = kittens::py::tensor_to_gl<gate_up_fp8_gl>(up_fp8_routed),
        .up_sc_routed = kittens::py::tensor_to_gl<sc_gl>(up_sc_routed),
        .hidden_shared = kittens::py::tensor_to_gl<mlp_bf16_gl>(hidden_shared),
        .hidden_fp8_routed = kittens::py::tensor_to_gl<mlp_fp8_gl>(hidden_fp8_routed),
        .hidden_sc_routed = kittens::py::tensor_to_gl<sc_gl>(hidden_sc_routed),
        .hidden_fp8_t_routed = kittens::py::tensor_to_gl<mlp_fp8_gl>(hidden_fp8_t_routed),
        .hidden_sc_t_routed = kittens::py::tensor_to_gl<sc_gl>(hidden_sc_t_routed),
        .y_shared = kittens::py::tensor_to_gl<epi_bf16_gl>(y_shared),
        .y_routed = kittens::py::tensor_to_gl<epi_bf16_gl>(y_routed),
        .x_routed_send_buffer = x_routed_send_buffer_data,
        .y_routed_recv_buffer = y_routed_recv_buffer_data,
        .w_shared_gate = kittens::py::tensor_to_gl<weight_bf16_gl>(w_shared_gate),
        .w_routed_gate = kittens::py::tensor_to_gl<weight_fp8_gl>(w_routed_gate),
        .w_routed_gate_sc = kittens::py::tensor_to_gl<sc_gl>(w_routed_gate_sc),
        .w_shared_up = kittens::py::tensor_to_gl<weight_bf16_gl>(w_shared_up),
        .w_routed_up = kittens::py::tensor_to_gl<weight_fp8_gl>(w_routed_up),
        .w_routed_up_sc = kittens::py::tensor_to_gl<sc_gl>(w_routed_up_sc),
        .w_shared_down = kittens::py::tensor_to_gl<weight_bf16_gl>(w_shared_down),
        .w_routed_down = kittens::py::tensor_to_gl<weight_fp8_gl>(w_routed_down),
        .w_routed_down_sc = kittens::py::tensor_to_gl<sc_gl>(w_routed_down_sc),
        .schedule_peer_rank = kittens::py::tensor_to_gl<index_gl>(schedule_peer_rank),
        .schedule_peer_token_idx = kittens::py::tensor_to_gl<index_gl>(schedule_peer_token_idx),
        .num_tokens = kittens::py::tensor_to_gl<index_gl>(num_tokens),
        .tokens_per_expert = kittens::py::tensor_to_gl<index_gl>(tokens_per_expert),
        .gate_up_tile_ready = kittens::py::tensor_to_gl<index_gl>(gate_up_tile_ready),
        .hidden_row_block_ready = kittens::py::tensor_to_gl<index_gl>(hidden_row_block_ready),
        .x_routed_ready = kittens::py::tensor_to_gl<index_gl>(x_routed_ready),
        .y_routed_ready = kittens::py::tensor_to_gl<index_gl>(y_routed_ready),
        .y_routed_done = kittens::py::tensor_to_gl<index_gl>(y_routed_done),
        .topk = topk,
        .num_comm_sms = num_comm_sms,
        .macrobatch_size = macrobatch_size,
        .minibatch_size = minibatch_size
    };

    kittens::py::launch_kernel<config, globals_fwd, dispatch_mlp_swiglu_combine_fwd_kernel>(g);

    return {x_fp8_t_routed, x_sc_t_routed,
            gate_shared, gate_fp8_routed, gate_sc_routed,
            up_shared, up_fp8_routed, up_sc_routed,
            hidden_shared, hidden_fp8_t_routed, hidden_sc_t_routed,
            y_shared, y_routed};
}

static __host__ __forceinline__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                           at::Tensor, at::Tensor, at::Tensor, at::Tensor>
dispatch_mlp_swiglu_combine_fwd_bf16(
    const at::Tensor &x,
    const std::vector<int64_t> &x_ptrs,
    const at::Tensor &combine_buffer,
    const std::vector<int64_t> &combine_buffer_ptrs,
    const at::Tensor &w_shared_gate,
    const at::Tensor &w_routed_gate,
    const at::Tensor &w_shared_up,
    const at::Tensor &w_routed_up,
    const at::Tensor &w_shared_down,
    const at::Tensor &w_routed_down,
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens,
    const at::Tensor &tokens_per_expert,
    int topk,
    int num_comm_sms,
    int macrobatch_size,
    int minibatch_size
) {
    static_assert(!USE_MXFP8);
    const int num_local_tokens = x.size(0);
    const int schedule_capacity = schedule_peer_rank.size(0);
    const int hidden_dim = x.size(1);
    const int intermediate_dim = w_shared_gate.size(0);
    const int num_global_minibatches = (schedule_capacity + minibatch_size - 1) / minibatch_size;
    const int num_global_row_blocks = schedule_capacity / (config::MLP_Mb / config::CLUSTER_SIZE);
    const int shared_row_blocks = num_local_tokens / config::MLP_Mb;
    const int routed_row_blocks = schedule_capacity / config::MLP_Mb;
    const int shared_gate_up_tasks = shared_row_blocks * (intermediate_dim / config::MLP_Nb);
    const int routed_gate_up_tasks = routed_row_blocks * (intermediate_dim / config::MLP_Nb);

    activation_bf16_pgl x_routed_send_buffer_data;
    activation_bf16_pgl y_routed_recv_buffer_data;
    for (int i = 0; i < NUM_DEVICES; ++i) {
        x_routed_send_buffer_data[i] = reinterpret_cast<bf16*>(x_ptrs[i]);
        y_routed_recv_buffer_data[i] = reinterpret_cast<bf16*>(combine_buffer_ptrs[i]);
    }

    at::Tensor x_routed = at::empty({macrobatch_size, hidden_dim}, x.options());
    at::Tensor gate_shared = at::empty({num_local_tokens, intermediate_dim}, x.options());
    at::Tensor gate_routed = at::empty({macrobatch_size, intermediate_dim}, x.options());
    at::Tensor up_shared = at::empty({num_local_tokens, intermediate_dim}, x.options());
    at::Tensor up_routed = at::empty({macrobatch_size, intermediate_dim}, x.options());
    at::Tensor hidden_shared = at::empty({num_local_tokens, intermediate_dim}, x.options());
    at::Tensor hidden_routed = at::empty({macrobatch_size, intermediate_dim}, x.options());
    at::Tensor y_shared = at::empty_like(x);
    at::Tensor y_routed = at::empty_like(x_routed);
    at::Tensor x_routed_ready = at::zeros({num_global_minibatches}, tokens_per_expert.options());
    at::Tensor gate_up_tile_ready = at::zeros({shared_gate_up_tasks + routed_gate_up_tasks}, tokens_per_expert.options());
    at::Tensor hidden_row_block_ready = at::zeros({shared_row_blocks + routed_row_blocks}, tokens_per_expert.options());
    at::Tensor y_routed_ready = at::zeros({num_global_minibatches}, tokens_per_expert.options());
    at::Tensor y_routed_done = at::zeros({num_global_row_blocks}, tokens_per_expert.options());

    globals_fwd g {
        .x_shared = kittens::py::tensor_to_gl<mlp_bf16_gl>(x),
        .x_fp8_routed = kittens::py::tensor_to_gl<routed_activation_gl>(x_routed),
        .x_sc_routed = {},
        .x_fp8_t_routed = kittens::py::tensor_to_gl<routed_transposed_gl>(x_routed),
        .x_sc_t_routed = {},
        .gate_shared = kittens::py::tensor_to_gl<epi_bf16_gl>(gate_shared),
        .gate_routed = kittens::py::tensor_to_gl<epi_bf16_gl>(gate_routed),
        .gate_fp8_routed = kittens::py::tensor_to_gl<routed_gate_up_gl>(gate_routed),
        .gate_sc_routed = {},
        .up_shared = kittens::py::tensor_to_gl<epi_bf16_gl>(up_shared),
        .up_routed = kittens::py::tensor_to_gl<epi_bf16_gl>(up_routed),
        .up_fp8_routed = kittens::py::tensor_to_gl<routed_gate_up_gl>(up_routed),
        .up_sc_routed = {},
        .hidden_shared = kittens::py::tensor_to_gl<mlp_bf16_gl>(hidden_shared),
        .hidden_fp8_routed = kittens::py::tensor_to_gl<routed_activation_gl>(hidden_routed),
        .hidden_sc_routed = {},
        .hidden_fp8_t_routed = kittens::py::tensor_to_gl<routed_transposed_gl>(hidden_routed),
        .hidden_sc_t_routed = {},
        .y_shared = kittens::py::tensor_to_gl<epi_bf16_gl>(y_shared),
        .y_routed = kittens::py::tensor_to_gl<epi_bf16_gl>(y_routed),
        .x_routed_send_buffer = x_routed_send_buffer_data,
        .y_routed_recv_buffer = y_routed_recv_buffer_data,
        .w_shared_gate = kittens::py::tensor_to_gl<weight_bf16_gl>(w_shared_gate),
        .w_routed_gate = kittens::py::tensor_to_gl<routed_weight_gl>(w_routed_gate),
        .w_routed_gate_sc = {},
        .w_shared_up = kittens::py::tensor_to_gl<weight_bf16_gl>(w_shared_up),
        .w_routed_up = kittens::py::tensor_to_gl<routed_weight_gl>(w_routed_up),
        .w_routed_up_sc = {},
        .w_shared_down = kittens::py::tensor_to_gl<weight_bf16_gl>(w_shared_down),
        .w_routed_down = kittens::py::tensor_to_gl<routed_weight_gl>(w_routed_down),
        .w_routed_down_sc = {},
        .schedule_peer_rank = kittens::py::tensor_to_gl<index_gl>(schedule_peer_rank),
        .schedule_peer_token_idx = kittens::py::tensor_to_gl<index_gl>(schedule_peer_token_idx),
        .num_tokens = kittens::py::tensor_to_gl<index_gl>(num_tokens),
        .tokens_per_expert = kittens::py::tensor_to_gl<index_gl>(tokens_per_expert),
        .gate_up_tile_ready = kittens::py::tensor_to_gl<index_gl>(gate_up_tile_ready),
        .hidden_row_block_ready = kittens::py::tensor_to_gl<index_gl>(hidden_row_block_ready),
        .x_routed_ready = kittens::py::tensor_to_gl<index_gl>(x_routed_ready),
        .y_routed_ready = kittens::py::tensor_to_gl<index_gl>(y_routed_ready),
        .y_routed_done = kittens::py::tensor_to_gl<index_gl>(y_routed_done),
        .topk = topk,
        .num_comm_sms = num_comm_sms,
        .macrobatch_size = macrobatch_size,
        .minibatch_size = minibatch_size
    };

    kittens::py::launch_kernel<config, globals_fwd, dispatch_mlp_swiglu_combine_fwd_kernel>(g);
    return {x_routed, gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, y_shared, y_routed};
}

static __device__ __forceinline__ void dispatch_mlp_swiglu_combine_bwd_kernel(const globals_bwd &g) {
    const int num_local_experts = g.w_routed_gate.depth();
    const int intermediate_dim_col_blocks = g.hidden_shared.cols() / config::MLP_Nb;
    const int hidden_dim_col_blocks = g.d_y_shared.cols() / config::MLP_Nb;

    int cluster_idx = clusterIdx().x;
    const int cta_rank = cluster_ctarank();

    const int shared_row_blocks = g.d_y_shared.rows() / config::MLP_Mb;
    const int shared_dgrad_down_tasks = shared_row_blocks * intermediate_dim_col_blocks;
    const int shared_swiglu_bwd_tiles = (g.hidden_shared.rows() / config::SWIGLU_Mb) * (g.hidden_shared.cols() / config::SWIGLU_Nb);
    const int shared_swiglu_bwd_tasks = (shared_swiglu_bwd_tiles + config::CLUSTER_SIZE * config::SWIGLU_BWD_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_BWD_PIPE_DEPTH);
    const int shared_dgrad_gate_up_tasks = shared_row_blocks * hidden_dim_col_blocks;
    const int shared_wgrad_tasks = intermediate_dim_col_blocks * hidden_dim_col_blocks; // per weight matrix
    const int shared_tasks = shared_dgrad_down_tasks + shared_swiglu_bwd_tasks + shared_dgrad_gate_up_tasks + 3 * shared_wgrad_tasks;

    const int minibatch_routed_row_blocks = g.minibatch_size / config::MLP_Mb;
    const int minibatch_routed_dgrad_down_tasks = minibatch_routed_row_blocks * intermediate_dim_col_blocks;
    const int minibatch_routed_swiglu_tiles = (g.minibatch_size / config::SWIGLU_Mb) * (g.hidden_fp8_routed.cols() / config::SWIGLU_Nb);
    const int minibatch_routed_swiglu_bwd_tasks = (minibatch_routed_swiglu_tiles + config::CLUSTER_SIZE * config::SWIGLU_BWD_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_BWD_PIPE_DEPTH);
    const int minibatch_routed_dgrad_gate_up_tasks = minibatch_routed_row_blocks * hidden_dim_col_blocks;
    const int minibatch_routed_bwd_tasks = minibatch_routed_dgrad_down_tasks + minibatch_routed_swiglu_bwd_tasks + minibatch_routed_dgrad_gate_up_tasks;

    const int wgrad_matrix_tasks = num_local_experts * intermediate_dim_col_blocks * hidden_dim_col_blocks;
    const int wgrad_tasks = 3 * wgrad_matrix_tasks;

    const int minibatch_routed_gate_up_tasks = minibatch_routed_row_blocks * intermediate_dim_col_blocks;
    const int minibatch_routed_swiglu_fwd_tasks = (minibatch_routed_swiglu_tiles + config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH);
    const int minibatch_routed_replay_tasks = 2 * minibatch_routed_gate_up_tasks + minibatch_routed_swiglu_fwd_tasks;

    const int comm_clusters = g.num_comm_sms / config::CLUSTER_SIZE;
    const int macrobatch_size = g.macrobatch_size;
    const int num_tokens = g.num_tokens[{0}];
    const int num_macrobatches = (num_tokens + macrobatch_size - 1) / macrobatch_size;
    const int minibatches_per_macrobatch = macrobatch_size / g.minibatch_size;

    auto num_minibatches_of = [&](int macrobatch_idx) { return (min(num_tokens - macrobatch_idx * macrobatch_size, macrobatch_size) + g.minibatch_size - 1) / g.minibatch_size; };
    auto num_dispatch_tasks_of = [&](int macrobatch_idx) {
        const int macrobatch_tokens = min(macrobatch_size, num_tokens - macrobatch_idx * macrobatch_size);
        const int col_blocks = (g.d_y_shared.cols() + config::DISPATCH_Nb - 1) / config::DISPATCH_Nb;
        return (macrobatch_tokens / config::DISPATCH_Mb) * col_blocks;
    };
    auto num_combine_tasks_of = [&](int macrobatch_idx) {
        const int macrobatch_tokens = min(macrobatch_size, num_tokens - macrobatch_idx * macrobatch_size);
        const int col_blocks = (g.d_y_shared.cols() + config::COMBINE_Nb - 1) / config::COMBINE_Nb;
        const int tiles = (macrobatch_tokens / config::COMBINE_Mb) * col_blocks;
        return (tiles + config::COMBINE_PIPE_DEPTH - 1) / config::COMBINE_PIPE_DEPTH;
    };
    auto routed_buffers_done_required_count_of = [&](int macrobatch_idx) {
        return config::CLUSTER_SIZE * (num_minibatches_of(macrobatch_idx) * minibatch_routed_bwd_tasks + wgrad_tasks) + num_combine_tasks_of(macrobatch_idx);
    };

    const int num_minibatches = (num_tokens + g.minibatch_size - 1) / g.minibatch_size;
    const int saved_macrobatch_num_minibatches = num_minibatches_of(0);
    const int saved_macrobatch_tasks = saved_macrobatch_num_minibatches * minibatch_routed_bwd_tasks + wgrad_tasks;
    const int replayed_macrobatch_tasks = minibatches_per_macrobatch * (minibatch_routed_replay_tasks + minibatch_routed_bwd_tasks) + wgrad_tasks;
    const int num_replay_minibatches = num_minibatches - saved_macrobatch_num_minibatches;
    const int true_num_clusters = comm_clusters + shared_tasks + num_minibatches * minibatch_routed_bwd_tasks +
                                  num_replay_minibatches * minibatch_routed_replay_tasks + num_macrobatches * wgrad_tasks;
    if (cluster_idx >= true_num_clusters) return;

    warpgroup::increase_registers<256>();

    extern __shared__ int __shm[];
    const uint64_t smem_base_addr = (reinterpret_cast<uint64_t>(&__shm[0]) + 1023) & ~uint64_t(1023);

    uint32_t gemm_bitfield = 0xFFFF0000;
    uint32_t swiglu_fwd_bitfield = 0xFFFF0000;
    uint32_t swiglu_bwd_bitfield = 0xFFFF0000;
    uint32_t dispatch_bitfield = 0xFFFF0000;
    uint32_t combine_bitfield = 0xFFFF0000;

    __shared__ clc::handle clc_handle[config::CLC_PIPE_DEPTH];
    __shared__ clc::handle clc_drain_handle[config::CLC_DRAIN_PIPE_DEPTH];
    __shared__ semaphore schedule_arrived[config::CLC_PIPE_DEPTH], schedule_finished[config::CLC_PIPE_DEPTH];
    __shared__ semaphore drain_schedule_arrived[config::CLC_DRAIN_PIPE_DEPTH];
    __shared__ semaphore drain_schedule_finished[config::CLC_DRAIN_PIPE_DEPTH];
    __shared__ semaphore swiglu_fwd_inputs_arrived[config::SWIGLU_FWD_PIPE_DEPTH];
    __shared__ semaphore swiglu_bwd_inputs_arrived[config::SWIGLU_BWD_PIPE_DEPTH];
    __shared__ semaphore gemm_inputs_arrived[config::MLP_LOAD_PIPE_DEPTH];
    __shared__ semaphore gemm_scales_arrived[config::MLP_LOAD_PIPE_DEPTH];
    __shared__ semaphore gemm_inputs_finished[config::MLP_LOAD_PIPE_DEPTH];
    __shared__ semaphore gemm_scales_finished[config::MLP_LOAD_PIPE_DEPTH];
    __shared__ semaphore gemm_outputs_arrived, gemm_outputs_finished;
    __shared__ semaphore dispatch_inputs_arrived;
    __shared__ semaphore combine_inputs_arrived[config::COMBINE_PIPE_DEPTH];

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < config::SWIGLU_FWD_PIPE_DEPTH; ++i) {
            init_semaphore(swiglu_fwd_inputs_arrived[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < config::SWIGLU_BWD_PIPE_DEPTH; ++i) {
            init_semaphore(swiglu_bwd_inputs_arrived[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < config::MLP_LOAD_PIPE_DEPTH; ++i) {
            init_semaphore(gemm_inputs_arrived[i], 0, 1);
            init_semaphore(gemm_scales_arrived[i], 0, 1);
            init_semaphore(gemm_inputs_finished[i], 0, 1);
            init_semaphore(gemm_scales_finished[i], 0, 1);
        }
        init_semaphore(gemm_outputs_arrived, 0, 1);
        init_semaphore(gemm_outputs_finished, 0, config::CLUSTER_SIZE);
        #pragma unroll
        for (int i = 0; i < config::CLC_PIPE_DEPTH; ++i) {
            init_semaphore(schedule_arrived[i], 0, 1);
            init_semaphore(schedule_finished[i], 0, config::CLUSTER_SIZE * config::NUM_WARPS);
        }
        #pragma unroll
        for (int i = 0; i < config::CLC_DRAIN_PIPE_DEPTH; ++i) {
            init_semaphore(drain_schedule_arrived[i], 0, 1);
            init_semaphore(drain_schedule_finished[i], 0, config::CLUSTER_SIZE);
        }
        init_semaphore(dispatch_inputs_arrived, 0, 1);
        #pragma unroll
        for (int i = 0; i < config::COMBINE_PIPE_DEPTH; ++i) {
            init_semaphore(combine_inputs_arrived[i], 0, 1);
        }
    }

    tensor_allocator<1, config::CLUSTER_SIZE> tm_alloc{};
    tt<float, config::MLP_Mb / 2, config::MLP_Nb> d_tt = tm_alloc.template allocate<tt<float, config::MLP_Mb / 2, config::MLP_Nb>>(0);
    full_tt_fp8e8m0<16 * config::MLP_LOAD_PIPE_DEPTH> a_sc_tt = tm_alloc.template allocate<full_tt_fp8e8m0<16 * config::MLP_LOAD_PIPE_DEPTH>>(256);
    full_tt_fp8e8m0<32 * config::MLP_LOAD_PIPE_DEPTH> b_sc_tt = tm_alloc.template allocate<full_tt_fp8e8m0<32 * config::MLP_LOAD_PIPE_DEPTH>>(384);
    everyone::tma::cluster::sync();

    if (cluster_idx < comm_clusters) {
        const int comm_cta_idx = cluster_idx * config::CLUSTER_SIZE + cta_rank;
        auto reverse_combine = [&](int macrobatch_idx, int task_idx) {
            dispatch_kernel<true>(g.d_y_buffer, g.d_y_fp8_routed, &g.d_y_sc_routed, &g.d_y_fp8_t_routed, &g.d_y_sc_t_routed,
                                     &g.router_weights, g.schedule_peer_rank, g.schedule_peer_token_idx,
                                     nullptr, nullptr, g.d_y_routed_ready,
                                     dispatch_inputs_arrived, dispatch_bitfield,
                                     num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, task_idx, g.topk,
                                     -1, 0, smem_base_addr);
        };
        auto reverse_dispatch = [&](int macrobatch_idx, int task_idx) {
            combine_kernel(g.d_x_routed_buffer, g.d_x_routed, &g.d_router_weight_buffer, &g.d_router_weight_partials,
                           g.schedule_peer_rank, g.schedule_peer_token_idx,
                           g.d_x_routed_ready, num_macrobatches > 1 ? &g.routed_buffers_done : nullptr,
                           combine_inputs_arrived, combine_bitfield,
                           num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, task_idx, smem_base_addr);
        };
        auto replay_dispatch = [&](int macrobatch_idx, int task_idx) {
            dispatch_kernel<false>(g.x_routed_send_buffer, g.x_fp8_routed, &g.x_sc_routed, &g.x_fp8_t_routed, &g.x_sc_t_routed,
                                     nullptr, g.schedule_peer_rank, g.schedule_peer_token_idx,
                                     nullptr, nullptr, g.replayed_x_routed_ready,
                                     dispatch_inputs_arrived, dispatch_bitfield,
                                     num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, task_idx, g.topk,
                                     -1, 0, smem_base_addr);
        };
        preload_router_weights_kernel(g.router_weight_buffer, g.router_weights,
                                      g.schedule_peer_rank, g.schedule_peer_token_idx,
                                      nullptr, g.router_weights_ready,
                                      num_tokens, macrobatch_size, 0, comm_cta_idx, g.num_comm_sms, -1, 0);
        for (int task_idx = comm_cta_idx; task_idx < num_dispatch_tasks_of(0); task_idx += g.num_comm_sms)
            reverse_combine(0, task_idx);
        for (int macrobatch_idx = 0; macrobatch_idx < num_macrobatches; ++macrobatch_idx) {
            // All reverse-dispatch tasks must complete before this CTA moves on: the next macrobatch's pulls
            // wait on routed_buffers_done, which counts every rank's reverse-dispatch arrivals (including this CTA's)
            for (int task_idx = comm_cta_idx; task_idx < num_combine_tasks_of(macrobatch_idx); task_idx += g.num_comm_sms)
                reverse_dispatch(macrobatch_idx, task_idx);
            if (macrobatch_idx + 1 < num_macrobatches) {
                preload_router_weights_kernel(g.router_weight_buffer, g.router_weights,
                                              g.schedule_peer_rank, g.schedule_peer_token_idx,
                                              &g.routed_buffers_done, g.router_weights_ready,
                                              num_tokens, macrobatch_size, macrobatch_idx + 1, comm_cta_idx, g.num_comm_sms,
                                              macrobatch_idx, routed_buffers_done_required_count_of(macrobatch_idx));
                for (int task_idx = comm_cta_idx; task_idx < num_dispatch_tasks_of(macrobatch_idx + 1); task_idx += g.num_comm_sms) {
                    reverse_combine(macrobatch_idx + 1, task_idx);
                    replay_dispatch(macrobatch_idx + 1, task_idx);
                }
            }
        }
        return;
    }

    // Swiglu (forward and backward) tasks are CTA-local, GEMM is not
    auto is_cta_local_task = [&](int compute_cluster_idx) {
        if (compute_cluster_idx < 0) return false;
        else if (compute_cluster_idx < shared_dgrad_down_tasks) return false; // shared dgrad down
        else if (compute_cluster_idx < shared_dgrad_down_tasks + shared_swiglu_bwd_tasks) return true; // shared swiglu bwd
        else if (compute_cluster_idx < shared_tasks) return false; // shared dgrad/wgrad
        else if (compute_cluster_idx >= true_num_clusters - comm_clusters) return false;

        int idx = compute_cluster_idx - shared_tasks;
        int macrobatch_num_minibatches, macrobatch_task_idx;
        if (idx < saved_macrobatch_tasks) {
            macrobatch_num_minibatches = saved_macrobatch_num_minibatches;
            macrobatch_task_idx = idx;
        } else {
            idx -= saved_macrobatch_tasks;
            const int macrobatch_idx = 1 + idx / replayed_macrobatch_tasks;
            macrobatch_num_minibatches = num_minibatches_of(macrobatch_idx);
            macrobatch_task_idx = idx % replayed_macrobatch_tasks;
            if (macrobatch_task_idx < macrobatch_num_minibatches * minibatch_routed_replay_tasks)
                return macrobatch_task_idx % minibatch_routed_replay_tasks >= 2 * minibatch_routed_gate_up_tasks; // swiglu fwd replay
            macrobatch_task_idx -= macrobatch_num_minibatches * minibatch_routed_replay_tasks;
        }
        if (macrobatch_task_idx >= macrobatch_num_minibatches * minibatch_routed_bwd_tasks) return false; // wgrad
        const int minibatch_task_idx = macrobatch_task_idx % minibatch_routed_bwd_tasks;
        return minibatch_task_idx >= minibatch_routed_dgrad_down_tasks &&
               minibatch_task_idx < minibatch_routed_dgrad_down_tasks + minibatch_routed_swiglu_bwd_tasks; // swiglu bwd
    };

    const int d_gate_up_row_block_ready_required_count = (config::MLP_Mb / config::SWIGLU_Mb) * (g.hidden_shared.cols() / config::SWIGLU_Nb);
    const index_gl *buffer_done = num_macrobatches > 1 ? &g.routed_buffers_done : nullptr;

    for (int task_iter = 0; cluster_idx >= 0 && cluster_idx < true_num_clusters; ++task_iter) {
        const int clc_stage = task_iter % config::CLC_PIPE_DEPTH;
        if (warpgroup::groupid() == config::NUM_CONSUMERS && warpgroup::warpid() == 1 && warp::elect_leader()) { // warp not used by the gemms
            if (cta_rank == 0) {
                wait(schedule_finished[clc_stage], ((task_iter + config::CLC_PIPE_DEPTH) / config::CLC_PIPE_DEPTH) % 2);
                clc::schedule(clc_handle[clc_stage], schedule_arrived[clc_stage]);
            }
            tma::expect_bytes(schedule_arrived[clc_stage], sizeof(clc_handle[clc_stage]));
        }

        const int compute_cluster_idx = cluster_idx - comm_clusters;
        const bool current_is_cta_local = is_cta_local_task(compute_cluster_idx);

        if (compute_cluster_idx < shared_dgrad_down_tasks) {
            // Shared dgrad down: d_hidden_shared = d_y_shared @ w_shared_down
            const int task_idx = compute_cluster_idx;
            expert_grouped_gemm_kernel<true, false, true>(g.d_y_shared, g.w_shared_down, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                      g.d_hidden_shared, nullptr, nullptr,
                                      g.tokens_per_expert, nullptr, nullptr, nullptr, &g.d_hidden_ready, nullptr, nullptr,
                                      d_tt, a_sc_tt, b_sc_tt,
                                      gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                      num_tokens, macrobatch_size, g.minibatch_size, 0, 0, task_idx, cta_rank,
                                      0, 0, 0, 0, 0, smem_base_addr);
        } else if (compute_cluster_idx < shared_dgrad_down_tasks + shared_swiglu_bwd_tasks) {
            // Shared Swiglu bwd: d_gate_shared, d_up_shared = swiglu_bwd(d_hidden_shared, gate_shared, up_shared)
            const int task_idx = compute_cluster_idx - shared_dgrad_down_tasks;
            swiglu_bwd_kernel<true>(g.d_hidden_shared, g.gate_shared, g.up_shared, g.d_gate_shared, g.d_up_shared,
                             nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                             nullptr, nullptr, nullptr,
                             g.d_hidden_ready, nullptr, g.d_gate_up_ready, nullptr,
                             swiglu_bwd_inputs_arrived, swiglu_bwd_bitfield,
                             g.gate_shared.rows(), macrobatch_size, g.minibatch_size, 0, 0, task_idx, cta_rank,
                             0, 0, 0, 0, smem_base_addr);
        } else if (compute_cluster_idx < shared_dgrad_down_tasks + shared_swiglu_bwd_tasks + shared_dgrad_gate_up_tasks) {
            // Shared dgrad gate+up: d_x_shared = d_gate_shared @ w_shared_gate + d_up_shared @ w_shared_up
            const int task_idx = compute_cluster_idx - shared_dgrad_down_tasks - shared_swiglu_bwd_tasks;
            expert_grouped_gemm_kernel<true, false, true>(g.d_gate_shared, g.w_shared_gate, nullptr, nullptr, &g.d_up_shared, &g.w_shared_up, nullptr, nullptr,
                                      g.d_x_shared, nullptr, nullptr,
                                      g.tokens_per_expert, nullptr, &g.d_gate_up_ready, nullptr, nullptr, nullptr, nullptr,
                                      d_tt, a_sc_tt, b_sc_tt,
                                      gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                      num_tokens, macrobatch_size, g.minibatch_size, 0, 0, task_idx, cta_rank,
                                      0, 0, d_gate_up_row_block_ready_required_count, 0, 0, smem_base_addr);
        } else if (compute_cluster_idx < shared_dgrad_down_tasks + shared_swiglu_bwd_tasks + shared_dgrad_gate_up_tasks + shared_wgrad_tasks) {
            // Shared wgrad down: d_w_shared_down += d_y_shared^T @ hidden_shared
            const int task_idx = compute_cluster_idx - shared_dgrad_down_tasks - shared_swiglu_bwd_tasks - shared_dgrad_gate_up_tasks;
            expert_grouped_gemm_kernel<true, true>(g.d_y_shared, g.hidden_shared, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                            g.d_w_shared_down, nullptr, nullptr,
                                            g.tokens_per_expert, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                            d_tt, a_sc_tt, b_sc_tt,
                                            gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                            num_tokens, macrobatch_size, g.minibatch_size, 0, 0, task_idx, cta_rank,
                                            0, 0, 0, 0, 0, smem_base_addr);
        } else if (compute_cluster_idx < shared_tasks - shared_wgrad_tasks) {
            // Shared wgrad gate: d_w_shared_gate += d_gate_shared^T @ x_shared
            const int task_idx = compute_cluster_idx - shared_dgrad_down_tasks - shared_swiglu_bwd_tasks - shared_dgrad_gate_up_tasks - shared_wgrad_tasks;
            expert_grouped_gemm_kernel<true, true>(g.d_gate_shared, g.x_shared, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                            g.d_w_shared_gate, nullptr, nullptr,
                                            g.tokens_per_expert, nullptr, &g.d_gate_up_ready, nullptr, nullptr, nullptr, nullptr,
                                            d_tt, a_sc_tt, b_sc_tt,
                                            gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                            num_tokens, macrobatch_size, g.minibatch_size, 0, 0, task_idx, cta_rank,
                                            0, 0, d_gate_up_row_block_ready_required_count, 0, 0, smem_base_addr);
        } else if (compute_cluster_idx < shared_tasks) {
            // Shared wgrad up: d_w_shared_up += d_up_shared^T @ x_shared
            const int task_idx = compute_cluster_idx - shared_dgrad_down_tasks - shared_swiglu_bwd_tasks - shared_dgrad_gate_up_tasks - 2 * shared_wgrad_tasks;
            expert_grouped_gemm_kernel<true, true>(g.d_up_shared, g.x_shared, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                            g.d_w_shared_up, nullptr, nullptr,
                                            g.tokens_per_expert, nullptr, &g.d_gate_up_ready, nullptr, nullptr, nullptr, nullptr,
                                            d_tt, a_sc_tt, b_sc_tt,
                                            gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                            num_tokens, macrobatch_size, g.minibatch_size, 0, 0, task_idx, cta_rank,
                                            0, 0, d_gate_up_row_block_ready_required_count, 0, 0, smem_base_addr);
        } else {
            // Routed / replay tasks
            const int global_routed_task_idx = compute_cluster_idx - shared_tasks;
            const bool replayed = global_routed_task_idx >= saved_macrobatch_tasks;
            const int replayed_task_idx = global_routed_task_idx - saved_macrobatch_tasks;
            const int macrobatch_idx = replayed ? 1 + replayed_task_idx / replayed_macrobatch_tasks : 0;
            const int macrobatch_task_idx = replayed ? replayed_task_idx % replayed_macrobatch_tasks : global_routed_task_idx;
            const int macrobatch_num_minibatches = num_minibatches_of(macrobatch_idx);
            const int num_replay_tasks = replayed ? macrobatch_num_minibatches * minibatch_routed_replay_tasks : 0;

            if (macrobatch_task_idx < num_replay_tasks) {
                const int minibatch_idx = macrobatch_task_idx / minibatch_routed_replay_tasks;
                const int minibatch_task_idx = macrobatch_task_idx % minibatch_routed_replay_tasks;
                if (minibatch_task_idx < minibatch_routed_gate_up_tasks) {
                    // Replay gate GEMM refreshes the routed activation.
                    const int task_idx = minibatch_task_idx;
                    expert_grouped_gemm_kernel<false>(g.x_fp8_routed, g.w_routed_gate, &g.x_sc_routed, &g.w_routed_gate_sc, nullptr, nullptr, nullptr, nullptr,
                                               g.gate_routed, &g.gate_fp8_routed, &g.gate_sc_routed,
                                               g.tokens_per_expert, &g.replayed_x_routed_ready, nullptr, nullptr, &g.replayed_gate_up_ready, nullptr, nullptr,
                                               d_tt, a_sc_tt, b_sc_tt,
                                               gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                               num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, minibatch_idx, task_idx, cta_rank,
                                               0, 0, 0, 0, 0, smem_base_addr);
                } else if (minibatch_task_idx < minibatch_routed_gate_up_tasks * 2) {
                    // Replay up GEMM refreshes the routed activation.
                    const int task_idx = minibatch_task_idx - minibatch_routed_gate_up_tasks;
                    expert_grouped_gemm_kernel<false>(g.x_fp8_routed, g.w_routed_up, &g.x_sc_routed, &g.w_routed_up_sc, nullptr, nullptr, nullptr, nullptr,
                                               g.up_routed, &g.up_fp8_routed, &g.up_sc_routed,
                                               g.tokens_per_expert, &g.replayed_x_routed_ready, nullptr, nullptr, &g.replayed_gate_up_ready, nullptr, nullptr,
                                               d_tt, a_sc_tt, b_sc_tt,
                                               gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                               num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, minibatch_idx, task_idx, cta_rank,
                                               0, 0, 0, 0, 0, smem_base_addr);
                } else {
                    // Replay Swiglu refreshes the routed hidden activation.
                    const int task_idx = minibatch_task_idx - minibatch_routed_gate_up_tasks * 2;
                    swiglu_fwd_kernel<false>(g.gate_routed, g.up_routed, g.hidden_fp8_routed,
                                      &g.hidden_sc_routed, &g.hidden_fp8_t_routed, &g.hidden_sc_t_routed,
                                      g.replayed_gate_up_ready, g.replayed_hidden_ready,
                                      swiglu_fwd_inputs_arrived, swiglu_fwd_bitfield,
                                      num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, minibatch_idx,
                                      task_idx, cta_rank, 0, 0, smem_base_addr);
                }
            } else {
                const int num_routed_tasks = macrobatch_num_minibatches * minibatch_routed_bwd_tasks;
                const int routed_task_idx = macrobatch_task_idx - num_replay_tasks;
                const int minibatch_idx = routed_task_idx / minibatch_routed_bwd_tasks;
                const int minibatch_task_idx = routed_task_idx % minibatch_routed_bwd_tasks;
                if (routed_task_idx < num_routed_tasks && minibatch_task_idx < minibatch_routed_dgrad_down_tasks) {
                    // Dgrad down: d_hidden_routed = d_y_routed @ w_routed_down
                    const int task_idx = minibatch_task_idx;
                    expert_grouped_gemm_kernel<false, false, !USE_MXFP8>(g.d_y_fp8_routed, g.w_routed_down_T, &g.d_y_sc_routed, &g.w_routed_down_T_sc, nullptr, nullptr, nullptr, nullptr,
                                               g.d_hidden_routed, nullptr, nullptr,
                                               g.tokens_per_expert, &g.d_y_routed_ready, nullptr, nullptr, &g.d_hidden_ready, nullptr, buffer_done,
                                               d_tt, a_sc_tt, b_sc_tt,
                                               gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                               num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, minibatch_idx, task_idx, cta_rank,
                                               0, 0, 0, shared_dgrad_down_tasks, macrobatch_idx, smem_base_addr);
                } else if (routed_task_idx < num_routed_tasks &&
                           minibatch_task_idx < minibatch_routed_dgrad_down_tasks + minibatch_routed_swiglu_bwd_tasks) {
                    // Routed Swiglu backward
                    const int task_idx = minibatch_task_idx - minibatch_routed_dgrad_down_tasks;
                    swiglu_bwd_kernel<false>(g.d_hidden_routed, g.gate_fp8_routed, g.up_fp8_routed, g.d_gate_fp8_routed, g.d_up_fp8_routed,
                                      &g.gate_sc_routed, &g.up_sc_routed, &g.d_gate_sc_routed, &g.d_up_sc_routed,
                                      &g.d_gate_fp8_t_routed, &g.d_gate_sc_t_routed, &g.d_up_fp8_t_routed, &g.d_up_sc_t_routed,
                                      &g.router_weights, &g.d_router_weight_partials, &g.schedule_peer_rank,
                                      g.d_hidden_ready, replayed ? &g.replayed_gate_up_ready : nullptr, g.d_gate_up_ready, buffer_done,
                                      swiglu_bwd_inputs_arrived, swiglu_bwd_bitfield,
                                      num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, minibatch_idx,
                                      task_idx, cta_rank, shared_dgrad_down_tasks, 0, shared_row_blocks, macrobatch_idx, smem_base_addr);
                } else if (routed_task_idx < num_routed_tasks) {
                    // Dgrad gate+up: d_x_routed = d_gate @ w_routed_gate + d_up @ w_routed_up
                    const int task_idx = minibatch_task_idx - minibatch_routed_dgrad_down_tasks - minibatch_routed_swiglu_bwd_tasks;
                    expert_grouped_gemm_kernel<false, false, !USE_MXFP8>(g.d_gate_fp8_routed, g.w_routed_gate_T, &g.d_gate_sc_routed, &g.w_routed_gate_T_sc,
                                               &g.d_up_fp8_routed, &g.w_routed_up_T, &g.d_up_sc_routed, &g.w_routed_up_T_sc,
                                               g.d_x_routed, nullptr, nullptr,
                                               g.tokens_per_expert, nullptr, &g.d_gate_up_ready, nullptr, nullptr, &g.d_x_routed_ready, buffer_done,
                                               d_tt, a_sc_tt, b_sc_tt,
                                               gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                               num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, minibatch_idx, task_idx, cta_rank,
                                               0, shared_row_blocks, d_gate_up_row_block_ready_required_count, 0, macrobatch_idx, smem_base_addr);
                } else if (routed_task_idx < num_routed_tasks + wgrad_matrix_tasks) {
                    // Wgrad down: d_w_routed_down += d_y_routed^T @ hidden_routed
                    const int task_idx = routed_task_idx - num_routed_tasks;
                    expert_grouped_gemm_kernel<false, true>(g.d_y_fp8_t_routed, g.hidden_fp8_t_routed, &g.d_y_sc_t_routed, &g.hidden_sc_t_routed, nullptr, nullptr, nullptr, nullptr,
                                                     g.d_w_routed_down, nullptr, nullptr,
                                                     g.tokens_per_expert, &g.d_y_routed_ready, replayed ? &g.replayed_hidden_ready : nullptr,
                                                     nullptr, nullptr, nullptr, buffer_done,
                                                     d_tt, a_sc_tt, b_sc_tt,
                                                     gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                                     num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, 0, task_idx, cta_rank,
                                                     g.d_y_shared.cols(), 0, d_gate_up_row_block_ready_required_count, 0, macrobatch_idx, smem_base_addr);
                } else if (routed_task_idx < num_routed_tasks + 2 * wgrad_matrix_tasks) {
                    // Wgrad gate: d_w_routed_gate += d_gate_routed^T @ x_routed
                    const int task_idx = routed_task_idx - num_routed_tasks - wgrad_matrix_tasks;
                    expert_grouped_gemm_kernel<false, true>(g.d_gate_fp8_t_routed, g.x_fp8_t_routed, &g.d_gate_sc_t_routed, &g.x_sc_t_routed, nullptr, nullptr, nullptr, nullptr,
                                                     g.d_w_routed_gate, nullptr, nullptr,
                                                     g.tokens_per_expert, replayed ? &g.replayed_x_routed_ready : nullptr,
                                                     &g.d_gate_up_ready, nullptr, nullptr, nullptr, buffer_done,
                                                     d_tt, a_sc_tt, b_sc_tt,
                                                     gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                                     num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, 0, task_idx, cta_rank,
                                                     g.d_y_shared.cols(), shared_row_blocks, d_gate_up_row_block_ready_required_count,
                                                     0, macrobatch_idx, smem_base_addr);
                } else {
                    // Wgrad up: d_w_routed_up += d_up_routed^T @ x_routed
                    const int task_idx = routed_task_idx - num_routed_tasks - 2 * wgrad_matrix_tasks;
                    expert_grouped_gemm_kernel<false, true>(g.d_up_fp8_t_routed, g.x_fp8_t_routed, &g.d_up_sc_t_routed, &g.x_sc_t_routed, nullptr, nullptr, nullptr, nullptr,
                                                     g.d_w_routed_up, nullptr, nullptr,
                                                     g.tokens_per_expert, replayed ? &g.replayed_x_routed_ready : nullptr,
                                                     &g.d_gate_up_ready, nullptr, nullptr, nullptr, buffer_done,
                                                     d_tt, a_sc_tt, b_sc_tt,
                                                     gemm_inputs_arrived, gemm_scales_arrived, gemm_inputs_finished, gemm_scales_finished, gemm_outputs_arrived, gemm_outputs_finished, gemm_bitfield,
                                                     num_tokens, macrobatch_size, g.minibatch_size, macrobatch_idx, 0, task_idx, cta_rank,
                                                     g.d_y_shared.cols(), shared_row_blocks, d_gate_up_row_block_ready_required_count,
                                                     0, macrobatch_idx, smem_base_addr);
                }
            }
        }

        wait(schedule_arrived[clc_stage], (task_iter / config::CLC_PIPE_DEPTH) % 2);
        const auto schedule = clc::query(clc_handle[clc_stage]);
        cluster_idx = schedule.success ? static_cast<int>(schedule.x / config::CLUSTER_SIZE) : -1;
        __syncwarp();
        warp::tma::cluster::arrive(schedule_finished[clc_stage], 0);

        // SWIGLU -> GEMM requires a cluster-wide sync
        const int next_compute_cluster_idx = cluster_idx - comm_clusters;
        if (current_is_cta_local && cluster_idx >= 0 && !is_cta_local_task(next_compute_cluster_idx))
            everyone::tma::cluster::sync();
    }

    everyone::tma::cluster::sync();

    // CLC drain for no-op threadblocks
    if (cluster_idx >= 0 && warp::laneid() == 0) {
        const int stage = warpid();
        int iter = 0;
        if (cta_rank == 0)
            clc::schedule(clc_drain_handle[stage], drain_schedule_arrived[stage]);
        tma::expect_bytes(drain_schedule_arrived[stage], sizeof(clc::handle));
        while (true) {
            wait(drain_schedule_arrived[stage], iter % 2);
            const auto schedule = clc::query(clc_drain_handle[stage]);
            warp::tma::cluster::arrive(drain_schedule_finished[stage], 0);
            if (cta_rank == 0)
                wait(drain_schedule_finished[stage], iter % 2);
            if (!schedule.success)
                break;
            if (cta_rank == 0)
                clc::schedule(clc_drain_handle[stage], drain_schedule_arrived[stage]);
            tma::expect_bytes(drain_schedule_arrived[stage], sizeof(clc::handle));
            ++iter;
        }
    }
}

static __host__ __forceinline__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
dispatch_mlp_swiglu_combine_bwd_mxfp8(
    // Symmetric buffers (input/output gradients and router weights)
    const at::Tensor &d_y_buffer,               // (num_local_tokens, H)
    const std::vector<int64_t> &d_y_buffer_ptrs,
    const at::Tensor &d_x_routed_buffer,        // (num_local_tokens * topk, H)
    const std::vector<int64_t> &d_x_routed_buffer_ptrs,
    const at::Tensor &router_weight_buffer,     // (num_local_tokens, topk)
    const std::vector<int64_t> &router_weight_buffer_ptrs,
    const at::Tensor &d_router_weight_buffer,   // (num_local_tokens, topk)
    const std::vector<int64_t> &d_router_weight_buffer_ptrs,

    // Weights (routed transposes pre-quantized to MXFP8)
    const at::Tensor &w_shared_gate,            // (I, H)
    const at::Tensor &w_routed_gate_T,          // (E, H, I) fp8
    const at::Tensor &w_routed_gate_T_sc,       // (E * H / 128, I / 128, 32, 16)
    const at::Tensor &w_shared_up,              // (I, H)
    const at::Tensor &w_routed_up_T,            // (E, H, I) fp8
    const at::Tensor &w_routed_up_T_sc,         // (E * H / 128, I / 128, 32, 16)
    const at::Tensor &w_shared_down,            // (H, I)
    const at::Tensor &w_routed_down_T,          // (E, I, H) fp8
    const at::Tensor &w_routed_down_T_sc,       // (E * I / 128, H / 128, 32, 16)

    // Activations saved from the forward (routed ones already MXFP8; replay overwrites them)
    const at::Tensor &x_fp8_t_routed,           // (H, macrobatch_size) fp8
    const at::Tensor &x_sc_t_routed,            // (H / 128, macrobatch_size / 128, 32, 16)
    const at::Tensor &gate_shared,              // (num_local_tokens, I)
    const at::Tensor &gate_fp8_routed,          // (macrobatch_size, I) fp8
    const at::Tensor &gate_sc_routed,           // (macrobatch_size / 128, I / 128, 32, 16)
    const at::Tensor &up_shared,                // (num_local_tokens, I)
    const at::Tensor &up_fp8_routed,            // (macrobatch_size, I) fp8
    const at::Tensor &up_sc_routed,             // (macrobatch_size / 128, I / 128, 32, 16)
    const at::Tensor &hidden_shared,            // (num_local_tokens, I)
    const at::Tensor &hidden_fp8_t_routed,      // (I, macrobatch_size) fp8
    const at::Tensor &hidden_sc_t_routed,       // (I / 128, macrobatch_size / 128, 32, 16)

    // Activations and weights for forward replay
    const at::Tensor &x,                        // (num_local_tokens, H)
    const std::vector<int64_t> &x_ptrs,
    const at::Tensor &w_routed_gate,            // (E, I, H) fp8
    const at::Tensor &w_routed_gate_sc,         // (E * I / 128, H / 128, 32, 16)
    const at::Tensor &w_routed_up,              // (E, I, H) fp8
    const at::Tensor &w_routed_up_sc,           // (E * I / 128, H / 128, 32, 16)

    // Dispatch/combine schedule saved from the forward
    const at::Tensor &schedule_peer_rank,       // (schedule_capacity,)
    const at::Tensor &schedule_peer_token_idx,  // (schedule_capacity,)
    const at::Tensor &num_tokens,               // (1,)
    const at::Tensor &tokens_per_expert,        // (E,)

    // Metadata
    int topk,
    int num_comm_sms,
    int macrobatch_size,
    int minibatch_size
) {
    const int num_local_tokens = x.size(0);
    const int schedule_capacity = schedule_peer_rank.size(0);
    const int hidden_dim = x.size(1);
    const int intermediate_dim = w_shared_gate.size(0);
    const int num_local_experts = w_routed_gate.size(0);
    const int num_global_minibatches = (schedule_capacity + minibatch_size - 1) / minibatch_size;
    const int num_macrobatches = (schedule_capacity + macrobatch_size - 1) / macrobatch_size;
    const int shared_row_blocks = num_local_tokens / config::MLP_Mb;
    const int routed_row_blocks = schedule_capacity / config::MLP_Mb;
    const int intermediate_dim_col_blocks = intermediate_dim / config::MLP_Nb;

    activation_bf16_pgl x_routed_send_buffer_data;
    activation_bf16_pgl d_y_buffer_data;
    activation_bf16_pgl d_x_routed_buffer_data;
    router_weight_pgl router_weight_buffer_data;
    router_weight_pgl d_router_weight_buffer_data;
    for (int i = 0; i < NUM_DEVICES; ++i) {
        x_routed_send_buffer_data[i] = reinterpret_cast<bf16*>(x_ptrs[i]);
        d_y_buffer_data[i] = reinterpret_cast<bf16*>(d_y_buffer_ptrs[i]);
        d_x_routed_buffer_data[i] = reinterpret_cast<bf16*>(d_x_routed_buffer_ptrs[i]);
        router_weight_buffer_data[i] = reinterpret_cast<float*>(router_weight_buffer_ptrs[i]);
        d_router_weight_buffer_data[i] = reinterpret_cast<float*>(d_router_weight_buffer_ptrs[i]);
    }

    // Replayed forward activations
    at::Tensor x_fp8_routed = at::empty({macrobatch_size, hidden_dim}, x.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor x_sc_routed = at::empty({macrobatch_size / 128, hidden_dim / 128, 32, 16}, x.options().dtype(at::kByte));
    at::Tensor gate_routed = at::empty({macrobatch_size, intermediate_dim}, x.options());
    at::Tensor up_routed = at::empty({macrobatch_size, intermediate_dim}, x.options());
    at::Tensor hidden_fp8_routed = at::empty({macrobatch_size, intermediate_dim}, x.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor hidden_sc_routed = at::empty({macrobatch_size / 128, intermediate_dim / 128, 32, 16}, x.options().dtype(at::kByte));
    at::Tensor router_weights = at::empty({macrobatch_size}, router_weight_buffer.options());
    at::Tensor d_router_weight_partials = at::empty({macrobatch_size, intermediate_dim / config::SWIGLU_Nb}, router_weight_buffer.options());

    // Gradient tensors
    at::Tensor d_y_fp8_routed = at::empty({macrobatch_size, hidden_dim}, d_y_buffer.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor d_y_sc_routed = at::empty({macrobatch_size / 128, hidden_dim / 128, 32, 16}, d_y_buffer.options().dtype(at::kByte));
    at::Tensor d_y_fp8_t_routed = at::empty({hidden_dim, macrobatch_size}, d_y_buffer.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor d_y_sc_t_routed = at::empty({hidden_dim / 128, macrobatch_size / 128, 32, 16}, d_y_buffer.options().dtype(at::kByte));
    at::Tensor d_hidden_shared = at::empty({num_local_tokens, intermediate_dim}, d_y_buffer.options());
    at::Tensor d_hidden_routed = at::empty({macrobatch_size, intermediate_dim}, d_y_buffer.options());
    at::Tensor d_gate_shared = at::empty_like(d_hidden_shared);
    at::Tensor d_gate_fp8_routed = at::empty({macrobatch_size, intermediate_dim}, d_y_buffer.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor d_gate_sc_routed = at::empty({macrobatch_size / 128, intermediate_dim / 128, 32, 16}, d_y_buffer.options().dtype(at::kByte));
    at::Tensor d_gate_fp8_t_routed = at::empty({intermediate_dim, macrobatch_size}, d_y_buffer.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor d_gate_sc_t_routed = at::empty({intermediate_dim / 128, macrobatch_size / 128, 32, 16}, d_y_buffer.options().dtype(at::kByte));
    at::Tensor d_up_fp8_routed = at::empty_like(d_gate_fp8_routed);
    at::Tensor d_up_sc_routed = at::empty_like(d_gate_sc_routed);
    at::Tensor d_up_fp8_t_routed = at::empty_like(d_gate_fp8_t_routed);
    at::Tensor d_up_sc_t_routed = at::empty_like(d_gate_sc_t_routed);
    at::Tensor d_up_shared = at::empty_like(d_hidden_shared);
    at::Tensor d_x_shared = at::empty({num_local_tokens, hidden_dim}, d_y_buffer.options());
    at::Tensor d_x_routed = at::empty({macrobatch_size, hidden_dim}, d_y_buffer.options());
    at::Tensor d_w_shared_gate = at::empty({intermediate_dim, hidden_dim}, d_y_buffer.options());
    at::Tensor d_w_routed_gate = at::empty({num_local_experts, intermediate_dim, hidden_dim}, d_y_buffer.options());
    at::Tensor d_w_shared_up = at::empty({intermediate_dim, hidden_dim}, d_y_buffer.options());
    at::Tensor d_w_routed_up = at::empty({num_local_experts, intermediate_dim, hidden_dim}, d_y_buffer.options());
    at::Tensor d_w_shared_down = at::empty({hidden_dim, intermediate_dim}, d_y_buffer.options());
    at::Tensor d_w_routed_down = at::empty({num_local_experts, hidden_dim, intermediate_dim}, d_y_buffer.options());

    // Counters
    at::Tensor d_y_routed_ready = at::zeros({num_global_minibatches}, tokens_per_expert.options());
    at::Tensor d_hidden_ready = at::zeros({(shared_row_blocks + routed_row_blocks) * intermediate_dim_col_blocks}, tokens_per_expert.options());
    at::Tensor d_gate_up_ready = at::zeros({shared_row_blocks + routed_row_blocks}, tokens_per_expert.options());
    at::Tensor d_x_routed_ready = at::zeros({num_global_minibatches}, tokens_per_expert.options());
    at::Tensor replayed_x_routed_ready = at::zeros({num_global_minibatches}, tokens_per_expert.options());
    at::Tensor replayed_gate_up_ready = at::zeros({routed_row_blocks * intermediate_dim_col_blocks}, tokens_per_expert.options());
    at::Tensor replayed_hidden_ready = at::zeros({routed_row_blocks}, tokens_per_expert.options());
    at::Tensor routed_buffers_done = at::zeros({num_macrobatches}, tokens_per_expert.options());
    at::Tensor router_weights_ready = at::zeros({num_macrobatches}, tokens_per_expert.options());

    globals_bwd g {
        .x_shared = kittens::py::tensor_to_gl<wgrad_bf16_gl>(x),
        .x_fp8_routed = kittens::py::tensor_to_gl<mlp_fp8_gl>(x_fp8_routed),
        .x_sc_routed = kittens::py::tensor_to_gl<sc_gl>(x_sc_routed),
        .x_fp8_t_routed = kittens::py::tensor_to_gl<mlp_fp8_gl>(x_fp8_t_routed),
        .x_sc_t_routed = kittens::py::tensor_to_gl<sc_gl>(x_sc_t_routed),
        .gate_shared = kittens::py::tensor_to_gl<swiglu_bf16_gl>(gate_shared),
        .gate_routed = kittens::py::tensor_to_gl<epi_bf16_gl>(gate_routed),
        .gate_fp8_routed = kittens::py::tensor_to_gl<gate_up_fp8_gl>(gate_fp8_routed),
        .gate_sc_routed = kittens::py::tensor_to_gl<sc_gl>(gate_sc_routed),
        .up_shared = kittens::py::tensor_to_gl<swiglu_bf16_gl>(up_shared),
        .up_routed = kittens::py::tensor_to_gl<epi_bf16_gl>(up_routed),
        .up_fp8_routed = kittens::py::tensor_to_gl<gate_up_fp8_gl>(up_fp8_routed),
        .up_sc_routed = kittens::py::tensor_to_gl<sc_gl>(up_sc_routed),
        .hidden_shared = kittens::py::tensor_to_gl<wgrad_bf16_gl>(hidden_shared),
        .hidden_fp8_routed = kittens::py::tensor_to_gl<mlp_fp8_gl>(hidden_fp8_routed),
        .hidden_sc_routed = kittens::py::tensor_to_gl<sc_gl>(hidden_sc_routed),
        .hidden_fp8_t_routed = kittens::py::tensor_to_gl<mlp_fp8_gl>(hidden_fp8_t_routed),
        .hidden_sc_t_routed = kittens::py::tensor_to_gl<sc_gl>(hidden_sc_t_routed),
        .d_y_shared = kittens::py::tensor_to_gl<mlp_bf16_gl>(d_y_buffer),
        .d_y_fp8_routed = kittens::py::tensor_to_gl<mlp_fp8_gl>(d_y_fp8_routed),
        .d_y_sc_routed = kittens::py::tensor_to_gl<sc_gl>(d_y_sc_routed),
        .d_y_fp8_t_routed = kittens::py::tensor_to_gl<mlp_fp8_gl>(d_y_fp8_t_routed),
        .d_y_sc_t_routed = kittens::py::tensor_to_gl<sc_gl>(d_y_sc_t_routed),
        .d_hidden_shared = kittens::py::tensor_to_gl<epi_bf16_gl>(d_hidden_shared),
        .d_hidden_routed = kittens::py::tensor_to_gl<epi_bf16_gl>(d_hidden_routed),
        .d_gate_shared = kittens::py::tensor_to_gl<mlp_bf16_gl>(d_gate_shared),
        .d_gate_fp8_routed = kittens::py::tensor_to_gl<mlp_fp8_gl>(d_gate_fp8_routed),
        .d_gate_sc_routed = kittens::py::tensor_to_gl<sc_gl>(d_gate_sc_routed),
        .d_gate_fp8_t_routed = kittens::py::tensor_to_gl<mlp_fp8_gl>(d_gate_fp8_t_routed),
        .d_gate_sc_t_routed = kittens::py::tensor_to_gl<sc_gl>(d_gate_sc_t_routed),
        .d_up_shared = kittens::py::tensor_to_gl<mlp_bf16_gl>(d_up_shared),
        .d_up_fp8_routed = kittens::py::tensor_to_gl<mlp_fp8_gl>(d_up_fp8_routed),
        .d_up_sc_routed = kittens::py::tensor_to_gl<sc_gl>(d_up_sc_routed),
        .d_up_fp8_t_routed = kittens::py::tensor_to_gl<mlp_fp8_gl>(d_up_fp8_t_routed),
        .d_up_sc_t_routed = kittens::py::tensor_to_gl<sc_gl>(d_up_sc_t_routed),
        .d_x_shared = kittens::py::tensor_to_gl<epi_bf16_gl>(d_x_shared),
        .d_x_routed = kittens::py::tensor_to_gl<epi_bf16_gl>(d_x_routed),
        .x_routed_send_buffer = x_routed_send_buffer_data,
        .d_y_buffer = d_y_buffer_data,
        .d_x_routed_buffer = d_x_routed_buffer_data,
        .router_weight_buffer = router_weight_buffer_data,
        .d_router_weight_buffer = d_router_weight_buffer_data,
        .router_weights = kittens::py::tensor_to_gl<router_weight_gl>(router_weights),
        .d_router_weight_partials = kittens::py::tensor_to_gl<d_router_weight_partials_gl>(d_router_weight_partials),
        .w_routed_gate = kittens::py::tensor_to_gl<weight_fp8_gl>(w_routed_gate),
        .w_routed_gate_sc = kittens::py::tensor_to_gl<sc_gl>(w_routed_gate_sc),
        .w_routed_up = kittens::py::tensor_to_gl<weight_fp8_gl>(w_routed_up),
        .w_routed_up_sc = kittens::py::tensor_to_gl<sc_gl>(w_routed_up_sc),
        .w_shared_gate = kittens::py::tensor_to_gl<weight_bf16_gl>(w_shared_gate),
        .w_routed_gate_T = kittens::py::tensor_to_gl<weight_fp8_gl>(w_routed_gate_T),
        .w_routed_gate_T_sc = kittens::py::tensor_to_gl<sc_gl>(w_routed_gate_T_sc),
        .w_shared_up = kittens::py::tensor_to_gl<weight_bf16_gl>(w_shared_up),
        .w_routed_up_T = kittens::py::tensor_to_gl<weight_fp8_gl>(w_routed_up_T),
        .w_routed_up_T_sc = kittens::py::tensor_to_gl<sc_gl>(w_routed_up_T_sc),
        .w_shared_down = kittens::py::tensor_to_gl<weight_bf16_gl>(w_shared_down),
        .w_routed_down_T = kittens::py::tensor_to_gl<weight_fp8_gl>(w_routed_down_T),
        .w_routed_down_T_sc = kittens::py::tensor_to_gl<sc_gl>(w_routed_down_T_sc),
        .d_w_shared_gate = kittens::py::tensor_to_gl<d_weight_bf16_gl>(d_w_shared_gate),
        .d_w_routed_gate = kittens::py::tensor_to_gl<d_weight_bf16_gl>(d_w_routed_gate),
        .d_w_shared_up = kittens::py::tensor_to_gl<d_weight_bf16_gl>(d_w_shared_up),
        .d_w_routed_up = kittens::py::tensor_to_gl<d_weight_bf16_gl>(d_w_routed_up),
        .d_w_shared_down = kittens::py::tensor_to_gl<d_weight_bf16_gl>(d_w_shared_down),
        .d_w_routed_down = kittens::py::tensor_to_gl<d_weight_bf16_gl>(d_w_routed_down),
        .schedule_peer_rank = kittens::py::tensor_to_gl<index_gl>(schedule_peer_rank),
        .schedule_peer_token_idx = kittens::py::tensor_to_gl<index_gl>(schedule_peer_token_idx),
        .num_tokens = kittens::py::tensor_to_gl<index_gl>(num_tokens),
        .tokens_per_expert = kittens::py::tensor_to_gl<index_gl>(tokens_per_expert),
        .router_weights_ready = kittens::py::tensor_to_gl<index_gl>(router_weights_ready),
        .d_y_routed_ready = kittens::py::tensor_to_gl<index_gl>(d_y_routed_ready),
        .d_hidden_ready = kittens::py::tensor_to_gl<index_gl>(d_hidden_ready),
        .d_gate_up_ready = kittens::py::tensor_to_gl<index_gl>(d_gate_up_ready),
        .d_x_routed_ready = kittens::py::tensor_to_gl<index_gl>(d_x_routed_ready),
        .replayed_x_routed_ready = kittens::py::tensor_to_gl<index_gl>(replayed_x_routed_ready),
        .replayed_gate_up_ready = kittens::py::tensor_to_gl<index_gl>(replayed_gate_up_ready),
        .replayed_hidden_ready = kittens::py::tensor_to_gl<index_gl>(replayed_hidden_ready),
        .routed_buffers_done = kittens::py::tensor_to_gl<index_gl>(routed_buffers_done),
        .topk = topk,
        .num_comm_sms = num_comm_sms,
        .macrobatch_size = macrobatch_size,
        .minibatch_size = minibatch_size
    };

    kittens::py::launch_kernel<config, globals_bwd, dispatch_mlp_swiglu_combine_bwd_kernel>(g);
    const int64_t elements_per_expert = d_w_routed_gate.numel() / num_local_experts;
    utils::zero_empty_routed_wgrads<<<dim3(128, num_local_experts), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<uint16_t *>(d_w_routed_gate.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint16_t *>(d_w_routed_up.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint16_t *>(d_w_routed_down.data_ptr<at::BFloat16>()),
        tokens_per_expert.data_ptr<int>(),
        elements_per_expert);

    return {d_x_shared, d_x_routed,
            d_gate_shared, d_gate_fp8_routed, d_gate_sc_routed,
            d_up_shared, d_up_fp8_routed, d_up_sc_routed,
            d_hidden_shared, d_hidden_routed, d_y_fp8_routed, d_y_sc_routed,
            d_w_shared_gate, d_w_routed_gate, d_w_shared_up, d_w_routed_up, d_w_shared_down, d_w_routed_down};
}

static __host__ __forceinline__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
dispatch_mlp_swiglu_combine_bwd_bf16(
    const at::Tensor &d_y_buffer,
    const std::vector<int64_t> &d_y_buffer_ptrs,
    const at::Tensor &d_x_routed_buffer,
    const std::vector<int64_t> &d_x_routed_buffer_ptrs,
    const at::Tensor &router_weight_buffer,
    const std::vector<int64_t> &router_weight_buffer_ptrs,
    const at::Tensor &d_router_weight_buffer,
    const std::vector<int64_t> &d_router_weight_buffer_ptrs,
    const at::Tensor &w_shared_gate,
    const at::Tensor &w_routed_gate,
    const at::Tensor &w_shared_up,
    const at::Tensor &w_routed_up,
    const at::Tensor &w_shared_down,
    const at::Tensor &w_routed_down,
    const at::Tensor &x_routed,
    const at::Tensor &gate_shared,
    const at::Tensor &gate_routed,
    const at::Tensor &up_shared,
    const at::Tensor &up_routed,
    const at::Tensor &hidden_shared,
    const at::Tensor &hidden_routed,
    const at::Tensor &x,
    const std::vector<int64_t> &x_ptrs,
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens,
    const at::Tensor &tokens_per_expert,
    int topk,
    int num_comm_sms,
    int macrobatch_size,
    int minibatch_size
) {
    static_assert(!USE_MXFP8);
    const int num_local_tokens = x.size(0);
    const int schedule_capacity = schedule_peer_rank.size(0);
    const int hidden_dim = x.size(1);
    const int intermediate_dim = w_shared_gate.size(0);
    const int num_local_experts = w_routed_gate.size(0);
    const int num_global_minibatches = (schedule_capacity + minibatch_size - 1) / minibatch_size;
    const int num_macrobatches = (schedule_capacity + macrobatch_size - 1) / macrobatch_size;
    const int shared_row_blocks = num_local_tokens / config::MLP_Mb;
    const int routed_row_blocks = schedule_capacity / config::MLP_Mb;
    const int intermediate_dim_col_blocks = intermediate_dim / config::MLP_Nb;

    activation_bf16_pgl x_routed_send_buffer_data;
    activation_bf16_pgl d_y_buffer_data;
    activation_bf16_pgl d_x_routed_buffer_data;
    router_weight_pgl router_weight_buffer_data;
    router_weight_pgl d_router_weight_buffer_data;
    for (int i = 0; i < NUM_DEVICES; ++i) {
        x_routed_send_buffer_data[i] = reinterpret_cast<bf16*>(x_ptrs[i]);
        d_y_buffer_data[i] = reinterpret_cast<bf16*>(d_y_buffer_ptrs[i]);
        d_x_routed_buffer_data[i] = reinterpret_cast<bf16*>(d_x_routed_buffer_ptrs[i]);
        router_weight_buffer_data[i] = reinterpret_cast<float*>(router_weight_buffer_ptrs[i]);
        d_router_weight_buffer_data[i] = reinterpret_cast<float*>(d_router_weight_buffer_ptrs[i]);
    }

    at::Tensor router_weights = at::empty({macrobatch_size}, router_weight_buffer.options());
    at::Tensor d_router_weight_partials = at::empty({macrobatch_size, intermediate_dim / config::SWIGLU_Nb}, router_weight_buffer.options());
    at::Tensor d_y_routed = at::empty({macrobatch_size, hidden_dim}, d_y_buffer.options());
    at::Tensor d_hidden_shared = at::empty({num_local_tokens, intermediate_dim}, d_y_buffer.options());
    at::Tensor d_hidden_routed = at::empty({macrobatch_size, intermediate_dim}, d_y_buffer.options());
    at::Tensor d_gate_shared = at::empty_like(d_hidden_shared);
    at::Tensor d_gate_routed = at::empty_like(d_hidden_routed);
    at::Tensor d_up_shared = at::empty_like(d_hidden_shared);
    at::Tensor d_up_routed = at::empty_like(d_hidden_routed);
    at::Tensor d_x_shared = at::empty({num_local_tokens, hidden_dim}, d_y_buffer.options());
    at::Tensor d_x_routed = at::empty({macrobatch_size, hidden_dim}, d_y_buffer.options());
    at::Tensor d_w_shared_gate = at::empty({intermediate_dim, hidden_dim}, d_y_buffer.options());
    at::Tensor d_w_routed_gate = at::empty(w_routed_gate.sizes(), w_routed_gate.options());
    at::Tensor d_w_shared_up = at::empty({intermediate_dim, hidden_dim}, d_y_buffer.options());
    at::Tensor d_w_routed_up = at::empty(w_routed_up.sizes(), w_routed_up.options());
    at::Tensor d_w_shared_down = at::empty({hidden_dim, intermediate_dim}, d_y_buffer.options());
    at::Tensor d_w_routed_down = at::empty(w_routed_down.sizes(), w_routed_down.options());

    at::Tensor d_y_routed_ready = at::zeros({num_global_minibatches}, tokens_per_expert.options());
    at::Tensor d_hidden_ready = at::zeros({(shared_row_blocks + routed_row_blocks) * intermediate_dim_col_blocks}, tokens_per_expert.options());
    at::Tensor d_gate_up_ready = at::zeros({shared_row_blocks + routed_row_blocks}, tokens_per_expert.options());
    at::Tensor d_x_routed_ready = at::zeros({num_global_minibatches}, tokens_per_expert.options());
    at::Tensor replayed_x_routed_ready = at::zeros({num_global_minibatches}, tokens_per_expert.options());
    at::Tensor replayed_gate_up_ready = at::zeros({routed_row_blocks * intermediate_dim_col_blocks}, tokens_per_expert.options());
    at::Tensor replayed_hidden_ready = at::zeros({routed_row_blocks}, tokens_per_expert.options());
    at::Tensor routed_buffers_done = at::zeros({num_macrobatches}, tokens_per_expert.options());
    at::Tensor router_weights_ready = at::zeros({num_macrobatches}, tokens_per_expert.options());

    globals_bwd g {
        .x_shared = kittens::py::tensor_to_gl<wgrad_bf16_gl>(x),
        .x_fp8_routed = kittens::py::tensor_to_gl<routed_activation_gl>(x_routed),
        .x_sc_routed = {},
        .x_fp8_t_routed = kittens::py::tensor_to_gl<routed_transposed_gl>(x_routed),
        .x_sc_t_routed = {},
        .gate_shared = kittens::py::tensor_to_gl<swiglu_bf16_gl>(gate_shared),
        .gate_routed = kittens::py::tensor_to_gl<epi_bf16_gl>(gate_routed),
        .gate_fp8_routed = kittens::py::tensor_to_gl<routed_gate_up_gl>(gate_routed),
        .gate_sc_routed = {},
        .up_shared = kittens::py::tensor_to_gl<swiglu_bf16_gl>(up_shared),
        .up_routed = kittens::py::tensor_to_gl<epi_bf16_gl>(up_routed),
        .up_fp8_routed = kittens::py::tensor_to_gl<routed_gate_up_gl>(up_routed),
        .up_sc_routed = {},
        .hidden_shared = kittens::py::tensor_to_gl<wgrad_bf16_gl>(hidden_shared),
        .hidden_fp8_routed = kittens::py::tensor_to_gl<routed_activation_gl>(hidden_routed),
        .hidden_sc_routed = {},
        .hidden_fp8_t_routed = kittens::py::tensor_to_gl<routed_transposed_gl>(hidden_routed),
        .hidden_sc_t_routed = {},
        .d_y_shared = kittens::py::tensor_to_gl<mlp_bf16_gl>(d_y_buffer),
        .d_y_fp8_routed = kittens::py::tensor_to_gl<routed_activation_gl>(d_y_routed),
        .d_y_sc_routed = {},
        .d_y_fp8_t_routed = kittens::py::tensor_to_gl<routed_transposed_gl>(d_y_routed),
        .d_y_sc_t_routed = {},
        .d_hidden_shared = kittens::py::tensor_to_gl<epi_bf16_gl>(d_hidden_shared),
        .d_hidden_routed = kittens::py::tensor_to_gl<epi_bf16_gl>(d_hidden_routed),
        .d_gate_shared = kittens::py::tensor_to_gl<mlp_bf16_gl>(d_gate_shared),
        .d_gate_fp8_routed = kittens::py::tensor_to_gl<routed_activation_gl>(d_gate_routed),
        .d_gate_sc_routed = {},
        .d_gate_fp8_t_routed = kittens::py::tensor_to_gl<routed_transposed_gl>(d_gate_routed),
        .d_gate_sc_t_routed = {},
        .d_up_shared = kittens::py::tensor_to_gl<mlp_bf16_gl>(d_up_shared),
        .d_up_fp8_routed = kittens::py::tensor_to_gl<routed_activation_gl>(d_up_routed),
        .d_up_sc_routed = {},
        .d_up_fp8_t_routed = kittens::py::tensor_to_gl<routed_transposed_gl>(d_up_routed),
        .d_up_sc_t_routed = {},
        .d_x_shared = kittens::py::tensor_to_gl<epi_bf16_gl>(d_x_shared),
        .d_x_routed = kittens::py::tensor_to_gl<epi_bf16_gl>(d_x_routed),
        .x_routed_send_buffer = x_routed_send_buffer_data,
        .d_y_buffer = d_y_buffer_data,
        .d_x_routed_buffer = d_x_routed_buffer_data,
        .router_weight_buffer = router_weight_buffer_data,
        .d_router_weight_buffer = d_router_weight_buffer_data,
        .router_weights = kittens::py::tensor_to_gl<router_weight_gl>(router_weights),
        .d_router_weight_partials = kittens::py::tensor_to_gl<d_router_weight_partials_gl>(d_router_weight_partials),
        .w_routed_gate = kittens::py::tensor_to_gl<routed_weight_gl>(w_routed_gate),
        .w_routed_gate_sc = {},
        .w_routed_up = kittens::py::tensor_to_gl<routed_weight_gl>(w_routed_up),
        .w_routed_up_sc = {},
        .w_shared_gate = kittens::py::tensor_to_gl<weight_bf16_gl>(w_shared_gate),
        .w_routed_gate_T = kittens::py::tensor_to_gl<routed_weight_gl>(w_routed_gate),
        .w_routed_gate_T_sc = {},
        .w_shared_up = kittens::py::tensor_to_gl<weight_bf16_gl>(w_shared_up),
        .w_routed_up_T = kittens::py::tensor_to_gl<routed_weight_gl>(w_routed_up),
        .w_routed_up_T_sc = {},
        .w_shared_down = kittens::py::tensor_to_gl<weight_bf16_gl>(w_shared_down),
        .w_routed_down_T = kittens::py::tensor_to_gl<routed_weight_gl>(w_routed_down),
        .w_routed_down_T_sc = {},
        .d_w_shared_gate = kittens::py::tensor_to_gl<d_weight_bf16_gl>(d_w_shared_gate),
        .d_w_routed_gate = kittens::py::tensor_to_gl<d_weight_bf16_gl>(d_w_routed_gate),
        .d_w_shared_up = kittens::py::tensor_to_gl<d_weight_bf16_gl>(d_w_shared_up),
        .d_w_routed_up = kittens::py::tensor_to_gl<d_weight_bf16_gl>(d_w_routed_up),
        .d_w_shared_down = kittens::py::tensor_to_gl<d_weight_bf16_gl>(d_w_shared_down),
        .d_w_routed_down = kittens::py::tensor_to_gl<d_weight_bf16_gl>(d_w_routed_down),
        .schedule_peer_rank = kittens::py::tensor_to_gl<index_gl>(schedule_peer_rank),
        .schedule_peer_token_idx = kittens::py::tensor_to_gl<index_gl>(schedule_peer_token_idx),
        .num_tokens = kittens::py::tensor_to_gl<index_gl>(num_tokens),
        .tokens_per_expert = kittens::py::tensor_to_gl<index_gl>(tokens_per_expert),
        .router_weights_ready = kittens::py::tensor_to_gl<index_gl>(router_weights_ready),
        .d_y_routed_ready = kittens::py::tensor_to_gl<index_gl>(d_y_routed_ready),
        .d_hidden_ready = kittens::py::tensor_to_gl<index_gl>(d_hidden_ready),
        .d_gate_up_ready = kittens::py::tensor_to_gl<index_gl>(d_gate_up_ready),
        .d_x_routed_ready = kittens::py::tensor_to_gl<index_gl>(d_x_routed_ready),
        .replayed_x_routed_ready = kittens::py::tensor_to_gl<index_gl>(replayed_x_routed_ready),
        .replayed_gate_up_ready = kittens::py::tensor_to_gl<index_gl>(replayed_gate_up_ready),
        .replayed_hidden_ready = kittens::py::tensor_to_gl<index_gl>(replayed_hidden_ready),
        .routed_buffers_done = kittens::py::tensor_to_gl<index_gl>(routed_buffers_done),
        .topk = topk,
        .num_comm_sms = num_comm_sms,
        .macrobatch_size = macrobatch_size,
        .minibatch_size = minibatch_size
    };

    kittens::py::launch_kernel<config, globals_bwd, dispatch_mlp_swiglu_combine_bwd_kernel>(g);
    const int64_t elements_per_expert = d_w_routed_gate.numel() / num_local_experts;
    utils::zero_empty_routed_wgrads<<<dim3(128, num_local_experts), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<uint16_t *>(d_w_routed_gate.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint16_t *>(d_w_routed_up.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint16_t *>(d_w_routed_down.data_ptr<at::BFloat16>()),
        tokens_per_expert.data_ptr<int>(),
        elements_per_expert);
    return {d_x_shared, d_x_routed, d_gate_shared, d_gate_routed, d_up_shared, d_up_routed,
            d_hidden_shared, d_hidden_routed, d_y_routed,
            d_w_shared_gate, d_w_routed_gate, d_w_shared_up, d_w_routed_up, d_w_shared_down, d_w_routed_down};
}

}; // struct dispatch_mlp_swiglu_combiner

template <GluActivation ACTIVATION>
static __host__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                           at::Tensor, at::Tensor, at::Tensor>
dispatch_mlp_glu_combine_fwd_mxfp8(
    // Inputs and communication buffers
    const at::Tensor &x,
    const std::vector<int64_t> &x_ptrs,
    const at::Tensor &combine_buffer,
    const std::vector<int64_t> &combine_buffer_ptrs,

    // Weights
    const at::Tensor &w_shared_gate,
    const at::Tensor &w_routed_gate,
    const at::Tensor &w_routed_gate_sc,
    const at::Tensor &w_shared_up,
    const at::Tensor &w_routed_up,
    const at::Tensor &w_routed_up_sc,
    const at::Tensor &w_shared_down,
    const at::Tensor &w_routed_down,
    const at::Tensor &w_routed_down_sc,

    // Dispatch/combine schedule
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens,
    const at::Tensor &tokens_per_expert,

    // Metadata
    int topk,
    int num_comm_sms,
    int macrobatch_size,
    int minibatch_size
) {
    const int num_devices = static_cast<int>(x_ptrs.size());

    switch (num_devices) {
        case 4:
            return dispatch_mlp_swiglu_combiner<4, utils::RoutedPrecision::MXFP8, ACTIVATION>::dispatch_mlp_swiglu_combine_fwd_mxfp8(
                x, x_ptrs, combine_buffer, combine_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_routed_gate_sc,
                w_shared_up, w_routed_up, w_routed_up_sc,
                w_shared_down, w_routed_down, w_routed_down_sc,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, num_comm_sms, macrobatch_size, minibatch_size);
        case 8:
            return dispatch_mlp_swiglu_combiner<8, utils::RoutedPrecision::MXFP8, ACTIVATION>::dispatch_mlp_swiglu_combine_fwd_mxfp8(
                x, x_ptrs, combine_buffer, combine_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_routed_gate_sc,
                w_shared_up, w_routed_up, w_routed_up_sc,
                w_shared_down, w_routed_down, w_routed_down_sc,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, num_comm_sms, macrobatch_size, minibatch_size);
        case 16:
            return dispatch_mlp_swiglu_combiner<16, utils::RoutedPrecision::MXFP8, ACTIVATION>::dispatch_mlp_swiglu_combine_fwd_mxfp8(
                x, x_ptrs, combine_buffer, combine_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_routed_gate_sc,
                w_shared_up, w_routed_up, w_routed_up_sc,
                w_shared_down, w_routed_down, w_routed_down_sc,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, num_comm_sms, macrobatch_size, minibatch_size);
        case 32:
            return dispatch_mlp_swiglu_combiner<32, utils::RoutedPrecision::MXFP8, ACTIVATION>::dispatch_mlp_swiglu_combine_fwd_mxfp8(
                x, x_ptrs, combine_buffer, combine_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_routed_gate_sc,
                w_shared_up, w_routed_up, w_routed_up_sc,
                w_shared_down, w_routed_down, w_routed_down_sc,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, num_comm_sms, macrobatch_size, minibatch_size);
        case 64:
            return dispatch_mlp_swiglu_combiner<64, utils::RoutedPrecision::MXFP8, ACTIVATION>::dispatch_mlp_swiglu_combine_fwd_mxfp8(
                x, x_ptrs, combine_buffer, combine_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_routed_gate_sc,
                w_shared_up, w_routed_up, w_routed_up_sc,
                w_shared_down, w_routed_down, w_routed_down_sc,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, num_comm_sms, macrobatch_size, minibatch_size);
        default:
            throw std::runtime_error("MoK: dispatch_mlp_swiglu_combine_fwd_mxfp8 unsupported num_devices=" +
                                     std::to_string(num_devices) + " (supported: 4, 8, 16, 32, 64)");
    }
}

template <GluActivation ACTIVATION>
static __host__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                           at::Tensor, at::Tensor, at::Tensor, at::Tensor>
dispatch_mlp_glu_combine_fwd_bf16(
    const at::Tensor &x,
    const std::vector<int64_t> &x_ptrs,
    const at::Tensor &combine_buffer,
    const std::vector<int64_t> &combine_buffer_ptrs,
    const at::Tensor &w_shared_gate,
    const at::Tensor &w_routed_gate,
    const at::Tensor &w_shared_up,
    const at::Tensor &w_routed_up,
    const at::Tensor &w_shared_down,
    const at::Tensor &w_routed_down,
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens,
    const at::Tensor &tokens_per_expert,
    int topk,
    int num_comm_sms,
    int macrobatch_size,
    int minibatch_size
) {
    const int num_devices = static_cast<int>(x_ptrs.size());
    switch (num_devices) {
        case 4:
            return dispatch_mlp_swiglu_combiner<4, utils::RoutedPrecision::BF16, ACTIVATION>::dispatch_mlp_swiglu_combine_fwd_bf16(
                x, x_ptrs, combine_buffer, combine_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, num_comm_sms, macrobatch_size, minibatch_size);
        case 8:
            return dispatch_mlp_swiglu_combiner<8, utils::RoutedPrecision::BF16, ACTIVATION>::dispatch_mlp_swiglu_combine_fwd_bf16(
                x, x_ptrs, combine_buffer, combine_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, num_comm_sms, macrobatch_size, minibatch_size);
        case 16:
            return dispatch_mlp_swiglu_combiner<16, utils::RoutedPrecision::BF16, ACTIVATION>::dispatch_mlp_swiglu_combine_fwd_bf16(
                x, x_ptrs, combine_buffer, combine_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, num_comm_sms, macrobatch_size, minibatch_size);
        case 32:
            return dispatch_mlp_swiglu_combiner<32, utils::RoutedPrecision::BF16, ACTIVATION>::dispatch_mlp_swiglu_combine_fwd_bf16(
                x, x_ptrs, combine_buffer, combine_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, num_comm_sms, macrobatch_size, minibatch_size);
        case 64:
            return dispatch_mlp_swiglu_combiner<64, utils::RoutedPrecision::BF16, ACTIVATION>::dispatch_mlp_swiglu_combine_fwd_bf16(
                x, x_ptrs, combine_buffer, combine_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, num_comm_sms, macrobatch_size, minibatch_size);
        default:
            throw std::runtime_error("MoK: dispatch_mlp_swiglu_combine_fwd_bf16 unsupported num_devices=" +
                                     std::to_string(num_devices) + " (supported: 4, 8, 16, 32, 64)");
    }
}

template <GluActivation ACTIVATION>
static __host__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
dispatch_mlp_glu_combine_bwd_mxfp8(
    // Symmetric buffers (input/output gradients and router weights)
    const at::Tensor &d_y_buffer,
    const std::vector<int64_t> &d_y_buffer_ptrs,
    const at::Tensor &d_x_routed_buffer,
    const std::vector<int64_t> &d_x_routed_buffer_ptrs,
    const at::Tensor &router_weight_buffer,
    const std::vector<int64_t> &router_weight_buffer_ptrs,
    const at::Tensor &d_router_weight_buffer,
    const std::vector<int64_t> &d_router_weight_buffer_ptrs,

    // Weights (routed transposes pre-quantized to MXFP8)
    const at::Tensor &w_shared_gate,
    const at::Tensor &w_routed_gate_T,
    const at::Tensor &w_routed_gate_T_sc,
    const at::Tensor &w_shared_up,
    const at::Tensor &w_routed_up_T,
    const at::Tensor &w_routed_up_T_sc,
    const at::Tensor &w_shared_down,
    const at::Tensor &w_routed_down_T,
    const at::Tensor &w_routed_down_T_sc,

    // Activations saved from the forward
    const at::Tensor &x_fp8_t_routed,
    const at::Tensor &x_sc_t_routed,
    const at::Tensor &gate_shared,
    const at::Tensor &gate_fp8_routed,
    const at::Tensor &gate_sc_routed,
    const at::Tensor &up_shared,
    const at::Tensor &up_fp8_routed,
    const at::Tensor &up_sc_routed,
    const at::Tensor &hidden_shared,
    const at::Tensor &hidden_fp8_t_routed,
    const at::Tensor &hidden_sc_t_routed,

    // Activations and weights for forward replay
    const at::Tensor &x,
    const std::vector<int64_t> &x_ptrs,
    const at::Tensor &w_routed_gate,
    const at::Tensor &w_routed_gate_sc,
    const at::Tensor &w_routed_up,
    const at::Tensor &w_routed_up_sc,

    // Dispatch/combine schedule
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens,
    const at::Tensor &tokens_per_expert,

    // Metadata
    int topk,
    int num_comm_sms,
    int macrobatch_size,
    int minibatch_size
) {
    const int num_devices = static_cast<int>(x_ptrs.size());

    switch (num_devices) {
        case 4:
            return dispatch_mlp_swiglu_combiner<4, utils::RoutedPrecision::MXFP8, ACTIVATION>::dispatch_mlp_swiglu_combine_bwd_mxfp8(
                d_y_buffer, d_y_buffer_ptrs, d_x_routed_buffer, d_x_routed_buffer_ptrs,
                router_weight_buffer, router_weight_buffer_ptrs, d_router_weight_buffer, d_router_weight_buffer_ptrs,
                w_shared_gate, w_routed_gate_T, w_routed_gate_T_sc,
                w_shared_up, w_routed_up_T, w_routed_up_T_sc,
                w_shared_down, w_routed_down_T, w_routed_down_T_sc,
                x_fp8_t_routed, x_sc_t_routed,
                gate_shared, gate_fp8_routed, gate_sc_routed,
                up_shared, up_fp8_routed, up_sc_routed,
                hidden_shared, hidden_fp8_t_routed, hidden_sc_t_routed,
                x, x_ptrs,
                w_routed_gate, w_routed_gate_sc,
                w_routed_up, w_routed_up_sc,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, num_comm_sms, macrobatch_size, minibatch_size);
        case 8:
            return dispatch_mlp_swiglu_combiner<8, utils::RoutedPrecision::MXFP8, ACTIVATION>::dispatch_mlp_swiglu_combine_bwd_mxfp8(
                d_y_buffer, d_y_buffer_ptrs, d_x_routed_buffer, d_x_routed_buffer_ptrs,
                router_weight_buffer, router_weight_buffer_ptrs, d_router_weight_buffer, d_router_weight_buffer_ptrs,
                w_shared_gate, w_routed_gate_T, w_routed_gate_T_sc,
                w_shared_up, w_routed_up_T, w_routed_up_T_sc,
                w_shared_down, w_routed_down_T, w_routed_down_T_sc,
                x_fp8_t_routed, x_sc_t_routed,
                gate_shared, gate_fp8_routed, gate_sc_routed,
                up_shared, up_fp8_routed, up_sc_routed,
                hidden_shared, hidden_fp8_t_routed, hidden_sc_t_routed,
                x, x_ptrs,
                w_routed_gate, w_routed_gate_sc,
                w_routed_up, w_routed_up_sc,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, num_comm_sms, macrobatch_size, minibatch_size);
        case 16:
            return dispatch_mlp_swiglu_combiner<16, utils::RoutedPrecision::MXFP8, ACTIVATION>::dispatch_mlp_swiglu_combine_bwd_mxfp8(
                d_y_buffer, d_y_buffer_ptrs, d_x_routed_buffer, d_x_routed_buffer_ptrs,
                router_weight_buffer, router_weight_buffer_ptrs, d_router_weight_buffer, d_router_weight_buffer_ptrs,
                w_shared_gate, w_routed_gate_T, w_routed_gate_T_sc,
                w_shared_up, w_routed_up_T, w_routed_up_T_sc,
                w_shared_down, w_routed_down_T, w_routed_down_T_sc,
                x_fp8_t_routed, x_sc_t_routed,
                gate_shared, gate_fp8_routed, gate_sc_routed,
                up_shared, up_fp8_routed, up_sc_routed,
                hidden_shared, hidden_fp8_t_routed, hidden_sc_t_routed,
                x, x_ptrs,
                w_routed_gate, w_routed_gate_sc,
                w_routed_up, w_routed_up_sc,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, num_comm_sms, macrobatch_size, minibatch_size);
        case 32:
            return dispatch_mlp_swiglu_combiner<32, utils::RoutedPrecision::MXFP8, ACTIVATION>::dispatch_mlp_swiglu_combine_bwd_mxfp8(
                d_y_buffer, d_y_buffer_ptrs, d_x_routed_buffer, d_x_routed_buffer_ptrs,
                router_weight_buffer, router_weight_buffer_ptrs, d_router_weight_buffer, d_router_weight_buffer_ptrs,
                w_shared_gate, w_routed_gate_T, w_routed_gate_T_sc,
                w_shared_up, w_routed_up_T, w_routed_up_T_sc,
                w_shared_down, w_routed_down_T, w_routed_down_T_sc,
                x_fp8_t_routed, x_sc_t_routed,
                gate_shared, gate_fp8_routed, gate_sc_routed,
                up_shared, up_fp8_routed, up_sc_routed,
                hidden_shared, hidden_fp8_t_routed, hidden_sc_t_routed,
                x, x_ptrs,
                w_routed_gate, w_routed_gate_sc,
                w_routed_up, w_routed_up_sc,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, num_comm_sms, macrobatch_size, minibatch_size);
        case 64:
            return dispatch_mlp_swiglu_combiner<64, utils::RoutedPrecision::MXFP8, ACTIVATION>::dispatch_mlp_swiglu_combine_bwd_mxfp8(
                d_y_buffer, d_y_buffer_ptrs, d_x_routed_buffer, d_x_routed_buffer_ptrs,
                router_weight_buffer, router_weight_buffer_ptrs, d_router_weight_buffer, d_router_weight_buffer_ptrs,
                w_shared_gate, w_routed_gate_T, w_routed_gate_T_sc,
                w_shared_up, w_routed_up_T, w_routed_up_T_sc,
                w_shared_down, w_routed_down_T, w_routed_down_T_sc,
                x_fp8_t_routed, x_sc_t_routed,
                gate_shared, gate_fp8_routed, gate_sc_routed,
                up_shared, up_fp8_routed, up_sc_routed,
                hidden_shared, hidden_fp8_t_routed, hidden_sc_t_routed,
                x, x_ptrs,
                w_routed_gate, w_routed_gate_sc,
                w_routed_up, w_routed_up_sc,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, num_comm_sms, macrobatch_size, minibatch_size);
        default:
            throw std::runtime_error("MoK: dispatch_mlp_swiglu_combine_bwd_mxfp8 unsupported num_devices=" +
                                     std::to_string(num_devices) + " (supported: 4, 8, 16, 32, 64)");
    }
}

template <GluActivation ACTIVATION>
static __host__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
dispatch_mlp_glu_combine_bwd_bf16(
    const at::Tensor &d_y_buffer,
    const std::vector<int64_t> &d_y_buffer_ptrs,
    const at::Tensor &d_x_routed_buffer,
    const std::vector<int64_t> &d_x_routed_buffer_ptrs,
    const at::Tensor &router_weight_buffer,
    const std::vector<int64_t> &router_weight_buffer_ptrs,
    const at::Tensor &d_router_weight_buffer,
    const std::vector<int64_t> &d_router_weight_buffer_ptrs,
    const at::Tensor &w_shared_gate,
    const at::Tensor &w_routed_gate,
    const at::Tensor &w_shared_up,
    const at::Tensor &w_routed_up,
    const at::Tensor &w_shared_down,
    const at::Tensor &w_routed_down,
    const at::Tensor &x_routed,
    const at::Tensor &gate_shared,
    const at::Tensor &gate_routed,
    const at::Tensor &up_shared,
    const at::Tensor &up_routed,
    const at::Tensor &hidden_shared,
    const at::Tensor &hidden_routed,
    const at::Tensor &x,
    const std::vector<int64_t> &x_ptrs,
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens,
    const at::Tensor &tokens_per_expert,
    int topk,
    int num_comm_sms,
    int macrobatch_size,
    int minibatch_size
) {
    const int num_devices = static_cast<int>(x_ptrs.size());
    switch (num_devices) {
        case 4:
            return dispatch_mlp_swiglu_combiner<4, utils::RoutedPrecision::BF16, ACTIVATION>::dispatch_mlp_swiglu_combine_bwd_bf16(
                d_y_buffer, d_y_buffer_ptrs, d_x_routed_buffer, d_x_routed_buffer_ptrs,
                router_weight_buffer, router_weight_buffer_ptrs, d_router_weight_buffer, d_router_weight_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down,
                x_routed, gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, x, x_ptrs,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, num_comm_sms, macrobatch_size, minibatch_size);
        case 8:
            return dispatch_mlp_swiglu_combiner<8, utils::RoutedPrecision::BF16, ACTIVATION>::dispatch_mlp_swiglu_combine_bwd_bf16(
                d_y_buffer, d_y_buffer_ptrs, d_x_routed_buffer, d_x_routed_buffer_ptrs,
                router_weight_buffer, router_weight_buffer_ptrs, d_router_weight_buffer, d_router_weight_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down,
                x_routed, gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, x, x_ptrs,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, num_comm_sms, macrobatch_size, minibatch_size);
        case 16:
            return dispatch_mlp_swiglu_combiner<16, utils::RoutedPrecision::BF16, ACTIVATION>::dispatch_mlp_swiglu_combine_bwd_bf16(
                d_y_buffer, d_y_buffer_ptrs, d_x_routed_buffer, d_x_routed_buffer_ptrs,
                router_weight_buffer, router_weight_buffer_ptrs, d_router_weight_buffer, d_router_weight_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down,
                x_routed, gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, x, x_ptrs,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, num_comm_sms, macrobatch_size, minibatch_size);
        case 32:
            return dispatch_mlp_swiglu_combiner<32, utils::RoutedPrecision::BF16, ACTIVATION>::dispatch_mlp_swiglu_combine_bwd_bf16(
                d_y_buffer, d_y_buffer_ptrs, d_x_routed_buffer, d_x_routed_buffer_ptrs,
                router_weight_buffer, router_weight_buffer_ptrs, d_router_weight_buffer, d_router_weight_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down,
                x_routed, gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, x, x_ptrs,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, num_comm_sms, macrobatch_size, minibatch_size);
        case 64:
            return dispatch_mlp_swiglu_combiner<64, utils::RoutedPrecision::BF16, ACTIVATION>::dispatch_mlp_swiglu_combine_bwd_bf16(
                d_y_buffer, d_y_buffer_ptrs, d_x_routed_buffer, d_x_routed_buffer_ptrs,
                router_weight_buffer, router_weight_buffer_ptrs, d_router_weight_buffer, d_router_weight_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down,
                x_routed, gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, x, x_ptrs,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, num_comm_sms, macrobatch_size, minibatch_size);
        default:
            throw std::runtime_error("MoK: dispatch_mlp_swiglu_combine_bwd_bf16 unsupported num_devices=" +
                                     std::to_string(num_devices) + " (supported: 4, 8, 16, 32, 64)");
    }
}
