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
using quant_bf16_tile = mxfp8::globals::x_bf16_tile; // st_bf<128, 128, false>
using quant_fp8_tile = mxfp8::globals::x_fp8_tile;   // st_fp8e4m3<128, 128, false>
using quant_sc_tile = mxfp8::globals::x_sc_tile;     // st_fp8e8m0<32, 16, false>

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
    const float swiglu_limit;
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
    const float swiglu_limit;
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

struct globals_recompute_forward_context {
    // Recomputed forward activations
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

    // Symmetric buffers
    activation_bf16_pgl x_routed_send_buffer; // (num_local_tokens, H)

    // Weights
    weight_bf16_gl w_shared_gate;             // (I, H)
    routed_weight_gl w_routed_gate;           // (num_local_experts, I, H)
    routed_sc_gl w_routed_gate_sc;            // MXFP8 only: (num_local_experts * I / 128, H / 128, 32, 16)
    weight_bf16_gl w_shared_up;               // (I, H)
    routed_weight_gl w_routed_up;             // (num_local_experts, I, H)
    routed_sc_gl w_routed_up_sc;              // MXFP8 only: (num_local_experts * I / 128, H / 128, 32, 16)

    // Schedules
    index_gl schedule_peer_rank;              // (schedule_capacity,)
    index_gl schedule_peer_token_idx;         // (schedule_capacity,)
    index_gl num_tokens;                      // (1,)
    index_gl tokens_per_expert;               // (num_local_experts,)

    // Barrier
    index_gl gate_up_tile_ready;              // (shared_gate_up_tasks + routed_gate_up_tasks,)
    index_gl hidden_row_block_ready;          // (shared_row_blocks + routed_row_blocks,)
    index_gl x_routed_ready;                  // (num_minibatches,)

    const int topk;
    const float swiglu_limit;
    const int num_comm_sms;
    const int macrobatch_size;
    const int minibatch_size;

    __host__ inline dim3 grid() const {
        const int routed_capacity = min(schedule_peer_rank.cols(), macrobatch_size);
        const int num_minibatches = (routed_capacity + minibatch_size - 1) / minibatch_size;
        const int shared_row_blocks = x_shared.rows() / config::MLP_Mb;
        const int minibatch_routed_row_blocks = minibatch_size / config::MLP_Mb;
        const int shared_gate_up_tasks = shared_row_blocks * (w_shared_gate.rows() / config::MLP_Nb);
        const int minibatch_routed_gate_up_tasks = minibatch_routed_row_blocks * (w_routed_gate.rows() / config::MLP_Nb);
        const int shared_swiglu_tiles = (hidden_shared.rows() / config::SWIGLU_Mb) * (hidden_shared.cols() / config::SWIGLU_Nb);
        const int minibatch_routed_swiglu_tiles = (minibatch_size / config::SWIGLU_Mb) * (hidden_fp8_routed.cols() / config::SWIGLU_Nb);
        const int shared_swiglu_tasks = (shared_swiglu_tiles + config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH);
        const int minibatch_routed_swiglu_tasks = (minibatch_routed_swiglu_tiles + config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH - 1) / (config::CLUSTER_SIZE * config::SWIGLU_FWD_PIPE_DEPTH);
        const int shared_tasks = 2 * shared_gate_up_tasks + shared_swiglu_tasks;
        const int minibatch_tasks = 2 * minibatch_routed_gate_up_tasks + minibatch_routed_swiglu_tasks;
        return dim3(config::CLUSTER_SIZE * (shared_tasks + num_minibatches * minibatch_tasks) + num_comm_sms);
    }
};
