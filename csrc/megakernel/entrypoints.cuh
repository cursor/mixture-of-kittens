#pragma once

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include "megakernel.cuh"

template <GluActivation ACTIVATION>
static __host__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                           at::Tensor, at::Tensor, at::Tensor>
dispatch_mlp_swiglu_combine_fwd_mxfp8_entrypoint(
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
    std::optional<float> swiglu_limit,
    int num_comm_sms,
    int macrobatch_size,
    int minibatch_size
) {
    const int num_devices = static_cast<int>(x_ptrs.size());

    switch (num_devices) {
        case 1:
            return dispatch_mlp_swiglu_combiner<1, RoutedPrecision::MXFP8, ACTIVATION>::dispatch_mlp_swiglu_combine_fwd_mxfp8(
                x, x_ptrs, combine_buffer, combine_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_routed_gate_sc,
                w_shared_up, w_routed_up, w_routed_up_sc,
                w_shared_down, w_routed_down, w_routed_down_sc,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        case 4:
            return dispatch_mlp_swiglu_combiner<4, RoutedPrecision::MXFP8, ACTIVATION>::dispatch_mlp_swiglu_combine_fwd_mxfp8(
                x, x_ptrs, combine_buffer, combine_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_routed_gate_sc,
                w_shared_up, w_routed_up, w_routed_up_sc,
                w_shared_down, w_routed_down, w_routed_down_sc,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        case 8:
            return dispatch_mlp_swiglu_combiner<8, RoutedPrecision::MXFP8, ACTIVATION>::dispatch_mlp_swiglu_combine_fwd_mxfp8(
                x, x_ptrs, combine_buffer, combine_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_routed_gate_sc,
                w_shared_up, w_routed_up, w_routed_up_sc,
                w_shared_down, w_routed_down, w_routed_down_sc,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        case 16:
            return dispatch_mlp_swiglu_combiner<16, RoutedPrecision::MXFP8, ACTIVATION>::dispatch_mlp_swiglu_combine_fwd_mxfp8(
                x, x_ptrs, combine_buffer, combine_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_routed_gate_sc,
                w_shared_up, w_routed_up, w_routed_up_sc,
                w_shared_down, w_routed_down, w_routed_down_sc,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        case 32:
            return dispatch_mlp_swiglu_combiner<32, RoutedPrecision::MXFP8, ACTIVATION>::dispatch_mlp_swiglu_combine_fwd_mxfp8(
                x, x_ptrs, combine_buffer, combine_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_routed_gate_sc,
                w_shared_up, w_routed_up, w_routed_up_sc,
                w_shared_down, w_routed_down, w_routed_down_sc,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        case 64:
            return dispatch_mlp_swiglu_combiner<64, RoutedPrecision::MXFP8, ACTIVATION>::dispatch_mlp_swiglu_combine_fwd_mxfp8(
                x, x_ptrs, combine_buffer, combine_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_routed_gate_sc,
                w_shared_up, w_routed_up, w_routed_up_sc,
                w_shared_down, w_routed_down, w_routed_down_sc,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        default:
            throw std::runtime_error("MoK: dispatch_mlp_swiglu_combine_fwd_mxfp8 unsupported num_devices=" +
                                     std::to_string(num_devices) + " (supported: 1, 4, 8, 16, 32, 64)");
    }
}

template <GluActivation ACTIVATION>
static __host__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                           at::Tensor, at::Tensor, at::Tensor, at::Tensor>
dispatch_mlp_swiglu_combine_fwd_bf16_entrypoint(
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
    std::optional<float> swiglu_limit,
    int num_comm_sms,
    int macrobatch_size,
    int minibatch_size
) {
    const int num_devices = static_cast<int>(x_ptrs.size());
    switch (num_devices) {
        case 1:
            return dispatch_mlp_swiglu_combiner<1, RoutedPrecision::BF16, ACTIVATION>::dispatch_mlp_swiglu_combine_fwd_bf16(
                x, x_ptrs, combine_buffer, combine_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        case 4:
            return dispatch_mlp_swiglu_combiner<4, RoutedPrecision::BF16, ACTIVATION>::dispatch_mlp_swiglu_combine_fwd_bf16(
                x, x_ptrs, combine_buffer, combine_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        case 8:
            return dispatch_mlp_swiglu_combiner<8, RoutedPrecision::BF16, ACTIVATION>::dispatch_mlp_swiglu_combine_fwd_bf16(
                x, x_ptrs, combine_buffer, combine_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        case 16:
            return dispatch_mlp_swiglu_combiner<16, RoutedPrecision::BF16, ACTIVATION>::dispatch_mlp_swiglu_combine_fwd_bf16(
                x, x_ptrs, combine_buffer, combine_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        case 32:
            return dispatch_mlp_swiglu_combiner<32, RoutedPrecision::BF16, ACTIVATION>::dispatch_mlp_swiglu_combine_fwd_bf16(
                x, x_ptrs, combine_buffer, combine_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        case 64:
            return dispatch_mlp_swiglu_combiner<64, RoutedPrecision::BF16, ACTIVATION>::dispatch_mlp_swiglu_combine_fwd_bf16(
                x, x_ptrs, combine_buffer, combine_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        default:
            throw std::runtime_error("MoK: dispatch_mlp_swiglu_combine_fwd_bf16 unsupported num_devices=" +
                                     std::to_string(num_devices) + " (supported: 1, 4, 8, 16, 32, 64)");
    }
}

template <GluActivation ACTIVATION>
static __host__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
dispatch_mlp_swiglu_combine_bwd_mxfp8_entrypoint(
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
    std::optional<float> swiglu_limit,
    int num_comm_sms,
    int macrobatch_size,
    int minibatch_size
) {
    const int num_devices = static_cast<int>(x_ptrs.size());

    switch (num_devices) {
        case 1:
            return dispatch_mlp_swiglu_combiner<1, RoutedPrecision::MXFP8, ACTIVATION>::dispatch_mlp_swiglu_combine_bwd_mxfp8(
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
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        case 4:
            return dispatch_mlp_swiglu_combiner<4, RoutedPrecision::MXFP8, ACTIVATION>::dispatch_mlp_swiglu_combine_bwd_mxfp8(
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
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        case 8:
            return dispatch_mlp_swiglu_combiner<8, RoutedPrecision::MXFP8, ACTIVATION>::dispatch_mlp_swiglu_combine_bwd_mxfp8(
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
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        case 16:
            return dispatch_mlp_swiglu_combiner<16, RoutedPrecision::MXFP8, ACTIVATION>::dispatch_mlp_swiglu_combine_bwd_mxfp8(
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
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        case 32:
            return dispatch_mlp_swiglu_combiner<32, RoutedPrecision::MXFP8, ACTIVATION>::dispatch_mlp_swiglu_combine_bwd_mxfp8(
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
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        case 64:
            return dispatch_mlp_swiglu_combiner<64, RoutedPrecision::MXFP8, ACTIVATION>::dispatch_mlp_swiglu_combine_bwd_mxfp8(
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
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        default:
            throw std::runtime_error("MoK: dispatch_mlp_swiglu_combine_bwd_mxfp8 unsupported num_devices=" +
                                     std::to_string(num_devices) + " (supported: 1, 4, 8, 16, 32, 64)");
    }
}

template <GluActivation ACTIVATION>
static __host__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
dispatch_mlp_swiglu_combine_bwd_bf16_entrypoint(
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
    std::optional<float> swiglu_limit,
    int num_comm_sms,
    int macrobatch_size,
    int minibatch_size
) {
    const int num_devices = static_cast<int>(x_ptrs.size());
    switch (num_devices) {
        case 1:
            return dispatch_mlp_swiglu_combiner<1, RoutedPrecision::BF16, ACTIVATION>::dispatch_mlp_swiglu_combine_bwd_bf16(
                d_y_buffer, d_y_buffer_ptrs, d_x_routed_buffer, d_x_routed_buffer_ptrs,
                router_weight_buffer, router_weight_buffer_ptrs, d_router_weight_buffer, d_router_weight_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down,
                x_routed, gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, x, x_ptrs,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        case 4:
            return dispatch_mlp_swiglu_combiner<4, RoutedPrecision::BF16, ACTIVATION>::dispatch_mlp_swiglu_combine_bwd_bf16(
                d_y_buffer, d_y_buffer_ptrs, d_x_routed_buffer, d_x_routed_buffer_ptrs,
                router_weight_buffer, router_weight_buffer_ptrs, d_router_weight_buffer, d_router_weight_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down,
                x_routed, gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, x, x_ptrs,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        case 8:
            return dispatch_mlp_swiglu_combiner<8, RoutedPrecision::BF16, ACTIVATION>::dispatch_mlp_swiglu_combine_bwd_bf16(
                d_y_buffer, d_y_buffer_ptrs, d_x_routed_buffer, d_x_routed_buffer_ptrs,
                router_weight_buffer, router_weight_buffer_ptrs, d_router_weight_buffer, d_router_weight_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down,
                x_routed, gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, x, x_ptrs,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        case 16:
            return dispatch_mlp_swiglu_combiner<16, RoutedPrecision::BF16, ACTIVATION>::dispatch_mlp_swiglu_combine_bwd_bf16(
                d_y_buffer, d_y_buffer_ptrs, d_x_routed_buffer, d_x_routed_buffer_ptrs,
                router_weight_buffer, router_weight_buffer_ptrs, d_router_weight_buffer, d_router_weight_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down,
                x_routed, gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, x, x_ptrs,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        case 32:
            return dispatch_mlp_swiglu_combiner<32, RoutedPrecision::BF16, ACTIVATION>::dispatch_mlp_swiglu_combine_bwd_bf16(
                d_y_buffer, d_y_buffer_ptrs, d_x_routed_buffer, d_x_routed_buffer_ptrs,
                router_weight_buffer, router_weight_buffer_ptrs, d_router_weight_buffer, d_router_weight_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down,
                x_routed, gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, x, x_ptrs,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        case 64:
            return dispatch_mlp_swiglu_combiner<64, RoutedPrecision::BF16, ACTIVATION>::dispatch_mlp_swiglu_combine_bwd_bf16(
                d_y_buffer, d_y_buffer_ptrs, d_x_routed_buffer, d_x_routed_buffer_ptrs,
                router_weight_buffer, router_weight_buffer_ptrs, d_router_weight_buffer, d_router_weight_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down,
                x_routed, gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, x, x_ptrs,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        default:
            throw std::runtime_error("MoK: dispatch_mlp_swiglu_combine_bwd_bf16 unsupported num_devices=" +
                                     std::to_string(num_devices) + " (supported: 1, 4, 8, 16, 32, 64)");
    }
}

static __host__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
recompute_forward_context_mxfp8_entrypoint(
    // Input buffers
    const at::Tensor &x,
    const std::vector<int64_t> &x_ptrs,

    // Weights
    const at::Tensor &w_shared_gate,
    const at::Tensor &w_routed_gate,
    const at::Tensor &w_routed_gate_sc,
    const at::Tensor &w_shared_up,
    const at::Tensor &w_routed_up,
    const at::Tensor &w_routed_up_sc,

    // Dispatch schedule
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens,
    const at::Tensor &tokens_per_expert,

    // Metadata
    int topk,
    std::optional<float> swiglu_limit,
    int num_comm_sms,
    int macrobatch_size,
    int minibatch_size
) {
    const int num_devices = static_cast<int>(x_ptrs.size());

    switch (num_devices) {
        case 1:
            return dispatch_mlp_swiglu_combiner<1>::recompute_forward_context_mxfp8(
                x, x_ptrs,
                w_shared_gate, w_routed_gate, w_routed_gate_sc,
                w_shared_up, w_routed_up, w_routed_up_sc,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        case 4:
            return dispatch_mlp_swiglu_combiner<4>::recompute_forward_context_mxfp8(
                x, x_ptrs,
                w_shared_gate, w_routed_gate, w_routed_gate_sc,
                w_shared_up, w_routed_up, w_routed_up_sc,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        case 8:
            return dispatch_mlp_swiglu_combiner<8>::recompute_forward_context_mxfp8(
                x, x_ptrs,
                w_shared_gate, w_routed_gate, w_routed_gate_sc,
                w_shared_up, w_routed_up, w_routed_up_sc,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        case 16:
            return dispatch_mlp_swiglu_combiner<16>::recompute_forward_context_mxfp8(
                x, x_ptrs,
                w_shared_gate, w_routed_gate, w_routed_gate_sc,
                w_shared_up, w_routed_up, w_routed_up_sc,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        case 32:
            return dispatch_mlp_swiglu_combiner<32>::recompute_forward_context_mxfp8(
                x, x_ptrs,
                w_shared_gate, w_routed_gate, w_routed_gate_sc,
                w_shared_up, w_routed_up, w_routed_up_sc,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        case 64:
            return dispatch_mlp_swiglu_combiner<64>::recompute_forward_context_mxfp8(
                x, x_ptrs,
                w_shared_gate, w_routed_gate, w_routed_gate_sc,
                w_shared_up, w_routed_up, w_routed_up_sc,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        default:
            throw std::runtime_error("MoK: recompute_forward_context_mxfp8 unsupported num_devices=" +
                                     std::to_string(num_devices) + " (supported: 1, 4, 8, 16, 32, 64)");
    }
}

static __host__ std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                           at::Tensor, at::Tensor, at::Tensor>
recompute_forward_context_bf16_entrypoint(
    // Input buffers
    const at::Tensor &x,
    const std::vector<int64_t> &x_ptrs,

    // Weights
    const at::Tensor &w_shared_gate,
    const at::Tensor &w_routed_gate,
    const at::Tensor &w_shared_up,
    const at::Tensor &w_routed_up,

    // Dispatch schedule
    const at::Tensor &schedule_peer_rank,
    const at::Tensor &schedule_peer_token_idx,
    const at::Tensor &num_tokens,
    const at::Tensor &tokens_per_expert,

    // Metadata
    int topk,
    std::optional<float> swiglu_limit,
    int num_comm_sms,
    int macrobatch_size,
    int minibatch_size
) {
    const int num_devices = static_cast<int>(x_ptrs.size());

    switch (num_devices) {
        case 1:
            return dispatch_mlp_swiglu_combiner<1, RoutedPrecision::BF16>::recompute_forward_context_bf16(
                x, x_ptrs,
                w_shared_gate, w_routed_gate, w_shared_up, w_routed_up,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        case 4:
            return dispatch_mlp_swiglu_combiner<4, RoutedPrecision::BF16>::recompute_forward_context_bf16(
                x, x_ptrs,
                w_shared_gate, w_routed_gate, w_shared_up, w_routed_up,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        case 8:
            return dispatch_mlp_swiglu_combiner<8, RoutedPrecision::BF16>::recompute_forward_context_bf16(
                x, x_ptrs,
                w_shared_gate, w_routed_gate, w_shared_up, w_routed_up,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        case 16:
            return dispatch_mlp_swiglu_combiner<16, RoutedPrecision::BF16>::recompute_forward_context_bf16(
                x, x_ptrs,
                w_shared_gate, w_routed_gate, w_shared_up, w_routed_up,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        case 32:
            return dispatch_mlp_swiglu_combiner<32, RoutedPrecision::BF16>::recompute_forward_context_bf16(
                x, x_ptrs,
                w_shared_gate, w_routed_gate, w_shared_up, w_routed_up,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        case 64:
            return dispatch_mlp_swiglu_combiner<64, RoutedPrecision::BF16>::recompute_forward_context_bf16(
                x, x_ptrs,
                w_shared_gate, w_routed_gate, w_shared_up, w_routed_up,
                schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size);
        default:
            throw std::runtime_error("MoK: recompute_forward_context_bf16 unsupported num_devices=" +
                                     std::to_string(num_devices) + " (supported: 1, 4, 8, 16, 32, 64)");
    }
}
