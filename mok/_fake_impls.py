import torch

from . import ops as _ops  # noqa: F401


@torch.library.register_fake("mok::all_gather_top_experts")
def _all_gather_top_experts_fake(
    top_experts: torch.Tensor,
    all_gather_top_experts_buffer: torch.Tensor,
    all_gather_top_experts_buffer_multicast_ptr: int,
    rank: int,
    chunk_bytes: int,
) -> None:
    return None


@torch.library.register_fake("mok::barrier_all")
def _barrier_all_fake(
    barrier_buffer: torch.Tensor,
    barrier_buffer_ptrs: list[int],
    barrier_buffer_multicast_ptr: int,
    target: torch.Tensor,
) -> None:
    return None


@torch.library.register_fake("mok::schedule")
def _schedule_fake(
    topk_all: torch.Tensor,
    num_local_experts: int,
    schedule_capacity: int,
    rank: int,
) -> tuple[
    torch.Tensor, torch.Tensor,  # schedule_peer_rank, schedule_peer_token_idx
    torch.Tensor, torch.Tensor,  # num_tokens, tokens_per_expert
]:
    return (
        topk_all.new_empty((schedule_capacity,), dtype=torch.int32),
        topk_all.new_empty((schedule_capacity,), dtype=torch.int32),
        topk_all.new_empty((1,), dtype=torch.int32),
        topk_all.new_empty((num_local_experts,), dtype=torch.int32),
    )


@torch.library.register_fake("mok::mxfp8_quantize")
def _mxfp8_quantize_fake(
    x_bf16: torch.Tensor,
    return_normal: bool,
    return_transposed: bool,
) -> tuple[
    torch.Tensor | None, torch.Tensor | None,  # x_fp8, x_sc
    torch.Tensor | None, torch.Tensor | None,  # x_fp8_t, x_sc_t
]:
    expert_count = x_bf16.shape[0] if x_bf16.ndim == 3 else 1
    rows = x_bf16.shape[-2]
    columns = x_bf16.shape[-1]
    x_fp8_t_shape = ((expert_count, columns, rows)
                     if x_bf16.ndim == 3 else (columns, rows))
    return (
        x_bf16.new_empty(x_bf16.shape, dtype=torch.float8_e4m3fn) if return_normal else None,
        x_bf16.new_empty((expert_count * rows // 128, columns // 128, 32, 16), dtype=torch.uint8) if return_normal else None,
        x_bf16.new_empty(x_fp8_t_shape, dtype=torch.float8_e4m3fn) if return_transposed else None,
        x_bf16.new_empty((expert_count * columns // 128, rows // 128, 32, 16), dtype=torch.uint8) if return_transposed else None,
    )


@torch.library.register_fake("mok::dispatch_mlp_swiglu_combine_fwd_mxfp8")
def _dispatch_mlp_swiglu_combine_fwd_mxfp8_fake(
    x: torch.Tensor,
    x_ptrs: list[int],
    combine_buffer: torch.Tensor,
    combine_buffer_ptrs: list[int],
    w_shared_gate: torch.Tensor,
    w_routed_gate: torch.Tensor,
    w_routed_gate_sc: torch.Tensor,
    w_shared_up: torch.Tensor,
    w_routed_up: torch.Tensor,
    w_routed_up_sc: torch.Tensor,
    w_shared_down: torch.Tensor,
    w_routed_down: torch.Tensor,
    w_routed_down_sc: torch.Tensor,
    schedule_peer_rank: torch.Tensor,
    schedule_peer_token_idx: torch.Tensor,
    num_tokens: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk: int,
    swiglu_limit: float | None,
    num_comm_sms: int,
    macrobatch_size: int,
    minibatch_size: int,
) -> tuple[
    torch.Tensor, torch.Tensor,  # x_fp8_t_routed, x_sc_t_routed
    torch.Tensor, torch.Tensor, torch.Tensor,  # gate_shared, gate_fp8_routed, gate_sc_routed
    torch.Tensor, torch.Tensor, torch.Tensor,  # up_shared, up_fp8_routed, up_sc_routed
    torch.Tensor, torch.Tensor, torch.Tensor,  # hidden_shared, hidden_fp8_t_routed, hidden_sc_t_routed
    torch.Tensor, torch.Tensor,  # y_shared, y_routed
]:
    num_local_tokens, hidden_size = x.shape
    intermediate_size = w_shared_gate.shape[0]
    return (
        x.new_empty((hidden_size, macrobatch_size), dtype=torch.float8_e4m3fn),
        x.new_empty((hidden_size // 128, macrobatch_size // 128, 32, 16), dtype=torch.uint8),
        x.new_empty((num_local_tokens, intermediate_size)),
        x.new_empty((macrobatch_size, intermediate_size), dtype=torch.float8_e4m3fn),
        x.new_empty((macrobatch_size // 128, intermediate_size // 128, 32, 16), dtype=torch.uint8),
        x.new_empty((num_local_tokens, intermediate_size)),
        x.new_empty((macrobatch_size, intermediate_size), dtype=torch.float8_e4m3fn),
        x.new_empty((macrobatch_size // 128, intermediate_size // 128, 32, 16), dtype=torch.uint8),
        x.new_empty((num_local_tokens, intermediate_size)),
        x.new_empty((intermediate_size, macrobatch_size), dtype=torch.float8_e4m3fn),
        x.new_empty((intermediate_size // 128, macrobatch_size // 128, 32, 16), dtype=torch.uint8),
        x.new_empty(x.shape),
        x.new_empty((macrobatch_size, hidden_size)),
    )


@torch.library.register_fake("mok::dispatch_mlp_swiglu_combine_fwd_bf16")
def _dispatch_mlp_swiglu_combine_fwd_bf16_fake(
    x: torch.Tensor,
    x_ptrs: list[int],
    combine_buffer: torch.Tensor,
    combine_buffer_ptrs: list[int],
    w_shared_gate: torch.Tensor,
    w_routed_gate: torch.Tensor,
    w_shared_up: torch.Tensor,
    w_routed_up: torch.Tensor,
    w_shared_down: torch.Tensor,
    w_routed_down: torch.Tensor,
    schedule_peer_rank: torch.Tensor,
    schedule_peer_token_idx: torch.Tensor,
    num_tokens: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk: int,
    swiglu_limit: float | None,
    num_comm_sms: int,
    macrobatch_size: int,
    minibatch_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
]:
    num_local_tokens, hidden_size = x.shape
    intermediate_size = w_shared_gate.shape[0]
    return (
        x.new_empty((macrobatch_size, hidden_size)),
        x.new_empty((num_local_tokens, intermediate_size)),
        x.new_empty((macrobatch_size, intermediate_size)),
        x.new_empty((num_local_tokens, intermediate_size)),
        x.new_empty((macrobatch_size, intermediate_size)),
        x.new_empty((num_local_tokens, intermediate_size)),
        x.new_empty((macrobatch_size, intermediate_size)),
        x.new_empty(x.shape),
        x.new_empty((macrobatch_size, hidden_size)),
    )


@torch.library.register_fake("mok::dispatch_mlp_swiglu_combine_bwd_mxfp8")
def _dispatch_mlp_swiglu_combine_bwd_mxfp8_fake(
    d_y_buffer: torch.Tensor,
    d_y_buffer_ptrs: list[int],
    d_x_routed_buffer: torch.Tensor,
    d_x_routed_buffer_ptrs: list[int],
    router_weight_buffer: torch.Tensor,
    router_weight_buffer_ptrs: list[int],
    d_router_weight_buffer: torch.Tensor,
    d_router_weight_buffer_ptrs: list[int],
    w_shared_gate: torch.Tensor,
    w_routed_gate_T: torch.Tensor,
    w_routed_gate_T_sc: torch.Tensor,
    w_shared_up: torch.Tensor,
    w_routed_up_T: torch.Tensor,
    w_routed_up_T_sc: torch.Tensor,
    w_shared_down: torch.Tensor,
    w_routed_down_T: torch.Tensor,
    w_routed_down_T_sc: torch.Tensor,
    x_fp8_t_routed: torch.Tensor,
    x_sc_t_routed: torch.Tensor,
    gate_shared: torch.Tensor,
    gate_fp8_routed: torch.Tensor,
    gate_sc_routed: torch.Tensor,
    up_shared: torch.Tensor,
    up_fp8_routed: torch.Tensor,
    up_sc_routed: torch.Tensor,
    hidden_shared: torch.Tensor,
    hidden_fp8_t_routed: torch.Tensor,
    hidden_sc_t_routed: torch.Tensor,
    x: torch.Tensor,
    x_ptrs: list[int],
    w_routed_gate: torch.Tensor,
    w_routed_gate_sc: torch.Tensor,
    w_routed_up: torch.Tensor,
    w_routed_up_sc: torch.Tensor,
    schedule_peer_rank: torch.Tensor,
    schedule_peer_token_idx: torch.Tensor,
    num_tokens: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk: int,
    swiglu_limit: float | None,
    num_comm_sms: int,
    macrobatch_size: int,
    minibatch_size: int,
) -> tuple[
    torch.Tensor, torch.Tensor,  # d_x_shared, d_x_routed
    torch.Tensor, torch.Tensor, torch.Tensor,  # d_gate_shared, d_gate_fp8_routed, d_gate_sc_routed
    torch.Tensor, torch.Tensor, torch.Tensor,  # d_up_shared, d_up_fp8_routed, d_up_sc_routed
    torch.Tensor, torch.Tensor,  # d_hidden_shared, d_hidden_routed
    torch.Tensor, torch.Tensor,  # d_y_fp8_routed, d_y_sc_routed
    torch.Tensor, torch.Tensor,  # d_w_shared_gate, d_w_routed_gate
    torch.Tensor, torch.Tensor,  # d_w_shared_up, d_w_routed_up
    torch.Tensor, torch.Tensor,  # d_w_shared_down, d_w_routed_down
]:
    num_local_tokens, hidden_size = x.shape
    num_local_experts = w_routed_gate.shape[0]
    intermediate_size = w_shared_gate.shape[0]
    return (
        d_y_buffer.new_empty((num_local_tokens, hidden_size)),
        d_y_buffer.new_empty((macrobatch_size, hidden_size)),
        d_y_buffer.new_empty((num_local_tokens, intermediate_size)),
        d_y_buffer.new_empty((macrobatch_size, intermediate_size), dtype=torch.float8_e4m3fn),
        d_y_buffer.new_empty((macrobatch_size // 128, intermediate_size // 128, 32, 16), dtype=torch.uint8),
        d_y_buffer.new_empty((num_local_tokens, intermediate_size)),
        d_y_buffer.new_empty((macrobatch_size, intermediate_size), dtype=torch.float8_e4m3fn),
        d_y_buffer.new_empty((macrobatch_size // 128, intermediate_size // 128, 32, 16), dtype=torch.uint8),
        d_y_buffer.new_empty((num_local_tokens, intermediate_size)),
        d_y_buffer.new_empty((macrobatch_size, intermediate_size)),
        d_y_buffer.new_empty((macrobatch_size, hidden_size), dtype=torch.float8_e4m3fn),
        d_y_buffer.new_empty((macrobatch_size // 128, hidden_size // 128, 32, 16), dtype=torch.uint8),
        d_y_buffer.new_empty((intermediate_size, hidden_size)),
        d_y_buffer.new_empty((num_local_experts, intermediate_size, hidden_size)),
        d_y_buffer.new_empty((intermediate_size, hidden_size)),
        d_y_buffer.new_empty((num_local_experts, intermediate_size, hidden_size)),
        d_y_buffer.new_empty((hidden_size, intermediate_size)),
        d_y_buffer.new_empty((num_local_experts, hidden_size, intermediate_size)),
    )


@torch.library.register_fake("mok::dispatch_mlp_swiglu_combine_bwd_bf16")
def _dispatch_mlp_swiglu_combine_bwd_bf16_fake(
    d_y_buffer: torch.Tensor,
    d_y_buffer_ptrs: list[int],
    d_x_routed_buffer: torch.Tensor,
    d_x_routed_buffer_ptrs: list[int],
    router_weight_buffer: torch.Tensor,
    router_weight_buffer_ptrs: list[int],
    d_router_weight_buffer: torch.Tensor,
    d_router_weight_buffer_ptrs: list[int],
    w_shared_gate: torch.Tensor,
    w_routed_gate: torch.Tensor,
    w_shared_up: torch.Tensor,
    w_routed_up: torch.Tensor,
    w_shared_down: torch.Tensor,
    w_routed_down: torch.Tensor,
    x_routed: torch.Tensor,
    gate_shared: torch.Tensor,
    gate_routed: torch.Tensor,
    up_shared: torch.Tensor,
    up_routed: torch.Tensor,
    hidden_shared: torch.Tensor,
    hidden_routed: torch.Tensor,
    x: torch.Tensor,
    x_ptrs: list[int],
    schedule_peer_rank: torch.Tensor,
    schedule_peer_token_idx: torch.Tensor,
    num_tokens: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk: int,
    swiglu_limit: float | None,
    num_comm_sms: int,
    macrobatch_size: int,
    minibatch_size: int,
) -> tuple[
    torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
    torch.Tensor,
    torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
]:
    num_local_tokens, hidden_size = x.shape
    num_local_experts = w_routed_gate.shape[0]
    intermediate_size = w_shared_gate.shape[0]
    return (
        d_y_buffer.new_empty((num_local_tokens, hidden_size)),
        d_y_buffer.new_empty((macrobatch_size, hidden_size)),
        d_y_buffer.new_empty((num_local_tokens, intermediate_size)),
        d_y_buffer.new_empty((macrobatch_size, intermediate_size)),
        d_y_buffer.new_empty((num_local_tokens, intermediate_size)),
        d_y_buffer.new_empty((macrobatch_size, intermediate_size)),
        d_y_buffer.new_empty((num_local_tokens, intermediate_size)),
        d_y_buffer.new_empty((macrobatch_size, intermediate_size)),
        d_y_buffer.new_empty((macrobatch_size, hidden_size)),
        d_y_buffer.new_empty((intermediate_size, hidden_size)),
        d_y_buffer.new_empty((num_local_experts, intermediate_size, hidden_size)),
        d_y_buffer.new_empty((intermediate_size, hidden_size)),
        d_y_buffer.new_empty((num_local_experts, intermediate_size, hidden_size)),
        d_y_buffer.new_empty((hidden_size, intermediate_size)),
        d_y_buffer.new_empty((num_local_experts, hidden_size, intermediate_size)),
    )


# situ-glu has the same saved-tensor and gradient layouts as swiglu.
torch.library.register_fake("mok::dispatch_mlp_situglu_combine_fwd_mxfp8")(
    _dispatch_mlp_swiglu_combine_fwd_mxfp8_fake
)
torch.library.register_fake("mok::dispatch_mlp_situglu_combine_fwd_bf16")(
    _dispatch_mlp_swiglu_combine_fwd_bf16_fake
)
torch.library.register_fake("mok::dispatch_mlp_situglu_combine_bwd_mxfp8")(
    _dispatch_mlp_swiglu_combine_bwd_mxfp8_fake
)
torch.library.register_fake("mok::dispatch_mlp_situglu_combine_bwd_bf16")(
    _dispatch_mlp_swiglu_combine_bwd_bf16_fake
)


@torch.library.register_fake("mok::fwd_epilogue")
def _fwd_epilogue_fake(
    y_shared: torch.Tensor,
    combine_buffer: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:  # output
    return y_shared.new_empty(y_shared.shape)


@torch.library.register_fake("mok::bwd_epilogue")
def _bwd_epilogue_fake(
    d_x_shared: torch.Tensor,
    d_x_routed_buffer: torch.Tensor,
) -> torch.Tensor:  # d_x
    return d_x_shared.new_empty(d_x_shared.shape)
