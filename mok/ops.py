import torch

from . import _C


@torch.library.custom_op("mok::all_gather_top_experts", mutates_args=("all_gather_top_experts_buffer",))
def all_gather_top_experts(
    top_experts: torch.Tensor,
    all_gather_top_experts_buffer: torch.Tensor,
    all_gather_top_experts_buffer_multicast_ptr: int,
    rank: int,
    chunk_bytes: int,
) -> None:
    """All-gathers top-expert assignments across expert-parallel ranks.

    Inputs:
        top_experts:                                 int32 [num_local_tokens, topk]
        all_gather_top_experts_buffer:               int32 [ep_size, num_local_tokens, topk]
        all_gather_top_experts_buffer_multicast_ptr: int
        rank:                                        int
        chunk_bytes:                                 int

    Outputs:
        None
    """
    if (not top_experts.is_cuda or top_experts.dtype != torch.int32 or not top_experts.is_contiguous() or top_experts.ndim != 2):
        raise ValueError("top_experts must be contiguous CUDA int32 [num_local_tokens, topk]")
    if not all_gather_top_experts_buffer.is_cuda:
        raise ValueError("all_gather_top_experts_buffer must be a CUDA tensor")
    if all_gather_top_experts_buffer.dtype != torch.int32:
        raise TypeError("all_gather_top_experts_buffer must have dtype torch.int32")
    if not all_gather_top_experts_buffer.is_contiguous():
        raise ValueError("all_gather_top_experts_buffer must be contiguous")
    if all_gather_top_experts_buffer.ndim != 3:
        raise ValueError("all_gather_top_experts_buffer must have shape (ep_size, num_local_tokens, topk)")
    if any(size <= 0 for size in top_experts.shape):
        raise ValueError("top_experts dimensions must be positive")
    if any(size <= 0 for size in all_gather_top_experts_buffer.shape):
        raise ValueError("all_gather_top_experts_buffer dimensions must be positive")
    ep_size = all_gather_top_experts_buffer.shape[0]
    if ep_size not in (1, 4, 8, 16, 32, 64):
        raise ValueError("all_gather_top_experts_buffer ep_size must be one of 1, 4, 8, 16, 32, 64")
    if (all_gather_top_experts_buffer.device != top_experts.device
            or tuple(all_gather_top_experts_buffer.shape[1:]) != tuple(top_experts.shape)):
        raise ValueError("all_gather_top_experts_buffer must match top_experts shape and device")
    if type(all_gather_top_experts_buffer_multicast_ptr) is not int or all_gather_top_experts_buffer_multicast_ptr <= 0:
        raise TypeError("all_gather_top_experts_buffer_multicast_ptr must be a positive integer")
    if type(rank) is not int or not 0 <= rank < ep_size:
        raise ValueError("rank must be an integer in [0, ep_size)")
    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be a positive integer")
    if chunk_bytes % 16 != 0:
        raise ValueError("chunk_bytes must be divisible by 16")
    rank_buffer_bytes = top_experts.numel() * top_experts.element_size()
    if rank_buffer_bytes % chunk_bytes != 0:
        raise ValueError("chunk_bytes must divide one rank's route-buffer bytes")

    if ep_size == 1:
        all_gather_top_experts_buffer[0].copy_(top_experts)
    else:
        _C.all_gather_top_experts(top_experts, all_gather_top_experts_buffer, all_gather_top_experts_buffer_multicast_ptr, rank, chunk_bytes)


@torch.library.custom_op("mok::barrier_all", mutates_args=("barrier_buffer", "target"))
def barrier_all(
    barrier_buffer: torch.Tensor,
    barrier_buffer_ptrs: list[int],
    barrier_buffer_multicast_ptr: int,
    target: torch.Tensor,
) -> None:
    """Synchronizes all expert-parallel ranks at device-side.

    Inputs:
        barrier_buffer:               int32 [1]
        barrier_buffer_ptrs:          list[int] [ep_size]
        barrier_buffer_multicast_ptr: int
        target:                       int32 [1]

    Outputs:
        None
    """
    if not barrier_buffer.is_cuda:
        raise ValueError("barrier_buffer must be a CUDA tensor")
    if barrier_buffer.dtype != torch.int32:
        raise TypeError("barrier_buffer must have dtype torch.int32")
    if not barrier_buffer.is_contiguous():
        raise ValueError("barrier_buffer must be contiguous")
    if tuple(barrier_buffer.shape) != (1,):
        raise ValueError("barrier_buffer must have shape (1,)")
    if not isinstance(barrier_buffer_ptrs, list) or any(
        type(pointer) is not int or pointer <= 0 for pointer in barrier_buffer_ptrs):
        raise TypeError("barrier_buffer_ptrs must be a list of positive integers")
    ep_size = len(barrier_buffer_ptrs)
    if ep_size not in (1, 4, 8, 16, 32, 64):
        raise ValueError("barrier_buffer_ptrs length must be one of 1, 4, 8, 16, 32, 64")
    if type(barrier_buffer_multicast_ptr) is not int or barrier_buffer_multicast_ptr <= 0:
        raise TypeError("barrier_buffer_multicast_ptr must be a positive integer")
    if (not target.is_cuda or target.device != barrier_buffer.device
            or target.dtype != torch.int32 or not target.is_contiguous()
            or tuple(target.shape) != (1,)):
        raise ValueError("target must be contiguous int32 [1] on the barrier CUDA device")

    if ep_size == 1:
        return
    else:
        _C.barrier_all(barrier_buffer, barrier_buffer_ptrs, barrier_buffer_multicast_ptr, target)


@torch.library.custom_op("mok::schedule", mutates_args=())
def schedule(
    topk_all: torch.Tensor,
    num_local_experts: int,
    schedule_capacity: int,
    rank: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Builds the routed-token schedule for the current expert-parallel rank.

    Inputs:
        topk_all:          int32 [ep_size, num_local_tokens, topk]
        num_local_experts: int
        schedule_capacity: int
        rank:              int

    Outputs:
        schedule_peer_rank:      int32 [schedule_capacity]
        schedule_peer_token_idx: int32 [schedule_capacity]
        num_tokens:              int32 [1]
        tokens_per_expert:       int32 [num_local_experts]
    """
    if topk_all.ndim != 3:
        raise ValueError("topk_all must have shape (ep_size, num_local_tokens, topk)")
    ep_size, num_local_tokens, topk = topk_all.shape
    if ep_size not in (1, 4, 8, 16, 32, 64):
        raise ValueError("topk_all ep_size must be one of 1, 4, 8, 16, 32, 64")
    if num_local_tokens < 512 or num_local_tokens % 256 != 0:
        raise ValueError(
            "topk_all num_local_tokens must be at least 512 and divisible by 256"
        )
    if not 0 < topk <= 255:
        raise ValueError("topk_all topk must be in [1, 255]")
    if type(num_local_experts) is not int or num_local_experts <= 0:
        raise ValueError("num_local_experts must be a positive integer")
    if (type(schedule_capacity) is not int or schedule_capacity <= 0
            or schedule_capacity % 256 != 0):
        raise ValueError("schedule_capacity must be positive and divisible by 256")
    if schedule_capacity < num_local_tokens * topk:
        raise ValueError("schedule_capacity must hold at least one rank's routed tokens")
    if type(rank) is not int or not 0 <= rank < ep_size:
        raise ValueError("rank must be an integer in [0, ep_size)")

    return _C.schedule(topk_all, num_local_experts, schedule_capacity, rank)


@torch.library.custom_op(
    "mok::mxfp8_quantize", mutates_args=(),
    schema="(Tensor x_bf16, bool return_normal, bool return_transposed) -> (Tensor?, Tensor?, Tensor?, Tensor?)",
)
def mxfp8_quantize(
    x_bf16: torch.Tensor,
    return_normal: bool,
    return_transposed: bool,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Quantizes BF16 matrices to MXFP8 in normal and/or transposed layouts.

    Inputs:
        x_bf16:           bfloat16 [M, N] or [E, M, N]
        return_normal:     bool
        return_transposed: bool

    Outputs:
        x_fp8:   float8_e4m3fn [M, N] or [E, M, N] | None
        x_sc:    uint8 [E * M // 128, N // 128, 32, 16] | None
        x_fp8_t: float8_e4m3fn [N, M] or [E, N, M] | None
        x_sc_t:  uint8 [E * N // 128, M // 128, 32, 16] | None
    """
    if x_bf16.ndim not in (2, 3):
        raise ValueError("x_bf16 must have shape (M, N) or (E, M, N)")
    if any(size <= 0 for size in x_bf16.shape):
        raise ValueError("x_bf16 dimensions must be positive")
    if x_bf16.shape[-2] % 128 != 0 or x_bf16.shape[-1] % 128 != 0:
        raise ValueError("x_bf16 M and N dimensions must be divisible by 128")
    if type(return_normal) is not bool or type(return_transposed) is not bool:
        raise TypeError("return_normal and return_transposed must be booleans")
    if not return_normal and not return_transposed:
        raise ValueError("at least one quantized layout must be requested")

    return _C.mxfp8_quantize(x_bf16, return_normal, return_transposed)


@torch.library.custom_op("mok::dispatch_mlp_swiglu_combine_fwd_mxfp8", mutates_args=("combine_buffer",))
def dispatch_mlp_swiglu_combine_fwd_mxfp8(
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
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Runs the fused MXFP8 MoE forward pass.

    Inputs:
        x:                       bfloat16 [num_local_tokens, hidden_size]
        x_ptrs:                  list[int] [ep_size]
        combine_buffer:          bfloat16 [token_capacity * topk, hidden_size]
        combine_buffer_ptrs:     list[int] [ep_size]
        w_shared_gate:           bfloat16 [intermediate_size, hidden_size]
        w_routed_gate:           float8_e4m3fn [num_local_experts, intermediate_size, hidden_size]
        w_routed_gate_sc:        uint8 [num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16]
        w_shared_up:             bfloat16 [intermediate_size, hidden_size]
        w_routed_up:             float8_e4m3fn [num_local_experts, intermediate_size, hidden_size]
        w_routed_up_sc:          uint8 [num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16]
        w_shared_down:           bfloat16 [hidden_size, intermediate_size]
        w_routed_down:           float8_e4m3fn [num_local_experts, hidden_size, intermediate_size]
        w_routed_down_sc:        uint8 [num_local_experts * hidden_size // 128, intermediate_size // 128, 32, 16]
        schedule_peer_rank:      int32 [schedule_capacity]
        schedule_peer_token_idx: int32 [schedule_capacity]
        num_tokens:              int32 [1]
        tokens_per_expert:       int32 [num_local_experts]
        topk:                    int
        swiglu_limit:            float | None
        num_comm_sms:            int
        macrobatch_size:         int
        minibatch_size:          int

    Outputs:
        x_fp8_t_routed:      float8_e4m3fn [hidden_size, macrobatch_size]
        x_sc_t_routed:       uint8 [hidden_size // 128, macrobatch_size // 128, 32, 16]
        gate_shared:         bfloat16 [num_local_tokens, intermediate_size]
        gate_fp8_routed:     float8_e4m3fn [macrobatch_size, intermediate_size]
        gate_sc_routed:      uint8 [macrobatch_size // 128, intermediate_size // 128, 32, 16]
        up_shared:           bfloat16 [num_local_tokens, intermediate_size]
        up_fp8_routed:       float8_e4m3fn [macrobatch_size, intermediate_size]
        up_sc_routed:        uint8 [macrobatch_size // 128, intermediate_size // 128, 32, 16]
        hidden_shared:       bfloat16 [num_local_tokens, intermediate_size]
        hidden_fp8_t_routed: float8_e4m3fn [intermediate_size, macrobatch_size]
        hidden_sc_t_routed:  uint8 [intermediate_size // 128, macrobatch_size // 128, 32, 16]
        y_shared:            bfloat16 [num_local_tokens, hidden_size]
        y_routed:            bfloat16 [macrobatch_size, hidden_size]
    """
    if x.ndim != 2:
        raise ValueError("x must have shape (num_local_tokens, hidden_size)")
    num_local_tokens, hidden_size = x.shape
    if hidden_size <= 0 or hidden_size % 256 != 0:
        raise ValueError("hidden_size must be positive and divisible by 256")
    if type(topk) is not int or not 0 < topk <= 255:
        raise ValueError("topk must be an integer in [1, 255]")
    if swiglu_limit is not None and (type(swiglu_limit) not in (int, float) or swiglu_limit < 0):
        raise ValueError("swiglu_limit must be None or a non-negative number")
    if type(num_comm_sms) is not int or num_comm_sms <= 0 or num_comm_sms % 2 != 0:
        raise ValueError("num_comm_sms must be a positive even integer")
    if (type(minibatch_size) is not int or minibatch_size <= 0
            or minibatch_size % 256 != 0):
        raise ValueError("minibatch_size must be positive and divisible by 256")
    if (type(macrobatch_size) is not int or macrobatch_size <= 0
            or macrobatch_size % minibatch_size != 0):
        raise ValueError("macrobatch_size must be a positive multiple of minibatch_size")
    for pointer_name, pointers in (("x_ptrs", x_ptrs),
                                   ("combine_buffer_ptrs", combine_buffer_ptrs)):
        if not isinstance(pointers, list) or any(
            type(pointer) is not int or pointer <= 0 for pointer in pointers
        ):
            raise TypeError(f"{pointer_name} must be a list of positive integers")
    ep_size = len(x_ptrs)
    if ep_size not in (1, 4, 8, 16, 32, 64):
        raise ValueError("x_ptrs length must be one of 1, 4, 8, 16, 32, 64")
    if len(combine_buffer_ptrs) != ep_size:
        raise ValueError("combine_buffer_ptrs length must match x_ptrs")
    if w_shared_gate.ndim != 2:
        raise ValueError("w_shared_gate must have shape (intermediate_size, hidden_size)")
    intermediate_size = w_shared_gate.shape[0]
    if intermediate_size <= 0 or intermediate_size % 256 != 0:
        raise ValueError("intermediate_size must be positive and divisible by 256")
    if w_routed_gate.ndim != 3 or w_routed_gate.shape[0] <= 0:
        raise ValueError("w_routed_gate must have shape "
                         "(num_local_experts, intermediate_size, hidden_size)")
    num_local_experts = w_routed_gate.shape[0]
    if (combine_buffer.ndim != 2 or combine_buffer.shape[1] != hidden_size
            or combine_buffer.shape[0] % topk != 0 or combine_buffer.shape[0] < num_local_tokens * topk):
        raise ValueError("combine_buffer must have shape (token_capacity * topk, hidden_size) "
                         "with token_capacity >= num_local_tokens")
    expected_shapes = (
        ("w_shared_gate", w_shared_gate, (intermediate_size, hidden_size)),
        ("w_routed_gate", w_routed_gate, (num_local_experts, intermediate_size, hidden_size)),
        ("w_routed_gate_sc", w_routed_gate_sc,
         (num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16)),
        ("w_shared_up", w_shared_up, (intermediate_size, hidden_size)),
        ("w_routed_up", w_routed_up, (num_local_experts, intermediate_size, hidden_size)),
        ("w_routed_up_sc", w_routed_up_sc,
         (num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16)),
        ("w_shared_down", w_shared_down, (hidden_size, intermediate_size)),
        ("w_routed_down", w_routed_down, (num_local_experts, hidden_size, intermediate_size)),
        ("w_routed_down_sc", w_routed_down_sc,
         (num_local_experts * hidden_size // 128, intermediate_size // 128, 32, 16)),
    )
    for tensor_name, tensor, expected_shape in expected_shapes:
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{tensor_name} must have shape {expected_shape}")
    for tensor_name, tensor in (
        ("combine_buffer", combine_buffer),
        ("w_shared_gate", w_shared_gate),
        ("w_routed_gate", w_routed_gate),
        ("w_routed_gate_sc", w_routed_gate_sc),
        ("w_shared_up", w_shared_up),
        ("w_routed_up", w_routed_up),
        ("w_routed_up_sc", w_routed_up_sc),
        ("w_shared_down", w_shared_down),
        ("w_routed_down", w_routed_down),
        ("w_routed_down_sc", w_routed_down_sc),
        ("schedule_peer_rank", schedule_peer_rank),
        ("schedule_peer_token_idx", schedule_peer_token_idx),
        ("num_tokens", num_tokens),
        ("tokens_per_expert", tokens_per_expert),
    ):
        if tensor.device != x.device:
            raise ValueError(f"{tensor_name} must be on {x.device}")
    if schedule_peer_rank.ndim != 1 or schedule_peer_rank.numel() == 0:
        raise ValueError("schedule_peer_rank must be a nonempty 1D tensor")
    schedule_capacity = schedule_peer_rank.numel()
    if schedule_capacity % 256 != 0:
        raise ValueError("schedule_capacity must be divisible by 256")
    if tuple(schedule_peer_token_idx.shape) != (schedule_capacity,):
        raise ValueError("schedule_peer_token_idx must have shape (schedule_capacity,)")
    if tuple(num_tokens.shape) != (1,):
        raise ValueError("num_tokens must have shape (1,)")
    if tuple(tokens_per_expert.shape) != (num_local_experts,):
        raise ValueError("tokens_per_expert must have shape (num_local_experts,)")

    return _C.dispatch_mlp_swiglu_combine_fwd_mxfp8(
        x, x_ptrs, combine_buffer, combine_buffer_ptrs,
        w_shared_gate, w_routed_gate, w_routed_gate_sc,
        w_shared_up, w_routed_up, w_routed_up_sc,
        w_shared_down, w_routed_down, w_routed_down_sc,
        schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
        topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size,
    )


@torch.library.custom_op("mok::dispatch_mlp_swiglu_combine_fwd_bf16", mutates_args=("combine_buffer",))
def dispatch_mlp_swiglu_combine_fwd_bf16(
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
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Runs the fused BF16 MoE forward pass.

    Inputs:
        x:                       bfloat16 [num_local_tokens, hidden_size]
        x_ptrs:                  list[int] [ep_size]
        combine_buffer:          bfloat16 [token_capacity * topk, hidden_size]
        combine_buffer_ptrs:     list[int] [ep_size]
        w_shared_gate:           bfloat16 [intermediate_size, hidden_size]
        w_routed_gate:           bfloat16 [num_local_experts, intermediate_size, hidden_size]
        w_shared_up:             bfloat16 [intermediate_size, hidden_size]
        w_routed_up:             bfloat16 [num_local_experts, intermediate_size, hidden_size]
        w_shared_down:           bfloat16 [hidden_size, intermediate_size]
        w_routed_down:           bfloat16 [num_local_experts, hidden_size, intermediate_size]
        schedule_peer_rank:      int32 [schedule_capacity]
        schedule_peer_token_idx: int32 [schedule_capacity]
        num_tokens:              int32 [1]
        tokens_per_expert:       int32 [num_local_experts]
        topk:                    int
        swiglu_limit:            float | None
        num_comm_sms:            int
        macrobatch_size:         int
        minibatch_size:          int

    Outputs:
        x_routed:      bfloat16 [macrobatch_size, hidden_size]
        gate_shared:   bfloat16 [num_local_tokens, intermediate_size]
        gate_routed:   bfloat16 [macrobatch_size, intermediate_size]
        up_shared:     bfloat16 [num_local_tokens, intermediate_size]
        up_routed:     bfloat16 [macrobatch_size, intermediate_size]
        hidden_shared: bfloat16 [num_local_tokens, intermediate_size]
        hidden_routed: bfloat16 [macrobatch_size, intermediate_size]
        y_shared:      bfloat16 [num_local_tokens, hidden_size]
        y_routed:      bfloat16 [macrobatch_size, hidden_size]
    """
    if x.ndim != 2:
        raise ValueError("x must have shape (num_local_tokens, hidden_size)")
    num_local_tokens, hidden_size = x.shape
    if hidden_size <= 0 or hidden_size % 256 != 0:
        raise ValueError("hidden_size must be positive and divisible by 256")
    if w_shared_gate.ndim != 2 or w_shared_gate.shape[1] != hidden_size:
        raise ValueError("w_shared_gate must have shape (intermediate_size, hidden_size)")
    intermediate_size = w_shared_gate.shape[0]
    if intermediate_size <= 0 or intermediate_size % 256 != 0:
        raise ValueError("intermediate_size must be positive and divisible by 256")
    if w_routed_gate.ndim != 3 or w_routed_gate.shape[0] <= 0:
        raise ValueError("w_routed_gate must have shape (num_local_experts, intermediate_size, hidden_size)")
    num_local_experts = w_routed_gate.shape[0]
    if type(topk) is not int or not 0 < topk <= 255:
        raise ValueError("topk must be an integer in [1, 255]")
    if swiglu_limit is not None and (type(swiglu_limit) not in (int, float) or swiglu_limit < 0):
        raise ValueError("swiglu_limit must be None or a non-negative number")
    if type(num_comm_sms) is not int or num_comm_sms <= 0 or num_comm_sms % 2 != 0:
        raise ValueError("num_comm_sms must be a positive even integer")
    if type(minibatch_size) is not int or minibatch_size <= 0 or minibatch_size % 256 != 0:
        raise ValueError("minibatch_size must be positive and divisible by 256")
    if type(macrobatch_size) is not int or macrobatch_size <= 0 or macrobatch_size % minibatch_size != 0:
        raise ValueError("macrobatch_size must be a positive multiple of minibatch_size")
    for pointer_name, pointers in (("x_ptrs", x_ptrs), ("combine_buffer_ptrs", combine_buffer_ptrs)):
        if not isinstance(pointers, list) or any(
            type(pointer) is not int or pointer <= 0 for pointer in pointers
        ):
            raise TypeError(f"{pointer_name} must be a list of positive integers")
    ep_size = len(x_ptrs)
    if ep_size not in (1, 4, 8, 16, 32, 64):
        raise ValueError("x_ptrs length must be one of 1, 4, 8, 16, 32, 64")
    if len(combine_buffer_ptrs) != ep_size:
        raise ValueError("combine_buffer_ptrs length must match x_ptrs")
    if schedule_peer_rank.ndim != 1 or schedule_peer_rank.numel() == 0:
        raise ValueError("schedule_peer_rank must be a nonempty 1D tensor")
    schedule_capacity = schedule_peer_rank.numel()
    if schedule_capacity % 256 != 0:
        raise ValueError("schedule_capacity must be divisible by 256")
    if (combine_buffer.ndim != 2 or combine_buffer.shape[1] != hidden_size
            or combine_buffer.shape[0] % topk != 0 or combine_buffer.shape[0] < num_local_tokens * topk):
        raise ValueError("combine_buffer must have shape (token_capacity * topk, hidden_size) "
                         "with token_capacity >= num_local_tokens")
    expected_shapes = (
        ("w_shared_gate", w_shared_gate, (intermediate_size, hidden_size)),
        ("w_routed_gate", w_routed_gate, (num_local_experts, intermediate_size, hidden_size)),
        ("w_shared_up", w_shared_up, (intermediate_size, hidden_size)),
        ("w_routed_up", w_routed_up, (num_local_experts, intermediate_size, hidden_size)),
        ("w_shared_down", w_shared_down, (hidden_size, intermediate_size)),
        ("w_routed_down", w_routed_down, (num_local_experts, hidden_size, intermediate_size)),
        ("schedule_peer_token_idx", schedule_peer_token_idx, (schedule_capacity,)),
        ("num_tokens", num_tokens, (1,)),
        ("tokens_per_expert", tokens_per_expert, (num_local_experts,)),
    )
    for tensor_name, tensor, expected_shape in expected_shapes:
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{tensor_name} must have shape {expected_shape}")
    for tensor_name, tensor in (
        ("combine_buffer", combine_buffer),
        ("w_shared_gate", w_shared_gate),
        ("w_routed_gate", w_routed_gate),
        ("w_shared_up", w_shared_up),
        ("w_routed_up", w_routed_up),
        ("w_shared_down", w_shared_down),
        ("w_routed_down", w_routed_down),
        ("schedule_peer_rank", schedule_peer_rank),
        ("schedule_peer_token_idx", schedule_peer_token_idx),
        ("num_tokens", num_tokens),
        ("tokens_per_expert", tokens_per_expert),
    ):
        if tensor.device != x.device:
            raise ValueError(f"{tensor_name} must be on {x.device}")

    return _C.dispatch_mlp_swiglu_combine_fwd_bf16(
        x, x_ptrs, combine_buffer, combine_buffer_ptrs,
        w_shared_gate, w_routed_gate, w_shared_up, w_routed_up,
        w_shared_down, w_routed_down,
        schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
        topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size,
    )


@torch.library.custom_op("mok::recompute_forward_context_mxfp8", mutates_args=())
def recompute_forward_context_mxfp8(
    x: torch.Tensor,
    x_ptrs: list[int],
    w_shared_gate: torch.Tensor,
    w_routed_gate: torch.Tensor,
    w_routed_gate_sc: torch.Tensor,
    w_shared_up: torch.Tensor,
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
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Recomputes the MXFP8 MoE forward context.

    Inputs:
        x:                       bfloat16 [num_local_tokens, hidden_size]
        x_ptrs:                  list[int] [ep_size]
        w_shared_gate:           bfloat16 [intermediate_size, hidden_size]
        w_routed_gate:           float8_e4m3fn [num_local_experts, intermediate_size, hidden_size]
        w_routed_gate_sc:        uint8 [num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16]
        w_shared_up:             bfloat16 [intermediate_size, hidden_size]
        w_routed_up:             float8_e4m3fn [num_local_experts, intermediate_size, hidden_size]
        w_routed_up_sc:          uint8 [num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16]
        schedule_peer_rank:      int32 [schedule_capacity]
        schedule_peer_token_idx: int32 [schedule_capacity]
        num_tokens:              int32 [1]
        tokens_per_expert:       int32 [num_local_experts]
        topk:                    int
        swiglu_limit:            float | None
        num_comm_sms:            int
        macrobatch_size:         int
        minibatch_size:          int

    Outputs:
        x_fp8_t_routed:      float8_e4m3fn [hidden_size, macrobatch_size]
        x_sc_t_routed:       uint8 [hidden_size // 128, macrobatch_size // 128, 32, 16]
        gate_shared:         bfloat16 [num_local_tokens, intermediate_size]
        gate_fp8_routed:     float8_e4m3fn [macrobatch_size, intermediate_size]
        gate_sc_routed:      uint8 [macrobatch_size // 128, intermediate_size // 128, 32, 16]
        up_shared:           bfloat16 [num_local_tokens, intermediate_size]
        up_fp8_routed:       float8_e4m3fn [macrobatch_size, intermediate_size]
        up_sc_routed:        uint8 [macrobatch_size // 128, intermediate_size // 128, 32, 16]
        hidden_shared:       bfloat16 [num_local_tokens, intermediate_size]
        hidden_fp8_t_routed: float8_e4m3fn [intermediate_size, macrobatch_size]
        hidden_sc_t_routed:  uint8 [intermediate_size // 128, macrobatch_size // 128, 32, 16]
    """
    if x.ndim != 2:
        raise ValueError("x must have shape (num_local_tokens, hidden_size)")
    num_local_tokens, hidden_size = x.shape
    if hidden_size <= 0 or hidden_size % 256 != 0:
        raise ValueError("hidden_size must be positive and divisible by 256")
    if type(topk) is not int or not 0 < topk <= 255:
        raise ValueError("topk must be an integer in [1, 255]")
    if swiglu_limit is not None and (type(swiglu_limit) not in (int, float) or swiglu_limit < 0):
        raise ValueError("swiglu_limit must be None or a non-negative number")
    if type(num_comm_sms) is not int or num_comm_sms <= 0 or num_comm_sms % 2 != 0:
        raise ValueError("num_comm_sms must be a positive even integer")
    if (type(minibatch_size) is not int or minibatch_size <= 0 or minibatch_size % 256 != 0):
        raise ValueError("minibatch_size must be positive and divisible by 256")
    if (type(macrobatch_size) is not int or macrobatch_size <= 0 or macrobatch_size % minibatch_size != 0):
        raise ValueError("macrobatch_size must be a positive multiple of minibatch_size")
    if not isinstance(x_ptrs, list) or any(type(pointer) is not int or pointer <= 0 for pointer in x_ptrs):
        raise TypeError("x_ptrs must be a list of positive integers")
    ep_size = len(x_ptrs)
    if ep_size not in (1, 4, 8, 16, 32, 64):
        raise ValueError("x_ptrs length must be one of 1, 4, 8, 16, 32, 64")
    if w_shared_gate.ndim != 2:
        raise ValueError("w_shared_gate must have shape (intermediate_size, hidden_size)")
    intermediate_size = w_shared_gate.shape[0]
    if intermediate_size <= 0 or intermediate_size % 256 != 0:
        raise ValueError("intermediate_size must be positive and divisible by 256")
    if w_routed_gate.ndim != 3 or w_routed_gate.shape[0] <= 0:
        raise ValueError("w_routed_gate must have shape "
                         "(num_local_experts, intermediate_size, hidden_size)")
    num_local_experts = w_routed_gate.shape[0]
    expected_shapes = (
        ("w_shared_gate", w_shared_gate, (intermediate_size, hidden_size)),
        ("w_routed_gate", w_routed_gate, (num_local_experts, intermediate_size, hidden_size)),
        ("w_routed_gate_sc", w_routed_gate_sc,
         (num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16)),
        ("w_shared_up", w_shared_up, (intermediate_size, hidden_size)),
        ("w_routed_up", w_routed_up, (num_local_experts, intermediate_size, hidden_size)),
        ("w_routed_up_sc", w_routed_up_sc,
         (num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16)),
    )
    for tensor_name, tensor, expected_shape in expected_shapes:
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{tensor_name} must have shape {expected_shape}")
    for tensor_name, tensor in (
        ("w_shared_gate", w_shared_gate),
        ("w_routed_gate", w_routed_gate),
        ("w_routed_gate_sc", w_routed_gate_sc),
        ("w_shared_up", w_shared_up),
        ("w_routed_up", w_routed_up),
        ("w_routed_up_sc", w_routed_up_sc),
        ("schedule_peer_rank", schedule_peer_rank),
        ("schedule_peer_token_idx", schedule_peer_token_idx),
        ("num_tokens", num_tokens),
        ("tokens_per_expert", tokens_per_expert),
    ):
        if tensor.device != x.device:
            raise ValueError(f"{tensor_name} must be on {x.device}")
    if schedule_peer_rank.ndim != 1 or schedule_peer_rank.numel() == 0:
        raise ValueError("schedule_peer_rank must be a nonempty 1D tensor")
    schedule_capacity = schedule_peer_rank.numel()
    if schedule_capacity % 256 != 0:
        raise ValueError("schedule_capacity must be divisible by 256")
    if tuple(schedule_peer_token_idx.shape) != (schedule_capacity,):
        raise ValueError("schedule_peer_token_idx must have shape (schedule_capacity,)")
    if tuple(num_tokens.shape) != (1,):
        raise ValueError("num_tokens must have shape (1,)")
    if tuple(tokens_per_expert.shape) != (num_local_experts,):
        raise ValueError("tokens_per_expert must have shape (num_local_experts,)")

    return _C.recompute_forward_context_mxfp8(
        x, x_ptrs,
        w_shared_gate, w_routed_gate, w_routed_gate_sc,
        w_shared_up, w_routed_up, w_routed_up_sc,
        schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
        topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size,
    )


@torch.library.custom_op("mok::recompute_forward_context_bf16", mutates_args=())
def recompute_forward_context_bf16(
    x: torch.Tensor,
    x_ptrs: list[int],
    w_shared_gate: torch.Tensor,
    w_routed_gate: torch.Tensor,
    w_shared_up: torch.Tensor,
    w_routed_up: torch.Tensor,
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
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Recomputes the BF16 MoE forward context.

    Inputs:
        x:                       bfloat16 [num_local_tokens, hidden_size]
        x_ptrs:                  list[int] [ep_size]
        w_shared_gate:           bfloat16 [intermediate_size, hidden_size]
        w_routed_gate:           bfloat16 [num_local_experts, intermediate_size, hidden_size]
        w_shared_up:             bfloat16 [intermediate_size, hidden_size]
        w_routed_up:             bfloat16 [num_local_experts, intermediate_size, hidden_size]
        schedule_peer_rank:      int32 [schedule_capacity]
        schedule_peer_token_idx: int32 [schedule_capacity]
        num_tokens:              int32 [1]
        tokens_per_expert:       int32 [num_local_experts]
        topk:                    int
        swiglu_limit:            float | None
        num_comm_sms:            int
        macrobatch_size:         int
        minibatch_size:          int

    Outputs:
        x_routed:      bfloat16 [macrobatch_size, hidden_size]
        gate_shared:   bfloat16 [num_local_tokens, intermediate_size]
        gate_routed:   bfloat16 [macrobatch_size, intermediate_size]
        up_shared:     bfloat16 [num_local_tokens, intermediate_size]
        up_routed:     bfloat16 [macrobatch_size, intermediate_size]
        hidden_shared: bfloat16 [num_local_tokens, intermediate_size]
        hidden_routed: bfloat16 [macrobatch_size, intermediate_size]
    """
    if x.ndim != 2:
        raise ValueError("x must have shape (num_local_tokens, hidden_size)")
    num_local_tokens, hidden_size = x.shape
    if hidden_size <= 0 or hidden_size % 256 != 0:
        raise ValueError("hidden_size must be positive and divisible by 256")
    if w_shared_gate.ndim != 2 or w_shared_gate.shape[1] != hidden_size:
        raise ValueError("w_shared_gate must have shape (intermediate_size, hidden_size)")
    intermediate_size = w_shared_gate.shape[0]
    if intermediate_size <= 0 or intermediate_size % 256 != 0:
        raise ValueError("intermediate_size must be positive and divisible by 256")
    if w_routed_gate.ndim != 3 or w_routed_gate.shape[0] <= 0:
        raise ValueError("w_routed_gate must have shape (num_local_experts, intermediate_size, hidden_size)")
    num_local_experts = w_routed_gate.shape[0]
    if type(topk) is not int or not 0 < topk <= 255:
        raise ValueError("topk must be an integer in [1, 255]")
    if swiglu_limit is not None and (type(swiglu_limit) not in (int, float) or swiglu_limit < 0):
        raise ValueError("swiglu_limit must be None or a non-negative number")
    if type(num_comm_sms) is not int or num_comm_sms <= 0 or num_comm_sms % 2 != 0:
        raise ValueError("num_comm_sms must be a positive even integer")
    if type(minibatch_size) is not int or minibatch_size <= 0 or minibatch_size % 256 != 0:
        raise ValueError("minibatch_size must be positive and divisible by 256")
    if type(macrobatch_size) is not int or macrobatch_size <= 0 or macrobatch_size % minibatch_size != 0:
        raise ValueError("macrobatch_size must be a positive multiple of minibatch_size")
    if not isinstance(x_ptrs, list) or any(
        type(pointer) is not int or pointer <= 0 for pointer in x_ptrs
    ):
        raise TypeError("x_ptrs must be a list of positive integers")
    ep_size = len(x_ptrs)
    if ep_size not in (1, 4, 8, 16, 32, 64):
        raise ValueError("x_ptrs length must be one of 1, 4, 8, 16, 32, 64")
    if schedule_peer_rank.ndim != 1 or schedule_peer_rank.numel() == 0:
        raise ValueError("schedule_peer_rank must be a nonempty 1D tensor")
    schedule_capacity = schedule_peer_rank.numel()
    if schedule_capacity % 256 != 0:
        raise ValueError("schedule_capacity must be divisible by 256")
    expected_shapes = (
        ("w_shared_gate", w_shared_gate, (intermediate_size, hidden_size)),
        ("w_routed_gate", w_routed_gate, (num_local_experts, intermediate_size, hidden_size)),
        ("w_shared_up", w_shared_up, (intermediate_size, hidden_size)),
        ("w_routed_up", w_routed_up, (num_local_experts, intermediate_size, hidden_size)),
        ("schedule_peer_token_idx", schedule_peer_token_idx, (schedule_capacity,)),
        ("num_tokens", num_tokens, (1,)),
        ("tokens_per_expert", tokens_per_expert, (num_local_experts,)),
    )
    for tensor_name, tensor, expected_shape in expected_shapes:
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{tensor_name} must have shape {expected_shape}")
    for tensor_name, tensor, _ in expected_shapes:
        if tensor.device != x.device:
            raise ValueError(f"{tensor_name} must be on {x.device}")
    if schedule_peer_rank.device != x.device:
        raise ValueError(f"schedule_peer_rank must be on {x.device}")

    return _C.recompute_forward_context_bf16(
        x, x_ptrs,
        w_shared_gate, w_routed_gate, w_shared_up, w_routed_up,
        schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
        topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size,
    )


@torch.library.custom_op(
    "mok::dispatch_mlp_swiglu_combine_bwd_mxfp8",
    mutates_args=("d_x_routed_buffer", "d_router_weight_buffer",
                  "x_fp8_t_routed", "x_sc_t_routed",
                  "gate_fp8_routed", "gate_sc_routed", "up_fp8_routed", "up_sc_routed",
                  "hidden_fp8_t_routed", "hidden_sc_t_routed"),
)
def dispatch_mlp_swiglu_combine_bwd_mxfp8(
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
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Runs the fused MXFP8 MoE backward pass.

    Inputs:
        d_y_buffer:                    bfloat16 [num_local_tokens, hidden_size]
        d_y_buffer_ptrs:               list[int] [ep_size]
        d_x_routed_buffer:             bfloat16 [token_capacity * topk, hidden_size]
        d_x_routed_buffer_ptrs:        list[int] [ep_size]
        router_weight_buffer:          float32 [token_capacity, topk]
        router_weight_buffer_ptrs:     list[int] [ep_size]
        d_router_weight_buffer:        float32 [token_capacity, topk]
        d_router_weight_buffer_ptrs:   list[int] [ep_size]
        w_shared_gate:                 bfloat16 [intermediate_size, hidden_size]
        w_routed_gate_T:               float8_e4m3fn [num_local_experts, hidden_size, intermediate_size]
        w_routed_gate_T_sc:            uint8 [num_local_experts * hidden_size // 128, intermediate_size // 128, 32, 16]
        w_shared_up:                   bfloat16 [intermediate_size, hidden_size]
        w_routed_up_T:                 float8_e4m3fn [num_local_experts, hidden_size, intermediate_size]
        w_routed_up_T_sc:              uint8 [num_local_experts * hidden_size // 128, intermediate_size // 128, 32, 16]
        w_shared_down:                 bfloat16 [hidden_size, intermediate_size]
        w_routed_down_T:               float8_e4m3fn [num_local_experts, intermediate_size, hidden_size]
        w_routed_down_T_sc:            uint8 [num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16]
        x_fp8_t_routed:                float8_e4m3fn [hidden_size, macrobatch_size]
        x_sc_t_routed:                 uint8 [hidden_size // 128, macrobatch_size // 128, 32, 16]
        gate_shared:                   bfloat16 [num_local_tokens, intermediate_size]
        gate_fp8_routed:               float8_e4m3fn [macrobatch_size, intermediate_size]
        gate_sc_routed:                uint8 [macrobatch_size // 128, intermediate_size // 128, 32, 16]
        up_shared:                     bfloat16 [num_local_tokens, intermediate_size]
        up_fp8_routed:                 float8_e4m3fn [macrobatch_size, intermediate_size]
        up_sc_routed:                  uint8 [macrobatch_size // 128, intermediate_size // 128, 32, 16]
        hidden_shared:                 bfloat16 [num_local_tokens, intermediate_size]
        hidden_fp8_t_routed:           float8_e4m3fn [intermediate_size, macrobatch_size]
        hidden_sc_t_routed:            uint8 [intermediate_size // 128, macrobatch_size // 128, 32, 16]
        x:                             bfloat16 [num_local_tokens, hidden_size]
        x_ptrs:                        list[int] [ep_size]
        w_routed_gate:                 float8_e4m3fn [num_local_experts, intermediate_size, hidden_size]
        w_routed_gate_sc:              uint8 [num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16]
        w_routed_up:                   float8_e4m3fn [num_local_experts, intermediate_size, hidden_size]
        w_routed_up_sc:                uint8 [num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16]
        schedule_peer_rank:            int32 [schedule_capacity]
        schedule_peer_token_idx:       int32 [schedule_capacity]
        num_tokens:                    int32 [1]
        tokens_per_expert:             int32 [num_local_experts]
        topk:                          int
        swiglu_limit:                  float | None
        num_comm_sms:                  int
        macrobatch_size:               int
        minibatch_size:                int

    Outputs:
        d_x_shared:          bfloat16 [num_local_tokens, hidden_size]
        d_x_routed:          bfloat16 [macrobatch_size, hidden_size]
        d_gate_shared:       bfloat16 [num_local_tokens, intermediate_size]
        d_gate_fp8_routed:   float8_e4m3fn [macrobatch_size, intermediate_size]
        d_gate_sc_routed:    uint8 [macrobatch_size // 128, intermediate_size // 128, 32, 16]
        d_up_shared:         bfloat16 [num_local_tokens, intermediate_size]
        d_up_fp8_routed:     float8_e4m3fn [macrobatch_size, intermediate_size]
        d_up_sc_routed:      uint8 [macrobatch_size // 128, intermediate_size // 128, 32, 16]
        d_hidden_shared:     bfloat16 [num_local_tokens, intermediate_size]
        d_hidden_routed:     bfloat16 [macrobatch_size, intermediate_size]
        d_y_fp8_routed:      float8_e4m3fn [macrobatch_size, hidden_size]
        d_y_sc_routed:       uint8 [macrobatch_size // 128, hidden_size // 128, 32, 16]
        d_w_shared_gate:     bfloat16 [intermediate_size, hidden_size]
        d_w_routed_gate:     bfloat16 [num_local_experts, intermediate_size, hidden_size]
        d_w_shared_up:       bfloat16 [intermediate_size, hidden_size]
        d_w_routed_up:       bfloat16 [num_local_experts, intermediate_size, hidden_size]
        d_w_shared_down:     bfloat16 [hidden_size, intermediate_size]
        d_w_routed_down:     bfloat16 [num_local_experts, hidden_size, intermediate_size]
    """
    if x.ndim != 2:
        raise ValueError("x must have shape (num_local_tokens, hidden_size)")
    num_local_tokens, hidden_size = x.shape
    if hidden_size <= 0 or hidden_size % 256 != 0:
        raise ValueError("hidden_size must be positive and divisible by 256")
    if type(topk) is not int or not 0 < topk <= 255:
        raise ValueError("topk must be an integer in [1, 255]")
    if swiglu_limit is not None and (type(swiglu_limit) not in (int, float) or swiglu_limit < 0):
        raise ValueError("swiglu_limit must be None or a non-negative number")
    if type(num_comm_sms) is not int or num_comm_sms <= 0 or num_comm_sms % 2 != 0:
        raise ValueError("num_comm_sms must be a positive even integer")
    if (type(minibatch_size) is not int or minibatch_size <= 0
            or minibatch_size % 256 != 0):
        raise ValueError("minibatch_size must be positive and divisible by 256")
    if (type(macrobatch_size) is not int or macrobatch_size <= 0
            or macrobatch_size % minibatch_size != 0):
        raise ValueError("macrobatch_size must be a positive multiple of minibatch_size")
    for pointer_name, pointers in (
        ("d_y_buffer_ptrs", d_y_buffer_ptrs),
        ("d_x_routed_buffer_ptrs", d_x_routed_buffer_ptrs),
        ("router_weight_buffer_ptrs", router_weight_buffer_ptrs),
        ("d_router_weight_buffer_ptrs", d_router_weight_buffer_ptrs),
        ("x_ptrs", x_ptrs),
    ):
        if not isinstance(pointers, list) or any(
            type(pointer) is not int or pointer <= 0 for pointer in pointers
        ):
            raise TypeError(f"{pointer_name} must be a list of positive integers")
    ep_size = len(x_ptrs)
    if ep_size not in (1, 4, 8, 16, 32, 64):
        raise ValueError("x_ptrs length must be one of 1, 4, 8, 16, 32, 64")
    for pointer_name, pointers in (
        ("d_y_buffer_ptrs", d_y_buffer_ptrs),
        ("d_x_routed_buffer_ptrs", d_x_routed_buffer_ptrs),
        ("router_weight_buffer_ptrs", router_weight_buffer_ptrs),
        ("d_router_weight_buffer_ptrs", d_router_weight_buffer_ptrs),
    ):
        if len(pointers) != ep_size:
            raise ValueError(f"{pointer_name} length must match x_ptrs")
    if w_shared_gate.ndim != 2 or w_shared_gate.shape[1] != hidden_size:
        raise ValueError("w_shared_gate must have shape (intermediate_size, hidden_size)")
    intermediate_size = w_shared_gate.shape[0]
    if intermediate_size <= 0 or intermediate_size % 256 != 0:
        raise ValueError("intermediate_size must be positive and divisible by 256")
    if w_routed_gate.ndim != 3 or w_routed_gate.shape[0] <= 0:
        raise ValueError("w_routed_gate must have shape "
                         "(num_local_experts, intermediate_size, hidden_size)")
    num_local_experts = w_routed_gate.shape[0]
    mb_i_sc = (macrobatch_size // 128, intermediate_size // 128, 32, 16)
    i_mb_sc = (intermediate_size // 128, macrobatch_size // 128, 32, 16)
    h_mb_sc = (hidden_size // 128, macrobatch_size // 128, 32, 16)
    e_i_h_sc = (num_local_experts * intermediate_size // 128, hidden_size // 128, 32, 16)
    e_h_i_sc = (num_local_experts * hidden_size // 128, intermediate_size // 128, 32, 16)
    token_capacity = router_weight_buffer.shape[0] if router_weight_buffer.ndim == 2 else -1
    if token_capacity < num_local_tokens:
        raise ValueError("router_weight_buffer must have shape (token_capacity, topk) "
                         "with token_capacity >= num_local_tokens")
    expected_shapes = (
        ("d_y_buffer", d_y_buffer, (num_local_tokens, hidden_size)),
        ("d_x_routed_buffer", d_x_routed_buffer, (token_capacity * topk, hidden_size)),
        ("router_weight_buffer", router_weight_buffer, (token_capacity, topk)),
        ("d_router_weight_buffer", d_router_weight_buffer, (token_capacity, topk)),
        ("w_shared_gate", w_shared_gate, (intermediate_size, hidden_size)),
        ("w_routed_gate_T", w_routed_gate_T,
         (num_local_experts, hidden_size, intermediate_size)),
        ("w_routed_gate_T_sc", w_routed_gate_T_sc, e_h_i_sc),
        ("w_shared_up", w_shared_up, (intermediate_size, hidden_size)),
        ("w_routed_up_T", w_routed_up_T, (num_local_experts, hidden_size, intermediate_size)),
        ("w_routed_up_T_sc", w_routed_up_T_sc, e_h_i_sc),
        ("w_shared_down", w_shared_down, (hidden_size, intermediate_size)),
        ("w_routed_down_T", w_routed_down_T,
         (num_local_experts, intermediate_size, hidden_size)),
        ("w_routed_down_T_sc", w_routed_down_T_sc, e_i_h_sc),
        ("x_fp8_t_routed", x_fp8_t_routed, (hidden_size, macrobatch_size)),
        ("x_sc_t_routed", x_sc_t_routed, h_mb_sc),
        ("gate_shared", gate_shared, (num_local_tokens, intermediate_size)),
        ("gate_fp8_routed", gate_fp8_routed, (macrobatch_size, intermediate_size)),
        ("gate_sc_routed", gate_sc_routed, mb_i_sc),
        ("up_shared", up_shared, (num_local_tokens, intermediate_size)),
        ("up_fp8_routed", up_fp8_routed, (macrobatch_size, intermediate_size)),
        ("up_sc_routed", up_sc_routed, mb_i_sc),
        ("hidden_shared", hidden_shared, (num_local_tokens, intermediate_size)),
        ("hidden_fp8_t_routed", hidden_fp8_t_routed, (intermediate_size, macrobatch_size)),
        ("hidden_sc_t_routed", hidden_sc_t_routed, i_mb_sc),
        ("w_routed_gate", w_routed_gate,
         (num_local_experts, intermediate_size, hidden_size)),
        ("w_routed_gate_sc", w_routed_gate_sc, e_i_h_sc),
        ("w_routed_up", w_routed_up, (num_local_experts, intermediate_size, hidden_size)),
        ("w_routed_up_sc", w_routed_up_sc, e_i_h_sc),
    )
    for tensor_name, tensor, expected_shape in expected_shapes:
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{tensor_name} must have shape {expected_shape}")
    for tensor_name, tensor in (
        ("d_y_buffer", d_y_buffer),
        ("d_x_routed_buffer", d_x_routed_buffer),
        ("router_weight_buffer", router_weight_buffer),
        ("d_router_weight_buffer", d_router_weight_buffer),
        ("w_shared_gate", w_shared_gate),
        ("w_routed_gate_T", w_routed_gate_T),
        ("w_routed_gate_T_sc", w_routed_gate_T_sc),
        ("w_shared_up", w_shared_up),
        ("w_routed_up_T", w_routed_up_T),
        ("w_routed_up_T_sc", w_routed_up_T_sc),
        ("w_shared_down", w_shared_down),
        ("w_routed_down_T", w_routed_down_T),
        ("w_routed_down_T_sc", w_routed_down_T_sc),
        ("x_fp8_t_routed", x_fp8_t_routed),
        ("x_sc_t_routed", x_sc_t_routed),
        ("gate_shared", gate_shared),
        ("gate_fp8_routed", gate_fp8_routed),
        ("gate_sc_routed", gate_sc_routed),
        ("up_shared", up_shared),
        ("up_fp8_routed", up_fp8_routed),
        ("up_sc_routed", up_sc_routed),
        ("hidden_shared", hidden_shared),
        ("hidden_fp8_t_routed", hidden_fp8_t_routed),
        ("hidden_sc_t_routed", hidden_sc_t_routed),
        ("w_routed_gate", w_routed_gate),
        ("w_routed_gate_sc", w_routed_gate_sc),
        ("w_routed_up", w_routed_up),
        ("w_routed_up_sc", w_routed_up_sc),
        ("schedule_peer_rank", schedule_peer_rank),
        ("schedule_peer_token_idx", schedule_peer_token_idx),
        ("num_tokens", num_tokens),
        ("tokens_per_expert", tokens_per_expert),
    ):
        if tensor.device != x.device:
            raise ValueError(f"{tensor_name} must be on {x.device}")
    if schedule_peer_rank.ndim != 1 or schedule_peer_rank.numel() == 0:
        raise ValueError("schedule_peer_rank must be a nonempty 1D tensor")
    schedule_capacity = schedule_peer_rank.numel()
    if schedule_capacity % 256 != 0:
        raise ValueError("schedule_capacity must be divisible by 256")
    if tuple(schedule_peer_token_idx.shape) != (schedule_capacity,):
        raise ValueError("schedule_peer_token_idx must have shape (schedule_capacity,)")
    if tuple(num_tokens.shape) != (1,):
        raise ValueError("num_tokens must have shape (1,)")
    if tuple(tokens_per_expert.shape) != (num_local_experts,):
        raise ValueError("tokens_per_expert must have shape (num_local_experts,)")

    return _C.dispatch_mlp_swiglu_combine_bwd_mxfp8(
        d_y_buffer, d_y_buffer_ptrs, d_x_routed_buffer, d_x_routed_buffer_ptrs,
        router_weight_buffer, router_weight_buffer_ptrs,
        d_router_weight_buffer, d_router_weight_buffer_ptrs,
        w_shared_gate, w_routed_gate_T, w_routed_gate_T_sc,
        w_shared_up, w_routed_up_T, w_routed_up_T_sc,
        w_shared_down, w_routed_down_T, w_routed_down_T_sc,
        x_fp8_t_routed, x_sc_t_routed,
        gate_shared, gate_fp8_routed, gate_sc_routed,
        up_shared, up_fp8_routed, up_sc_routed,
        hidden_shared, hidden_fp8_t_routed, hidden_sc_t_routed,
        x, x_ptrs, w_routed_gate, w_routed_gate_sc, w_routed_up, w_routed_up_sc,
        schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
        topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size,
    )


@torch.library.custom_op(
    "mok::dispatch_mlp_swiglu_combine_bwd_bf16",
    mutates_args=("d_x_routed_buffer", "d_router_weight_buffer", "x_routed", "gate_routed", "up_routed", "hidden_routed"),
)
def dispatch_mlp_swiglu_combine_bwd_bf16(
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
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Runs the fused BF16 MoE backward pass.

    Inputs:
        d_y_buffer:                    bfloat16 [num_local_tokens, hidden_size]
        d_y_buffer_ptrs:               list[int] [ep_size]
        d_x_routed_buffer:             bfloat16 [token_capacity * topk, hidden_size]
        d_x_routed_buffer_ptrs:        list[int] [ep_size]
        router_weight_buffer:          float32 [token_capacity, topk]
        router_weight_buffer_ptrs:     list[int] [ep_size]
        d_router_weight_buffer:        float32 [token_capacity, topk]
        d_router_weight_buffer_ptrs:   list[int] [ep_size]
        w_shared_gate:                 bfloat16 [intermediate_size, hidden_size]
        w_routed_gate:                 bfloat16 [num_local_experts, intermediate_size, hidden_size]
        w_shared_up:                   bfloat16 [intermediate_size, hidden_size]
        w_routed_up:                   bfloat16 [num_local_experts, intermediate_size, hidden_size]
        w_shared_down:                 bfloat16 [hidden_size, intermediate_size]
        w_routed_down:                 bfloat16 [num_local_experts, hidden_size, intermediate_size]
        x_routed:                      bfloat16 [macrobatch_size, hidden_size]
        gate_shared:                   bfloat16 [num_local_tokens, intermediate_size]
        gate_routed:                   bfloat16 [macrobatch_size, intermediate_size]
        up_shared:                     bfloat16 [num_local_tokens, intermediate_size]
        up_routed:                     bfloat16 [macrobatch_size, intermediate_size]
        hidden_shared:                 bfloat16 [num_local_tokens, intermediate_size]
        hidden_routed:                 bfloat16 [macrobatch_size, intermediate_size]
        x:                             bfloat16 [num_local_tokens, hidden_size]
        x_ptrs:                        list[int] [ep_size]
        schedule_peer_rank:            int32 [schedule_capacity]
        schedule_peer_token_idx:       int32 [schedule_capacity]
        num_tokens:                    int32 [1]
        tokens_per_expert:             int32 [num_local_experts]
        topk:                          int
        swiglu_limit:                  float | None
        num_comm_sms:                  int
        macrobatch_size:               int
        minibatch_size:                int

    Outputs:
        d_x_shared:      bfloat16 [num_local_tokens, hidden_size]
        d_x_routed:      bfloat16 [macrobatch_size, hidden_size]
        d_gate_shared:   bfloat16 [num_local_tokens, intermediate_size]
        d_gate_routed:   bfloat16 [macrobatch_size, intermediate_size]
        d_up_shared:     bfloat16 [num_local_tokens, intermediate_size]
        d_up_routed:     bfloat16 [macrobatch_size, intermediate_size]
        d_hidden_shared: bfloat16 [num_local_tokens, intermediate_size]
        d_hidden_routed: bfloat16 [macrobatch_size, intermediate_size]
        d_y_routed:      bfloat16 [macrobatch_size, hidden_size]
        d_w_shared_gate: bfloat16 [intermediate_size, hidden_size]
        d_w_routed_gate: bfloat16 [num_local_experts, intermediate_size, hidden_size]
        d_w_shared_up:   bfloat16 [intermediate_size, hidden_size]
        d_w_routed_up:   bfloat16 [num_local_experts, intermediate_size, hidden_size]
        d_w_shared_down: bfloat16 [hidden_size, intermediate_size]
        d_w_routed_down: bfloat16 [num_local_experts, hidden_size, intermediate_size]
    """
    if x.ndim != 2:
        raise ValueError("x must have shape (num_local_tokens, hidden_size)")
    num_local_tokens, hidden_size = x.shape
    if hidden_size <= 0 or hidden_size % 256 != 0:
        raise ValueError("hidden_size must be positive and divisible by 256")
    if w_shared_gate.ndim != 2 or w_shared_gate.shape[1] != hidden_size:
        raise ValueError("w_shared_gate must have shape (intermediate_size, hidden_size)")
    intermediate_size = w_shared_gate.shape[0]
    if intermediate_size <= 0 or intermediate_size % 256 != 0:
        raise ValueError("intermediate_size must be positive and divisible by 256")
    if w_routed_gate.ndim != 3 or w_routed_gate.shape[0] <= 0:
        raise ValueError("w_routed_gate must have shape (num_local_experts, intermediate_size, hidden_size)")
    num_local_experts = w_routed_gate.shape[0]
    if type(topk) is not int or not 0 < topk <= 255:
        raise ValueError("topk must be an integer in [1, 255]")
    if swiglu_limit is not None and (type(swiglu_limit) not in (int, float) or swiglu_limit < 0):
        raise ValueError("swiglu_limit must be None or a non-negative number")
    if type(num_comm_sms) is not int or num_comm_sms <= 0 or num_comm_sms % 2 != 0:
        raise ValueError("num_comm_sms must be a positive even integer")
    if type(minibatch_size) is not int or minibatch_size <= 0 or minibatch_size % 256 != 0:
        raise ValueError("minibatch_size must be positive and divisible by 256")
    if type(macrobatch_size) is not int or macrobatch_size <= 0 or macrobatch_size % minibatch_size != 0:
        raise ValueError("macrobatch_size must be a positive multiple of minibatch_size")
    pointer_lists = (
        ("d_y_buffer_ptrs", d_y_buffer_ptrs),
        ("d_x_routed_buffer_ptrs", d_x_routed_buffer_ptrs),
        ("router_weight_buffer_ptrs", router_weight_buffer_ptrs),
        ("d_router_weight_buffer_ptrs", d_router_weight_buffer_ptrs),
        ("x_ptrs", x_ptrs),
    )
    for pointer_name, pointers in pointer_lists:
        if not isinstance(pointers, list) or any(
            type(pointer) is not int or pointer <= 0 for pointer in pointers
        ):
            raise TypeError(f"{pointer_name} must be a list of positive integers")
    ep_size = len(x_ptrs)
    if ep_size not in (1, 4, 8, 16, 32, 64):
        raise ValueError("x_ptrs length must be one of 1, 4, 8, 16, 32, 64")
    for pointer_name, pointers in pointer_lists[:-1]:
        if len(pointers) != ep_size:
            raise ValueError(f"{pointer_name} length must match x_ptrs")
    if schedule_peer_rank.ndim != 1 or schedule_peer_rank.numel() == 0:
        raise ValueError("schedule_peer_rank must be a nonempty 1D tensor")
    schedule_capacity = schedule_peer_rank.numel()
    if schedule_capacity % 256 != 0:
        raise ValueError("schedule_capacity must be divisible by 256")
    token_capacity = router_weight_buffer.shape[0] if router_weight_buffer.ndim == 2 else -1
    if token_capacity < num_local_tokens:
        raise ValueError("router_weight_buffer must have shape (token_capacity, topk) "
                         "with token_capacity >= num_local_tokens")
    expected_shapes = (
        ("d_y_buffer", d_y_buffer, (num_local_tokens, hidden_size)),
        ("d_x_routed_buffer", d_x_routed_buffer, (token_capacity * topk, hidden_size)),
        ("router_weight_buffer", router_weight_buffer, (token_capacity, topk)),
        ("d_router_weight_buffer", d_router_weight_buffer, (token_capacity, topk)),
        ("w_shared_gate", w_shared_gate, (intermediate_size, hidden_size)),
        ("w_routed_gate", w_routed_gate, (num_local_experts, intermediate_size, hidden_size)),
        ("w_shared_up", w_shared_up, (intermediate_size, hidden_size)),
        ("w_routed_up", w_routed_up, (num_local_experts, intermediate_size, hidden_size)),
        ("w_shared_down", w_shared_down, (hidden_size, intermediate_size)),
        ("w_routed_down", w_routed_down, (num_local_experts, hidden_size, intermediate_size)),
        ("x_routed", x_routed, (macrobatch_size, hidden_size)),
        ("gate_shared", gate_shared, (num_local_tokens, intermediate_size)),
        ("gate_routed", gate_routed, (macrobatch_size, intermediate_size)),
        ("up_shared", up_shared, (num_local_tokens, intermediate_size)),
        ("up_routed", up_routed, (macrobatch_size, intermediate_size)),
        ("hidden_shared", hidden_shared, (num_local_tokens, intermediate_size)),
        ("hidden_routed", hidden_routed, (macrobatch_size, intermediate_size)),
        ("schedule_peer_token_idx", schedule_peer_token_idx, (schedule_capacity,)),
        ("num_tokens", num_tokens, (1,)),
        ("tokens_per_expert", tokens_per_expert, (num_local_experts,)),
    )
    for tensor_name, tensor, expected_shape in expected_shapes:
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{tensor_name} must have shape {expected_shape}")
    for tensor_name, tensor, _ in expected_shapes:
        if tensor.device != x.device:
            raise ValueError(f"{tensor_name} must be on {x.device}")
    if schedule_peer_rank.device != x.device:
        raise ValueError(f"schedule_peer_rank must be on {x.device}")

    return _C.dispatch_mlp_swiglu_combine_bwd_bf16(
        d_y_buffer, d_y_buffer_ptrs, d_x_routed_buffer, d_x_routed_buffer_ptrs,
        router_weight_buffer, router_weight_buffer_ptrs,
        d_router_weight_buffer, d_router_weight_buffer_ptrs,
        w_shared_gate, w_routed_gate, w_shared_up, w_routed_up,
        w_shared_down, w_routed_down,
        x_routed, gate_shared, gate_routed, up_shared, up_routed,
        hidden_shared, hidden_routed,
        x, x_ptrs,
        schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
        topk, swiglu_limit, num_comm_sms, macrobatch_size, minibatch_size,
    )


@torch.library.custom_op("mok::fwd_epilogue", mutates_args=())
def fwd_epilogue(
    y_shared: torch.Tensor,
    combine_buffer: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Combines shared and router-weighted routed expert outputs.

    Inputs:
        y_shared:       bfloat16 [num_local_tokens, hidden_size]
        combine_buffer: bfloat16 [num_local_tokens * topk, hidden_size]
        topk_weights:   float32 [num_local_tokens, topk]

    Outputs:
        output: bfloat16 [num_local_tokens, hidden_size]
    """
    if y_shared.ndim != 2:
        raise ValueError("y_shared must have shape (num_local_tokens, hidden_size)")
    num_local_tokens, hidden_size = y_shared.shape
    if hidden_size <= 0 or hidden_size % 256 != 0:
        raise ValueError("hidden_size must be positive and divisible by 256")
    if (topk_weights.device != y_shared.device
            or combine_buffer.device != y_shared.device):
        raise ValueError("all tensors must be on the same CUDA device")
    if topk_weights.ndim != 2 or topk_weights.shape[0] != num_local_tokens:
        raise ValueError("topk_weights must have shape (num_local_tokens, topk)")
    topk = topk_weights.shape[1]
    if not 0 < topk <= 255:
        raise ValueError("topk must be in [1, 255]")
    if tuple(combine_buffer.shape) != (num_local_tokens * topk, hidden_size):
        raise ValueError("combine_buffer must have shape (num_local_tokens * topk, hidden_size)")

    return _C.fwd_epilogue(y_shared, combine_buffer, topk_weights)


@torch.library.custom_op("mok::bwd_epilogue", mutates_args=())
def bwd_epilogue(
    d_x_shared: torch.Tensor,
    d_x_routed_buffer: torch.Tensor,
) -> torch.Tensor:
    """Combines shared and routed input gradients.

    topk is inferred as d_x_routed_buffer rows / num_local_tokens, so pass exactly the leading
    num_local_tokens * topk rows of a capacity-sized buffer, not the whole buffer.

    Inputs:
        d_x_shared:        bfloat16 [num_local_tokens, hidden_size]
        d_x_routed_buffer: bfloat16 [num_local_tokens * topk, hidden_size]

    Outputs:
        d_x: bfloat16 [num_local_tokens, hidden_size]
    """
    if d_x_shared.ndim != 2:
        raise ValueError("d_x_shared must have shape (num_local_tokens, hidden_size)")
    num_local_tokens, hidden_size = d_x_shared.shape
    if hidden_size <= 0 or hidden_size % 256 != 0:
        raise ValueError("hidden_size must be positive and divisible by 256")
    if d_x_routed_buffer.device != d_x_shared.device:
        raise ValueError("d_x_routed_buffer must be on the same CUDA device")
    if d_x_routed_buffer.ndim != 2:
        raise ValueError("d_x_routed_buffer must have shape "
                         "(num_local_tokens * topk, hidden_size)")
    num_routes = d_x_routed_buffer.shape[0]
    if num_local_tokens == 0:
        valid_num_routes = num_routes == 0
    else:
        valid_num_routes = num_routes % num_local_tokens == 0 and 0 < num_routes <= num_local_tokens * 255
    if d_x_routed_buffer.shape[1] != hidden_size or not valid_num_routes:
        raise ValueError("d_x_routed_buffer must have shape "
                         "(num_local_tokens * topk, hidden_size)")

    return _C.bwd_epilogue(d_x_shared, d_x_routed_buffer)
