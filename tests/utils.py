import math

import torch
import torch.distributed as dist


BF16_TOLERANCE = (0.5, 0.01)
MXFP8_TOLERANCE = (1.0, 0.1)


def situglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    # h = [beta1 * tanh(g / beta1) * sigmoid(g)] * [beta2 * tanh(u / beta2)].
    gate_factor = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
    up_factor = 25.0 * torch.tanh(up / 25.0)
    return gate_factor * up_factor


def apply_glu(
    gate: torch.Tensor,
    up: torch.Tensor,
    activation: str,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    if activation == "swiglu":
        return run_swiglu_reference(gate, up, swiglu_limit)
    if activation == "situglu":
        return situglu(gate, up)
    raise ValueError("activation must be 'swiglu' or 'situglu'")


def shapes(world_size: int) -> tuple[tuple[str, int, int, int, int, int], ...]:
    return (  # (name, routed experts, hidden dim, intermediate dim, top-k, num local tokens)
        ("Kimi K2.7 Code", 384, 7168, 2048, 8, 7168),
        ("GLM-5.2", 256, 6144, 2048, 8, 7168),
        ("Qwen3.5-397B-A17B", 512, 4096, 1024, 10, 7168),
        ("DeepSeek-V4-Pro", 384, 7168, 3072, 6, 7168),
        ("All minimums", world_size, 256, 256, 1, 512),
        ("Minimum top-k", 384, 7168, 2048, 1, 7168),
        ("Minimum local experts", world_size, 7168, 2048, min(8, world_size), 7168),
        ("Minimum hidden dimension", 384, 256, 2048, 8, 7168),
        ("Minimum intermediate dimension", 384, 7168, 256, 8, 7168),
        ("Minimum local tokens", 384, 7168, 2048, 8, 512),
        ("Local tokens 1024", 384, 7168, 2048, 8, 1024),
        ("Local tokens 16384", 384, 7168, 2048, 8, 2048),
    )


def mok_params() -> tuple[tuple[str, int, int, int, int], ...]:
    return (  # (name, forward comm SMs, backward comm SMs, minibatch size, macrobatch size)
        ("Default", 40, 28, 4096, 131072),
        ("Minimum forward comm SMs", 2, 28, 4096, 131072),
        ("Minimum backward comm SMs", 40, 2, 4096, 131072),
        ("Minimum minibatch", 40, 28, 256, 131072),
        ("Minimum macrobatch", 40, 28, 4096, 4096),
        ("2x macrobatch", 40, 28, 4096, 8192),
        ("4x macrobatch", 40, 28, 4096, 16384)
    )


def swiglu_params() -> tuple[tuple[str, float | None], ...]:
    return (
        ("Unclamped SwiGLU", None),
        ("Clamped SwiGLU", 4.0),
    )


def generate_topk_experts(
    rank: int,
    device: torch.device,
    num_experts: int,
    topk: int,
    num_local_tokens: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(1234 + rank)
    router_logits = torch.randn(num_local_tokens, num_experts, generator=generator, device=device)
    return torch.topk(router_logits, topk, dim=1).indices


def generate_inputs(
    rank: int,
    device: torch.device,
    num_experts: int,
    num_local_experts: int,
    topk: int,
    num_local_tokens: int,
    hidden_dim: int,
    intermediate_dim: int,
) -> tuple[
    torch.Tensor,  # x
    torch.Tensor,  # topk_experts
    torch.Tensor,  # router_weights
    torch.Tensor,  # w_shared_gate
    torch.Tensor,  # w_shared_up
    torch.Tensor,  # w_shared_down
    torch.Tensor,  # w_routed_gate
    torch.Tensor,  # w_routed_up
    torch.Tensor,  # w_routed_down
    torch.Tensor,  # d_output
]:
    generator = torch.Generator(device=device).manual_seed(1234 + rank)

    router_logits = torch.randn(num_local_tokens, num_experts, generator=generator, device=device)
    topk_values, topk_experts = torch.topk(router_logits, topk, dim=1)
    router_weights = torch.softmax(topk_values.float(), dim=-1)

    x = torch.randn(num_local_tokens, hidden_dim, generator=generator, device=device, dtype=torch.bfloat16)
    w_shared_gate = torch.randn(intermediate_dim, hidden_dim, generator=generator, device=device, dtype=torch.bfloat16) * hidden_dim**-0.5
    w_shared_up = torch.randn(intermediate_dim, hidden_dim, generator=generator, device=device, dtype=torch.bfloat16) * hidden_dim**-0.5
    w_shared_down = torch.randn(hidden_dim, intermediate_dim, generator=generator, device=device, dtype=torch.bfloat16) * intermediate_dim**-0.5
    w_routed_gate = torch.randn(num_local_experts, intermediate_dim, hidden_dim, generator=generator, device=device, dtype=torch.bfloat16) * hidden_dim**-0.5
    w_routed_up = torch.randn(num_local_experts, intermediate_dim, hidden_dim, generator=generator, device=device, dtype=torch.bfloat16) * hidden_dim**-0.5
    w_routed_down = torch.randn(num_local_experts, hidden_dim, intermediate_dim, generator=generator, device=device, dtype=torch.bfloat16) * intermediate_dim**-0.5
    d_output = torch.randn(num_local_tokens, hidden_dim, generator=generator, device=device, dtype=torch.bfloat16) * hidden_dim**-0.5

    return (
        x,
        topk_experts,
        router_weights,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        w_routed_gate,
        w_routed_up,
        w_routed_down,
        d_output,
    )


def all_to_all(x: torch.Tensor, count: int, output_splits: list[int], input_splits: list[int]) -> torch.Tensor:
    output = x.new_empty((count, *x.shape[1:]))
    dist.all_to_all_single(output, x, output_splits, input_splits)
    return output


def run_all_gather_top_experts_reference(top_experts: torch.Tensor) -> torch.Tensor:
    world_size = dist.get_world_size()
    top_experts_all = top_experts.new_empty((world_size, *top_experts.shape))
    dist.all_gather_into_tensor(top_experts_all, top_experts)
    return top_experts_all


def run_schedule_reference(
    topk_all: torch.Tensor,
    num_local_experts: int,
    schedule_capacity: int,
    rank: int,
) -> tuple[
    torch.Tensor, torch.Tensor,  # schedule_peer_rank, schedule_peer_token_idx
    torch.Tensor, torch.Tensor,  # num_tokens, tokens_per_expert
]:
    world_size, num_local_tokens, topk = topk_all.shape
    num_routes = num_local_tokens * topk
    first_expert = rank * num_local_experts

    schedule_peer_rank = torch.full((schedule_capacity,), -1, dtype=torch.int32, device=topk_all.device)
    schedule_peer_token_idx = torch.full_like(schedule_peer_rank, -1)
    tokens_per_expert = torch.empty(num_local_experts, dtype=torch.int32, device=topk_all.device)

    # The CUDA scheduler visits all routes owned by thread 0, then thread 1, etc.
    num_schedule_threads = 1024
    route_order = torch.arange(
        math.ceil(num_routes / num_schedule_threads) * num_schedule_threads,
        device=topk_all.device,
    ).view(-1, num_schedule_threads).T.flatten()
    route_order = route_order[route_order < num_routes]
    topk_flat = topk_all.view(world_size, num_routes)

    offset = 0
    for local_expert in range(num_local_experts):
        expert = first_expert + local_expert
        peer_ranks = []
        peer_token_indices = []
        peer_offsets = []
        for peer_rank in range(world_size):
            peer_tokens = route_order[topk_flat[peer_rank, route_order] == expert]
            peer_ranks.append(torch.full_like(peer_tokens, peer_rank, dtype=torch.int32))
            peer_token_indices.append(peer_tokens.to(torch.int32))
            peer_offsets.append(torch.arange(peer_tokens.numel(), device=topk_all.device))

        num_expert_tokens = sum(tokens.numel() for tokens in peer_token_indices)
        padded_num_expert_tokens = math.ceil(num_expert_tokens / 256) * 256
        tokens_per_expert[local_expert] = padded_num_expert_tokens
        if offset + padded_num_expert_tokens > schedule_capacity:
            raise ValueError("schedule_capacity is too small for the reference schedule")

        if num_expert_tokens:
            expert_peer_ranks = torch.cat(peer_ranks)
            expert_peer_token_indices = torch.cat(peer_token_indices)
            interleave_order = torch.argsort(torch.cat(peer_offsets) * world_size + expert_peer_ranks, stable=True)
            expert_slice = slice(offset, offset + num_expert_tokens)
            schedule_peer_rank[expert_slice] = expert_peer_ranks[interleave_order]
            schedule_peer_token_idx[expert_slice] = expert_peer_token_indices[interleave_order]
        offset += padded_num_expert_tokens

    num_tokens = torch.tensor([offset], dtype=torch.int32, device=topk_all.device)
    return schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert


def run_mxfp8_quantize_normal_reference(
    x_bf16: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    is_2d = x_bf16.ndim == 2
    x_bf16_3d = x_bf16.unsqueeze(0) if is_2d else x_bf16
    num_experts, rows, columns = x_bf16_3d.shape

    x_float = x_bf16_3d.float().reshape(num_experts * rows, columns)
    dest_max = torch.tensor(448.0, dtype=torch.float32, device=x_bf16.device)
    min_exp = torch.tensor(-127.0, dtype=torch.float32, device=x_bf16.device)
    fp8e8m0_bias = torch.tensor(127.0, dtype=torch.float32, device=x_bf16.device)

    block_amax = x_float.abs().view(num_experts * rows, columns // 32, 32).amax(dim=-1)
    decode_scale = block_amax / dest_max
    scale_exponent = torch.clamp(torch.ceil(torch.log2(decode_scale)), min=min_exp)
    x_fp8 = (x_float / (2 ** scale_exponent.repeat_interleave(32, dim=-1))).to(torch.float8_e4m3fn)
    x_fp8 = x_fp8.reshape(num_experts, rows, columns)

    scale_bytes = (scale_exponent + fp8e8m0_bias).to(torch.uint8)
    scales = scale_bytes.reshape(num_experts * rows // 128, 128, columns // 128, 4).transpose(1, 2)
    scales = scales.reshape(num_experts * rows // 128, columns // 128, 4, 32, 4).transpose(-2, -3)
    scales = scales.reshape(num_experts * rows // 128, columns // 128, 32, 16).contiguous()

    return (x_fp8.squeeze(0) if is_2d else x_fp8), scales


def run_mxfp8_quantize_reference(
    x_bf16: torch.Tensor,
    return_normal: bool,
    return_transposed: bool,
) -> tuple[
    torch.Tensor | None, torch.Tensor | None,  # x_fp8, x_sc
    torch.Tensor | None, torch.Tensor | None,  # x_fp8_t, x_sc_t
]:
    if return_normal:
        x_fp8, x_sc = run_mxfp8_quantize_normal_reference(x_bf16)
    else:
        x_fp8, x_sc = None, None
    if return_transposed:
        x_fp8_t, x_sc_t = run_mxfp8_quantize_normal_reference(x_bf16.transpose(-2, -1).contiguous())
    else:
        x_fp8_t, x_sc_t = None, None
    return x_fp8, x_sc, x_fp8_t, x_sc_t


def run_swiglu_reference(
    gate: torch.Tensor,
    up: torch.Tensor,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    if swiglu_limit is None:
        return torch.nn.functional.silu(gate) * up
    else:
        dtype = gate.dtype
        gate_fp32 = gate.float().clamp(max=swiglu_limit)
        up_fp32 = up.float().clamp(min=-swiglu_limit, max=swiglu_limit)
        return (torch.nn.functional.silu(gate_fp32) * up_fp32).to(dtype)


def run_forward_reference_bf16(
    x: torch.Tensor,              # [T, H]
    topk_experts: torch.Tensor,   # [T, TOPK]
    w_shared_gate: torch.Tensor,  # [I, H]
    w_shared_up: torch.Tensor,    # [I, H]
    w_shared_down: torch.Tensor,  # [H, I]
    w_routed_gate: torch.Tensor,  # [E, I, H]
    w_routed_up: torch.Tensor,    # [E, I, H]
    w_routed_down: torch.Tensor,  # [E, H, I]
    swiglu_limit: float | None = None,
    *,
    activation: str = "swiglu",
) -> tuple[
    torch.Tensor,  # combine_buffer
    torch.Tensor,  # gate_shared
    torch.Tensor,  # up_shared
    torch.Tensor,  # hidden_shared
    torch.Tensor,  # y_shared
]:
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    num_local_experts = w_routed_gate.shape[0]
    num_local_tokens, hidden = x.shape
    topk = topk_experts.shape[1]
    num_routes = num_local_tokens * topk

    topk_experts_flat = topk_experts.flatten()
    destination_ranks = topk_experts_flat // num_local_experts
    dispatch_order = torch.argsort(destination_ranks, stable=True)
    send_counts = torch.bincount(destination_ranks, minlength=world_size)
    all_send_counts = torch.empty(world_size, world_size, dtype=torch.int64, device=x.device)
    dist.all_gather_into_tensor(all_send_counts, send_counts)
    send_splits = send_counts.tolist()
    recv_splits = all_send_counts[:, rank].tolist()
    num_recv = sum(recv_splits)
    send_x = x[dispatch_order // topk]
    send_local_experts = (topk_experts_flat[dispatch_order] % num_local_experts).contiguous()
    recv_x = all_to_all(send_x, num_recv, recv_splits, send_splits)
    recv_local_experts = all_to_all(send_local_experts, num_recv, recv_splits, send_splits)

    recv_output = torch.zeros_like(recv_x)
    for expert_idx in range(num_local_experts):
        rows = (recv_local_experts == expert_idx).nonzero().flatten()
        expert_x = recv_x[rows]
        gate = expert_x @ w_routed_gate[expert_idx].T
        up = expert_x @ w_routed_up[expert_idx].T
        hidden_activations = apply_glu(gate, up, activation, swiglu_limit)
        recv_output = recv_output.index_copy(
            0, rows, hidden_activations @ w_routed_down[expert_idx].T)

    returned_output = all_to_all(recv_output, num_routes, send_splits, recv_splits)
    combine_buffer = torch.empty_like(returned_output)
    combine_buffer[dispatch_order] = returned_output

    gate_shared = x @ w_shared_gate.T
    up_shared = x @ w_shared_up.T
    hidden_shared = apply_glu(gate_shared, up_shared, activation, swiglu_limit)
    y_shared = hidden_shared @ w_shared_down.T

    return combine_buffer, gate_shared, up_shared, hidden_shared, y_shared


def run_fwd_epilogue_reference(
    y_shared: torch.Tensor,
    combine_buffer: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    num_local_tokens, hidden_size = y_shared.shape
    topk = topk_weights.shape[1]
    routed = combine_buffer.view(num_local_tokens, topk, hidden_size)
    return (y_shared.float() + (routed.float() * topk_weights.unsqueeze(2)).sum(dim=1)).to(torch.bfloat16)


def run_bwd_epilogue_reference(
    d_x_shared: torch.Tensor,
    d_x_routed_buffer: torch.Tensor,
) -> torch.Tensor:
    num_local_tokens, hidden_size = d_x_shared.shape
    topk = d_x_routed_buffer.shape[0] // num_local_tokens
    d_x_routed = d_x_routed_buffer.view(num_local_tokens, topk, hidden_size)
    return (d_x_shared.float() + d_x_routed.float().sum(dim=1)).to(torch.bfloat16)


def run_reference_bf16(
    x: torch.Tensor,               # [T, H]
    topk_experts: torch.Tensor,    # [T, TOPK]
    router_weights: torch.Tensor,  # [T, TOPK]
    w_shared_gate: torch.Tensor,   # [I, H]
    w_shared_up: torch.Tensor,     # [I, H]
    w_shared_down: torch.Tensor,   # [H, I]
    w_routed_gate: torch.Tensor,   # [E, I, H]
    w_routed_up: torch.Tensor,     # [E, I, H]
    w_routed_down: torch.Tensor,   # [E, H, I]
    d_output: torch.Tensor,        # [T, H]
    swiglu_limit: float | None = None,
    *,
    activation: str = "swiglu",
) -> tuple[
    torch.Tensor,  # output
    torch.Tensor,  # d_x
    torch.Tensor,  # d_router_weights
    torch.Tensor,  # d_w_routed_gate
    torch.Tensor,  # d_w_routed_up
    torch.Tensor,  # d_w_routed_down
    torch.Tensor,  # d_w_shared_gate
    torch.Tensor,  # d_w_shared_up
    torch.Tensor,  # d_w_shared_down
]:
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    num_local_experts = w_routed_gate.shape[0]
    num_local_tokens, hidden = x.shape
    topk = topk_experts.shape[1]
    num_routes = num_local_tokens * topk

    # Dispatch all-to-all
    topk_experts_flat = topk_experts.flatten()
    destination_ranks = topk_experts_flat // num_local_experts
    dispatch_order = torch.argsort(destination_ranks, stable=True)
    send_counts = torch.bincount(destination_ranks, minlength=world_size)
    all_send_counts = torch.empty(world_size, world_size, dtype=torch.int64, device=x.device)
    dist.all_gather_into_tensor(all_send_counts, send_counts)
    send_splits = send_counts.tolist()
    recv_splits = all_send_counts[:, rank].tolist()
    num_recv = sum(recv_splits)
    send_x = x[dispatch_order // topk]
    send_local_experts = (topk_experts_flat[dispatch_order] % num_local_experts).contiguous()
    recv_x = all_to_all(send_x, num_recv, recv_splits, send_splits).requires_grad_()
    recv_local_experts = all_to_all(send_local_experts, num_recv, recv_splits, send_splits)

    # Routed expert FFN
    w_routed_gate = w_routed_gate.detach().requires_grad_()
    w_routed_up = w_routed_up.detach().requires_grad_()
    w_routed_down = w_routed_down.detach().requires_grad_()
    recv_output = torch.zeros_like(recv_x)
    for expert_idx in range(num_local_experts):
        rows = (recv_local_experts == expert_idx).nonzero().flatten()
        expert_x = recv_x[rows]
        gate = expert_x @ w_routed_gate[expert_idx].T
        up = expert_x @ w_routed_up[expert_idx].T
        hidden_activations = apply_glu(gate, up, activation, swiglu_limit)
        recv_output = recv_output.index_copy(0, rows, hidden_activations @ w_routed_down[expert_idx].T)

    # Routed token weighted sum
    returned_output = all_to_all(recv_output.detach(), num_routes, send_splits, recv_splits)
    flat_output = torch.empty_like(returned_output)
    flat_output[dispatch_order] = returned_output
    routed_output = (flat_output.view(num_local_tokens, topk, hidden).float() * router_weights.unsqueeze(2)).sum(1)

    # Shared expert FFN
    x_shared = x.detach().requires_grad_()
    w_shared_gate = w_shared_gate.detach().requires_grad_()
    w_shared_up = w_shared_up.detach().requires_grad_()
    w_shared_down = w_shared_down.detach().requires_grad_()
    gate_shared = x_shared @ w_shared_gate.T
    up_shared = x_shared @ w_shared_up.T
    shared_output = apply_glu(gate_shared, up_shared, activation, swiglu_limit) @ w_shared_down.T

    # Final sum
    output = (routed_output + shared_output.float()).to(torch.bfloat16)

    # Combine all-to-all
    d_flat_output = (d_output.unsqueeze(1).float() * router_weights.unsqueeze(2)).to(torch.bfloat16)
    d_send_output = d_flat_output.reshape(-1, hidden)[dispatch_order]
    d_recv_output = all_to_all(d_send_output, num_recv, recv_splits, send_splits).contiguous()
    d_recv_x, d_w_routed_gate, d_w_routed_up, d_w_routed_down = torch.autograd.grad(
        recv_output,
        (recv_x, w_routed_gate, w_routed_up, w_routed_down),
        d_recv_output,
    )

    # Backward
    returned_d_x = all_to_all(d_recv_x, num_routes, send_splits, recv_splits)
    flat_d_x = torch.empty_like(returned_d_x)
    flat_d_x[dispatch_order] = returned_d_x
    d_x_routed = flat_d_x.view(num_local_tokens, topk, hidden).float().sum(1)
    d_x_shared, d_w_shared_gate, d_w_shared_up, d_w_shared_down = torch.autograd.grad(
        shared_output,
        (x_shared, w_shared_gate, w_shared_up, w_shared_down),
        d_output,
    )
    d_x = (d_x_routed + d_x_shared.float()).to(torch.bfloat16)
    d_router_weights = (d_output.unsqueeze(1).float() * flat_output.view(num_local_tokens, topk, hidden).float()).sum(2)

    return (
        output,             # [T, H]
        d_x,                # [T, H]
        d_router_weights,   # [T, TOPK]
        d_w_routed_gate,    # [E, I, H]
        d_w_routed_up,      # [E, I, H]
        d_w_routed_down,    # [E, H, I]
        d_w_shared_gate,    # [I, H]
        d_w_shared_up,      # [I, H]
        d_w_shared_down,    # [H, I]
    )


def get_error_stats(
    reference: torch.Tensor,
    actual: torch.Tensor,
) -> tuple[float, float, float]:
    reference_float = reference.float()
    actual_float = actual.float()

    diff = (reference_float - actual_float).abs()
    diff_sum = diff.sum()
    diff_count = torch.tensor(diff.numel(), dtype=torch.float64, device=diff.device)
    diff_max = diff.max()
    reference_sum = reference_float.abs().sum()

    dist.all_reduce(diff_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(diff_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(diff_max, op=dist.ReduceOp.MAX)
    dist.all_reduce(reference_sum, op=dist.ReduceOp.SUM)

    abs_error_mean = (diff_sum / diff_count).item()
    abs_error_max = diff_max.item()
    relative_error = (diff_sum / reference_sum).item()

    return abs_error_mean, abs_error_max, relative_error


def check_correctness(
    name: str,
    reference: torch.Tensor,
    actual: torch.Tensor,
    tolerance: tuple[float, float],
    print_stats: bool = False,
) -> None:
    abs_error_mean, abs_error_max, relative_error = get_error_stats(reference, actual)
    absolute_tolerance, relative_tolerance = tolerance
    passed = (
        all(math.isfinite(value) for value in (abs_error_mean, abs_error_max, relative_error))
        and abs_error_max <= absolute_tolerance
        and relative_error <= relative_tolerance
    )
    if print_stats or not passed:
        print(f"{name}: abs error mean={abs_error_mean:.6e}, abs error max={abs_error_max:.6e}, relative error={relative_error:.6e}")
    assert passed, f"{name} exceeded tolerance: abs error max {abs_error_max:.6e} > {absolute_tolerance:.6e} or relative error {relative_error:.6e} > {relative_tolerance:.6e}"
