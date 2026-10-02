import itertools

import torch
import torch.distributed as dist

from mok import functional
from mok.functional import get_workspace
from mok.ops import mxfp8_quantize

from .utils import generate_inputs, run_reference_bf16

BF16_TOLERANCE = (0.5, 0.01)
MXFP8_TOLERANCE = (1.0, 0.1)
ROW_RESULT_NAMES = ("output", "gate_shared", "up_shared", "hidden_shared", "d_x", "d_router_weights")
GRADIENT_NAMES = (
    "d_x",
    "d_router_weights",
    "d_w_routed_gate",
    "d_w_routed_up",
    "d_w_routed_down",
    "d_w_shared_gate",
    "d_w_shared_up",
    "d_w_shared_down",
)
REFERENCE_NAMES = ("output", *GRADIENT_NAMES)


def ragged_shapes(world_size: int) -> tuple[tuple[str, int, int, int, int, int], ...]:
    return (  # (name, routed experts, hidden dim, intermediate dim, top-k, token capacity)
        ("GLM-5.2", 256, 6144, 2048, 8, 1024),
        ("Qwen3.5-397B-A17B", 512, 4096, 1024, 10, 1024),
        ("All minimums", world_size, 256, 256, 1, 512),
    )


def ragged_token_counts(capacity: int) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return (  # (name, per-rank token counts, cycled over ranks)
        ("Full", (capacity,)),
        ("Row block edges", (0, 1, 255, 257)),
        ("Near capacity", (511, capacity - 1, capacity, 0)),
        ("Mixed", (257, 0, capacity - 1, 1)),
        ("All empty", (0,)),
    )


def ragged_mok_params() -> tuple[tuple[str, int, int, int, int, float | None], ...]:
    return (  # (name, forward comm SMs, backward comm SMs, minibatch size, macrobatch size, SwiGLU limit)
        ("Default", 40, 28, 4096, 131072, None),
        ("Minimum macrobatch, clamped SwiGLU", 40, 28, 4096, 4096, 4.0),
    )


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    # Reinterpreting as integers makes the comparison exact, including -0.0 versus +0.0
    return tensor.contiguous().view({1: torch.uint8, 2: torch.int16, 4: torch.int32}[tensor.element_size()])


def assert_bitwise_equal(name: str, expected: torch.Tensor, actual: torch.Tensor) -> None:
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        mismatches = torch.tensor(-1, dtype=torch.int64, device=actual.device)
    else:
        mismatches = (_bits(expected) != _bits(actual)).sum()
    mismatch_flags = torch.stack([mismatches != 0, mismatches < 0]).to(torch.int64)
    dist.all_reduce(mismatch_flags)
    assert int(mismatch_flags.sum().item()) == 0, (
        f"{name}: local shape {tuple(actual.shape)} vs {tuple(expected.shape)}, "
        f"{int(mismatches.item())} mismatching elements on this rank"
    )


def check_close(
    name: str,
    reference: torch.Tensor,
    actual: torch.Tensor,
    tolerance: tuple[float, float],
    print_stats: bool = False,
) -> None:
    """Like utils.check_correctness, but also accepts ranks that hold zero rows."""
    assert reference.shape == actual.shape, f"{name}: shape {tuple(actual.shape)} != {tuple(reference.shape)}"
    diff = (reference.float() - actual.float()).abs()
    sums = torch.stack([
        diff.sum(dtype=torch.float64),
        reference.float().abs().sum(dtype=torch.float64),
        torch.tensor(float(diff.numel()), dtype=torch.float64, device=diff.device),
        (~torch.isfinite(actual.float())).sum().to(torch.float64),
    ])
    diff_max = diff.max().double() if diff.numel() else torch.zeros((), dtype=torch.float64, device=diff.device)
    dist.all_reduce(sums)
    dist.all_reduce(diff_max, op=dist.ReduceOp.MAX)
    diff_sum, reference_sum, count, non_finite = sums.tolist()
    abs_error_mean = diff_sum / max(count, 1.0)
    abs_error_max = float(diff_max.item())
    relative_error = 0.0 if diff_sum == 0.0 else diff_sum / reference_sum
    absolute_tolerance, relative_tolerance = tolerance
    passed = non_finite == 0 and abs_error_max <= absolute_tolerance and relative_error <= relative_tolerance
    if print_stats or not passed:
        print(f"{name}: abs error mean={abs_error_mean:.6e}, abs error max={abs_error_max:.6e}, relative error={relative_error:.6e}")
    assert passed, f"{name} exceeded tolerance: abs error max {abs_error_max:.6e} or relative error {relative_error:.6e}"


def pad_inputs(inputs: tuple[torch.Tensor, ...], num_rows: int) -> tuple[torch.Tensor, ...]:
    """The pre-ragged calling convention: rows past num_rows are zero activations with expert -1 and zero weight."""
    x, topk_experts, router_weights, *weights, d_output = inputs
    x, topk_experts, router_weights, d_output = (tensor.clone() for tensor in (x, topk_experts, router_weights, d_output))
    x[num_rows:] = 0
    topk_experts[num_rows:] = -1
    router_weights[num_rows:] = 0
    d_output[num_rows:] = 0
    return x, topk_experts, router_weights, *weights, d_output


def run_mok(
    config: functional.MoKConfig,
    workspace: functional.MoKWorkspace,
    precision: str,
    inputs: tuple[torch.Tensor, ...],
    num_local_experts: int,
    swiglu_limit: float | None,
) -> dict[str, torch.Tensor]:
    x, topk_experts, router_weights, w_shared_gate, w_shared_up, w_shared_down, w_routed_gate, w_routed_up, w_routed_down, d_output = inputs
    schedule = functional.build_schedule(workspace, config, topk_experts, num_local_experts=num_local_experts)
    if precision == "bf16":
        routed_gate_fwd = routed_gate_bwd = w_routed_gate
        routed_up_fwd = routed_up_bwd = w_routed_up
        routed_down_fwd = routed_down_bwd = w_routed_down
    else:
        gate_fp8, gate_sc, gate_t_fp8, gate_t_sc = mxfp8_quantize(w_routed_gate, True, True)
        up_fp8, up_sc, up_t_fp8, up_t_sc = mxfp8_quantize(w_routed_up, True, True)
        down_fp8, down_sc, down_t_fp8, down_t_sc = mxfp8_quantize(w_routed_down, True, True)
        routed_gate_fwd, routed_gate_bwd = (gate_fp8, gate_sc), (gate_fp8, gate_sc, gate_t_fp8, gate_t_sc)
        routed_up_fwd, routed_up_bwd = (up_fp8, up_sc), (up_fp8, up_sc, up_t_fp8, up_t_sc)
        routed_down_fwd, routed_down_bwd = (down_fp8, down_sc), (down_t_fp8, down_t_sc)

    output, forward_context = functional.forward(
        config, workspace, schedule, x, router_weights,
        w_shared_gate, w_shared_up, w_shared_down,
        routed_gate_fwd, routed_up_fwd, routed_down_fwd,
        swiglu_limit,
    )
    recomputed_context = functional.recompute_forward_context(
        config, workspace, schedule, x,
        w_shared_gate, w_shared_up, routed_gate_fwd, routed_up_fwd,
        swiglu_limit,
    )
    results = {"output": output}
    for context_name, context in (("", forward_context), ("recomputed/", recomputed_context)):
        results[f"{context_name}gate_shared"] = context.gate_shared
        results[f"{context_name}up_shared"] = context.up_shared
        results[f"{context_name}hidden_shared"] = context.hidden_shared
        gradients = functional.backward(
            config, workspace, schedule, context, d_output, x, router_weights,
            w_shared_gate, w_shared_up, w_shared_down,
            routed_gate_bwd, routed_up_bwd, routed_down_bwd,
            swiglu_limit,
        )
        results |= {f"{context_name}{name}": gradient for name, gradient in zip(GRADIENT_NAMES, gradients, strict=True)}
    return results


def test_ragged_e2e(context: tuple[int, int, torch.device]) -> None:
    rank, world_size, device = context
    for shape, params in itertools.product(ragged_shapes(world_size), ragged_mok_params()):
        shape_name, num_experts, hidden_dim, intermediate_dim, topk, capacity = shape
        params_name, fwd_num_comm_sms, bwd_num_comm_sms, minibatch_size, macrobatch_size, swiglu_limit = params
        assert num_experts % world_size == 0
        num_local_experts = num_experts // world_size
        config = functional.MoKConfig(
            fwd_num_comm_sms=fwd_num_comm_sms,
            bwd_num_comm_sms=bwd_num_comm_sms,
            minibatch_size=minibatch_size,
            macrobatch_size=macrobatch_size,
            schedule_capacity_multiplier=1.5,
        )
        workspace = get_workspace(
            config,
            dist.group.WORLD,
            device=device,
            num_local_tokens=capacity,
            hidden_size=hidden_dim,
            topk=topk,
        )
        full_inputs = generate_inputs(
            rank, device, num_experts, num_local_experts, topk, capacity, hidden_dim, intermediate_dim,
        )
        for counts_name, counts in ragged_token_counts(capacity):
            num_rows = counts[rank % len(counts)]
            ragged_inputs = tuple(
                tensor[:num_rows].contiguous() if index in (0, 1, 2, 9) else tensor
                for index, tensor in enumerate(full_inputs)
            )
            padded_inputs = pad_inputs(full_inputs, num_rows)
            reference = run_reference_bf16(*ragged_inputs, swiglu_limit)
            for precision in ("bf16", "mxfp8"):
                name = f"{shape_name}/{params_name}/{counts_name}/{precision}"
                ragged = run_mok(config, workspace, precision, ragged_inputs, num_local_experts, swiglu_limit)
                repeated = run_mok(config, workspace, precision, ragged_inputs, num_local_experts, swiglu_limit)
                padded = run_mok(config, workspace, precision, padded_inputs, num_local_experts, swiglu_limit)
                for result_name, result in ragged.items():
                    padded_result = padded[result_name]
                    if result_name.removeprefix("recomputed/") in ROW_RESULT_NAMES:
                        padded_result = padded_result[:num_rows]
                    assert_bitwise_equal(f"{name}/{result_name} vs padded", padded_result, result)
                    assert_bitwise_equal(f"{name}/{result_name} vs repeat", result, repeated[result_name])
                    if result_name.startswith("recomputed/"):
                        assert_bitwise_equal(f"{name}/{result_name} vs forward context", ragged[result_name.removeprefix("recomputed/")], result)
                for result_name, expected in zip(REFERENCE_NAMES, reference, strict=True):
                    check_close(
                        f"{name}/{result_name}",
                        expected,
                        ragged[result_name],
                        BF16_TOLERANCE if precision == "bf16" else MXFP8_TOLERANCE,
                        print_stats=rank == 0,
                    )


def test_ragged_input_validation(context: tuple[int, int, torch.device]) -> None:
    rank, world_size, device = context
    num_experts, hidden_dim, intermediate_dim, topk, capacity = world_size, 256, 256, 1, 512
    config = functional.MoKConfig(minibatch_size=256, macrobatch_size=4096)
    workspace = get_workspace(
        config, dist.group.WORLD, device=device, num_local_tokens=capacity, hidden_size=hidden_dim, topk=topk,
    )
    (x, topk_experts, router_weights, w_shared_gate, w_shared_up, w_shared_down,
     w_routed_gate, w_routed_up, w_routed_down, d_output) = generate_inputs(
        rank, device, num_experts, 1, topk, capacity + 256, hidden_dim, intermediate_dim,
    )

    try:
        functional.build_schedule(workspace, config, topk_experts, num_local_experts=1)
    except ValueError:
        pass
    else:
        raise AssertionError("top_experts beyond the workspace capacity should have failed")

    num_rows = 300
    schedule = functional.build_schedule(workspace, config, topk_experts[:num_rows], num_local_experts=1)
    assert schedule.num_local_tokens == num_rows
    valid_kwargs = {
        "config": config,
        "workspace": workspace,
        "schedule": schedule,
        "x": x[:num_rows],
        "router_weights": router_weights[:num_rows],
        "shared_gate_weights": w_shared_gate,
        "shared_up_weights": w_shared_up,
        "shared_down_weights": w_shared_down,
        "routed_gate_weights": w_routed_gate,
        "routed_up_weights": w_routed_up,
        "routed_down_weights": w_routed_down,
    }
    for failure_name, overrides in (
        ("rows beyond capacity", {"x": x, "router_weights": router_weights}),
        ("rows differ from the schedule", {"x": x[:num_rows - 1], "router_weights": router_weights[:num_rows - 1]}),
        ("router weight rows differ from x", {"router_weights": router_weights[:num_rows - 1]}),
        (
            "rows differ from a schedule without a row count, which means capacity",
            {"schedule": functional.MoKSchedule(
                peer_rank=schedule.peer_rank,
                peer_token_idx=schedule.peer_token_idx,
                num_tokens=schedule.num_tokens,
                tokens_per_expert=schedule.tokens_per_expert,
            )},
        ),
    ):
        try:
            functional.forward(**(valid_kwargs | overrides))
        except ValueError:
            pass
        else:
            raise AssertionError(f"{failure_name} should have failed")

    # A valid call still works after the rejected ones, and the schedule's peers stay in lockstep
    output, forward_context = functional.forward(**valid_kwargs)
    d_x, d_router_weights, *_ = functional.backward(
        config, workspace, schedule, forward_context, d_output[:num_rows], x[:num_rows], router_weights[:num_rows],
        w_shared_gate, w_shared_up, w_shared_down, w_routed_gate, w_routed_up, w_routed_down,
    )
    assert tuple(output.shape) == (num_rows, hidden_dim)
    assert tuple(forward_context.gate_shared.shape) == (num_rows, intermediate_dim)
    assert tuple(d_x.shape) == (num_rows, hidden_dim)
    assert tuple(d_router_weights.shape) == (num_rows, topk)
    torch.cuda.synchronize(device)


def test_ragged_fake_tensor_metadata(context: tuple[int, int, torch.device]) -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    from .test_misc import _make_fake_workspace

    _, _, real_device = context
    capacity, hidden_size, intermediate_size, num_local_experts, topk = 512, 1024, 256, 4, 2
    config = functional.MoKConfig(minibatch_size=256, macrobatch_size=512)
    with FakeTensorMode():
        device = torch.device("cuda", real_device.index)

        def tensor(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
            return torch.empty(shape, device=device, dtype=dtype)

        workspace = _make_fake_workspace(device, num_local_tokens=capacity, hidden_size=hidden_size, topk=topk)
        shared_gate, shared_up = (tensor((intermediate_size, hidden_size), torch.bfloat16) for _ in range(2))
        shared_down = tensor((hidden_size, intermediate_size), torch.bfloat16)
        routed_gate, routed_up = (tensor((num_local_experts, intermediate_size, hidden_size), torch.bfloat16) for _ in range(2))
        routed_down = tensor((num_local_experts, hidden_size, intermediate_size), torch.bfloat16)
        for num_rows in (0, 1, 300, capacity):
            x = tensor((num_rows, hidden_size), torch.bfloat16)
            router_weights = tensor((num_rows, topk), torch.float32)
            schedule = functional.build_schedule(
                workspace, config, tensor((num_rows, topk), torch.int64), num_local_experts=num_local_experts,
            )
            output, forward_context = functional.forward(
                config, workspace, schedule, x, router_weights,
                shared_gate, shared_up, shared_down, routed_gate, routed_up, routed_down,
            )
            recomputed_context = functional.recompute_forward_context(
                config, workspace, schedule, x, shared_gate, shared_up, routed_gate, routed_up,
            )
            gradients = functional.backward(
                config, workspace, schedule, forward_context, tensor((num_rows, hidden_size), torch.bfloat16),
                x, router_weights, shared_gate, shared_up, shared_down, routed_gate, routed_up, routed_down,
            )
            assert tuple(output.shape) == (num_rows, hidden_size)
            for context_ in (forward_context, recomputed_context):
                for shared in (context_.gate_shared, context_.up_shared, context_.hidden_shared):
                    assert tuple(shared.shape) == (num_rows, intermediate_size)
            assert [tuple(gradient.shape) for gradient in gradients] == [
                (num_rows, hidden_size),
                (num_rows, topk),
                (num_local_experts, intermediate_size, hidden_size),
                (num_local_experts, intermediate_size, hidden_size),
                (num_local_experts, hidden_size, intermediate_size),
                (intermediate_size, hidden_size),
                (intermediate_size, hidden_size),
                (hidden_size, intermediate_size),
            ]
