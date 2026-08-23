"""Miscellaneous tests that we let AI agents freely add."""

import math
from collections.abc import Callable

import pytest
import torch
import torch.distributed as dist

from mok import functional, ops

from .utils import (
    check_correctness,
    run_reference_bf16,
)


BF16_TOLERANCE = (0.5, 0.01)
MXFP8_TOLERANCE = (1.0, 0.1)
BF16_GRADIENT_MIN_COSINE = 0.9999
MXFP8_GRADIENT_MIN_COSINE = 0.996
RESULT_NAMES = (
    "output",
    "d_x",
    "d_router_weights",
    "d_w_routed_gate",
    "d_w_routed_up",
    "d_w_routed_down",
    "d_w_shared_gate",
    "d_w_shared_up",
    "d_w_shared_down",
)


def _run_e2e_case(
    context: tuple[int, int, torch.device],
    *,
    name: str,
    hidden_size: int,
    intermediate_size: int,
    num_local_experts: int,
    topk: int,
    config: functional.MoKConfig,
    precisions: tuple[str, ...],
    seed: int = 1234,
    grad_seed: int | None = None,
    top_experts: torch.Tensor | None = None,
    group: dist.ProcessGroup | None = None,
) -> None:
    rank, _, device = context
    ep_group = dist.group.WORLD if group is None else group
    num_local_tokens = 512
    num_experts = num_local_experts * dist.get_world_size(ep_group)
    generator = torch.Generator(device=device).manual_seed(seed + rank)
    router_logits = torch.randn(
        num_local_tokens,
        num_experts,
        generator=generator,
        device=device,
    )
    topk_values, generated_top_experts = torch.topk(router_logits, topk, dim=1)
    router_weights = torch.softmax(topk_values.float(), dim=-1)
    x = torch.randn(
        num_local_tokens,
        hidden_size,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    w_shared_gate = (
        torch.randn(
            intermediate_size,
            hidden_size,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        * hidden_size**-0.5
    )
    w_shared_up = (
        torch.randn(
            intermediate_size,
            hidden_size,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        * hidden_size**-0.5
    )
    w_shared_down = (
        torch.randn(
            hidden_size,
            intermediate_size,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        * intermediate_size**-0.5
    )
    w_routed_gate = (
        torch.randn(
            num_local_experts,
            intermediate_size,
            hidden_size,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        * hidden_size**-0.5
    )
    w_routed_up = (
        torch.randn(
            num_local_experts,
            intermediate_size,
            hidden_size,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        * hidden_size**-0.5
    )
    w_routed_down = (
        torch.randn(
            num_local_experts,
            hidden_size,
            intermediate_size,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        * intermediate_size**-0.5
    )
    if grad_seed is None:
        grad_generator = generator
    else:
        grad_generator = torch.Generator(device=device).manual_seed(
            grad_seed + rank
        )
    d_output = (
        torch.randn(
            num_local_tokens,
            hidden_size,
            generator=grad_generator,
            device=device,
            dtype=torch.bfloat16,
        )
        * hidden_size**-0.5
    )
    input_tuple = (
        x,
        generated_top_experts if top_experts is None else top_experts,
        router_weights,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        w_routed_gate,
        w_routed_up,
        w_routed_down,
        d_output,
    )
    reference = run_reference_bf16(*input_tuple, group=ep_group)
    workspace = functional.get_workspace(
        config,
        ep_group,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
    )
    schedule = functional.build_schedule(
        workspace,
        config,
        input_tuple[1],
        num_local_experts=num_local_experts,
    )

    for precision in precisions:
        if precision == "bf16":
            output, forward_context = functional.forward(
                config,
                workspace,
                schedule,
                x,
                router_weights,
                w_shared_gate,
                w_shared_up,
                w_shared_down,
                w_routed_gate,
                w_routed_up,
                w_routed_down,
            )
            gradients = functional.backward(
                config,
                workspace,
                schedule,
                forward_context,
                d_output,
                x,
                router_weights,
                w_shared_gate,
                w_shared_up,
                w_shared_down,
                w_routed_gate,
                w_routed_up,
                w_routed_down,
            )
            actual = (output, *gradients)
            tolerance = BF16_TOLERANCE
            gradient_min_cosine = BF16_GRADIENT_MIN_COSINE
        elif precision == "mxfp8":
            (
                w_routed_gate_fp8,
                w_routed_gate_sc,
                w_routed_gate_t_fp8,
                w_routed_gate_t_sc,
            ) = ops.mxfp8_quantize(w_routed_gate, True, True)
            (
                w_routed_up_fp8,
                w_routed_up_sc,
                w_routed_up_t_fp8,
                w_routed_up_t_sc,
            ) = ops.mxfp8_quantize(w_routed_up, True, True)
            (
                w_routed_down_fp8,
                w_routed_down_sc,
                w_routed_down_t_fp8,
                w_routed_down_t_sc,
            ) = ops.mxfp8_quantize(w_routed_down, True, True)
            assert all(
                tensor is not None
                for tensor in (
                    w_routed_gate_fp8,
                    w_routed_gate_sc,
                    w_routed_gate_t_fp8,
                    w_routed_gate_t_sc,
                    w_routed_up_fp8,
                    w_routed_up_sc,
                    w_routed_up_t_fp8,
                    w_routed_up_t_sc,
                    w_routed_down_fp8,
                    w_routed_down_sc,
                    w_routed_down_t_fp8,
                    w_routed_down_t_sc,
                )
            )
            output, forward_context = functional.forward(
                config,
                workspace,
                schedule,
                x,
                router_weights,
                w_shared_gate,
                w_shared_up,
                w_shared_down,
                (w_routed_gate_fp8, w_routed_gate_sc),
                (w_routed_up_fp8, w_routed_up_sc),
                (w_routed_down_fp8, w_routed_down_sc),
            )
            gradients = functional.backward(
                config,
                workspace,
                schedule,
                forward_context,
                d_output,
                x,
                router_weights,
                w_shared_gate,
                w_shared_up,
                w_shared_down,
                (
                    w_routed_gate_fp8,
                    w_routed_gate_sc,
                    w_routed_gate_t_fp8,
                    w_routed_gate_t_sc,
                ),
                (
                    w_routed_up_fp8,
                    w_routed_up_sc,
                    w_routed_up_t_fp8,
                    w_routed_up_t_sc,
                ),
                (
                    w_routed_down_t_fp8,
                    w_routed_down_t_sc,
                ),
            )
            actual = (output, *gradients)
            tolerance = MXFP8_TOLERANCE
            gradient_min_cosine = MXFP8_GRADIENT_MIN_COSINE
        else:
            raise AssertionError(f"unsupported precision {precision!r}")
        for result_name, expected, result in zip(
            RESULT_NAMES,
            reference,
            actual,
            strict=True,
        ):
            check_correctness(
                f"{name}/{precision}/{result_name}",
                expected,
                result,
                tolerance,
                print_stats=rank == 0,
            )
        for result_name, expected, result in zip(
            RESULT_NAMES[1:],
            reference[1:],
            actual[1:],
            strict=True,
        ):
            expected_flat = expected.float().reshape(-1)
            result_flat = result.float().reshape(-1)
            denominator = expected_flat.norm() * result_flat.norm()
            if float(denominator.item()) <= 1e-12:
                local_cosine = (
                    1.0 if torch.equal(expected_flat, result_flat) else 0.0
                )
            else:
                local_cosine = float(
                    ((expected_flat @ result_flat) / denominator).item()
                )
            finite = bool(torch.isfinite(expected_flat).all().item()) and bool(
                torch.isfinite(result_flat).all().item()
            )
            stats = torch.tensor(
                [local_cosine, int(finite)],
                device=device,
                dtype=torch.float32,
            )
            dist.all_reduce(stats[0:1], op=dist.ReduceOp.MIN)
            dist.all_reduce(stats[1:2], op=dist.ReduceOp.MIN)
            minimum_cosine = float(stats[0].item())
            if rank == 0:
                print(
                    f"{name}/{precision}/{result_name}: "
                    f"minimum cosine={minimum_cosine:.8f}"
                )
            assert bool(stats[1].item())
            assert minimum_cosine >= gradient_min_cosine


def _assert_metadata(
    tensors: tuple[torch.Tensor, ...],
    expected: tuple[tuple[tuple[int, ...], torch.dtype], ...],
) -> None:
    assert len(tensors) == len(expected)
    for tensor, (shape, dtype) in zip(tensors, expected, strict=True):
        assert isinstance(tensor, torch.Tensor)
        assert tuple(tensor.shape) == shape
        assert tensor.dtype == dtype
        assert tensor.device.type == "cuda"


def _make_fake_workspace(
    device: torch.device,
    *,
    num_local_tokens: int = 512,
    hidden_size: int = 1024,
    topk: int = 2,
    ep_size: int = 4,
    schedule_capacity: int = 4096,
) -> functional.MoKWorkspace:
    pointers = list(range(1, ep_size + 1))

    def tensor(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return torch.empty(shape, device=device, dtype=dtype)

    return functional.MoKWorkspace(
        group_name="fake-ep-group",
        ep_rank=0,
        ep_size=ep_size,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
        schedule_capacity=schedule_capacity,
        x_buffer=tensor((num_local_tokens, hidden_size), torch.bfloat16),
        x_buffer_handle=None,
        x_buffer_ptrs=pointers,
        combine_buffer=tensor(
            (num_local_tokens * topk, hidden_size),
            torch.bfloat16,
        ),
        combine_buffer_handle=None,
        combine_buffer_ptrs=pointers,
        d_y_buffer=tensor((num_local_tokens, hidden_size), torch.bfloat16),
        d_y_buffer_handle=None,
        d_y_buffer_ptrs=pointers,
        d_x_routed_buffer=tensor(
            (num_local_tokens * topk, hidden_size),
            torch.bfloat16,
        ),
        d_x_routed_buffer_handle=None,
        d_x_routed_buffer_ptrs=pointers,
        router_weight_buffer=tensor(
            (num_local_tokens, topk),
            torch.float32,
        ),
        router_weight_buffer_handle=None,
        router_weight_buffer_ptrs=pointers,
        d_router_weight_buffer=tensor(
            (num_local_tokens, topk),
            torch.float32,
        ),
        d_router_weight_buffer_handle=None,
        d_router_weight_buffer_ptrs=pointers,
        all_gather_top_experts_buffer=tensor(
            (ep_size, num_local_tokens, topk),
            torch.int32,
        ),
        all_gather_top_experts_buffer_handle=None,
        all_gather_top_experts_buffer_multicast_ptr=1,
        barrier_buffer=tensor((1,), torch.int32),
        barrier_buffer_handle=None,
        barrier_buffer_ptrs=pointers,
        barrier_buffer_multicast_ptr=1,
        barrier_target=tensor((1,), torch.int32),
    )


def test_ep1_on_each_rank(
    context: tuple[int, int, torch.device],
) -> None:
    rank, world_size, device = context
    singleton_groups = [
        dist.new_group(ranks=[group_rank])
        for group_rank in range(world_size)
    ]
    ep_group = singleton_groups[rank]
    assert isinstance(ep_group, dist.ProcessGroup)

    config = functional.MoKConfig(
        fwd_num_comm_sms=2,
        bwd_num_comm_sms=2,
        minibatch_size=256,
        macrobatch_size=512,
        all_gather_top_experts_chunk_bytes=16,
    )
    functional.clear_workspace_cache()
    try:
        workspace = functional.get_workspace(
            config,
            ep_group,
            device=device,
            num_local_tokens=512,
            hidden_size=256,
            topk=1,
        )

        assert workspace.ep_rank == 0
        assert workspace.ep_size == 1
        for buffer, handle, pointers in (
            (
                workspace.x_buffer,
                workspace.x_buffer_handle,
                workspace.x_buffer_ptrs,
            ),
            (
                workspace.combine_buffer,
                workspace.combine_buffer_handle,
                workspace.combine_buffer_ptrs,
            ),
            (
                workspace.d_y_buffer,
                workspace.d_y_buffer_handle,
                workspace.d_y_buffer_ptrs,
            ),
            (
                workspace.d_x_routed_buffer,
                workspace.d_x_routed_buffer_handle,
                workspace.d_x_routed_buffer_ptrs,
            ),
            (
                workspace.router_weight_buffer,
                workspace.router_weight_buffer_handle,
                workspace.router_weight_buffer_ptrs,
            ),
            (
                workspace.d_router_weight_buffer,
                workspace.d_router_weight_buffer_handle,
                workspace.d_router_weight_buffer_ptrs,
            ),
        ):
            assert handle is None
            assert pointers == [buffer.data_ptr()]

        assert workspace.all_gather_top_experts_buffer_handle is None
        assert (
            workspace.all_gather_top_experts_buffer_multicast_ptr
            == workspace.all_gather_top_experts_buffer.data_ptr()
        )
        assert workspace.barrier_buffer_handle is None
        assert workspace.barrier_buffer_ptrs == [
            workspace.barrier_buffer.data_ptr()
        ]
        assert (
            workspace.barrier_buffer_multicast_ptr
            == workspace.barrier_buffer.data_ptr()
        )

        top_experts = torch.zeros(
            512,
            1,
            dtype=torch.int32,
            device=device,
        )
        ops.all_gather_top_experts(
            top_experts,
            workspace.all_gather_top_experts_buffer,
            workspace.all_gather_top_experts_buffer_multicast_ptr,
            0,
            16,
        )
        assert torch.equal(
            workspace.all_gather_top_experts_buffer[0],
            top_experts,
        )

        ops.barrier_all(
            workspace.barrier_buffer,
            workspace.barrier_buffer_ptrs,
            workspace.barrier_buffer_multicast_ptr,
            workspace.barrier_target,
        )
        torch.cuda.synchronize(device)
        assert int(workspace.barrier_buffer.item()) == 0
        assert int(workspace.barrier_target.item()) == 0

        _run_e2e_case(
            context,
            name="ep1-minimum",
            hidden_size=256,
            intermediate_size=256,
            num_local_experts=1,
            topk=1,
            config=config,
            precisions=("bf16", "mxfp8"),
            group=ep_group,
        )
    finally:
        functional.clear_workspace_cache()
        dist.barrier()
        for group in singleton_groups:
            if isinstance(group, dist.ProcessGroup):
                dist.destroy_process_group(group)


def test_e2e_fixed_first_topk_experts_default_capacity(
    context: tuple[int, int, torch.device],
) -> None:
    _, _, device = context
    functional.clear_workspace_cache()
    topk = 8
    try:
        _run_e2e_case(
            context,
            name="fixed-first-topk/default-capacity",
            hidden_size=7168,
            intermediate_size=2048,
            num_local_experts=4,
            topk=topk,
            config=functional.MoKConfig(
                minibatch_size=256,
                macrobatch_size=512,
            ),
            precisions=("bf16", "mxfp8"),
            top_experts=torch.arange(
                topk,
                device=device,
                dtype=torch.int64,
            ).expand(512, topk).contiguous(),
        )
    finally:
        functional.clear_workspace_cache()


def test_workspace_cache_reuse_and_isolation(
    context: tuple[int, int, torch.device],
) -> None:
    _, world_size, device = context
    functional.clear_workspace_cache()
    duplicate_world = dist.new_group(ranks=list(range(world_size)))
    kwargs = {
        "device": device,
        "num_local_tokens": 512,
        "hidden_size": 256,
        "topk": 1,
    }
    default_config = functional.MoKConfig(
        minibatch_size=256,
        macrobatch_size=512,
    )
    larger_config = functional.MoKConfig(
        minibatch_size=256,
        macrobatch_size=512,
        schedule_capacity_multiplier=1.5,
    )
    try:
        created_workspace = functional.create_workspace(
            default_config,
            dist.group.WORLD,
            **kwargs,
        )
        default_workspace = functional.get_workspace(
            default_config,
            dist.group.WORLD,
            **kwargs,
        )
        reused_workspace = functional.get_workspace(
            default_config,
            dist.group.WORLD,
            **kwargs,
        )
        larger_workspace = functional.get_workspace(
            larger_config,
            dist.group.WORLD,
            **kwargs,
        )
        duplicate_group_workspace = functional.get_workspace(
            default_config,
            duplicate_world,
            **kwargs,
        )

        expected_default_capacity = 512 * max(
            2,
            math.ceil(world_size * default_config.schedule_capacity_multiplier),
        )
        expected_larger_capacity = 512 * max(
            2,
            math.ceil(world_size * larger_config.schedule_capacity_multiplier),
        )
        assert created_workspace is not default_workspace
        assert reused_workspace is default_workspace
        assert larger_workspace is not default_workspace
        assert duplicate_group_workspace is not default_workspace
        assert default_workspace.schedule_capacity == expected_default_capacity
        assert larger_workspace.schedule_capacity == expected_larger_capacity
        assert duplicate_group_workspace.group_name == duplicate_world.group_name
        assert duplicate_group_workspace.group_name != default_workspace.group_name
    finally:
        functional.clear_workspace_cache()
        dist.barrier()
        if isinstance(duplicate_world, dist.ProcessGroup):
            dist.destroy_process_group(duplicate_world)


def test_ep_subgroups_do_not_use_world(
    context: tuple[int, int, torch.device],
) -> None:
    rank, world_size, device = context
    if world_size < 8 or world_size % 4 != 0:
        pytest.skip("requires a world size divisible by four and at least eight")

    functional.clear_workspace_cache()
    subgroup_ranks = [
        list(range(start, start + 4))
        for start in range(0, world_size, 4)
    ]
    subgroups = [dist.new_group(ranks=ranks) for ranks in subgroup_ranks]
    subgroup_index = rank // 4
    subgroup = subgroups[subgroup_index]
    subgroup_rank = rank % 4
    topk = 2 + subgroup_index % 2
    num_local_experts = 4
    config = functional.MoKConfig(
        minibatch_size=256,
        macrobatch_size=512,
    )
    try:
        workspace = functional.get_workspace(
            config,
            subgroup,
            device=device,
            num_local_tokens=512,
            hidden_size=1024,
            topk=topk,
        )
        routes = (
            torch.arange(512 * topk, device=device, dtype=torch.int64)
            + subgroup_rank
        ).remainder(4 * num_local_experts).view(512, topk)
        schedule = functional.build_schedule(
            workspace,
            config,
            routes,
            num_local_experts=num_local_experts,
        )
        valid_peer_ranks = (
            (schedule.peer_rank == -1)
            | ((schedule.peer_rank >= 0) & (schedule.peer_rank < 4))
        )

        assert workspace.group_name == subgroup.group_name
        assert workspace.group_name != dist.group.WORLD.group_name
        assert workspace.ep_rank == subgroup_rank
        assert workspace.ep_size == 4
        assert workspace.topk == topk
        assert len(workspace.x_buffer_ptrs) == 4
        assert len(workspace.combine_buffer_ptrs) == 4
        assert len(workspace.barrier_buffer_ptrs) == 4
        assert bool(valid_peer_ranks.all().item())
    finally:
        functional.clear_workspace_cache()
        dist.barrier()
        for process_group in subgroups:
            if isinstance(process_group, dist.ProcessGroup):
                dist.destroy_process_group(process_group)


def test_fake_tensor_metadata(
    context: tuple[int, int, torch.device],
) -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    _, _, real_device = context
    num_local_tokens = 512
    hidden_size = 1024
    intermediate_size = 256
    num_local_experts = 4
    ep_size = 4
    topk = 2
    macrobatch_size = 512
    schedule_capacity = 4096
    pointers = list(range(1, ep_size + 1))

    with FakeTensorMode():
        device = torch.device("cuda", real_device.index)

        def tensor(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
            return torch.empty(shape, device=device, dtype=dtype)

        workspace = _make_fake_workspace(
            device,
            num_local_tokens=num_local_tokens,
            hidden_size=hidden_size,
            topk=topk,
            ep_size=ep_size,
            schedule_capacity=schedule_capacity,
        )
        top_experts = tensor((num_local_tokens, topk), torch.int32)
        assert (
            ops.all_gather_top_experts(
                top_experts,
                workspace.all_gather_top_experts_buffer,
                workspace.all_gather_top_experts_buffer_multicast_ptr,
                0,
                1024,
            )
            is None
        )
        assert (
            ops.barrier_all(
                workspace.barrier_buffer,
                pointers,
                workspace.barrier_buffer_multicast_ptr,
                workspace.barrier_target,
            )
            is None
        )
        schedule = ops.schedule(
            workspace.all_gather_top_experts_buffer,
            num_local_experts,
            schedule_capacity,
            0,
        )
        _assert_metadata(
            schedule,
            (
                ((schedule_capacity,), torch.int32),
                ((schedule_capacity,), torch.int32),
                ((1,), torch.int32),
                ((num_local_experts,), torch.int32),
            ),
        )

        x = tensor((num_local_tokens, hidden_size), torch.bfloat16)
        router_weights = tensor((num_local_tokens, topk), torch.float32)
        shared_gate = tensor(
            (intermediate_size, hidden_size),
            torch.bfloat16,
        )
        shared_up = tensor(
            (intermediate_size, hidden_size),
            torch.bfloat16,
        )
        shared_down = tensor(
            (hidden_size, intermediate_size),
            torch.bfloat16,
        )
        routed_gate_bf16 = tensor(
            (num_local_experts, intermediate_size, hidden_size),
            torch.bfloat16,
        )
        routed_up_bf16 = tensor(
            (num_local_experts, intermediate_size, hidden_size),
            torch.bfloat16,
        )
        routed_down_bf16 = tensor(
            (num_local_experts, hidden_size, intermediate_size),
            torch.bfloat16,
        )
        routed_gate = ops.mxfp8_quantize(routed_gate_bf16, True, True)
        routed_up = ops.mxfp8_quantize(routed_up_bf16, True, True)
        routed_down = ops.mxfp8_quantize(routed_down_bf16, True, True)
        _assert_metadata(
            routed_gate,
            (
                (
                    (num_local_experts, intermediate_size, hidden_size),
                    torch.float8_e4m3fn,
                ),
                (
                    (
                        num_local_experts * intermediate_size // 128,
                        hidden_size // 128,
                        32,
                        16,
                    ),
                    torch.uint8,
                ),
                (
                    (num_local_experts, hidden_size, intermediate_size),
                    torch.float8_e4m3fn,
                ),
                (
                    (
                        num_local_experts * hidden_size // 128,
                        intermediate_size // 128,
                        32,
                        16,
                    ),
                    torch.uint8,
                ),
            ),
        )
        normal_only = ops.mxfp8_quantize(routed_gate_bf16, True, False)
        transposed_only = ops.mxfp8_quantize(
            routed_gate_bf16,
            False,
            True,
        )
        assert normal_only[0] is not None
        assert normal_only[1] is not None
        assert normal_only[2:] == (None, None)
        assert tuple(normal_only[0].shape) == tuple(routed_gate_bf16.shape)
        assert normal_only[0].dtype == torch.float8_e4m3fn
        assert tuple(normal_only[1].shape) == (
            num_local_experts * intermediate_size // 128,
            hidden_size // 128,
            32,
            16,
        )
        assert normal_only[1].dtype == torch.uint8
        assert transposed_only[:2] == (None, None)
        assert transposed_only[2] is not None
        assert transposed_only[3] is not None
        assert tuple(transposed_only[2].shape) == (
            num_local_experts,
            hidden_size,
            intermediate_size,
        )
        assert transposed_only[2].dtype == torch.float8_e4m3fn
        assert tuple(transposed_only[3].shape) == (
            num_local_experts * hidden_size // 128,
            intermediate_size // 128,
            32,
            16,
        )
        assert transposed_only[3].dtype == torch.uint8

        mxfp8_forward = ops.dispatch_mlp_swiglu_combine_fwd_mxfp8(
            x,
            pointers,
            workspace.combine_buffer,
            pointers,
            shared_gate,
            routed_gate[0],
            routed_gate[1],
            shared_up,
            routed_up[0],
            routed_up[1],
            shared_down,
            routed_down[0],
            routed_down[1],
            *schedule,
            topk,
            None,
            2,
            macrobatch_size,
            256,
        )
        _assert_metadata(
            mxfp8_forward,
            (
                ((hidden_size, macrobatch_size), torch.float8_e4m3fn),
                (
                    (hidden_size // 128, macrobatch_size // 128, 32, 16),
                    torch.uint8,
                ),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((macrobatch_size, intermediate_size), torch.float8_e4m3fn),
                (
                    (
                        macrobatch_size // 128,
                        intermediate_size // 128,
                        32,
                        16,
                    ),
                    torch.uint8,
                ),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((macrobatch_size, intermediate_size), torch.float8_e4m3fn),
                (
                    (
                        macrobatch_size // 128,
                        intermediate_size // 128,
                        32,
                        16,
                    ),
                    torch.uint8,
                ),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((intermediate_size, macrobatch_size), torch.float8_e4m3fn),
                (
                    (
                        intermediate_size // 128,
                        macrobatch_size // 128,
                        32,
                        16,
                    ),
                    torch.uint8,
                ),
                ((num_local_tokens, hidden_size), torch.bfloat16),
                ((macrobatch_size, hidden_size), torch.bfloat16),
            ),
        )
        mxfp8_recomputed = ops.recompute_forward_context_mxfp8(
            x,
            pointers,
            shared_gate,
            routed_gate[0],
            routed_gate[1],
            shared_up,
            routed_up[0],
            routed_up[1],
            *schedule,
            topk,
            None,
            2,
            macrobatch_size,
            256,
        )
        _assert_metadata(
            mxfp8_recomputed,
            (
                ((hidden_size, macrobatch_size), torch.float8_e4m3fn),
                (
                    (hidden_size // 128, macrobatch_size // 128, 32, 16),
                    torch.uint8,
                ),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((macrobatch_size, intermediate_size), torch.float8_e4m3fn),
                (
                    (
                        macrobatch_size // 128,
                        intermediate_size // 128,
                        32,
                        16,
                    ),
                    torch.uint8,
                ),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((macrobatch_size, intermediate_size), torch.float8_e4m3fn),
                (
                    (
                        macrobatch_size // 128,
                        intermediate_size // 128,
                        32,
                        16,
                    ),
                    torch.uint8,
                ),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((intermediate_size, macrobatch_size), torch.float8_e4m3fn),
                (
                    (
                        intermediate_size // 128,
                        macrobatch_size // 128,
                        32,
                        16,
                    ),
                    torch.uint8,
                ),
            ),
        )
        output = ops.fwd_epilogue(
            mxfp8_forward[-2],
            workspace.combine_buffer,
            router_weights,
        )
        _assert_metadata(
            (output,),
            (((num_local_tokens, hidden_size), torch.bfloat16),),
        )

        mxfp8_backward = ops.dispatch_mlp_swiglu_combine_bwd_mxfp8(
            workspace.d_y_buffer,
            pointers,
            workspace.d_x_routed_buffer,
            pointers,
            workspace.router_weight_buffer,
            pointers,
            workspace.d_router_weight_buffer,
            pointers,
            shared_gate,
            routed_gate[2],
            routed_gate[3],
            shared_up,
            routed_up[2],
            routed_up[3],
            shared_down,
            routed_down[2],
            routed_down[3],
            *mxfp8_forward[:11],
            x,
            pointers,
            routed_gate[0],
            routed_gate[1],
            routed_up[0],
            routed_up[1],
            *schedule,
            topk,
            None,
            2,
            macrobatch_size,
            256,
        )
        _assert_metadata(
            mxfp8_backward,
            (
                ((num_local_tokens, hidden_size), torch.bfloat16),
                ((macrobatch_size, hidden_size), torch.bfloat16),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((macrobatch_size, intermediate_size), torch.float8_e4m3fn),
                (
                    (
                        macrobatch_size // 128,
                        intermediate_size // 128,
                        32,
                        16,
                    ),
                    torch.uint8,
                ),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((macrobatch_size, intermediate_size), torch.float8_e4m3fn),
                (
                    (
                        macrobatch_size // 128,
                        intermediate_size // 128,
                        32,
                        16,
                    ),
                    torch.uint8,
                ),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((macrobatch_size, intermediate_size), torch.bfloat16),
                ((macrobatch_size, hidden_size), torch.float8_e4m3fn),
                (
                    (
                        macrobatch_size // 128,
                        hidden_size // 128,
                        32,
                        16,
                    ),
                    torch.uint8,
                ),
                ((intermediate_size, hidden_size), torch.bfloat16),
                (
                    (num_local_experts, intermediate_size, hidden_size),
                    torch.bfloat16,
                ),
                ((intermediate_size, hidden_size), torch.bfloat16),
                (
                    (num_local_experts, intermediate_size, hidden_size),
                    torch.bfloat16,
                ),
                ((hidden_size, intermediate_size), torch.bfloat16),
                (
                    (num_local_experts, hidden_size, intermediate_size),
                    torch.bfloat16,
                ),
            ),
        )
        d_x = ops.bwd_epilogue(
            mxfp8_backward[0],
            workspace.d_x_routed_buffer,
        )
        _assert_metadata(
            (d_x,),
            (((num_local_tokens, hidden_size), torch.bfloat16),),
        )

        bf16_forward = ops.dispatch_mlp_swiglu_combine_fwd_bf16(
            x,
            pointers,
            workspace.combine_buffer,
            pointers,
            shared_gate,
            routed_gate_bf16,
            shared_up,
            routed_up_bf16,
            shared_down,
            routed_down_bf16,
            *schedule,
            topk,
            None,
            2,
            macrobatch_size,
            256,
        )
        _assert_metadata(
            bf16_forward,
            (
                ((macrobatch_size, hidden_size), torch.bfloat16),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((macrobatch_size, intermediate_size), torch.bfloat16),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((macrobatch_size, intermediate_size), torch.bfloat16),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((macrobatch_size, intermediate_size), torch.bfloat16),
                ((num_local_tokens, hidden_size), torch.bfloat16),
                ((macrobatch_size, hidden_size), torch.bfloat16),
            ),
        )
        bf16_recomputed = ops.recompute_forward_context_bf16(
            x,
            pointers,
            shared_gate,
            routed_gate_bf16,
            shared_up,
            routed_up_bf16,
            *schedule,
            topk,
            None,
            2,
            macrobatch_size,
            256,
        )
        _assert_metadata(
            bf16_recomputed,
            (
                ((macrobatch_size, hidden_size), torch.bfloat16),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((macrobatch_size, intermediate_size), torch.bfloat16),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((macrobatch_size, intermediate_size), torch.bfloat16),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((macrobatch_size, intermediate_size), torch.bfloat16),
            ),
        )
        bf16_backward = ops.dispatch_mlp_swiglu_combine_bwd_bf16(
            workspace.d_y_buffer,
            pointers,
            workspace.d_x_routed_buffer,
            pointers,
            workspace.router_weight_buffer,
            pointers,
            workspace.d_router_weight_buffer,
            pointers,
            shared_gate,
            routed_gate_bf16,
            shared_up,
            routed_up_bf16,
            shared_down,
            routed_down_bf16,
            *bf16_forward[:7],
            x,
            pointers,
            *schedule,
            topk,
            None,
            2,
            macrobatch_size,
            256,
        )
        _assert_metadata(
            bf16_backward,
            (
                ((num_local_tokens, hidden_size), torch.bfloat16),
                ((macrobatch_size, hidden_size), torch.bfloat16),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((macrobatch_size, intermediate_size), torch.bfloat16),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((macrobatch_size, intermediate_size), torch.bfloat16),
                ((num_local_tokens, intermediate_size), torch.bfloat16),
                ((macrobatch_size, intermediate_size), torch.bfloat16),
                ((macrobatch_size, hidden_size), torch.bfloat16),
                ((intermediate_size, hidden_size), torch.bfloat16),
                (
                    (num_local_experts, intermediate_size, hidden_size),
                    torch.bfloat16,
                ),
                ((intermediate_size, hidden_size), torch.bfloat16),
                (
                    (num_local_experts, intermediate_size, hidden_size),
                    torch.bfloat16,
                ),
                ((hidden_size, intermediate_size), torch.bfloat16),
                (
                    (num_local_experts, hidden_size, intermediate_size),
                    torch.bfloat16,
                ),
            ),
        )


def test_custom_op_mutation_schemas() -> None:
    expected = {
        "all_gather_top_experts": {"all_gather_top_experts_buffer"},
        "barrier_all": {"barrier_buffer", "target"},
        "schedule": set(),
        "mxfp8_quantize": set(),
        "dispatch_mlp_swiglu_combine_fwd_mxfp8": {"combine_buffer"},
        "dispatch_mlp_swiglu_combine_fwd_bf16": {"combine_buffer"},
        "dispatch_mlp_situglu_combine_fwd_mxfp8": {"combine_buffer"},
        "dispatch_mlp_situglu_combine_fwd_bf16": {"combine_buffer"},
        "recompute_forward_context_mxfp8": set(),
        "recompute_forward_context_bf16": set(),
        "dispatch_mlp_swiglu_combine_bwd_mxfp8": {
            "d_x_routed_buffer",
            "d_router_weight_buffer",
            "x_fp8_t_routed",
            "x_sc_t_routed",
            "gate_fp8_routed",
            "gate_sc_routed",
            "up_fp8_routed",
            "up_sc_routed",
            "hidden_fp8_t_routed",
            "hidden_sc_t_routed",
        },
        "dispatch_mlp_swiglu_combine_bwd_bf16": {
            "d_x_routed_buffer",
            "d_router_weight_buffer",
            "x_routed",
            "gate_routed",
            "up_routed",
            "hidden_routed",
        },
        "dispatch_mlp_situglu_combine_bwd_mxfp8": {
            "d_x_routed_buffer",
            "d_router_weight_buffer",
            "x_fp8_t_routed",
            "x_sc_t_routed",
            "gate_fp8_routed",
            "gate_sc_routed",
            "up_fp8_routed",
            "up_sc_routed",
            "hidden_fp8_t_routed",
            "hidden_sc_t_routed",
        },
        "dispatch_mlp_situglu_combine_bwd_bf16": {
            "d_x_routed_buffer",
            "d_router_weight_buffer",
            "x_routed",
            "gate_routed",
            "up_routed",
            "hidden_routed",
        },
        "fwd_epilogue": set(),
        "bwd_epilogue": set(),
    }
    actual = {}
    for name in expected:
        operation = getattr(torch.ops.mok, name)
        actual[name] = {
            argument.name
            for argument in operation.default._schema.arguments
            if argument.alias_info is not None and argument.alias_info.is_write
        }
    assert actual == expected


def test_compile_fullgraph(
    context: tuple[int, int, torch.device],
) -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    _, _, real_device = context
    num_local_tokens = 512
    hidden_size = 1024
    intermediate_size = 256
    num_local_experts = 4
    topk = 2
    config = functional.MoKConfig(
        minibatch_size=256,
        macrobatch_size=512,
    )
    expected_metadata = (
        ((num_local_tokens, hidden_size), torch.bfloat16),
        ((num_local_tokens, hidden_size), torch.bfloat16),
        ((num_local_tokens, topk), torch.float32),
        (
            (num_local_experts, intermediate_size, hidden_size),
            torch.bfloat16,
        ),
        (
            (num_local_experts, intermediate_size, hidden_size),
            torch.bfloat16,
        ),
        (
            (num_local_experts, hidden_size, intermediate_size),
            torch.bfloat16,
        ),
        ((intermediate_size, hidden_size), torch.bfloat16),
        ((intermediate_size, hidden_size), torch.bfloat16),
        ((hidden_size, intermediate_size), torch.bfloat16),
    )

    for precision in ("bf16", "mxfp8"):
        captured_graphs: list[torch.fx.GraphModule] = []

        def capture_backend(
            graph_module: torch.fx.GraphModule,
            _example_inputs: list[torch.Tensor],
        ) -> Callable[..., tuple[torch.Tensor, ...]]:
            captured_graphs.append(graph_module)
            return graph_module.forward

        with FakeTensorMode():
            device = torch.device("cuda", real_device.index)

            def tensor(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
                return torch.empty(shape, device=device, dtype=dtype)

            workspace = _make_fake_workspace(
                device,
                num_local_tokens=num_local_tokens,
                hidden_size=hidden_size,
                topk=topk,
                schedule_capacity=4096,
            )
            x = tensor((num_local_tokens, hidden_size), torch.bfloat16)
            top_experts = tensor((num_local_tokens, topk), torch.int64)
            router_weights = tensor((num_local_tokens, topk), torch.float32)
            grad_output = tensor(
                (num_local_tokens, hidden_size),
                torch.bfloat16,
            )
            shared_gate = tensor(
                (intermediate_size, hidden_size),
                torch.bfloat16,
            )
            shared_up = tensor(
                (intermediate_size, hidden_size),
                torch.bfloat16,
            )
            shared_down = tensor(
                (hidden_size, intermediate_size),
                torch.bfloat16,
            )
            routed_gate = tensor(
                (num_local_experts, intermediate_size, hidden_size),
                torch.bfloat16,
            )
            routed_up = tensor(
                (num_local_experts, intermediate_size, hidden_size),
                torch.bfloat16,
            )
            routed_down = tensor(
                (num_local_experts, hidden_size, intermediate_size),
                torch.bfloat16,
            )

            def complete_path(
                x: torch.Tensor,
                top_experts: torch.Tensor,
                router_weights: torch.Tensor,
                grad_output: torch.Tensor,
                shared_gate: torch.Tensor,
                shared_up: torch.Tensor,
                shared_down: torch.Tensor,
                routed_gate: torch.Tensor,
                routed_up: torch.Tensor,
                routed_down: torch.Tensor,
            ) -> tuple[torch.Tensor, ...]:
                schedule = functional.build_schedule(
                    workspace,
                    config,
                    top_experts,
                    num_local_experts=num_local_experts,
                )
                if precision == "bf16":
                    output, forward_context = functional.forward(
                        config,
                        workspace,
                        schedule,
                        x,
                        router_weights,
                        shared_gate,
                        shared_up,
                        shared_down,
                        routed_gate,
                        routed_up,
                        routed_down,
                    )
                    gradients = functional.backward(
                        config,
                        workspace,
                        schedule,
                        forward_context,
                        grad_output,
                        x,
                        router_weights,
                        shared_gate,
                        shared_up,
                        shared_down,
                        routed_gate,
                        routed_up,
                        routed_down,
                    )
                    return output, *gradients

                (
                    routed_gate_fp8,
                    routed_gate_sc,
                    routed_gate_t_fp8,
                    routed_gate_t_sc,
                ) = ops.mxfp8_quantize(routed_gate, True, True)
                (
                    routed_up_fp8,
                    routed_up_sc,
                    routed_up_t_fp8,
                    routed_up_t_sc,
                ) = ops.mxfp8_quantize(routed_up, True, True)
                (
                    routed_down_fp8,
                    routed_down_sc,
                    routed_down_t_fp8,
                    routed_down_t_sc,
                ) = ops.mxfp8_quantize(routed_down, True, True)
                output, forward_context = functional.forward(
                    config,
                    workspace,
                    schedule,
                    x,
                    router_weights,
                    shared_gate,
                    shared_up,
                    shared_down,
                    (routed_gate_fp8, routed_gate_sc),
                    (routed_up_fp8, routed_up_sc),
                    (routed_down_fp8, routed_down_sc),
                )
                gradients = functional.backward(
                    config,
                    workspace,
                    schedule,
                    forward_context,
                    grad_output,
                    x,
                    router_weights,
                    shared_gate,
                    shared_up,
                    shared_down,
                    (
                        routed_gate_fp8,
                        routed_gate_sc,
                        routed_gate_t_fp8,
                        routed_gate_t_sc,
                    ),
                    (
                        routed_up_fp8,
                        routed_up_sc,
                        routed_up_t_fp8,
                        routed_up_t_sc,
                    ),
                    (
                        routed_down_t_fp8,
                        routed_down_t_sc,
                    ),
                )
                return output, *gradients

            torch._dynamo.reset()
            compiled = torch.compile(
                complete_path,
                backend=capture_backend,
                fullgraph=True,
            )
            outputs = compiled(
                x,
                top_experts,
                router_weights,
                grad_output,
                shared_gate,
                shared_up,
                shared_down,
                routed_gate,
                routed_up,
                routed_down,
            )
            repeated_outputs = compiled(
                x,
                top_experts,
                router_weights,
                grad_output,
                shared_gate,
                shared_up,
                shared_down,
                routed_gate,
                routed_up,
                routed_down,
            )
            _assert_metadata(outputs, expected_metadata)
            _assert_metadata(repeated_outputs, expected_metadata)

        assert len(captured_graphs) == 1
        custom_op_targets = [
            str(node.target)
            for node in captured_graphs[0].graph.nodes
            if node.op == "call_function" and "mok" in str(node.target)
        ]
        expected_targets = [
            "mok.all_gather_top_experts.default",
            "mok.barrier_all.default",
            "mok.schedule.default",
            "mok.barrier_all.default",
            f"mok.dispatch_mlp_swiglu_combine_fwd_{precision}.default",
            "mok.barrier_all.default",
            "mok.fwd_epilogue.default",
            "mok.barrier_all.default",
            f"mok.dispatch_mlp_swiglu_combine_bwd_{precision}.default",
            "mok.barrier_all.default",
            "mok.bwd_epilogue.default",
        ]
        if precision == "mxfp8":
            expected_targets = [
                *expected_targets[:3],
                *(["mok.mxfp8_quantize.default"] * 3),
                *expected_targets[3:],
            ]
        assert custom_op_targets == expected_targets


def test_compile_fullgraph_recomputed_forward_context(
    context: tuple[int, int, torch.device],
) -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    _, _, real_device = context
    num_local_tokens = 512
    hidden_size = 1024
    intermediate_size = 256
    num_local_experts = 4
    topk = 2
    schedule_capacity = 4096
    config = functional.MoKConfig(
        minibatch_size=256,
        macrobatch_size=512,
    )
    expected_metadata = (
        ((num_local_tokens, hidden_size), torch.bfloat16),
        ((num_local_tokens, topk), torch.float32),
        (
            (num_local_experts, intermediate_size, hidden_size),
            torch.bfloat16,
        ),
        (
            (num_local_experts, intermediate_size, hidden_size),
            torch.bfloat16,
        ),
        (
            (num_local_experts, hidden_size, intermediate_size),
            torch.bfloat16,
        ),
        ((intermediate_size, hidden_size), torch.bfloat16),
        ((intermediate_size, hidden_size), torch.bfloat16),
        ((hidden_size, intermediate_size), torch.bfloat16),
    )

    for precision in ("bf16", "mxfp8"):
        captured_graphs: list[torch.fx.GraphModule] = []

        def capture_backend(
            graph_module: torch.fx.GraphModule,
            _example_inputs: list[torch.Tensor],
        ) -> Callable[..., tuple[torch.Tensor, ...]]:
            captured_graphs.append(graph_module)
            return graph_module.forward

        with FakeTensorMode():
            device = torch.device("cuda", real_device.index)

            def tensor(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
                return torch.empty(shape, device=device, dtype=dtype)

            workspace = _make_fake_workspace(
                device,
                num_local_tokens=num_local_tokens,
                hidden_size=hidden_size,
                topk=topk,
                schedule_capacity=schedule_capacity,
            )
            schedule = functional.MoKSchedule(
                peer_rank=tensor((schedule_capacity,), torch.int32),
                peer_token_idx=tensor((schedule_capacity,), torch.int32),
                num_tokens=tensor((1,), torch.int32),
                tokens_per_expert=tensor(
                    (num_local_experts,),
                    torch.int32,
                ),
            )
            x = tensor((num_local_tokens, hidden_size), torch.bfloat16)
            router_weights = tensor(
                (num_local_tokens, topk),
                torch.float32,
            )
            grad_output = tensor(
                (num_local_tokens, hidden_size),
                torch.bfloat16,
            )
            shared_gate = tensor(
                (intermediate_size, hidden_size),
                torch.bfloat16,
            )
            shared_up = tensor(
                (intermediate_size, hidden_size),
                torch.bfloat16,
            )
            shared_down = tensor(
                (hidden_size, intermediate_size),
                torch.bfloat16,
            )
            routed_gate_bf16 = tensor(
                (num_local_experts, intermediate_size, hidden_size),
                torch.bfloat16,
            )
            routed_up_bf16 = tensor(
                (num_local_experts, intermediate_size, hidden_size),
                torch.bfloat16,
            )
            routed_down_bf16 = tensor(
                (num_local_experts, hidden_size, intermediate_size),
                torch.bfloat16,
            )

            if precision == "bf16":
                routed_gate_forward = routed_gate_bf16
                routed_up_forward = routed_up_bf16
                routed_gate_backward = routed_gate_bf16
                routed_up_backward = routed_up_bf16
                routed_down_backward = routed_down_bf16
            else:
                routed_gate_quantized = ops.mxfp8_quantize(
                    routed_gate_bf16,
                    True,
                    True,
                )
                routed_up_quantized = ops.mxfp8_quantize(
                    routed_up_bf16,
                    True,
                    True,
                )
                routed_down_quantized = ops.mxfp8_quantize(
                    routed_down_bf16,
                    True,
                    True,
                )
                routed_gate_forward = routed_gate_quantized[:2]
                routed_up_forward = routed_up_quantized[:2]
                routed_gate_backward = routed_gate_quantized
                routed_up_backward = routed_up_quantized
                routed_down_backward = routed_down_quantized[2:]

            def recomputed_backward_path(
                x: torch.Tensor,
                router_weights: torch.Tensor,
                grad_output: torch.Tensor,
                shared_gate: torch.Tensor,
                shared_up: torch.Tensor,
                shared_down: torch.Tensor,
            ) -> tuple[torch.Tensor, ...]:
                forward_context = functional.recompute_forward_context(
                    config,
                    workspace,
                    schedule,
                    x,
                    shared_gate,
                    shared_up,
                    routed_gate_forward,
                    routed_up_forward,
                )
                return functional.backward(
                    config,
                    workspace,
                    schedule,
                    forward_context,
                    grad_output,
                    x,
                    router_weights,
                    shared_gate,
                    shared_up,
                    shared_down,
                    routed_gate_backward,
                    routed_up_backward,
                    routed_down_backward,
                )

            torch._dynamo.reset()
            compiled = torch.compile(
                recomputed_backward_path,
                backend=capture_backend,
                fullgraph=True,
            )
            outputs = compiled(
                x,
                router_weights,
                grad_output,
                shared_gate,
                shared_up,
                shared_down,
            )
            repeated_outputs = compiled(
                x,
                router_weights,
                grad_output,
                shared_gate,
                shared_up,
                shared_down,
            )
            _assert_metadata(outputs, expected_metadata)
            _assert_metadata(repeated_outputs, expected_metadata)

        assert len(captured_graphs) == 1
        custom_op_targets = [
            str(node.target)
            for node in captured_graphs[0].graph.nodes
            if node.op == "call_function" and "mok" in str(node.target)
        ]
        assert custom_op_targets == [
            "mok.barrier_all.default",
            f"mok.recompute_forward_context_{precision}.default",
            "mok.barrier_all.default",
            "mok.barrier_all.default",
            f"mok.dispatch_mlp_swiglu_combine_bwd_{precision}.default",
            "mok.barrier_all.default",
            "mok.bwd_epilogue.default",
        ]


def test_compiled_barrier_updates_state(
    context: tuple[int, int, torch.device],
) -> None:
    _, _, device = context
    functional.clear_workspace_cache()
    workspace = functional.get_workspace(
        functional.MoKConfig(
            minibatch_size=256,
            macrobatch_size=512,
        ),
        dist.group.WORLD,
        device=device,
        num_local_tokens=512,
        hidden_size=256,
        topk=1,
    )

    def run_barrier() -> None:
        ops.barrier_all(
            workspace.barrier_buffer,
            workspace.barrier_buffer_ptrs,
            workspace.barrier_buffer_multicast_ptr,
            workspace.barrier_target,
        )

    try:
        assert int(workspace.barrier_buffer.item()) == 0
        assert int(workspace.barrier_target.item()) == 0
        torch._dynamo.reset()
        compiled = torch.compile(
            run_barrier,
            backend="inductor",
            fullgraph=True,
        )
        compiled()
        torch.cuda.synchronize(device)
        compiled()
        torch.cuda.synchronize(device)
        dist.barrier()
        expected = 2 * workspace.ep_size
        assert int(workspace.barrier_buffer.item()) == expected
        assert int(workspace.barrier_target.item()) == expected
    finally:
        functional.clear_workspace_cache()
        torch._dynamo.reset()


def test_supported_shape_alignment(
    context: tuple[int, int, torch.device],
) -> None:
    _, _, device = context
    pointers = [1, 2, 3, 4]

    def run_shape(
        hidden_size: int,
        intermediate_size: int,
        *,
        num_comm_sms: int,
    ) -> None:
        num_local_tokens = 512
        topk = 2

        def tensor(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
            return torch.empty(shape, device=device, dtype=dtype)

        ops.dispatch_mlp_swiglu_combine_fwd_bf16(
            tensor((num_local_tokens, hidden_size), torch.bfloat16),
            pointers,
            tensor(
                (num_local_tokens * topk, hidden_size),
                torch.bfloat16,
            ),
            pointers,
            tensor((intermediate_size, hidden_size), torch.bfloat16),
            tensor((1, intermediate_size, hidden_size), torch.bfloat16),
            tensor((intermediate_size, hidden_size), torch.bfloat16),
            tensor((1, intermediate_size, hidden_size), torch.bfloat16),
            tensor((hidden_size, intermediate_size), torch.bfloat16),
            tensor((1, hidden_size, intermediate_size), torch.bfloat16),
            tensor((4096,), torch.int32),
            tensor((4096,), torch.int32),
            tensor((1,), torch.int32),
            tensor((1,), torch.int32),
            topk,
            None,
            num_comm_sms,
            512,
            256,
        )

    for hidden_size, intermediate_size in (
        (7168, 2048),
        (2048, 768),
        (2560, 2560),
        (1024, 1024),
        (1280, 1280),
    ):
        with pytest.raises(ValueError, match="num_comm_sms"):
            run_shape(
                hidden_size,
                intermediate_size,
                num_comm_sms=1,
            )
    for hidden_size, intermediate_size in (
        (640, 640),
        (1664, 1664),
        (1920, 1920),
        (128, 256),
        (0, 256),
        (256, 0),
    ):
        with pytest.raises(
            ValueError,
            match="hidden_size|intermediate_size",
        ):
            run_shape(
                hidden_size,
                intermediate_size,
                num_comm_sms=2,
            )


def test_e2e_requested_shapes_and_local_expert_counts(
    context: tuple[int, int, torch.device],
) -> None:
    functional.clear_workspace_cache()
    try:
        for hidden_size, intermediate_size in (
            (7168, 2048),
            (2048, 768),
            (2560, 2560),
        ):
            for num_local_experts in (4, 6, 8, 12, 16, 24, 32):
                _run_e2e_case(
                    context,
                    name=(
                        f"shape-h{hidden_size}-i{intermediate_size}"
                        f"/local-experts-{num_local_experts}"
                    ),
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_local_experts=num_local_experts,
                    topk=8,
                    config=functional.MoKConfig(
                        minibatch_size=256,
                        macrobatch_size=512,
                    ),
                    precisions=("mxfp8",),
                )
    finally:
        functional.clear_workspace_cache()


def test_e2e_different_topk(
    context: tuple[int, int, torch.device],
) -> None:
    functional.clear_workspace_cache()
    try:
        for topk in (1, 2, 4, 6, 8):
            _run_e2e_case(
                context,
                name=f"topk-{topk}",
                hidden_size=7168,
                intermediate_size=2048,
                num_local_experts=4,
                topk=topk,
                config=functional.MoKConfig(
                    minibatch_size=256,
                    macrobatch_size=512,
                ),
                precisions=("bf16", "mxfp8"),
            )
    finally:
        functional.clear_workspace_cache()


def test_e2e_finiteness_different_seeds(
    context: tuple[int, int, torch.device],
) -> None:
    functional.clear_workspace_cache()
    try:
        for seed in (42, 123, 456, 789):
            _run_e2e_case(
                context,
                name=f"finiteness/seed-{seed}",
                hidden_size=7168,
                intermediate_size=2048,
                num_local_experts=4,
                topk=8,
                config=functional.MoKConfig(
                    minibatch_size=256,
                    macrobatch_size=512,
                ),
                precisions=("mxfp8",),
                seed=seed,
                grad_seed=10_000 + seed,
            )
    finally:
        functional.clear_workspace_cache()


def test_epilogues_support_more_than_max_grid_y_tokens(
    context: tuple[int, int, torch.device],
) -> None:
    _, _, device = context
    num_local_tokens = 2 * (65_535 + 1)
    hidden_dim = 256
    shared = torch.full(
        (num_local_tokens, hidden_dim),
        1.0,
        device=device,
        dtype=torch.bfloat16,
    )
    routed = torch.full_like(shared, 2.0)
    topk_weights = torch.full(
        (num_local_tokens, 1),
        0.25,
        device=device,
        dtype=torch.float32,
    )

    output = ops.fwd_epilogue(shared, routed, topk_weights)
    assert torch.all(output == 1.5)

    d_x = ops.bwd_epilogue(shared, routed)
    assert torch.all(d_x == 3.0)
