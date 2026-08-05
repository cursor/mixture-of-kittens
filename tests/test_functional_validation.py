from types import SimpleNamespace

import pytest
import torch

from mok import functional


class FakeProcessGroup:
    group_name = "test"


@pytest.fixture
def validation_setup(monkeypatch):
    monkeypatch.setattr(functional.dist, "ProcessGroup", FakeProcessGroup)
    monkeypatch.setattr(functional.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(functional.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(functional.dist, "get_world_size", lambda group: 4)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(shared_memory_per_block_optin=1 << 20),
    )


@pytest.mark.parametrize("capability", [(10, 0), (10, 3)])
def test_validate_workspace_args_accepts_supported_architectures(
    monkeypatch, validation_setup, capability
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)

    functional.validate_workspace_args(
        functional.MoKConfig(),
        FakeProcessGroup(),
        device=torch.device("cuda", 0),
        num_local_tokens=512,
        hidden_size=256,
        topk=1,
    )


def test_validate_workspace_args_rejects_unsupported_architecture(
    monkeypatch, validation_setup
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))

    with pytest.raises(NotImplementedError, match="SM100 or SM103"):
        functional.validate_workspace_args(
            functional.MoKConfig(),
            FakeProcessGroup(),
            device=torch.device("cuda", 0),
            num_local_tokens=512,
            hidden_size=256,
            topk=1,
        )
