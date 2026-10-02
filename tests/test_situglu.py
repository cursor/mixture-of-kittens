import torch

from .utils import apply_glu, situglu


def test_situglu_is_bounded() -> None:
    values = torch.linspace(-1000.0, 1000.0, 257, dtype=torch.float64)
    hidden = situglu(values[:, None], values[None, :])

    assert torch.isfinite(hidden).all()
    assert hidden.abs().max() <= 100.0


def test_situglu_matches_swiglu_near_origin() -> None:
    gate = torch.tensor([-1e-4, 0.0, 1e-4], dtype=torch.float64)
    up = torch.tensor([1e-4, -1e-4, 1e-4], dtype=torch.float64)

    torch.testing.assert_close(
        situglu(gate, up),
        torch.nn.functional.silu(gate) * up,
        rtol=1e-8,
        atol=1e-16,
    )


def test_situglu_closed_form_derivatives() -> None:
    gate = torch.tensor([-7.0, -0.5, 3.0], dtype=torch.float64, requires_grad=True)
    up = torch.tensor([-30.0, 0.25, 40.0], dtype=torch.float64, requires_grad=True)
    situglu(gate, up).sum().backward()

    gate_tanh = torch.tanh(gate.detach() / 4.0)
    up_tanh = torch.tanh(up.detach() / 25.0)
    sigmoid = torch.sigmoid(gate.detach())
    gate_factor = 4.0 * gate_tanh * sigmoid
    up_factor = 25.0 * up_tanh
    d_gate_factor = (1.0 - gate_tanh.square()) * sigmoid + gate_factor * (1.0 - sigmoid)
    d_up_factor = 1.0 - up_tanh.square()

    torch.testing.assert_close(gate.grad, d_gate_factor * up_factor)
    torch.testing.assert_close(up.grad, gate_factor * d_up_factor)


def test_situglu_gradcheck() -> None:
    gate = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    up = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(situglu, (gate, up))


def test_apply_glu_rejects_unknown_activation() -> None:
    gate = torch.zeros(1)
    up = torch.zeros(1)

    try:
        apply_glu(gate, up, "unknown")
    except ValueError as error:
        assert str(error) == "activation must be 'swiglu' or 'situglu'"
    else:
        raise AssertionError("unknown activation was accepted")
