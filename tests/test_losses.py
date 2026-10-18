import math

import pytest
import torch

from gans.losses import d_loss, feature_matching, g_loss, gradient_penalty, r1_penalty
from gans.models import DOut


def test_bce_loss_values():
    zero = torch.zeros(4)
    assert d_loss("bce", zero, zero).item() == pytest.approx(2 * math.log(2))
    assert g_loss("bce", zero).item() == pytest.approx(math.log(2))


def test_saturating_loss_has_vanishing_gradient_when_d_wins():
    """When D confidently rejects fakes (logit -8), log(1 - D) gives almost no gradient."""
    f1 = torch.full((4,), -8.0, requires_grad=True)
    g_loss("bce", f1, saturating=True).backward()
    f2 = torch.full((4,), -8.0, requires_grad=True)
    g_loss("bce", f2).backward()
    assert f1.grad.abs().mean() < 1e-3 < f2.grad.abs().mean()


def test_ls_wgan_hinge_values():
    r, f = torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0])
    assert d_loss("ls", r, f).item() == 0
    assert g_loss("ls", torch.ones(3)).item() == 0
    assert d_loss("wgan", torch.tensor([3.0]), torch.tensor([1.0])).item() == -2
    assert d_loss("hinge", torch.tensor([2.0]), torch.tensor([-2.0])).item() == 0     # margins satisfied
    assert g_loss("hinge", torch.tensor([0.5])).item() == -0.5
    with pytest.raises(ValueError):
        d_loss("kl", r, f)


class Linear:
    """A critic D(x) = sum(w * x): its gradient norm is ||w|| everywhere."""

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, x, y=None):
        return DOut(self.scale * x.flatten(1).sum(1), x.flatten(1))


def test_gradient_penalty_is_zero_for_unit_lipschitz_critic():
    x = torch.randn(8, 1, 2, 2)
    w = 1 / math.sqrt(4)                        # ||grad|| = 4 * w^2 ... = 1
    assert gradient_penalty(Linear(w), x, torch.randn_like(x)).item() == pytest.approx(0, abs=1e-6)
    assert gradient_penalty(Linear(2 * w), x, torch.randn_like(x)).item() == pytest.approx(1, abs=1e-5)


def test_r1_and_feature_matching():
    x = torch.randn(8, 1, 2, 2)
    assert r1_penalty(Linear(1.0), x).item() == pytest.approx(0.5 * 4)
    f = torch.randn(8, 5)
    assert feature_matching(f, f).item() == 0
    assert feature_matching(f, f + 1).item() == pytest.approx(1)
