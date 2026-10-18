"""Adversarial losses and regularisers. Everything takes raw discriminator logits.

Notation: r = D(x) on real images, f = D(G(z)) on generated ones (logits, not probabilities).

| loss   | discriminator / critic minimises            | generator minimises           |
|--------|---------------------------------------------|-------------------------------|
| bce    | -log s(r) - log(1 - s(f))                   | -log s(f)  (non-saturating)   |
|        |                                             | log(1 - s(f)) (saturating)    |
| ls     | 1/2 (r - 1)^2 + 1/2 f^2                     | 1/2 (f - 1)^2                 |
| wgan   | f - r                                       | -f                            |
| hinge  | relu(1 - r) + relu(1 + f)                   | -f                            |

s is the sigmoid. With logits, -log s(r) = softplus(-r) and -log(1 - s(f)) = softplus(f), which
is what ``binary_cross_entropy_with_logits`` computes stably (the notebooks used Sigmoid + BCELoss,
which saturates to log(0) once D is confident).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

LOSSES = ["bce", "ls", "wgan", "hinge"]


def d_loss(kind: str, real: torch.Tensor, fake: torch.Tensor, real_target: float = 1.0) -> torch.Tensor:
    if kind == "bce":
        return (F.binary_cross_entropy_with_logits(real, torch.full_like(real, real_target))
                + F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake)))
    if kind == "ls":
        return 0.5 * ((real - real_target) ** 2).mean() + 0.5 * (fake**2).mean()
    if kind == "wgan":
        return fake.mean() - real.mean()
    if kind == "hinge":
        return F.relu(1 - real).mean() + F.relu(1 + fake).mean()
    raise ValueError(kind)


def g_loss(kind: str, fake: torch.Tensor, saturating: bool = False) -> torch.Tensor:
    if kind == "bce":
        if saturating:                       # minimise log(1 - D(G(z))): gradient vanishes when D wins
            return -F.softplus(fake).mean()
        return F.softplus(-fake).mean()      # maximise log D(G(z)): strong gradient early on
    if kind == "ls":
        return 0.5 * ((fake - 1) ** 2).mean()
    if kind in ("wgan", "hinge"):
        return -fake.mean()
    raise ValueError(kind)


def gradient_penalty(D, real: torch.Tensor, fake: torch.Tensor, y=None) -> torch.Tensor:
    """WGAN-GP: E[(||grad_xhat D(xhat)||_2 - 1)^2] on random interpolates xhat = a x + (1 - a) G(z).

    A 1-Lipschitz critic has gradient norm 1 almost everywhere along these lines, so the
    penalty enforces the Lipschitz constraint softly instead of clipping weights.
    """
    a = torch.rand(real.size(0), *([1] * (real.dim() - 1)), device=real.device)
    xhat = (a * real + (1 - a) * fake.detach()).requires_grad_(True)
    out = D(xhat, y).logit
    (grad,) = torch.autograd.grad(out.sum(), xhat, create_graph=True)
    return ((grad.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


def r1_penalty(D, real: torch.Tensor, y=None) -> torch.Tensor:
    """R1 (Mescheder et al. 2018): 1/2 E||grad_x D(x)||^2 on real data only.

    Pushes D to be flat around the data, which stops D and G from circling each other.
    """
    real = real.detach().requires_grad_(True)
    out = D(real, y).logit
    (grad,) = torch.autograd.grad(out.sum(), real, create_graph=True)
    return 0.5 * grad.flatten(1).pow(2).sum(1).mean()


def feature_matching(feat_real: torch.Tensor, feat_fake: torch.Tensor) -> torch.Tensor:
    """Salimans et al. 2016: match the mean of D's penultimate features on real and fake batches."""
    return F.mse_loss(feat_fake.mean(0), feat_real.detach().mean(0))
