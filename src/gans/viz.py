"""Figures: training curves (the notebooks' ``live_plot_gan``), sample grids, interpolations.

Every function writes a PNG when ``path`` is given and returns the matplotlib figure otherwise
(so a notebook can display it). Images are expected in [-1, 1].
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules and "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")                     # scripts and tests only write files
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid


def _finish(fig, path):
    if path is None:
        return fig
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return None


def _smooth(v, k: int = 5):
    v = np.asarray(v, dtype=float)
    if len(v) < k:
        return v
    return np.convolve(np.pad(v, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def grid_image(images: torch.Tensor, nrow: int = 8) -> np.ndarray:
    g = make_grid(images.clamp(-1, 1), nrow=nrow, padding=2, normalize=True, value_range=(-1, 1), pad_value=1)
    return g.permute(1, 2, 0).numpy()


def save_grid(images: torch.Tensor, path=None, nrow: int = 8, title: str | None = None):
    fig, ax = plt.subplots(figsize=(nrow * 0.45 + 0.4, (len(images) + nrow - 1) // nrow * 0.45 + 0.4))
    ax.imshow(grid_image(images, nrow), cmap="gray")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)
    return _finish(fig, path)


def live_plot_gan(history: dict, path=None, title: str = "GAN training"):
    """Three panels as in the coursework notebooks: losses, D outputs, parameter update ratios."""
    steps = history["step"]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 3.8))
    ax1.plot(steps, _smooth(history["d_loss"]), "r-", label="D loss")
    ax1b = ax1.twinx()
    ax1b.plot(steps, _smooth(history["g_loss"]), "b-", label="G loss")
    ax1.set_ylabel("D loss", color="r")
    ax1b.set_ylabel("G loss", color="b")
    ax1.set_title("losses")
    ax2.plot(steps, _smooth(history["d_real"]), "r--", label="D(x)")
    ax2.plot(steps, _smooth(history["d_fake"]), "g-", label="D(G(z))")
    ax2.set_title("discriminator outputs")
    ax2.legend(loc="best", fontsize=8)
    ax3.semilogy(steps, history["update_g"], "b-", label="G update ratio", alpha=0.8)
    ax3.semilogy(steps, history["update_d"], "r-", label="D update ratio", alpha=0.8)
    ax3.axhline(1e-3, color="gray", ls="--", label="1e-3")
    ax3.set_title("||dtheta|| / ||theta|| per step")
    ax3.legend(loc="best", fontsize=8)
    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("generator step")
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return _finish(fig, path)


def save_translation_grid(method, src: torch.Tensor, path=None, target: torch.Tensor | None = None):
    """Rows: input, (target), output. Works for pix2pix and CycleGAN (A -> B)."""
    out = method.translate(src)
    rows = [src] + ([target] if target is not None else []) + [out]
    return save_grid(torch.cat(rows), path, nrow=len(src))


@torch.no_grad()
def interpolation_grid(method, n_rows: int = 6, n_steps: int = 10, path=None, seed: int = 0):
    """Each row walks linearly from z_a to z_b; conditional models keep the row's label fixed."""
    g = torch.Generator().manual_seed(seed)
    za, zb = torch.randn(n_rows, method.cfg.z_dim, generator=g), torch.randn(n_rows, method.cfg.z_dim, generator=g)
    t = torch.linspace(0, 1, n_steps)
    z = (za[:, None] * (1 - t[None, :, None]) + zb[:, None] * t[None, :, None]).reshape(-1, method.cfg.z_dim)
    y = None
    if getattr(method, "n_classes", 0):
        y = (torch.arange(n_rows) % method.n_classes).repeat_interleave(n_steps)
    return save_grid(method.sample(len(z), y=y, z=z), path, nrow=n_steps)


def compare_curves(histories: dict[str, dict], key: str, path=None, title: str | None = None, logy=False):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, h in histories.items():
        if key in h:
            ax.plot(h["step"], _smooth(h[key], 9), label=name)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("generator step")
    ax.set_ylabel(key)
    ax.set_title(title or key)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return _finish(fig, path)
