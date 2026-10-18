"""Generators and discriminators.

Three families, kept small so they train on a CPU:

* ``MLPGenerator`` / ``MLPDiscriminator``: the 2014 GAN, fully connected on flattened pixels.
* ``DCGenerator`` / ``DCDiscriminator``: DCGAN for 28/32 px. G maps z to a 7x7 (or 8x8) map, two
  stride-2 transposed convolutions double it twice; D mirrors it with stride-2 convolutions.
  Both optionally take a class label (one-hot concatenated to z, and to the image as constant
  channel maps for D), and D can have an auxiliary classifier head (ACGAN) and spectral norm.
* ``UNetGenerator`` / ``PatchDiscriminator``: pix2pix, same blocks as the original coursework
  notebook (inconv / down / up / outconv and conv_block / out_block) but with a configurable
  depth so a 32 px input works.

Every discriminator returns ``DOut(logit, features, class_logits)``: raw scores (no sigmoid,
the losses work on logits), the penultimate features (used by feature matching) and the
auxiliary class logits (ACGAN only, else None).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm


class DOut(NamedTuple):
    logit: torch.Tensor
    features: torch.Tensor
    class_logits: torch.Tensor | None = None


def dcgan_init(m: nn.Module) -> None:
    """N(0, 0.02) weights, BatchNorm gamma ~ N(1, 0.02) (Radford et al.; same as the notebooks)."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def _sn(layer: nn.Module, use: bool) -> nn.Module:
    return spectral_norm(layer) if use else layer


def one_hot(y: torch.Tensor | None, n: int) -> torch.Tensor | None:
    return None if (y is None or n == 0) else F.one_hot(y, n).float()


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- MLP (GAN 2014)
class MLPGenerator(nn.Module):
    def __init__(self, z_dim: int, channels: int, size: int, width: int = 256, n_classes: int = 0):
        super().__init__()
        self.shape, self.n_classes = (channels, size, size), n_classes
        self.net = nn.Sequential(
            nn.Linear(z_dim + n_classes, width), nn.LeakyReLU(0.2),
            nn.Linear(width, width * 2), nn.BatchNorm1d(width * 2), nn.LeakyReLU(0.2),
            nn.Linear(width * 2, channels * size * size), nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        h = z if self.n_classes == 0 else torch.cat([z, one_hot(y, self.n_classes)], 1)
        return self.net(h).view(-1, *self.shape)


class MLPDiscriminator(nn.Module):
    def __init__(self, channels: int, size: int, width: int = 256, n_classes: int = 0,
                 aux_classes: int = 0, spectral: bool = False):
        super().__init__()
        self.n_classes = n_classes
        self.body = nn.Sequential(
            _sn(nn.Linear(channels * size * size + n_classes, width * 2), spectral), nn.LeakyReLU(0.2),
            _sn(nn.Linear(width * 2, width), spectral), nn.LeakyReLU(0.2),
        )
        self.head = _sn(nn.Linear(width, 1), spectral)
        self.aux = nn.Linear(width, aux_classes) if aux_classes else None

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> DOut:
        h = x.flatten(1)
        if self.n_classes:
            h = torch.cat([h, one_hot(y, self.n_classes)], 1)
        f = self.body(h)
        return DOut(self.head(f).squeeze(1), f, self.aux(f) if self.aux is not None else None)


# --------------------------------------------------------------------------- DCGAN
class DCGenerator(nn.Module):
    """z (+ one-hot y) -> Linear -> (2w, s/4, s/4) -> ConvT -> (w, s/2, s/2) -> ConvT -> (c, s, s)."""

    def __init__(self, z_dim: int, channels: int, size: int, width: int = 64, n_classes: int = 0):
        super().__init__()
        assert size % 4 == 0, "DCGenerator needs a size divisible by 4 (28 or 32)"
        self.s0, self.w, self.n_classes = size // 4, width, n_classes
        self.fc = nn.Linear(z_dim + n_classes, 2 * width * self.s0 * self.s0)
        self.net = nn.Sequential(
            nn.BatchNorm2d(2 * width), nn.ReLU(True),
            nn.ConvTranspose2d(2 * width, width, 4, 2, 1), nn.BatchNorm2d(width), nn.ReLU(True),
            nn.ConvTranspose2d(width, channels, 4, 2, 1), nn.Tanh(),
        )
        self.apply(dcgan_init)

    def forward(self, z: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        h = z if self.n_classes == 0 else torch.cat([z, one_hot(y, self.n_classes)], 1)
        return self.net(self.fc(h).view(-1, 2 * self.w, self.s0, self.s0))


class DCDiscriminator(nn.Module):
    """(c [+ n_classes label maps], s, s) -> conv s/2 -> conv s/4 -> flatten -> logit (+ class logits).

    ``norm``: "batch" (DCGAN), "none" (WGAN critics: a gradient penalty is per-sample, BatchNorm
    would mix samples) or "spectral" (SNGAN: every weight divided by its largest singular value).
    """

    def __init__(self, channels: int, size: int, width: int = 64, n_classes: int = 0,
                 aux_classes: int = 0, norm: str = "batch"):
        super().__init__()
        sn = norm == "spectral"
        self.n_classes = n_classes
        self.body = nn.Sequential(
            _sn(nn.Conv2d(channels + n_classes, width, 4, 2, 1), sn), nn.LeakyReLU(0.2, True),
            _sn(nn.Conv2d(width, 2 * width, 4, 2, 1), sn),
            nn.BatchNorm2d(2 * width) if norm == "batch" else nn.Identity(), nn.LeakyReLU(0.2, True),
            nn.Flatten(),
        )
        feat = 2 * width * (size // 4) ** 2
        self.head = _sn(nn.Linear(feat, 1), sn)
        self.aux = nn.Linear(feat, aux_classes) if aux_classes else None
        if not sn:
            self.apply(dcgan_init)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> DOut:
        if self.n_classes:
            maps = one_hot(y, self.n_classes)[:, :, None, None].expand(-1, -1, *x.shape[2:])
            x = torch.cat([x, maps], 1)
        f = self.body(x)
        return DOut(self.head(f).squeeze(1), f, self.aux(f) if self.aux is not None else None)


# --------------------------------------------------------------------------- pix2pix
class Down(nn.Module):
    """Conv 4x4 stride 2 (+ BatchNorm) + LeakyReLU: halves H and W (notebook: inconv / down)."""

    def __init__(self, cin: int, cout: int, norm: bool = True):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(cin, cout, 4, 2, 1),
                                  nn.BatchNorm2d(cout) if norm else nn.Identity(), nn.LeakyReLU(0.2, True))

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """ConvT 4x4 stride 2 + BatchNorm (+ Dropout) + ReLU, then concatenate the skip (notebook: up)."""

    def __init__(self, cin: int, cout: int, dropout: bool = False):
        super().__init__()
        self.conv = nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, 2, 1), nn.BatchNorm2d(cout),
                                  nn.Dropout(0.5) if dropout else nn.Identity(), nn.ReLU(True))

    def forward(self, x, skip):
        return torch.cat([self.conv(x), skip], 1)


class UNetGenerator(nn.Module):
    """Encoder of ``depth`` stride-2 Down blocks, decoder of Up blocks with skip connections.

    32 px, depth 4: 32 -> 16 -> 8 -> 4 -> 2 (bottleneck) -> 4 -> 8 -> 16 -> 32. The skips carry the
    high-resolution encoder features straight to the decoder, so the output keeps the input's
    edges; that is why pix2pix uses a U-Net and not a plain encoder-decoder.
    """

    def __init__(self, in_ch: int, out_ch: int, width: int = 32, depth: int = 4, dropout: bool = True):
        super().__init__()
        chs = [width * min(2**i, 8) for i in range(depth)]
        self.downs = nn.ModuleList([Down(in_ch, chs[0], norm=False)] +
                                   [Down(chs[i - 1], chs[i]) for i in range(1, depth)])
        self.ups = nn.ModuleList()
        cin = chs[-1]
        for i in range(depth - 2, -1, -1):
            self.ups.append(Up(cin, chs[i], dropout=dropout and i >= depth - 3))
            cin = 2 * chs[i]
        self.out = nn.Sequential(nn.ConvTranspose2d(cin, out_ch, 4, 2, 1), nn.Tanh())
        self.apply(dcgan_init)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
        x = skips.pop()
        for up in self.ups:
            x = up(x, skips.pop())
        return self.out(x)


class PatchDiscriminator(nn.Module):
    """PatchGAN: a fully convolutional D that outputs one real/fake logit per overlapping patch.

    pix2pix feeds it cat(condition, image); CycleGAN feeds it the image alone (cond_ch = 0).
    At 32 px: 32 -> 16 -> 8 -> 7 -> 6, i.e. a 6x6 grid of logits, each seeing a ~22 px patch.
    """

    def __init__(self, in_ch: int, cond_ch: int = 0, width: int = 32, norm: str = "batch"):
        super().__init__()
        bn = (lambda c: nn.BatchNorm2d(c)) if norm == "batch" else (
            (lambda c: nn.InstanceNorm2d(c, affine=True)) if norm == "instance" else (lambda c: nn.Identity()))
        self.body = nn.Sequential(
            nn.Conv2d(in_ch + cond_ch, width, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(width, 2 * width, 4, 2, 1), bn(2 * width), nn.LeakyReLU(0.2, True),
            nn.Conv2d(2 * width, 4 * width, 4, 1, 1), bn(4 * width), nn.LeakyReLU(0.2, True),
        )
        self.head = nn.Conv2d(4 * width, 1, 4, 1, 1)
        self.apply(dcgan_init)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> DOut:
        if cond is not None:
            x = torch.cat([cond, x], 1)
        f = self.body(x)
        return DOut(self.head(f), f.flatten(1))
