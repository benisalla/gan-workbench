"""Evaluation: a small classifier trained on the real data provides features and labels.

**Classifier-feature FID, not Inception FID.** The Frechet distance

    FID = ||mu_r - mu_g||^2 + Tr(S_r + S_g - 2 (S_r S_g)^(1/2))

is computed between Gaussians fitted to the 128-d penultimate features of a small CNN trained
on the same dataset (MNIST / FashionMNIST / shapes), not of Inception-v3 on ImageNet. Numbers
are therefore only comparable inside this repo, for the same dataset and classifier. Its
floor is the FID between two real splits (reported as ``real``).

The same classifier gives:

* **IS analogue**  exp(E_x KL(p(y|x) || p(y))): high when every sample is confidently one class
  (quality) and the classes are balanced (diversity). Its maximum is the number of classes.
* **mode coverage** number of classes that receive at least 1/(4K) of the samples (out of K),
  and the KL of the predicted class histogram from uniform. A collapsed GAN covers 1-3 digits.
* **label accuracy** (cGAN / ACGAN) how often the classifier agrees with the requested label.

For pix2pix: L1 and mask IoU against the true target; for CycleGAN: L1 / IoU of G_AB(a) against
the true outline of a (never seen paired in training) and the cycle error.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class FeatureClassifier(nn.Module):
    """conv-pool-conv-pool -> 128-d features -> K logits. About 99% on MNIST after one epoch."""

    def __init__(self, channels: int = 1, n_classes: int = 10, feat_dim: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveMaxPool2d(7),
            nn.Flatten(), nn.Linear(64 * 49, feat_dim), nn.ReLU(),
        )
        self.head = nn.Linear(feat_dim, n_classes)

    def forward(self, x):
        return self.head(self.body(x))

    @torch.no_grad()
    def features_and_probs(self, x: torch.Tensor, batch: int = 500):
        self.eval()
        feats, probs = [], []
        for i in range(0, len(x), batch):
            f = self.body(x[i:i + batch])
            feats.append(f)
            probs.append(F.softmax(self.head(f), 1))
        return torch.cat(feats), torch.cat(probs)


def train_classifier(data, epochs: int = 2, cache: str | Path | None = None, seed: int = 0,
                     verbose: bool = True) -> FeatureClassifier:
    """Train (or load from ``cache``) the evaluation classifier on a labeled dataset."""
    info = data.info
    clf = FeatureClassifier(info.channels, info.n_classes)
    if cache and Path(cache).exists():
        clf.load_state_dict(torch.load(cache, map_location="cpu", weights_only=True))
        return clf.eval()
    torch.manual_seed(seed)
    opt = torch.optim.Adam(clf.parameters(), 1e-3)
    loader = data.loader(128, seed)
    for _ in range(epochs):
        clf.train()
        for x, y in loader:
            loss = F.cross_entropy(clf(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    _, p = clf.features_and_probs(data.test[0])
    acc = (p.argmax(1) == data.test[1]).float().mean().item()
    if verbose:
        print(f"evaluation classifier on {info.name}: test accuracy {acc:.4f}")
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        torch.save(clf.state_dict(), cache)
    return clf.eval()


def _sqrtm_trace(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Tr((A B)^(1/2)) for SPD A, B, via the symmetric matrix A^(1/2) B A^(1/2) (same eigenvalues)."""
    ea, va = torch.linalg.eigh(a)
    sa = va @ torch.diag(ea.clamp_min(0).sqrt()) @ va.T
    ev = torch.linalg.eigvalsh(sa @ b @ sa)
    return ev.clamp_min(0).sqrt().sum()


def frechet_distance(f1: torch.Tensor, f2: torch.Tensor) -> float:
    f1, f2 = f1.double(), f2.double()
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = torch.cov(f1.T), torch.cov(f2.T)
    return float(((mu1 - mu2) ** 2).sum() + torch.trace(s1) + torch.trace(s2) - 2 * _sqrtm_trace(s1, s2))


def inception_score(probs: torch.Tensor, eps: float = 1e-12) -> float:
    py = probs.mean(0, keepdim=True)
    kl = (probs * ((probs + eps).log() - (py + eps).log())).sum(1)
    return float(kl.mean().exp())


def mode_coverage(probs: torch.Tensor) -> tuple[int, float]:
    k = probs.shape[1]
    hist = torch.bincount(probs.argmax(1), minlength=k).float() / len(probs)
    covered = int((hist >= 1.0 / (4 * k)).sum())
    kl_uniform = float((hist * ((hist + 1e-12) * k).log()).sum())
    return covered, kl_uniform


@torch.no_grad()
def evaluate_samples(clf: FeatureClassifier, fake: torch.Tensor, real: torch.Tensor,
                     labels: torch.Tensor | None = None) -> dict:
    """All metrics for generated images ``fake`` against real images ``real``."""
    fr, _ = clf.features_and_probs(real)
    ff, pf = clf.features_and_probs(fake)
    covered, kl = mode_coverage(pf)
    out = {"fid": frechet_distance(fr, ff), "is": inception_score(pf), "modes": covered,
           "n_classes": pf.shape[1], "kl_uniform": kl}
    if labels is not None:
        out["label_acc"] = float((pf.argmax(1) == labels).float().mean())
    return out


def iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mask IoU, both in [-1, 1] and thresholded at 0."""
    p, t = pred > 0, target > 0
    return float((p & t).sum() / (p | t).sum().clamp_min(1))
