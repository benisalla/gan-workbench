"""Data: MNIST, FashionMNIST and three offline synthetic datasets, all as in-memory tensors.

Every dataset is returned as tensors scaled to [-1, 1] (the range of the generators' tanh):

* ``mnist`` / ``fashion``   28x28 grey digits / clothes with 10 labels (torchvision download, 11-30 MB)
* ``cifar10``               32x32 colour images, 10 labels (170 MB download; meant for the GPU notebook)
* ``shapes``                28x28 white circle / square / triangle / cross on black, 4 labels, generated
* ``paired-shapes``         32x32 (noisy textured photo of shapes, binary mask) pairs, for pix2pix
* ``unpaired-shapes``       32x32 domain A = filled shapes, domain B = outlined shapes, sampled
                            independently (unpaired) for CycleGAN; the true B of every A is kept
                            for evaluation only

Why in-memory tensors: the images are tiny, so a ``TensorDataset`` avoids PIL transforms and
worker processes and keeps one training step on a single CPU thread fast.

CLI: ``gans-data mnist`` downloads (if needed) and writes a preview grid to ``data/previews``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

SHAPE_NAMES = ["circle", "square", "triangle", "cross"]
DATASETS = ["mnist", "fashion", "cifar10", "shapes", "paired-shapes", "unpaired-shapes"]


@dataclass
class DataInfo:
    name: str
    kind: str                     # "labeled" | "paired" | "unpaired"
    channels: int
    size: int
    n_classes: int = 0
    class_names: list[str] = field(default_factory=list)


@dataclass
class GanData:
    info: DataInfo
    train: tuple[torch.Tensor, ...]   # labeled: (x, y); paired: (src, tgt); unpaired: (a, b)
    test: tuple[torch.Tensor, ...]    # unpaired test: (a, b_true_for_a)

    def loader(self, batch_size: int, seed: int = 0) -> DataLoader:
        g = torch.Generator().manual_seed(seed)
        return DataLoader(TensorDataset(*self.train), batch_size=batch_size, shuffle=True,
                          drop_last=True, generator=g)


# --------------------------------------------------------------------------- synthetic shapes
def _shape_mask(kind: int, cx: float, cy: float, r: float, size: int) -> np.ndarray:
    """Boolean mask of one shape of 'radius' r centred at (cx, cy)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) + 0.5
    dx, dy = xx - cx, yy - cy
    if kind == 0:                                    # circle
        return dx**2 + dy**2 <= r**2
    if kind == 1:                                    # square
        return (np.abs(dx) <= r * 0.85) & (np.abs(dy) <= r * 0.85)
    if kind == 2:                                    # triangle, apex up
        t = (dy + r) / (2 * r)                       # 0 at the apex, 1 at the base
        return (t >= 0) & (t <= 1) & (np.abs(dx) <= t * r)
    arm = r / 3                                      # cross
    return ((np.abs(dx) <= arm) & (np.abs(dy) <= r)) | ((np.abs(dy) <= arm) & (np.abs(dx) <= r))


def _random_geometry(rng: np.random.Generator, size: int) -> tuple[float, float, float]:
    r = rng.uniform(0.22, 0.36) * size
    cx, cy = rng.uniform(r, size - r, size=2)
    return float(cx), float(cy), float(r)


def make_shapes(n: int, size: int = 28, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """n single white shapes on black, labels 0..3 (see SHAPE_NAMES)."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 1, size, size), np.float32)
    y = rng.integers(0, 4, size=n)
    for i in range(n):
        x[i, 0] = _shape_mask(int(y[i]), *_random_geometry(rng, size), size)
    return torch.from_numpy(x * 2 - 1), torch.from_numpy(y).long()


def make_paired_shapes(n: int, size: int = 32, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """(photo, mask) pairs: 1-3 shapes with random brightness gradients on a noisy background."""
    rng = np.random.default_rng(seed)
    photos = np.zeros((n, 1, size, size), np.float32)
    masks = np.zeros((n, 1, size, size), np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    for i in range(n):
        bg = rng.uniform(0.0, 0.35) + 0.15 * (xx * rng.uniform(-1, 1) + yy * rng.uniform(-1, 1))
        img = bg + rng.normal(0, 0.06, (size, size))
        m = np.zeros((size, size), bool)
        for _ in range(rng.integers(1, 4)):
            cx, cy, r = _random_geometry(rng, size)
            s = _shape_mask(int(rng.integers(0, 4)), cx, cy, r * 0.7, size)
            shade = rng.uniform(0.55, 0.95) + 0.25 * (xx - cx / size) * rng.uniform(-1, 1)
            img = np.where(s, shade + rng.normal(0, 0.04, (size, size)), img)
            m |= s
        photos[i, 0] = np.clip(img, 0, 1)
        masks[i, 0] = m
    return torch.from_numpy(photos * 2 - 1), torch.from_numpy(masks * 2 - 1)


def make_filled_and_outline(n: int, size: int = 32, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """The same n shapes drawn filled (domain A) and as a 2-pixel outline (domain B)."""
    rng = np.random.default_rng(seed)
    a = np.zeros((n, 1, size, size), np.float32)
    b = np.zeros_like(a)
    for i in range(n):
        kind = int(rng.integers(0, 4))
        cx, cy, r = _random_geometry(rng, size)
        outer = _shape_mask(kind, cx, cy, r, size)
        inner = _shape_mask(kind, cx, cy, r - 2.5, size)
        a[i, 0], b[i, 0] = outer, outer & ~inner
    return torch.from_numpy(a * 2 - 1), torch.from_numpy(b * 2 - 1)


# --------------------------------------------------------------------------- torchvision sets
def _torchvision(name: str, root: str, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    from torchvision import datasets

    cls = {"mnist": datasets.MNIST, "fashion": datasets.FashionMNIST, "cifar10": datasets.CIFAR10}[name]
    ds = cls(root=root, train=train, download=True)
    if name == "cifar10":                                # numpy uint8 [N,32,32,3] -> [-1,1] [N,3,32,32]
        x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).float() / 127.5 - 1.0
        return x, torch.tensor(ds.targets).long()
    x = ds.data.unsqueeze(1).float() / 127.5 - 1.0     # uint8 [N,28,28] -> [-1,1] [N,1,28,28]
    return x, ds.targets.long()


CIFAR_NAMES = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
FASHION_NAMES = ["t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "boot"]


def load_data(name: str, data_dir: str = "data", n_train: int = 0, seed: int = 0) -> GanData:
    """Load one of DATASETS. ``n_train`` > 0 keeps a random subset of the training split."""
    if name in ("mnist", "fashion", "cifar10"):
        xtr, ytr = _torchvision(name, data_dir, True)
        xte, yte = _torchvision(name, data_dir, False)
        names = {"mnist": [str(i) for i in range(10)], "fashion": FASHION_NAMES, "cifar10": CIFAR_NAMES}[name]
        info = DataInfo(name, "labeled", xtr.shape[1], xtr.shape[-1], 10, names)
        train, test = (xtr, ytr), (xte, yte)
    elif name == "shapes":
        n = n_train or 12000
        info = DataInfo(name, "labeled", 1, 28, 4, SHAPE_NAMES)
        train, test = make_shapes(n, 28, seed), make_shapes(2000, 28, seed + 10_000)
        n_train = 0
    elif name == "paired-shapes":
        n = n_train or 4000
        info = DataInfo(name, "paired", 1, 32)
        train, test = make_paired_shapes(n, 32, seed), make_paired_shapes(256, 32, seed + 10_000)
        n_train = 0
    elif name == "unpaired-shapes":
        n = n_train or 4000
        a, _ = make_filled_and_outline(n, 32, seed)
        _, b = make_filled_and_outline(n, 32, seed + 1)            # independent draw: unpaired
        info = DataInfo(name, "unpaired", 1, 32)
        train, test = (a, b), make_filled_and_outline(256, 32, seed + 10_000)
        n_train = 0
    else:
        raise ValueError(f"unknown dataset {name!r}, choose from {DATASETS}")
    if n_train and n_train < len(train[0]):
        idx = torch.randperm(len(train[0]), generator=torch.Generator().manual_seed(seed))[:n_train]
        train = tuple(t[idx] for t in train)
    return GanData(info, train, test)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Download / generate a dataset and write a preview grid.")
    p.add_argument("dataset", choices=DATASETS)
    p.add_argument("--data-dir", default="data")
    args = p.parse_args(argv)
    from gans.viz import save_grid

    data = load_data(args.dataset, args.data_dir)
    out = Path(args.data_dir) / "previews" / f"{args.dataset}.png"
    first = data.train[0][:64]
    if data.info.kind != "labeled":                      # show input and target side by side
        first = torch.cat([data.train[0][:32], data.train[1][:32]])
    save_grid(first, out, nrow=8)
    print(f"{args.dataset}: kind={data.info.kind} train={len(data.train[0])} test={len(data.test[0])} "
          f"shape={tuple(data.train[0].shape[1:])} classes={data.info.n_classes} preview -> {out}")


if __name__ == "__main__":
    main()
