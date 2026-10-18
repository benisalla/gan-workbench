"""Draw from a trained checkpoint: ``gans-sample checkpoints/cgan/last.pt --out grid.png``.

Noise GANs: an 8x8 grid (conditional ones: one class per column, or ``--label 3`` for one
class), and with ``--interpolate`` a grid of straight-line walks in z. pix2pix / CycleGAN:
rows input / target / output on test images.
"""

from __future__ import annotations

import argparse

import torch

from gans.data import load_data
from gans.train import load_checkpoint
from gans.viz import interpolation_grid, save_grid, save_translation_grid


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Sample images from a GAN checkpoint.")
    p.add_argument("checkpoint")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--label", type=int, default=None, help="conditional models: draw only this class")
    p.add_argument("--interpolate", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="samples.png")
    args = p.parse_args(argv)
    torch.manual_seed(args.seed)
    method = load_checkpoint(args.checkpoint)
    if method.spec.family == "noise":
        if args.interpolate:
            interpolation_grid(method, path=args.out, seed=args.seed)
        else:
            y = None
            if getattr(method, "n_classes", 0):
                y = (torch.full((args.n,), args.label) if args.label is not None
                     else torch.arange(args.n) % method.n_classes)
            save_grid(method.sample(args.n, y=y), args.out, nrow=getattr(method, "n_classes", 0) or 8)
    else:
        src, tgt = load_data(method.cfg.dataset, args.data_dir).test
        save_translation_grid(method, src[:12], args.out, target=tgt[:12])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
