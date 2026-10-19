"""Before / after figures and training-progress GIFs from real runs.

    python scripts/make_progress.py                 # train + snapshot + draw (about 15 min on a CPU)
    python scripts/make_progress.py --from-saved    # redraw from assets/progress/*.pt, no training

For each noise-to-image method, a fixed batch of 16 z (and labels for cGAN) is pushed through G
at step 0 (untrained) and at roughly log-spaced steps up to the end, using ``train(on_step=...)``.
The snapshots are saved as small tensors in assets/progress/, then drawn as

* assets/before_after_mnist.png   untrained vs trained G, same z, one column pair per method
* assets/progress_mnist.gif       the same z improving over training, methods side by side
* assets/before_after_pix2pix.png pix2pix input / output / target, before and after training
* assets/progress_pix2pix.gif     pix2pix outputs on fixed test inputs over training
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.utils import make_grid

from gans.config import TrainConfig
from gans.train import train

OUT = Path("assets")
PROG = OUT / "progress"
NOISE_METHODS = {"dcgan": 1500, "cgan": 1500, "sngan": 1500, "wgan-gp": 1000}


def snapshot_steps(total: int) -> list[int]:
    steps = {0, total} | {s for s in (10, 25, 50, 100, 150, 200, 300, 400, 500, 650, 800, 1000, 1250, 1500,
                                      1750, 2000) if s < total}
    return sorted(steps)


def run_noise(name: str, steps: int) -> dict:
    keep = set(snapshot_steps(steps))
    torch.manual_seed(1234)
    z = torch.randn(16, 64)
    y = torch.arange(16) % 10
    frames = {}

    def on_step(step, method):
        if step in keep:
            frames[step] = method.sample(16, y=y if method.n_classes else None, z=z)

    train(TrainConfig(method=name, dataset="mnist", steps=steps, sample_every=0,
                      output_dir=f"checkpoints/progress/{name}"), on_step=on_step)
    rec = {"method": name, "steps": sorted(frames), "images": torch.stack([frames[s] for s in sorted(frames)])}
    torch.save(rec, PROG / f"{name}.pt")
    return rec


def run_pix2pix(steps: int = 800) -> dict:
    keep = set(snapshot_steps(steps))
    frames, holder = {}, {}

    def on_step(step, method):
        if step == 0:
            from gans.data import load_data

            holder["src"], holder["tgt"] = (t[:8] for t in load_data("paired-shapes").test)
        if step in keep:
            frames[step] = method.translate(holder["src"])

    train(TrainConfig(method="pix2pix", steps=steps, sample_every=0, output_dir="checkpoints/progress/pix2pix"),
          on_step=on_step)
    rec = {"method": "pix2pix", "steps": sorted(frames), "images": torch.stack([frames[s] for s in sorted(frames)]),
           "src": holder["src"], "tgt": holder["tgt"]}
    torch.save(rec, PROG / "pix2pix.pt")
    return rec


# --------------------------------------------------------------------------- drawing
def to_pil(images: torch.Tensor, nrow: int, scale: int = 2) -> Image.Image:
    g = make_grid(images.clamp(-1, 1), nrow=nrow, padding=2, normalize=True, value_range=(-1, 1), pad_value=1)
    arr = (g[0].numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr).resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.NEAREST)


def titled(img: Image.Image, title: str) -> Image.Image:
    out = Image.new("L", (img.width, img.height + 18), 255)
    out.paste(img, (0, 18))
    ImageDraw.Draw(out).text((4, 3), title, fill=0)
    return out


def hstack(images: list[Image.Image], gap: int = 10) -> Image.Image:
    w = sum(i.width for i in images) + gap * (len(images) - 1)
    out = Image.new("L", (w, max(i.height for i in images)), 255)
    x = 0
    for i in images:
        out.paste(i, (x, 0))
        x += i.width + gap
    return out


def vstack(images: list[Image.Image], gap: int = 6) -> Image.Image:
    out = Image.new("L", (max(i.width for i in images), sum(i.height for i in images) + gap * (len(images) - 1)), 255)
    y = 0
    for i in images:
        out.paste(i, (0, y))
        y += i.height + gap
    return out


def draw_noise(recs: list[dict]) -> None:
    panels = []
    for r in recs:
        before = titled(to_pil(r["images"][0], 4), f"{r['method']}: step 0")
        after = titled(to_pil(r["images"][-1], 4), f"step {r['steps'][-1]}")
        panels.append(hstack([before, after], gap=4))
    hstack(panels, gap=18).save(OUT / "before_after_mnist.png")
    # GIF: frame k shows every method at its k-th snapshot (fraction of its own run)
    n = max(len(r["steps"]) for r in recs)
    frames = []
    for k in range(n):
        row = []
        for r in recs:
            i = min(round(k * (len(r["steps"]) - 1) / (n - 1)), len(r["steps"]) - 1)
            row.append(titled(to_pil(r["images"][i], 4), f"{r['method']}  step {r['steps'][i]}"))
        frames.append(hstack(row, gap=18).convert("P"))
    durations = [500] * (n - 1) + [2500]
    frames[0].save(OUT / "progress_mnist.gif", save_all=True, append_images=frames[1:], duration=durations, loop=0,
                   optimize=True)


def draw_pix2pix(r: dict) -> None:
    def block(i, label):
        rows = torch.cat([r["src"], r["images"][i], r["tgt"]])
        return titled(to_pil(rows, 8), f"{label}: input / generated / target")

    vstack([block(0, "pix2pix, step 0 (untrained)"), block(-1, f"step {r['steps'][-1]}")]).save(
        OUT / "before_after_pix2pix.png")
    frames = [block(i, f"pix2pix, step {s}").convert("P") for i, s in enumerate(r["steps"])]
    frames[0].save(OUT / "progress_pix2pix.gif", save_all=True, append_images=frames[1:],
                   duration=[500] * (len(frames) - 1) + [2500], loop=0, optimize=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--from-saved", action="store_true", help="only redraw from assets/progress/*.pt")
    args = p.parse_args()
    PROG.mkdir(parents=True, exist_ok=True)
    if args.from_saved:
        recs = [torch.load(PROG / f"{m}.pt") for m in NOISE_METHODS]
        p2p = torch.load(PROG / "pix2pix.pt")
    else:
        recs = [run_noise(m, s) for m, s in NOISE_METHODS.items()]
        p2p = run_pix2pix()
    draw_noise(recs)
    draw_pix2pix(p2p)
    for f in ("before_after_mnist.png", "progress_mnist.gif", "before_after_pix2pix.png", "progress_pix2pix.gif"):
        print(f"{OUT / f}: {(OUT / f).stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
