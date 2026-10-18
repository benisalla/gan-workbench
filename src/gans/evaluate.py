"""Evaluate a checkpoint: ``gans-eval checkpoints/dcgan/last.pt --n 5000``.

Noise GANs: classifier-feature FID, IS analogue, mode coverage (and label accuracy for cGAN /
ACGAN) on ``n`` samples against ``n`` real test images. pix2pix: L1 and IoU on the test pairs.
CycleGAN: L1 / IoU of A -> B against the true outlines, and the cycle error. Writes
``metrics.json`` next to the checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from gans.data import load_data
from gans.metrics import evaluate_samples, iou, train_classifier


def classifier_for(dataset: str, data_dir: str = "data", cache_dir: str = "checkpoints", data=None):
    data = data if data is not None else load_data(dataset, data_dir)
    return train_classifier(data, cache=Path(cache_dir) / f"classifier_{dataset}.pt")


def evaluate_method(method, data, n: int = 5000, clf=None, cache_dir: str = "checkpoints") -> dict:
    fam = method.spec.family
    if fam == "noise":
        clf = clf or classifier_for(data.info.name, data=data, cache_dir=cache_dir)
        torch.manual_seed(0)
        real = data.test[0][:n]
        labels = None
        if method.n_classes:
            labels = torch.arange(len(real)) % method.n_classes
        fake = torch.cat([method.sample(min(500, len(real) - i),
                                        y=None if labels is None else labels[i:i + 500])
                          for i in range(0, len(real), 500)])
        return evaluate_samples(clf, fake, real, labels)
    src, tgt = data.test
    if fam == "pix2pix":
        out = method.translate(src)
        return {"l1": float(F.l1_loss(out, tgt)), "iou": iou(out, tgt)}
    fake_b = method.translate(src, "ab")
    rec_a = method.translate(fake_b, "ba")
    return {"l1": float(F.l1_loss(fake_b, tgt)), "iou": iou(fake_b, tgt),
            "cycle_l1": float(F.l1_loss(rec_a, src))}


@torch.no_grad()
def real_baseline(dataset: str, data_dir: str = "data", n: int = 5000, cache_dir: str = "checkpoints") -> dict:
    """Metrics of real train images against real test images: the best a generator can hope for."""
    data = load_data(dataset, data_dir)
    clf = classifier_for(dataset, data=data, cache_dir=cache_dir)
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(data.train[0]), generator=g)[:n]
    return evaluate_samples(clf, data.train[0][idx], data.test[0][:n])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate a GAN checkpoint.")
    p.add_argument("checkpoint", nargs="?", help="path to last.pt (omit with --real)")
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--real", default=None, help="print the real-vs-real baseline for this dataset")
    p.add_argument("--data-dir", default="data")
    args = p.parse_args(argv)
    if args.real:
        print(json.dumps(real_baseline(args.real, args.data_dir, args.n), indent=2))
        return
    from gans.train import load_checkpoint, set_threads

    set_threads(1)
    method = load_checkpoint(args.checkpoint)
    data = load_data(method.cfg.dataset, args.data_dir)
    metrics = evaluate_method(method, data, args.n)
    print(json.dumps(metrics, indent=2))
    (Path(args.checkpoint).parent / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
