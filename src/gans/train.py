"""The one training loop shared by every GAN type.

    for step in 1..steps:
        repeat n_critic times:  batch = next(data);  method.d_step(batch)   # D / critic
        method.g_step(batch)                                                # G
        ema.update(G)                                                       # optional
        every log_every steps: record losses, D outputs, update ratios

The method object decides *what* a D step and a G step are (loss, conditioning, penalties,
clipping); this file decides *when* they run and what is measured. That is why the ten types
can be compared fairly: same data order, same number of generator updates, same logging.

Measured on every logged step (the spirit of the coursework notebooks' ``live_plot_gan`` and
``compute_update_ratio``):

* ``d_loss`` / ``g_loss``, ``d_real`` / ``d_fake`` (D(x) and D(G(z)): probabilities for BCE,
  raw critic scores otherwise)
* ``update_g`` / ``update_d``: ||theta_after - theta_before|| / ||theta_before|| for the step,
  the real relative size of an optimiser step (about 1e-3 is healthy). The notebooks used
  lr * ||grad||, which ignores Adam's per-parameter scaling.

CLI: ``gans-train --method wgan-gp --dataset mnist --steps 2000``
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from gans.config import TrainConfig
from gans.data import load_data
from gans.methods import METHODS, GANMethod, build_method, get_spec, resolve
from gans.models import count_params

KIND_FOR_FAMILY = {"noise": "labeled", "pix2pix": "paired", "cyclegan": "unpaired"}


def set_threads(n: int) -> None:
    if n == 1:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(n)


class EMA:
    """Exponential moving average of G's weights: theta_ema <- d * theta_ema + (1 - d) * theta.

    Averages out the oscillation of adversarial training; samples from the EMA copy are
    usually smoother and score better. BatchNorm running statistics are copied as is.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for pe, p in zip(self.model.parameters(), model.parameters()):
            pe.lerp_(p, 1 - self.decay)
        for be, b in zip(self.model.buffers(), model.buffers()):
            be.copy_(b)


@contextmanager
def update_ratio(net: nn.Module, out: dict, key: str, enabled: bool = True):
    """Measure ||delta theta|| / ||theta|| across the optimiser step done inside the block."""
    if not enabled:
        yield
        return
    before = [p.detach().clone() for p in net.parameters()]
    yield
    num = sum(((p.detach() - b) ** 2).sum() for p, b in zip(net.parameters(), before))
    den = sum((b**2).sum() for b in before)
    out[key] = (num.sqrt() / den.sqrt().clamp_min(1e-12)).item()


def _forever(loader):
    while True:
        yield from loader


def train(cfg: TrainConfig, verbose: bool = True, data=None, on_step=None) -> dict:
    """Train one GAN type; returns {"method", "history", "seconds", "checkpoint", ...}.

    ``data`` (a ``GanData``) overrides ``cfg.dataset``, e.g. for the facades set in the notebook.
    ``on_step(step, method)`` is called once before training (step 0) and after every generator
    step, e.g. to snapshot samples for a progress GIF (``scripts/make_progress.py``).
    """
    cfg = resolve(cfg)
    set_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    spec = get_spec(cfg.method)
    data = data if data is not None else load_data(cfg.dataset, cfg.data_dir, cfg.n_train, cfg.seed)
    if data.info.kind != KIND_FOR_FAMILY[spec.family]:
        raise ValueError(f"{spec.name} needs a {KIND_FOR_FAMILY[spec.family]} dataset, "
                         f"{cfg.dataset} is {data.info.kind}")
    method = build_method(cfg, data.info)
    ema = EMA(method.G, cfg.ema) if cfg.ema else None
    method.ema_G = ema.model if ema else None
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_g, n_d = count_params(method.G), count_params(method.D)
    if verbose:
        print(f"[{spec.name}] {cfg.dataset} {data.info.size}px  G {n_g / 1e6:.2f}M  D {n_d / 1e6:.2f}M  "
              f"lr_g {cfg.lr_g:g} lr_d {cfg.lr_d:g} n_critic {cfg.n_critic}  steps {cfg.steps}")

    batches = _forever(data.loader(cfg.batch_size, cfg.seed))
    history: dict[str, list] = defaultdict(list)
    fixed = _fixed_inputs(method, data)
    if on_step:
        on_step(0, method)
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        log = step % cfg.log_every == 0 or step == 1
        stats: dict = {}
        for i in range(cfg.n_critic):
            batch = next(batches)
            with update_ratio(method.D, stats, "update_d", enabled=log and i == cfg.n_critic - 1):
                stats.update(method.d_step(batch))
        with update_ratio(method.G, stats, "update_g", enabled=log):
            stats.update(method.g_step(batch))
        if ema:
            ema.update(method.G)
        if on_step:
            on_step(step, method)
        if log:
            history["step"].append(step)
            history["time"].append(time.time() - t0)
            for k, v in stats.items():
                history[k].append(v)
            if verbose and (step % (cfg.log_every * 10) == 0 or step == 1):
                print(f"  step {step:5d}  d_loss {stats['d_loss']:+.3f}  g_loss {stats['g_loss']:+.3f}  "
                      f"D(x) {stats['d_real']:+.3f}  D(G(z)) {stats['d_fake']:+.3f}  "
                      f"upd_g {stats['update_g']:.1e}  upd_d {stats['update_d']:.1e}  "
                      f"{time.time() - t0:.0f}s", flush=True)
        if cfg.sample_every and step % cfg.sample_every == 0:
            _save_progress(method, fixed, out_dir / f"samples_{step:06d}.png")
    seconds = time.time() - t0
    ckpt = out_dir / "last.pt"
    save_checkpoint(method, dict(history), seconds, ckpt)
    _save_progress(method, fixed, out_dir / "samples.png")
    from gans.viz import live_plot_gan

    live_plot_gan(history, path=out_dir / "curves.png", title=f"{spec.name} on {cfg.dataset}")
    (out_dir / "history.json").write_text(json.dumps(dict(history)))
    if verbose:
        print(f"[{spec.name}] done in {seconds:.0f}s -> {ckpt}")
    return {"method": spec.name, "dataset": cfg.dataset, "history": dict(history), "seconds": seconds,
            "checkpoint": str(ckpt), "params_g": n_g, "params_d": n_d, "model": method, "data": data}


def _fixed_inputs(method: GANMethod, data):
    torch.manual_seed(1234)
    if hasattr(method, "noise"):
        return method.noise(8 * (method.n_classes or 8)).cpu()
    return data.test[0][:16]


def _save_progress(method: GANMethod, fixed, path: Path) -> None:
    from gans.viz import save_grid, save_translation_grid

    if hasattr(method, "noise"):
        save_grid(method.sample(len(fixed), z=fixed), path, nrow=method.n_classes or 8)  # a class per column
    else:
        save_translation_grid(method, fixed, path)


def save_checkpoint(method: GANMethod, history: dict, seconds: float, path: Path) -> None:
    torch.save({"config": method.cfg.to_dict(), "info": dict(method.info.__dict__),
                "G": method.G.state_dict(), "D": method.D.state_dict(),
                "G_ema": method.ema_G.state_dict() if method.ema_G is not None else None,
                "history": history, "seconds": seconds}, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> GANMethod:
    """Rebuild the method object (G, D, EMA copy) from a ``last.pt`` file."""
    from gans.data import DataInfo

    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = TrainConfig(**{**ck["config"], "device": device})
    method = build_method(cfg, DataInfo(**ck["info"]))
    method.G.load_state_dict(ck["G"])
    method.D.load_state_dict(ck["D"])
    if ck.get("G_ema") is not None:
        method.ema_G = copy.deepcopy(method.G).eval()
        method.ema_G.load_state_dict(ck["G_ema"])
    method.history, method.seconds = ck.get("history", {}), ck.get("seconds", 0.0)
    return method


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train one GAN type with the shared loop.")
    p.add_argument("--method", default="dcgan", help=f"one of {list(METHODS)} (alias: hinge = sngan)")
    p.add_argument("--dataset", default=None, help="mnist | fashion | shapes | paired-shapes | unpaired-shapes")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--n-train", type=int, default=0)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--z-dim", type=int, default=64)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--lr-g", type=float, default=None)
    p.add_argument("--lr-d", type=float, default=None)
    p.add_argument("--n-critic", type=int, default=None)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--ttur", action="store_true")
    p.add_argument("--ema", type=float, default=0.0)
    p.add_argument("--r1", type=float, default=0.0)
    p.add_argument("--feature-matching", action="store_true")
    p.add_argument("--saturating", action="store_true")
    p.add_argument("--lambda-l1", type=float, default=100.0)
    p.add_argument("--lambda-cycle", type=float, default=10.0)
    p.add_argument("--lambda-identity", type=float, default=0.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--sample-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = TrainConfig(**{k: v for k, v in vars(args).items()})
    train(cfg)


if __name__ == "__main__":
    main()
