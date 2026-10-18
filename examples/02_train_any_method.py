"""02 - Train ANY of the ten GAN types with one command, small CPU defaults.

    python examples/02_train_any_method.py --method wgan-gp
    python examples/02_train_any_method.py --method lsgan --steps 600
    python examples/02_train_any_method.py --method pix2pix          # paired shapes -> mask
    python examples/02_train_any_method.py --method cyclegan         # filled <-> outlined shapes

Noise-to-image methods use MNIST (20 000 images); ``--dataset shapes`` runs fully offline.
Prints the method's one-line idea and its defaults, then trains.
"""

import argparse

from gans.config import TrainConfig
from gans.methods import METHODS, get_spec, resolve
from gans.train import train

p = argparse.ArgumentParser()
p.add_argument("--method", default="sngan", choices=list(METHODS) + ["hinge"])
p.add_argument("--dataset", default=None)
p.add_argument("--steps", type=int, default=600)
args = p.parse_args()

spec = get_spec(args.method)
dataset = args.dataset or spec.dataset
cfg = resolve(TrainConfig(method=spec.name, dataset=dataset, steps=args.steps, width=32,
                          n_train=20000 if dataset == "mnist" else 0,
                          log_every=20, sample_every=0, output_dir=f"examples/output/02_{spec.name}"))
print(f"{spec.name}: {spec.summary}")
print(f"loss={spec.loss}  D norm={spec.d_norm}  n_critic={cfg.n_critic}  clip={spec.clip}  gp={spec.gp}  "
      f"optimizer={spec.optimizer} lr_g={cfg.lr_g} lr_d={cfg.lr_d} betas={cfg.betas}")
train(cfg)
print(f"samples -> examples/output/02_{spec.name}/samples.png")
