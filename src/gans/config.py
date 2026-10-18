"""Training configuration: one dataclass shared by every GAN type.

Every field has a small CPU-friendly default. Fields left as ``None`` are filled from the
method's own defaults (``methods.METHODS``), so ``TrainConfig(method="wgan-gp")`` already uses
the WGAN-GP paper settings (Adam 1e-4, betas (0, 0.9), 5 critic steps, penalty 10).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class TrainConfig:
    method: str = "dcgan"
    dataset: str | None = None          # None -> the method's default dataset
    data_dir: str = "data"
    n_train: int = 0                    # 0 = use the whole training split
    steps: int = 2000                   # generator steps
    batch_size: int = 64
    z_dim: int = 64
    width: int = 32                     # base channel / hidden width of the networks
    # optimisation (None -> method default)
    lr_g: float | None = None
    lr_d: float | None = None
    betas: tuple[float, float] | None = None
    n_critic: int | None = None         # discriminator steps per generator step
    # tricks
    label_smoothing: float = 0.0        # one-sided: real target = 1 - label_smoothing (BCE losses)
    ttur: bool = False                  # two time-scale update rule: lr_d = 4 * lr_g
    ema: float = 0.0                    # EMA decay of G weights (0 = off, 0.999 typical)
    r1: float = 0.0                     # R1 gradient penalty weight on real images
    feature_matching: bool = False      # add ||mean f(x) - mean f(G(z))||^2 to the G loss
    saturating: bool = False            # gan/dcgan: original minimax G loss log(1 - D(G(z)))
    # image-to-image
    lambda_l1: float = 100.0            # pix2pix
    lambda_cycle: float = 10.0          # cyclegan
    lambda_identity: float = 0.0        # cyclegan
    # bookkeeping
    log_every: int = 50
    sample_every: int = 500
    seed: int = 0
    threads: int = 1
    device: str = "cpu"
    output_dir: str | None = None       # None -> checkpoints/<method>
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
