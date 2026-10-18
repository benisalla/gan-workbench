"""01 - Train a DCGAN on MNIST (11 MB download on first use, about 3 minutes on one CPU thread).

What to look at afterwards:
* examples/output/01_dcgan/samples.png  64 generated digits from a fixed batch of z
* examples/output/01_dcgan/curves.png   D / G losses, D(x) vs D(G(z)), update ratios

Offline? use ``dataset="shapes"`` (synthetic; harder for a GAN than it looks, because the edges
are perfectly sharp and D spots any grey pixel).

The whole API in three lines: a ``TrainConfig``, ``train(cfg)``, and the returned method object.

    python examples/01_train_dcgan.py
"""

from gans.config import TrainConfig
from gans.train import train

cfg = TrainConfig(method="dcgan", dataset="mnist", n_train=20000, steps=800, width=32, log_every=20,
                  sample_every=200, output_dir="examples/output/01_dcgan")
result = train(cfg)
h = result["history"]
print(f"final D(x) = {h['d_real'][-1]:.2f}, D(G(z)) = {h['d_fake'][-1]:.2f} "
      "(both near 0.5 means D can no longer tell them apart)")
print("samples ->", "examples/output/01_dcgan/samples.png")
