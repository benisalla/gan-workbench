# Examples

Short scripts that use the `gans` package from Python, in the order one would use it. Each is
one command, runs on a CPU with small defaults, and writes to `examples/output/`. Run them from
the repository root after `pip install -e ".[dev]"`. Times are measured on one thread of a busy
shared 8-core machine; on an idle machine they are faster.

| script | what it shows | time | needs |
|---|---|---|---|
| [01_train_dcgan.py](01_train_dcgan.py) | `TrainConfig` + `train()`: a DCGAN on 20 000 MNIST digits, sample grids and the three-panel training curves | ~3 min | MNIST (11 MB) |
| [02_train_any_method.py](02_train_any_method.py) | any of the ten types in one command (`--method wgan-gp`, `--method pix2pix`, ...), prints its idea and defaults | 2-6 min | MNIST, or `--dataset shapes` offline |
| [03_conditional_mnist.py](03_conditional_mnist.py) | cGAN vs ACGAN on MNIST: one digit per column, "only sevens", label accuracy measured by a classifier | ~8 min | MNIST (11 MB download) |
| [04_evaluate_and_interpolate.py](04_evaluate_and_interpolate.py) | the evaluation classifier, classifier-feature FID / IS / mode coverage vs the real-data floor, a latent interpolation grid | ~2 min | 01 |
| [05_pix2pix_photo_to_mask.py](05_pix2pix_photo_to_mask.py) | pix2pix (U-Net + PatchGAN + L1) on noisy photos of shapes → masks, test L1 and IoU | ~3.5 min | nothing |
| [06_cyclegan_unpaired.py](06_cyclegan_unpaired.py) | CycleGAN from unpaired filled / outlined shapes: a → G_AB(a) → G_BA(G_AB(a)) and the true outline | ~6 min | nothing |

```bash
python examples/01_train_dcgan.py
python examples/02_train_any_method.py --method wgan-gp
python examples/03_conditional_mnist.py
python examples/04_evaluate_and_interpolate.py
python examples/05_pix2pix_photo_to_mask.py
python examples/06_cyclegan_unpaired.py
```

The same steps as CLI commands are in the main README (`gans-data`, `gans-train`,
`gans-sample`, `gans-eval`); the Kaggle/Colab notebook runs them longer on a GPU.
