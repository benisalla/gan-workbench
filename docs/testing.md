# Testing

```bash
OMP_NUM_THREADS=1 pytest -q
```

All tests run on a CPU, offline (synthetic shapes only), in about a minute.

| file | what it checks |
|---|---|
| `test_data.py` | synthetic sets: range, labels, determinism; masks match bright pixels; outlines lie inside the filled shapes; loader shapes |
| `test_models.py` | output shapes of every G and D, with and without labels, at 28 and 32 px; spectral norm bounds σ(W) ≤ 1; the 6×6 PatchGAN grid; U-Net skips are concatenated untouched |
| `test_losses.py` | loss values by hand (BCE at logit 0 = log 2, LS, WGAN, hinge); the saturating G loss has a ~0 gradient where the non-saturating one does not; gradient penalty = 0 for a 1-Lipschitz linear critic and 1 for a 2-Lipschitz one; R1 and feature matching values |
| `test_metrics.py` | Fréchet distance on Gaussians with known answers (mean shift, covariance scale); IS = K and 10/10 modes for balanced one-hot predictions, IS = 1 and 1 mode for a collapsed generator; IoU |
| `test_methods.py` | every one of the ten methods does a D step and a G step that change both networks' weights; method defaults and overrides (TTUR, n_critic); WGAN clipping holds after a step; cGAN vs ACGAN wiring; every trick flag runs |
| `test_train.py` | `train()` writes checkpoint, curves, samples, history; reload gives identical samples; EMA formula; update-ratio measurement; the `on_step` callback (step 0 = untrained); **DCGAN learns a single image** (the adversarial loop works end to end); **pix2pix overfits a batch**; evaluation of the translation methods |
| `test_cli.py` | `gans-train` → `gans-sample` (grid and interpolation) → `gans-eval` on shapes; `gans-data` |

The two "learns" tests are the important ones: unit tests can pass while the training loop is
wired wrong (as in the coursework notebook, where the optimisers held the parameters of
networks that were then replaced). A generator that converges to a single training image, and a
pix2pix that memorises 16 pairs, prove gradients flow from the losses to the right weights.
