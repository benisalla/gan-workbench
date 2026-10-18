# Evaluation

[`metrics.py`](../src/gans/metrics.py), [`evaluate.py`](../src/gans/evaluate.py).

```bash
gans-eval checkpoints/compare/mnist/dcgan/last.pt --n 5000     # writes metrics.json next to it
gans-eval --real mnist                                          # the real-vs-real floor
gans-sample checkpoints/compare/mnist/cgan/last.pt --label 7 --out sevens.png
gans-sample checkpoints/compare/mnist/dcgan/last.pt --interpolate --out walk.png
```

## A classifier instead of Inception

The standard FID and Inception Score use features of Inception-v3 trained on ImageNet. For
28-px grey digits those features are a poor fit (and the network is a 100 MB download). Here
we train a small CNN (two conv layers, 128-d feature layer, ~99% test accuracy on MNIST after
two epochs, cached as `checkpoints/classifier_<dataset>.pt`) and use it the same way. **These
are "classifier-feature" FID / IS: only comparable within this repo, same dataset, same
classifier.** They are not comparable to published FID numbers.

## Metrics for noise-to-image GANs (5 000 samples vs 5 000 real test images)

| metric | definition | reading |
|---|---|---|
| FID (clf) | ‖μ_r − μ_g‖² + Tr(Σ_r + Σ_g − 2(Σ_r Σ_g)^½) on 128-d features | lower is better; the floor is the real-vs-real value |
| IS (clf) | exp E_x KL(p(y∣x) ‖ p(y)) | high when each sample is confidently one class *and* classes are balanced; max = number of classes |
| modes | classes receiving ≥ 1/(4K) of the samples | 10/10 for a healthy MNIST generator, 1-3 when collapsed |
| KL to uniform | KL(predicted class histogram ‖ uniform) | 0 when balanced |
| label acc | cGAN/ACGAN: classifier agrees with the requested label | does conditioning work? |

Why both FID and IS: IS only looks at the generated samples (a generator that outputs one
perfect sample per class gets a perfect IS), FID compares feature *statistics* with real data
and also penalises low diversity within a class.

The matrix square root is computed without SciPy: Tr((Σ_r Σ_g)^½) equals the sum of square
roots of the eigenvalues of Σ_r^½ Σ_g Σ_r^½, which is symmetric, so `torch.linalg.eigh`
suffices. `tests/test_metrics.py` checks the formula on Gaussians with a known answer.

## Metrics for image-to-image methods

* **pix2pix** on `paired-shapes` test pairs: L1 between G(photo) and the true mask (in [-1, 1])
  and **mask IoU** after thresholding at 0.
* **CycleGAN** on `unpaired-shapes`: training never sees pairs, but the synthetic test set
  knows the true outline of every filled shape, so L1 / IoU of G_AB(a) against it is reported,
  plus the **cycle L1** ‖G_BA(G_AB(a)) − a‖.

## Pictures

* `samples.png` from a fixed z (conditional models: one class per column);
* `interp_*.png`: rows go from z_a to z_b in a straight line; smooth morphs mean G learned a
  continuous map rather than memorising training images;
* `curves.png` (see [training.md](training.md));
* `scripts/results_table.py` draws `compare_dfake_mnist.png` and `compare_update_g_mnist.png`
  across methods.
