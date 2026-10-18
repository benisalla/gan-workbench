# Training: one loop for every GAN type

[`train.py`](../src/gans/train.py), in words:

    for step in 1 .. steps:                       # "step" = one generator update
        repeat n_critic times:
            batch = next(data)
            D step:  fakes = G(z) without gradient;  loss_D(D(real), D(fake)) (+ penalties); Adam step
                     (WGAN: clip D's weights after the step)
        G step:      fakes = G(z) with gradient;     loss_G(D(fake)) (+ class / L1 / cycle / FM); Adam step
        EMA update of G (if --ema)
        every log_every steps: losses, D(x), D(G(z)), update ratios -> history

The method object ([`methods.py`](../src/gans/methods.py)) decides *what* a D step and a G
step are; the loop decides *when* they run and *what is measured*. Keeping the two apart is
what makes the comparison fair: the same seeds, data order and number of G updates for all.

**Why fakes are generated twice.** In the D step the fakes are made under `torch.no_grad()`
because D's loss must not update G; in the G step they are regenerated with gradient. Reusing
one fake batch for both (as the notebook did with `.detach()`) is equivalent and saves a
forward pass; here clarity wins.

## Commands

```bash
gans-train --method dcgan --dataset mnist --steps 3000            # ~ 3 min on one thread
gans-train --method wgan-gp --dataset mnist --steps 2000          # 5 critic steps per G step
gans-train --method cgan --dataset fashion --ema 0.999
gans-train --method dcgan --dataset shapes --ttur --label-smoothing 0.1 --r1 1
gans-train --method pix2pix --steps 1500                          # paired-shapes, photo -> mask
gans-train --method cyclegan --steps 1500                         # unpaired-shapes, filled <-> outline
```

Output folder (`checkpoints/<method>` or `--output-dir`): `last.pt` (config, G, D, EMA G,
history), `history.json`, `curves.png`, `samples.png` and `samples_<step>.png` every
`--sample-every` steps.

## What the curves tell you

`curves.png` has the three panels of the coursework notebooks' `live_plot_gan`:

1. **D and G losses.** For BCE they should hover around fixed levels (D ≈ 2 log 2 ≈ 1.39 at
   the equilibrium). A D loss going to 0 with a G loss climbing means D has won: G gets no
   signal. For WGAN the negative critic loss is an estimate of W1 and should *decrease* as
   samples improve; it is the one GAN loss you can read as a quality curve.
2. **D(x) and D(G(z)).** For BCE these are probabilities; both ≈ 0.5 means D can no longer
   tell. Raw scores for the other losses.
3. **Update ratio** ‖θ_after − θ_before‖ / ‖θ_before‖ per step, for G and D. ~1e-3 is the usual
   healthy range: much lower and the net is frozen, much higher and it is thrashing. This is
   measured on the real update, so it accounts for Adam's per-parameter scaling (the
   notebooks' `compute_update_ratio` used lr·‖grad‖, which overstates Adam's steps by orders
   of magnitude when gradients are small, and summed per-tensor norms).

## Defaults and CPU budget

`--width 64 --z-dim 64 --batch-size 64`, one CPU thread (`--threads 1` sets
`torch.set_num_threads(1)`). On one thread of a shared 8-core box: about 20-40 ms per G step for
the conv GANs with one D step, ×3 for 5 critic steps, and 60-100 ms for pix2pix / CycleGAN at
32 px. A 3 000-step DCGAN run is a few minutes. On a GPU (notebook) use `--device cuda`
and more steps.

## Known failure modes to try on purpose

* `gans-train --method gan --saturating`: G stalls early (vanishing gradient).
* `gans-train --method dcgan --lr-d 1e-3 --lr-g 1e-4`: D wins, D(G(z)) → 0, G loss climbs.
* `gans-train --method wgan --steps 3000`: the clipped critic learns slowly; compare to `wgan-gp`.
* `gans-train --method gan` on MNIST: count the digits in `samples.png` (mode coverage).
