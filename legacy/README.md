# Legacy notebooks

The two Centrale coursework notebooks this repository grew from, copied verbatim.

| notebook | what it did |
|---|---|
| [BE2_PART_001_GAN.ipynb](BE2_PART_001_GAN.ipynb) | DCGAN (4 transposed-conv G, 4 conv D, BatchNorm, Sigmoid + BCELoss) on MNIST resized to 32 px, train + test sets merged, 100 epochs planned; helpers `live_plot_gan` (losses, D(x) / D(G(z)), update ratios) and `compute_update_ratio`. The author's own comment: *"takes hours and hours and does not converge easily"*. |
| [BE2_PART_002_cGAN_IMG_MASK.ipynb](BE2_PART_002_cGAN_IMG_MASK.ipynb) | pix2pix on CMP Facades (kagglehub `kokeyehya/cmp-facade-db-base`) at 256 px: U-Net generator (`inconv`, 7 `down`, 7 `up` with dropout, `outconv`), PatchGAN discriminator (`conv_block`, `out_block`, 30×30 patches), BCE + 100·L1, 200 epochs, checkpoints at 100 and 200; answers to the course questions (bottleneck size 512×1×1, role of skip connections, 2 769 601 D parameters). |

## Where their ideas live now

| notebook piece | in the package |
|---|---|
| `Generator` / `Discriminator` (DCGAN) | `models.DCGenerator` / `models.DCDiscriminator` (28 px, 2 up/down levels) |
| `weights_init` / `weights_init_normal` | `models.dcgan_init` |
| `live_plot_gan` | `viz.live_plot_gan` (same three panels, written to `curves.png`) |
| `compute_update_ratio` | `train.update_ratio` (measures the real step, see below) |
| `U_Net` with `inconv` / `down` / `up` / `outconv` | `models.UNetGenerator` with `Down` / `Up`, configurable depth |
| `PatchGAN` with `conv_block` / `out_block` | `models.PatchDiscriminator` |
| training cell of part 2 | `methods.Pix2Pix.d_step` / `g_step` inside the shared loop |
| `plot2x3Array` (input / real / fake) | `viz.save_translation_grid` |

## Bugs found and how they are fixed here

| # | notebook | bug | effect | fix in this repo |
|---|---|---|---|---|
| 1 | part 1, cell 12 | `optimizerD` / `optimizerG` are created from the `netD` / `netG` of the earlier test cells, **then** `netG` and `netD` are re-created. The optimisers hold the parameters of the old networks. | The networks that are trained never receive an update: G stays at its random init no matter how long it runs. This is very likely the "does not converge" in the author's comment. | Each method builds its networks first and then its optimisers from them (`GANMethod._finish`); `tests/test_methods.py` checks that one step changes the weights of G and D for all ten methods. |
| 2 | part 1, cell 13 | `compute_update_ratio` is called once per epoch, after the last batch, and uses lr·‖grad‖. | With Adam the real step is lr·m/√v, not lr·grad, so the ratio is off by orders of magnitude; one batch per epoch is noise. It also sums per-tensor norms instead of taking the global norm, and loops over the parameters once per param group. | `train.update_ratio` snapshots the parameters and measures ‖θ_after − θ_before‖/‖θ_before‖ on every logged step. |
| 3 | both | D ends with `nn.Sigmoid()` + `nn.BCELoss()`. | log(0) once D saturates; unstable gradients. | D returns logits; losses use `binary_cross_entropy_with_logits` (`losses.py`). |
| 4 | part 1, cell 13 | `D(x)` and `D(G(z))` are taken from the last batch of each epoch only. | Very noisy curves. | Logged every `log_every` steps from the step's own batch. |
| 5 | part 1, cell 6 | MNIST loaded twice; train and test merged into one training set. | Nothing left to evaluate against. | Train split for training, test split as the reference for FID. |
| 6 | part 2, cell 4 | Images and masks are `ToTensor()` only, in [0, 1], while G ends in `tanh` ([-1, 1]). | Half of G's output range is never a valid target; the L1 term pushes against it. | Every dataset is scaled to [-1, 1]. |
| 7 | part 2, cells 3 and 18 | `batch_size = 16` is defined for training but the loader was built earlier with `b_size = 64`. | The hyper-parameter shown is not the one used. | One `TrainConfig`, used to build the loader. |
| 8 | part 2, cell 20 | Validation images are generated with G still in train mode (dropout + batch statistics). | Displayed samples change from call to call and depend on the batch. | `translate()` / `sample()` switch G to eval mode and back. |
| 9 | part 2, cell 8 | `load_model` prints a message and silently continues when the checkpoint is missing; `evaluate_checkpoint` then evaluates whatever G is in memory. | The "epoch 100 vs 200" comparison can show the same model twice. | `train.load_checkpoint` loads from an explicit path and fails loudly. |
| 10 | part 2 | The notebook is titled image → mask but trains `generator(masks)` → images (facade label map to photo). | Not a bug in the code, but a naming mismatch. | `pix2pix` here goes photo → mask on `paired-shapes` (the direction in the title); the notebook trains facades label → photo, as the coursework did. |
| 11 | part 2, cell 12 | A 7-level U-Net with BatchNorm on the 1×1 bottleneck. | With batch size 1 (the pix2pix default) BatchNorm on a 1×1 map is undefined. | The U-Net depth is a parameter; at 32 px the bottleneck is 2×2. |
