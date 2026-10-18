# Architecture: the ten GAN types, what each one changes and why

All ten share the same skeleton: a generator G turns an input (noise z, or an image) into an
image, a discriminator / critic D scores images, and the two are trained in turns by the one
loop in [`train.py`](../src/gans/train.py). What differs is a short list of choices, collected
per method in `METHODS` ([`methods.py`](../src/gans/methods.py)):

| method | networks | D loss / G loss | conditioning | how D is kept in check | optimiser defaults |
|---|---|---|---|---|---|
| `gan` | MLP | BCE / non-saturating BCE | - | - | Adam 2e-4, (0.5, 0.999) |
| `dcgan` | conv | BCE / non-saturating BCE | - | BatchNorm | Adam 2e-4, (0.5, 0.999) |
| `cgan` | conv | BCE | y into G and into D | BatchNorm | Adam 2e-4 |
| `acgan` | conv | BCE + class cross-entropy | y into G, D predicts y | BatchNorm | Adam 2e-4 |
| `lsgan` | conv | least squares | - | BatchNorm | Adam 2e-4 |
| `wgan` | conv | Wasserstein | - | weight clipping to ±0.01, 5 critic steps | RMSprop 5e-5 |
| `wgan-gp` | conv | Wasserstein | - | gradient penalty λ=10, 5 critic steps, no BatchNorm | Adam 1e-4, (0, 0.9) |
| `sngan` (`hinge`) | conv | hinge | - | spectral norm on every D layer | Adam 2e-4, (0, 0.9) |
| `pix2pix` | U-Net + PatchGAN | BCE + 100·L1 | the input image goes to G and D | BatchNorm | Adam 2e-4, (0.5, 0.999) |
| `cyclegan` | 2 U-Nets + 2 PatchGANs | least squares + 10·cycle L1 | - (unpaired) | InstanceNorm | Adam 2e-4, (0.5, 0.999) |

Shapes below are for MNIST (1×28×28), z of size 64 and `--width 64` (w).

## 1. `gan`: the original minimax game (Goodfellow et al. 2014)

    min_G max_D  E_x[log D(x)] + E_z[log(1 - D(G(z)))]

D is a binary classifier "real or generated", G tries to make it wrong. For a fixed G the best
D is D*(x) = p_data(x) / (p_data(x) + p_G(x)), and plugging it back gives
2·JS(p_data || p_G) - log 4: the game minimises the Jensen-Shannon divergence.

**Saturating vs non-saturating.** Early in training D rejects fakes easily, D(G(z)) ≈ 0, and
log(1 - D(G(z))) is flat there: G gets almost no gradient. The fix from the same paper, used by
default everywhere here, is to have G *maximise* log D(G(z)) instead (same fixed point, strong
gradient when G is bad). `--saturating` switches back so you can see the difference;
`tests/test_losses.py::test_saturating_loss_has_vanishing_gradient_when_d_wins` measures it.

Networks ([`models.py`](../src/gans/models.py)): G = Linear(64→256) → Linear(256→512) +
BatchNorm → Linear(512→784) → tanh; D = Linear(784→512) → Linear(512→256) → Linear(256→1). The
losses take D's **logit** (no sigmoid in D) and use `binary_cross_entropy_with_logits`, which
is numerically stable; the coursework notebooks used `Sigmoid` + `BCELoss`.

**Why it collapses.** G is only rewarded for fooling the *current* D. If one kind of digit
fools D best, G can move all of z onto it (mode collapse); D then learns to reject that digit,
G jumps to another, and so on. Nothing in the loss asks G to cover all modes. The mode
coverage metric ([evaluation.md](evaluation.md)) counts how many digits survive.

## 2. `dcgan`: same game, convolutional networks (Radford et al. 2015)

    G: z(64) → Linear → (128, 7, 7) → BN, ReLU → ConvT 4×4 s2 → (64, 14, 14) → BN, ReLU → ConvT 4×4 s2 → (1, 28, 28) → tanh
    D: (1, 28, 28) → Conv 4×4 s2 → (64, 14, 14) → LReLU → Conv 4×4 s2 → (128, 7, 7) → BN, LReLU → Linear → logit

The DCGAN recipe: strided convolutions instead of pooling, BatchNorm in both nets, ReLU in G,
LeakyReLU(0.2) in D, tanh output, weights from N(0, 0.02), Adam with β1 = 0.5. Convolutions
share weights across positions, so G and D learn local stroke patterns once instead of per
pixel; that is the main reason it trains much faster than the MLP. (The 64-px DCGAN of the
first coursework notebook had four ConvT layers; at 28 px two are enough.)

## 3. `cgan`: conditional GAN (Mirza & Osindero 2014)

    min_G max_D  E_{x,y}[log D(x, y)] + E_{z,y}[log(1 - D(G(z, y), y))]

The label y is given to **both** nets: concatenated to z for G, and as 10 constant one-hot
channel maps stacked on the image for D (input (11, 28, 28)). D now judges *pairs*: a perfect
"3" with the label "7" is fake. So G must produce the requested class. This also removes most
of the mode-collapse pressure: the class is chosen by us, not by G.

## 4. `acgan`: auxiliary classifier GAN (Odena et al. 2017)

G is conditioned as in cGAN, but D sees only the image and has two heads: real/fake and a
10-way class classifier. Losses:

    L_D = BCE(real/fake) + CE(class | real) + CE(class | fake)
    L_G = BCE(fake → "real") + CE(class | fake)

D learns to classify real digits; G is rewarded for images D classifies as the asked class.
Difference from cGAN: the class signal is a *classification* loss rather than part of the
real/fake decision, which gives sharper class features but can push G toward the easiest,
most "classifiable" version of each class (less diversity within a class).

## 5. `lsgan`: least squares (Mao et al. 2017)

    L_D = ½ (D(x) - 1)² + ½ D(G(z))²          L_G = ½ (D(G(z)) - 1)²

With the sigmoid + log loss, a fake that D classifies correctly *and confidently* gives G a
vanishing gradient even if it is far from the data. The squared loss keeps penalising samples
in proportion to their distance from the target value, so far-away samples still get pulled in.
Theoretically it minimises a Pearson χ² divergence.

## 6. `wgan`: Wasserstein GAN with weight clipping (Arjovsky et al. 2017)

    L_D = E[D(G(z))] - E[D(x)]                L_G = -E[D(G(z))]

D is now a *critic* (an unbounded score, no sigmoid). By Kantorovich-Rubinstein duality,
max over 1-Lipschitz f of E_x f(x) - E_z f(G(z)) equals the Wasserstein-1 (earth mover's)
distance between data and generated distributions. Why that matters: when the two
distributions do not overlap (always true early, on low-dimensional image manifolds) JS is a
constant log 2 and gives no gradient, while W1 still says how far apart they are. The critic
loss is therefore a meaningful "distance" curve, and the critic should be trained close to
optimal: 5 critic steps per G step.

The 1-Lipschitz constraint is enforced crudely: after every critic step all weights are
clipped to [-0.01, 0.01]. Momentum optimisers misbehave with this, so the paper uses RMSprop
with lr 5e-5. The cost: a clipped critic has very little capacity, and training is slow; expect
blurrier samples for the same number of G steps (see the results).

## 7. `wgan-gp`: gradient penalty (Gulrajani et al. 2017)

Same Wasserstein losses, but Lipschitzness is enforced softly:

    L_D = E[D(G(z))] - E[D(x)] + λ E_x̂[(||∇_x̂ D(x̂)||₂ - 1)²],   x̂ = a·x + (1-a)·G(z), a ~ U(0,1)

An optimal critic has gradient norm exactly 1 on lines between real and fake points, so the
penalty is placed there. The critic has **no BatchNorm** (the penalty is per sample; BatchNorm
would make each sample's score depend on the batch) and the optimiser is Adam(1e-4, β=(0, 0.9)).
[`losses.gradient_penalty`](../src/gans/losses.py) is 6 lines; a test checks it is 0 for a
critic with gradient norm 1 and 1 for norm 2.

## 8. `sngan`: hinge loss + spectral normalisation (Miyato et al. 2018)

    L_D = E[max(0, 1 - D(x))] + E[max(0, 1 + D(G(z)))]          L_G = -E[D(G(z))]

Spectral norm divides each weight matrix by its largest singular value σ(W) (estimated with
one step of power iteration per forward pass, `torch.nn.utils.spectral_norm`). A network of
layers with σ = 1 and 1-Lipschitz activations is itself at most 1-Lipschitz, so D's smoothness
is controlled layer by layer, cheaply, with no extra backward pass (unlike the gradient
penalty) and without destroying capacity (unlike clipping). The hinge loss stops pushing D
once a sample is beyond the margin, which keeps D from becoming over-confident.
`--method hinge` is an alias.

## 9. `pix2pix`: paired image-to-image translation (Isola et al. 2017)

Now the input is an image, not noise:

    L_D = ½ [BCE(D(x, y), 1) + BCE(D(x, G(x)), 0)]
    L_G = BCE(D(x, G(x)), 1) + λ ||G(x) - y||₁,   λ = 100

* **L1 alone** gives the right layout but blurry outputs: averaging plausible answers
  minimises L1. **The adversarial term** penalises "blurry is not realistic". They split the
  job: L1 for low frequencies, D for high frequencies.
* **U-Net generator** (as in the coursework notebook: `inconv`, `down`, `up` with dropout,
  `outconv`). At 32 px, depth 4: 32 → 16 → 8 → 4 → 2 → 4 → 8 → 16 → 32, each decoder level
  concatenates the encoder feature map of the same size. Without skips every detail would
  have to squeeze through the 2×2 bottleneck; with them, the edges of the input are directly
  available to the output.
* **PatchGAN discriminator**: fully convolutional, D(x, y) sees cat(input, output) and outputs a
  6×6 grid of logits, each looking at a ~22-px patch. It only judges local texture/sharpness,
  which is what L1 cannot do, with few parameters, and works on any image size.
* No z: pix2pix's noise comes from dropout only; the mapping is almost deterministic.

The coursework notebook's `patch_size = (1, im_size//2**3 - 2, ...)` (30×30 at 256 px) is the
same computation; here the patch grid is read from D's output so no formula is needed.

## 10. `cyclegan`: unpaired translation (Zhu et al. 2017)

Two generators G_AB, G_BA and two PatchGAN discriminators D_A, D_B; training sees a batch of A
and an *independent* batch of B (no pairs):

    L_D = ½ [LS(D_B(b), D_B(G_AB(a))) + LS(D_A(a), D_A(G_BA(b)))]
    L_G = LS_G(D_B(G_AB(a))) + LS_G(D_A(G_BA(b))) + λ (||G_BA(G_AB(a)) - a||₁ + ||G_AB(G_BA(b)) - b||₁)

The adversarial terms only say "G_AB(a) looks like *some* B". The cycle term says "and you can
get a back from it", so G_AB must keep the content of a and change only its style. Optional
identity loss (`--lambda-identity`) asks G_AB(b) ≈ b. Simplifications versus the paper, to
stay small: U-Net generators instead of ResNet ones, no history buffer of fakes, 32 px.

## Shared tricks (flags of `gans-train`)

| flag | what it does | why |
|---|---|---|
| `--label-smoothing 0.1` | real target 0.9 instead of 1 (BCE / LS) | an over-confident D gives G poor gradients |
| `--ttur` | lr_D = 4e-4, lr_G = 1e-4 (Heusel 2017) | D tracks G closely without more D steps |
| `--n-critic k` | k D steps per G step | Wasserstein critics should be near optimal |
| `--ema 0.999` | sample from an exponential moving average of G's weights | averages out the oscillation of the game |
| `--r1 γ` | γ · ½ E‖∇ₓD(x)‖² on real data (Mescheder 2018) | flat D around the data, stops D/G rotating around each other |
| `--feature-matching` | add ‖mean f(x) − mean f(G(z))‖² over D's penultimate features (Salimans 2016) | gives G a target that does not move as fast as D's decision |
| `--saturating` | original log(1 − D(G(z))) generator loss | to see the vanishing gradient |
