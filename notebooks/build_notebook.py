"""Build notebooks/train_colab_kaggle.ipynb from this file.

The notebook installs the package from GitHub and imports it, so it always runs the same code
as the repo. It explains every GAN type step by step in markdown between the code cells, then
runs the bigger comparisons on a GPU (MNIST at full length, CIFAR-10, pix2pix on CMP Facades).

    python notebooks/build_notebook.py            # writes notebooks/train_colab_kaggle.ipynb
"""

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).with_name("train_colab_kaggle.ipynb")
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# GAN workbench: ten GAN types, one training loop (Kaggle / Colab GPU)

This notebook uses the `gans` package from [benisalla/gan-workbench](https://github.com/benisalla/gan-workbench).
Every GAN type is a small *spec* (networks, loss, conditioning, how D is kept in check, optimiser)
plugged into the **same** training loop, data pipeline and evaluation, so differences you see
come from the method and not from the code around it.

Plan:
1. set up, look at the data;
2. the ten types one by one: the equation, why it exists, what to watch, a short run;
3. the full comparison on MNIST on the GPU, with classifier-feature FID / IS / mode coverage;
4. the shared tricks (EMA, TTUR, R1, feature matching, label smoothing);
5. CIFAR-10 (colour, 32 px);
6. pix2pix on synthetic pairs and on CMP Facades, CycleGAN on unpaired shapes.

Runtime: enable a GPU (Kaggle: Settings → Accelerator → GPU; Colab: Runtime → Change runtime type).
""")

code("""
# The repository is private: on Kaggle/Colab either make a GitHub token available as GH_TOKEN
# (Kaggle: Add-ons -> Secrets; Colab: the key icon) or upload the repo folder and `pip install -e` it.
import os, subprocess, sys
token = os.environ.get("GH_TOKEN", "")
try:  # Colab / Kaggle secret stores
    from google.colab import userdata; token = token or userdata.get("GH_TOKEN")
except Exception:
    pass
try:
    from kaggle_secrets import UserSecretsClient; token = token or UserSecretsClient().get_secret("GH_TOKEN")
except Exception:
    pass
auth = f"{token}@" if token else ""
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                f"git+https://{auth}github.com/benisalla/gan-workbench.git"], check=True)
""")

code("""
import json, time
import torch
import matplotlib.pyplot as plt
from IPython.display import Image, display, Markdown

from gans.config import TrainConfig
from gans.data import load_data, GanData, DataInfo
from gans.methods import METHODS, get_spec, resolve
from gans.train import train
from gans.evaluate import classifier_for, evaluate_method, real_baseline
from gans.viz import save_grid, interpolation_grid, live_plot_gan, save_translation_grid, compare_curves

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")
for name, spec in METHODS.items():
    print(f"{name:9s} {spec.summary}")
""")

md("""
## 1. Data

All datasets are in-memory tensors in [-1, 1] (the range of `tanh`). `mnist` and `fashion` are
28 px grey with 10 labels, `cifar10` is 32 px colour. Three synthetic sets are generated
offline: `shapes` (4 classes), `paired-shapes` (noisy photo → mask, for pix2pix) and
`unpaired-shapes` (filled ↔ outlined shapes drawn independently, for CycleGAN, with the true
answer kept for evaluation).
""")

code("""
mnist = load_data("mnist")
fig = save_grid(mnist.train[0][:64], title="MNIST")
display(fig)
pairs = load_data("paired-shapes")
display(save_grid(torch.cat([pairs.train[0][:16], pairs.train[1][:16]]), nrow=16, title="paired-shapes: photo / mask"))
""")

md("""
## 2. The ten types, one at a time

A helper trains one method for a few hundred steps on MNIST and shows the three panels of the
coursework's `live_plot_gan` (losses, D(x) vs D(G(z)), update ratios ‖Δθ‖/‖θ‖) and a sample grid.
""")

code("""
def quick(method, steps=1500, dataset="mnist", **kw):
    cfg = TrainConfig(method=method, dataset=dataset, steps=steps, width=64, device=DEVICE,
                      log_every=25, sample_every=0, output_dir=f"runs/quick/{method}", **kw)
    r = train(cfg, verbose=False)
    print(f"{method}: {r['seconds']:.0f}s, G {r['params_g']/1e6:.2f}M, D {r['params_d']/1e6:.2f}M")
    display(live_plot_gan(r["history"], title=method))
    display(Image(f"runs/quick/{method}/samples.png"))
    return r
""")

md("""
### 2.1 `gan`: the original minimax game

$$\\min_G \\max_D \\; \\mathbb{E}_x[\\log D(x)] + \\mathbb{E}_z[\\log(1 - D(G(z)))]$$

D is a real-vs-fake classifier; G tries to fool it. With the optimal D the game minimises the
Jensen-Shannon divergence between data and samples. In practice G maximises $\\log D(G(z))$
instead ("non-saturating"): early on D rejects every fake, and $\\log(1 - D(G(z)))$ is flat
there, so the original loss gives G almost no gradient. Run it twice to see it:
`saturating=True` stalls for a long time.

Both networks here are MLPs on flattened pixels, as in 2014. **Why it collapses:** G is only
rewarded for fooling the *current* D, so it can put all its mass on the one digit that works
best; nothing in the loss asks for diversity. Count the digits in the grid.
""")
code('r_gan = quick("gan")\nr_gan_sat = quick("gan", saturating=True)')

md("""
### 2.2 `dcgan`: convolutions

Same loss, but G is `Linear → (2w,7,7) → ConvT s2 → (w,14,14) → ConvT s2 → (1,28,28)` and D mirrors
it with strided convolutions, BatchNorm, LeakyReLU, N(0, 0.02) init and Adam β₁ = 0.5.
Convolutions learn a stroke once and reuse it everywhere, which is why this is far faster and
sharper than the MLP.
""")
code('r_dcgan = quick("dcgan")')

md("""
### 2.3 `cgan`: conditioning by showing the label to both players

$$\\min_G \\max_D \\; \\mathbb{E}[\\log D(x, y)] + \\mathbb{E}[\\log(1 - D(G(z, y), y))]$$

G gets $[z, \\text{onehot}(y)]$; D gets the image plus 10 constant label channels. D now judges
*pairs*, so a perfect "3" labelled "7" is fake, and G must draw what it is asked. Because we
choose the class, collapse onto one digit is no longer possible. Columns of the grid = digits 0-9.
""")
code('r_cgan = quick("cgan")')

md("""
### 2.4 `acgan`: D must classify, not just judge

G is conditioned as in cGAN, but D only sees the image and has two heads: real/fake and a
10-way classifier. $L_D$ = BCE + CE(class | real) + CE(class | fake); $L_G$ = BCE + CE(class | fake).
The class signal becomes a classification loss: very readable digits, sometimes less variety
inside a class (G draws the "most classifiable" 7).
""")
code('r_acgan = quick("acgan")')

md("""
### 2.5 `lsgan`: least squares

$$L_D = \\tfrac12 (D(x) - 1)^2 + \\tfrac12 D(G(z))^2, \\qquad L_G = \\tfrac12 (D(G(z)) - 1)^2$$

With a log loss, a fake that D classifies correctly with high confidence gives a vanishing
gradient even when it is far from the data. A squared loss keeps pulling samples in
proportion to how wrong they are.
""")
code('r_lsgan = quick("lsgan")')

md("""
### 2.6 `wgan`: a critic that measures distance

$$L_D = \\mathbb{E}[D(G(z))] - \\mathbb{E}[D(x)], \\qquad L_G = -\\mathbb{E}[D(G(z))]$$

If D is 1-Lipschitz, $\\max_D$ of this equals the Wasserstein-1 (earth mover's) distance. Unlike
JS, it does not saturate when the two distributions do not overlap, so G always gets a useful
gradient, and the critic loss is a meaningful curve: it should go down as samples improve.
The critic must be near optimal (5 critic steps per G step). Lipschitzness is forced by
**clipping** every weight to ±0.01 after each step, with RMSprop (lr 5e-5). Clipping wastes the
critic's capacity: expect slow progress.
""")
code('r_wgan = quick("wgan", steps=1000)')

md("""
### 2.7 `wgan-gp`: gradient penalty instead of clipping

$$L_D = \\mathbb{E}[D(G(z))] - \\mathbb{E}[D(x)] + \\lambda\\, \\mathbb{E}_{\\hat x}\\big[(\\|\\nabla_{\\hat x} D(\\hat x)\\|_2 - 1)^2\\big], \\quad \\hat x = a x + (1-a) G(z)$$

The optimal critic has gradient norm 1 on lines between real and fake samples, so the penalty
asks exactly that, softly. No BatchNorm in the critic (the penalty is per sample), Adam(1e-4, β=(0, 0.9)).
""")
code('r_wgangp = quick("wgan-gp", steps=1000)')

md("""
### 2.8 `sngan`: hinge loss + spectral normalisation

$$L_D = \\mathbb{E}[\\max(0, 1 - D(x))] + \\mathbb{E}[\\max(0, 1 + D(G(z)))], \\qquad L_G = -\\mathbb{E}[D(G(z))]$$

Every D layer's weight is divided by its largest singular value (one power-iteration step per
forward pass), which bounds D's Lipschitz constant layer by layer: no extra backward pass like
the penalty, no capacity loss like clipping. The hinge stops pushing once a sample is past the margin.
""")
code('r_sngan = quick("sngan")')

md("""
### 2.9 `pix2pix`: paired image-to-image

$$L_G = \\text{BCE}(D(x, G(x)), 1) + 100\\, \\|G(x) - y\\|_1$$

The input is now an image. L1 alone gives the right layout but blurry edges (the average of
plausible answers); the conditional **PatchGAN** D judges each ~22-px patch of cat(input, output)
and punishes blur. The **U-Net** skips pass the input's edges straight to the decoder. This is
the coursework's second notebook, here on synthetic photo → mask pairs (32 px) first.
""")
code("""
r_p2p = quick("pix2pix", dataset="paired-shapes", steps=2000)
m = r_p2p["model"]; src, tgt = r_p2p["data"].test
display(save_translation_grid(m, src[:12], target=tgt[:12]))   # rows: input / true mask / output
print(evaluate_method(m, r_p2p["data"]))
""")

md("""
### 2.10 `cyclegan`: unpaired translation

Two generators $G_{AB}, G_{BA}$, two PatchGAN critics, LSGAN losses, plus the cycle loss
$\\lambda(\\|G_{BA}(G_{AB}(a)) - a\\|_1 + \\|G_{AB}(G_{BA}(b)) - b\\|_1)$. The adversarial terms only say
"looks like some B"; the cycle term forces $G_{AB}$ to keep *which* shape it was. Training never
sees a pair; the synthetic test set knows the right outline, so we can still score it.
""")
code("""
r_cyc = quick("cyclegan", dataset="unpaired-shapes", steps=2000)
m = r_cyc["model"]; a, b_true = r_cyc["data"].test
fake_b = m.translate(a[:12], "ab")
display(save_grid(torch.cat([a[:12], fake_b, m.translate(fake_b, "ba"), b_true[:12]]), nrow=12))
print(evaluate_method(m, r_cyc["data"]))
""")

md("""
## 3. Full comparison on MNIST (GPU)

Same data, same number of generator steps (WGANs: 5 critic steps each), width 64, then the
same evaluation: **classifier-feature FID** (Fréchet distance on the 128-d features of a small
CNN trained on MNIST, *not* Inception FID, only comparable inside this repo), the IS analogue,
how many digits survive (mode coverage) and, for conditional models, how often the requested
digit is drawn.
""")
code("""
STEPS = 6000
clf = classifier_for("mnist", data=mnist)
rows, hists = {"real": real_baseline("mnist")}, {}
for name in ["gan", "dcgan", "cgan", "acgan", "lsgan", "wgan", "wgan-gp", "sngan"]:
    cfg = TrainConfig(method=name, dataset="mnist", steps=STEPS, width=64, device=DEVICE,
                      sample_every=0, output_dir=f"runs/mnist/{name}")
    r = train(cfg, verbose=False)
    met = evaluate_method(r["model"], r["data"], 10000, clf=clf)
    rows[name] = {"seconds": round(r["seconds"]), **{k: round(v, 3) if isinstance(v, float) else v for k, v in met.items()}}
    hists[name] = r["history"]
    print(name, rows[name])
    display(Image(f"runs/mnist/{name}/samples.png"))
""")
code("""
lines = ["| method | time | FID (clf) | IS (clf) | modes | label acc |", "|---|---|---|---|---|---|"]
for n, r in rows.items():
    lines.append(f"| {n} | {r.get('seconds', '-')} s | {r['fid']:.2f} | {r['is']:.2f} | {r['modes']}/10 | "
                 f"{r.get('label_acc', '-')} |")
display(Markdown("\\n".join(lines)))
display(compare_curves(hists, "d_fake", title="D(G(z))"))
display(compare_curves(hists, "update_g", title="G update ratio", logy=True))
""")

md("""
**Reading the table.** Lower FID is better, the real-vs-real row is the floor. A low IS or fewer
than 10 modes with a decent FID means the samples are fine but not diverse. Conditional models
should be near 10/10 modes by construction, so for them look at label accuracy.

## 4. The shared tricks

All are flags of the same loop. EMA keeps a slowly moving average of G's weights and samples
from it (averages out the oscillation of the game); TTUR gives D a 4× larger learning rate
instead of more D steps; R1 penalises D's gradient on real data (keeps D flat around the data);
feature matching adds a target to G that moves slower than D's decision; one-sided label
smoothing (real target 0.9) stops D from becoming over-confident.
""")
code("""
tricks = {"dcgan": {}, "+ema": {"ema": 0.999}, "+ttur": {"ttur": True}, "+r1": {"r1": 1.0},
          "+fm": {"feature_matching": True}, "+smooth": {"label_smoothing": 0.1}}
for tag, kw in tricks.items():
    r = train(TrainConfig(method="dcgan", dataset="mnist", steps=STEPS, width=64, device=DEVICE,
                          sample_every=0, output_dir=f"runs/tricks/{tag}", **kw), verbose=False)
    print(tag, evaluate_method(r["model"], r["data"], 10000, clf=clf))
""")

md("""
## 5. CIFAR-10 (colour, 32 px)

The same `DCGenerator` / `DCDiscriminator` at 32 px with 3 channels. The evaluation classifier
is the same small CNN trained on CIFAR-10 (about 70% accuracy), so these FIDs are only for
comparing the methods with each other. 20k steps at width 128 take roughly 10-20 minutes each on
a T4/P100.
""")
code("""
cifar = load_data("cifar10")
clf_c = classifier_for("cifar10", data=cifar)
for name in ["dcgan", "lsgan", "wgan-gp", "sngan", "cgan"]:
    r = train(TrainConfig(method=name, dataset="cifar10", steps=20000, width=128, z_dim=128, ema=0.999,
                          device=DEVICE, sample_every=0, output_dir=f"runs/cifar/{name}"), verbose=False, data=cifar)
    print(name, f"{r['seconds']:.0f}s", evaluate_method(r["model"], cifar, 10000, clf=clf_c))
    display(Image(f"runs/cifar/{name}/samples.png"))
""")

md("""
## 6. pix2pix on CMP Facades

The coursework data: 606 facade photos with colour label maps
(`kagglehub.dataset_download("kokeyehya/cmp-facade-db-base")`). As in the coursework notebook
the generator maps the **label map to the photo**. At 128 px a depth-6 U-Net (128 → 2 → 128)
and the PatchGAN (a 14×14 grid of patch decisions) train in a few minutes on a GPU.
""")
code("""
import glob, numpy as np, kagglehub
from PIL import Image as PILImage

root = kagglehub.dataset_download("kokeyehya/cmp-facade-db-base")
photos = sorted(glob.glob(root + "/base/*.jpg"))
SIZE = 128

def load(path, rgb=True):
    im = PILImage.open(path).convert("RGB").resize((SIZE, SIZE), PILImage.BICUBIC)
    return torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 127.5 - 1   # [-1, 1], unlike the notebook

x = torch.stack([load(p.replace(".jpg", ".png")) for p in photos])   # label maps (input)
y = torch.stack([load(p) for p in photos])                           # photos (target)
n_val = len(x) // 20
facades = GanData(DataInfo("facades", "paired", 3, SIZE), (x[n_val:], y[n_val:]), (x[:n_val], y[:n_val]))
cfg = TrainConfig(method="pix2pix", steps=8000, batch_size=4, width=128, device=DEVICE, sample_every=0,
                  output_dir="runs/facades", extra={"unet_depth": 6})
r = train(cfg, data=facades)
m = r["model"]
display(save_translation_grid(m, x[:8], target=y[:8]))       # rows: label map / real photo / generated
print(evaluate_method(m, facades))
""")

md("""
## What to take away

* **The loss decides what "close" means.** JS (gan/dcgan) saturates when the distributions do
  not overlap; least squares and Wasserstein do not.
* **Controlling D is most of the game.** Clipping, gradient penalty and spectral norm are three
  ways of making D smooth; the results show what each costs.
* **Conditioning is the cheapest anti-collapse tool.** Tell G which class to draw and it can no
  longer drop classes.
* **For image-to-image, the adversarial loss is a texture prior** added to a regression loss
  (pix2pix) or to a cycle constraint (CycleGAN).
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}, "accelerator": "GPU"}
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
