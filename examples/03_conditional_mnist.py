"""03 - Conditioning: cGAN vs ACGAN on MNIST, one digit per column (about 4 minutes, 11 MB download).

cGAN shows the label to D (D judges the pair (image, label)); ACGAN makes D predict the label.
Both feed the label to G. After training, ask for a given digit and check with a classifier how
often the generator draws the digit it was asked for ("label accuracy").

    python examples/03_conditional_mnist.py
"""

import torch

from gans.config import TrainConfig
from gans.evaluate import classifier_for, evaluate_method
from gans.train import train
from gans.viz import save_grid

for name in ("cgan", "acgan"):
    r = train(TrainConfig(method=name, dataset="mnist", n_train=20000, steps=700, width=32, log_every=20,
                          sample_every=0, output_dir=f"examples/output/03_{name}"))
    method, data = r["model"], r["data"]
    sevens = method.sample(10, y=torch.full((10,), 7))
    save_grid(sevens, f"examples/output/03_{name}/only_sevens.png", nrow=10)
    clf = classifier_for("mnist", data=data, cache_dir="examples/output")
    m = evaluate_method(method, data, n=2000, clf=clf)
    print(f"{name}: label accuracy {m['label_acc']:.1%}  FID(clf) {m['fid']:.2f}  modes {m['modes']}/10")
print("grids (one digit per column) -> examples/output/03_*/samples.png")
