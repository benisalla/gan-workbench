"""04 - Evaluate a generator and walk its latent space (needs 01; about 2 minutes).

* trains (once, cached) the small evaluation classifier on MNIST
* classifier-feature FID, IS analogue and mode coverage of the 01 DCGAN, next to the
  real-vs-real floor
* a latent interpolation grid: each row goes in a straight line from z_a to z_b; smooth
  morphs mean G learned a continuous map, not a lookup table of training images

    python examples/04_evaluate_and_interpolate.py
"""

from gans.data import load_data
from gans.evaluate import classifier_for, evaluate_method
from gans.metrics import evaluate_samples
from gans.train import load_checkpoint
from gans.viz import interpolation_grid

method = load_checkpoint("examples/output/01_dcgan/last.pt")
data = load_data("mnist")
clf = classifier_for("mnist", data=data, cache_dir="examples/output")
gen = evaluate_method(method, data, n=2000, clf=clf)
real = evaluate_samples(clf, data.train[0][:2000], data.test[0][:2000])
print(f"{'':12s} {'FID(clf)':>9s} {'IS(clf)':>8s} {'modes':>6s}")
for name, m in (("dcgan", gen), ("real data", real)):
    print(f"{name:12s} {m['fid']:9.2f} {m['is']:8.2f} {m['modes']:>4d}/{m['n_classes']}")
interpolation_grid(method, path="examples/output/04_interpolation.png")
print("interpolation -> examples/output/04_interpolation.png")
