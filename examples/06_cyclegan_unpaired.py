"""06 - CycleGAN: filled shapes <-> outlined shapes from UNPAIRED sets (about 4 minutes).

Training never sees a (filled, outline) pair: domain A and domain B are drawn independently.
The cycle loss ||G_BA(G_AB(a)) - a|| is what forces G_AB to keep the shape and only change the
style. We can still score it because the synthetic test set knows the true outline of each a.

    python examples/06_cyclegan_unpaired.py
"""

from gans.config import TrainConfig
from gans.evaluate import evaluate_method
from gans.train import train
from gans.viz import save_grid

r = train(TrainConfig(method="cyclegan", steps=500, log_every=20, sample_every=0,
                      output_dir="examples/output/06_cyclegan"))
method, data = r["model"], r["data"]
a, b_true = data.test[0][:12], data.test[1][:12]
fake_b = method.translate(a, "ab")
back = method.translate(fake_b, "ba")
import torch

save_grid(torch.cat([a, fake_b, back, b_true]), "examples/output/06_cyclegan/a_to_b_to_a.png", nrow=12)
m = evaluate_method(method, data)
print(f"A->B L1 vs true outline {m['l1']:.3f}  IoU {m['iou']:.3f}  cycle L1 {m['cycle_l1']:.3f}")
print("rows: a / G_AB(a) / G_BA(G_AB(a)) / true outline -> examples/output/06_cyclegan/a_to_b_to_a.png")
