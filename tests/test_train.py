import json

import torch
import torch.nn.functional as F

from gans.config import TrainConfig
from gans.data import load_data
from gans.evaluate import evaluate_method
from gans.methods import build_method
from gans.train import EMA, load_checkpoint, train, update_ratio


def test_train_writes_checkpoint_curves_and_reloads(small_cfg, tmp_path):
    r = train(small_cfg("cgan", steps=4, ema=0.9, sample_every=2), verbose=False)
    out = tmp_path / "cgan"
    for f in ("last.pt", "curves.png", "samples.png", "samples_000002.png", "history.json"):
        assert (out / f).exists(), f
    hist = json.loads((out / "history.json").read_text())
    assert hist["step"] == [1, 2, 3, 4] and all(v > 0 for v in hist["update_g"])
    m = load_checkpoint(r["checkpoint"])
    z, y = torch.randn(4, 8), torch.tensor([0, 1, 2, 3])
    assert torch.allclose(m.sample(4, y=y, z=z), r["model"].sample(4, y=y, z=z), atol=1e-6)
    assert m.ema_G is not None


def test_ema_moves_slowly_toward_the_model():
    net = torch.nn.Linear(2, 2)
    ema = EMA(net, 0.9)
    with torch.no_grad():
        net.weight.add_(1.0)
    w0 = ema.model.weight.clone()
    ema.update(net)
    assert torch.allclose(ema.model.weight, w0 + 0.1 * (net.weight - w0))


def test_update_ratio_measures_the_step():
    net = torch.nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        net.weight.fill_(1.0)                                  # ||theta|| = 2
    out = {}
    with update_ratio(net, out, "r"), torch.no_grad():
        net.weight.add_(0.01)                              # ||dtheta|| = 0.02
    assert abs(out["r"] - 0.01) < 1e-6


def test_dcgan_learns_a_single_image():
    """Sanity check of the whole adversarial loop: with one training image, G must converge to it."""
    torch.manual_seed(0)
    data = load_data("shapes", n_train=64)
    target = data.train[0][:1]
    data.train = (target.repeat(64, 1, 1, 1), torch.zeros(64, dtype=torch.long))
    cfg = TrainConfig(method="dcgan", width=16, z_dim=8, lr_g=1e-3, lr_d=1e-3)
    m = build_method(cfg, data.info)
    batch = next(iter(data.loader(16)))
    start = F.l1_loss(m.sample(16), target.expand(16, -1, -1, -1)).item()
    for _ in range(120):
        m.d_step(batch)
        m.g_step(batch)
    end = F.l1_loss(m.sample(16), target.expand(16, -1, -1, -1)).item()
    assert start > 0.5 and end < 0.25, (start, end)


def test_pix2pix_overfits_a_batch(small_cfg):
    torch.manual_seed(0)
    data = load_data("paired-shapes", n_train=16)
    m = build_method(small_cfg("pix2pix", width=32, lr_g=2e-3), data.info)
    batch = data.train
    first = m.g_step(batch)["l1"]
    for _ in range(80):
        m.d_step(batch)
        last = m.g_step(batch)["l1"]
    assert last < 0.3 * first, (first, last)


def test_evaluate_translation_methods(small_cfg):
    for name, ds in (("pix2pix", "paired-shapes"), ("cyclegan", "unpaired-shapes")):
        data = load_data(ds, n_train=16)
        m = build_method(small_cfg(name), data.info)
        metrics = evaluate_method(m, data)
        assert 0 <= metrics["iou"] <= 1 and metrics["l1"] > 0
