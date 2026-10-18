import pytest
import torch

from gans.data import load_data
from gans.methods import ALIASES, METHODS, build_method, get_spec, resolve


def _data_for(method):
    return load_data(get_spec(method).dataset if method in ("pix2pix", "cyclegan") else "shapes", n_train=64)


@pytest.mark.parametrize("name", list(METHODS))
def test_every_method_does_a_d_step_and_a_g_step(name, small_cfg):
    data = _data_for(name)
    m = build_method(small_cfg(name), data.info)
    batch = next(iter(data.loader(16)))
    before_g = [p.detach().clone() for p in m.G.parameters()]
    before_d = [p.detach().clone() for p in m.D.parameters()]
    d = m.d_step(batch)
    g = m.g_step(batch)
    assert all(torch.isfinite(torch.tensor(v)) for v in {**d, **g}.values())
    assert any(not torch.equal(a, b) for a, b in zip(before_g, m.G.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(before_d, m.D.parameters()))


def test_defaults_come_from_the_method_and_config_overrides_them(small_cfg):
    c = resolve(small_cfg("wgan-gp"))
    assert (c.lr_g, c.n_critic, c.betas) == (1e-4, 5, (0.0, 0.9))
    c = resolve(small_cfg("dcgan", ttur=True))
    assert (c.lr_g, c.lr_d) == (1e-4, 4e-4)
    assert resolve(small_cfg("dcgan", n_critic=3)).n_critic == 3
    assert get_spec("hinge").name == "sngan" and "hinge" in ALIASES
    with pytest.raises(ValueError):
        get_spec("stylegan")


def test_wgan_clips_every_critic_weight(small_cfg):
    data = _data_for("wgan")
    m = build_method(small_cfg("wgan"), data.info)
    m.d_step(next(iter(data.loader(16))))
    assert max(p.abs().max().item() for p in m.D.parameters()) <= 0.01 + 1e-7
    assert isinstance(m.opt_d, torch.optim.RMSprop)


def test_conditioning_wiring(small_cfg):
    info = _data_for("cgan").info
    cgan, acgan, dcgan = (build_method(small_cfg(n), info) for n in ("cgan", "acgan", "dcgan"))
    assert cgan.D.n_classes == 4 and cgan.D.aux is None       # cGAN: D is told the label
    assert acgan.D.n_classes == 0 and acgan.D.aux is not None  # ACGAN: D must predict it
    assert dcgan.n_classes == 0
    y = torch.tensor([2, 2, 2])
    assert cgan.sample(3, y=y).shape == (3, 1, 28, 28)


def test_conditional_methods_need_labels(small_cfg):
    with pytest.raises(ValueError):
        build_method(small_cfg("cgan"), load_data("paired-shapes", n_train=8).info)


@pytest.mark.parametrize("trick", [{"label_smoothing": 0.1}, {"r1": 1.0}, {"feature_matching": True},
                                   {"saturating": True}])
def test_tricks_run(trick, small_cfg):
    data = _data_for("dcgan")
    m = build_method(small_cfg("dcgan", **trick), data.info)
    batch = next(iter(data.loader(16)))
    assert torch.isfinite(torch.tensor(m.d_step(batch)["d_loss"]))
    assert torch.isfinite(torch.tensor(m.g_step(batch)["g_loss"]))
