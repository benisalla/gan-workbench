import torch

from gans.models import (
    DCDiscriminator,
    DCGenerator,
    MLPDiscriminator,
    MLPGenerator,
    PatchDiscriminator,
    UNetGenerator,
)


def test_mlp_and_dcgan_shapes_with_and_without_labels():
    z, y = torch.randn(4, 16), torch.tensor([0, 1, 2, 3])
    for size in (28, 32):
        g = DCGenerator(16, 1, size, 8, n_classes=4)
        x = g(z, y)
        assert x.shape == (4, 1, size, size) and x.abs().max() <= 1
        d = DCDiscriminator(1, size, 8, n_classes=4, aux_classes=4)
        out = d(x, y)
        assert out.logit.shape == (4,) and out.class_logits.shape == (4, 4) and out.features.dim() == 2
    assert MLPGenerator(16, 1, 28, 32)(z).shape == (4, 1, 28, 28)
    assert MLPDiscriminator(1, 28, 32)(torch.randn(4, 1, 28, 28)).logit.shape == (4,)


def test_spectral_norm_bounds_the_largest_singular_value():
    d = DCDiscriminator(1, 28, 8, norm="spectral")
    for _ in range(20):                       # power iteration converges during forward passes
        d(torch.randn(8, 1, 28, 28))
    w = d.head.weight                         # the normalised weight used in the forward pass
    assert torch.linalg.matrix_norm(w, ord=2) <= 1.05
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in d.modules())


def test_unet_and_patchgan_shapes():
    g = UNetGenerator(1, 1, width=8, depth=4)
    x = torch.randn(2, 1, 32, 32)
    assert g(x).shape == (2, 1, 32, 32)
    d = PatchDiscriminator(1, cond_ch=1, width=8)
    assert d(x, x).logit.shape == (2, 1, 6, 6)     # 6x6 patch decisions at 32 px


def test_unet_decoder_receives_the_skips():
    """Every Up block outputs its own channels + the encoder skip's channels (concatenation)."""
    g = UNetGenerator(1, 1, width=8, depth=4)
    x = torch.randn(1, 1, 32, 32)
    skips = []
    h = x
    for down in g.downs:
        h = down(h)
        skips.append(h)
    h = skips.pop()
    for up in g.ups:
        skip = skips.pop()
        out = up(h, skip)
        assert torch.equal(out[:, -skip.shape[1]:], skip)      # the skip is passed through untouched
        h = out
