import pytest
import torch

from gans.data import load_data, make_filled_and_outline, make_paired_shapes, make_shapes


def test_shapes_range_labels_and_determinism():
    x, y = make_shapes(32, 28, seed=3)
    assert x.shape == (32, 1, 28, 28) and x.min() == -1 and x.max() == 1
    assert set(y.tolist()) <= {0, 1, 2, 3}
    x2, y2 = make_shapes(32, 28, seed=3)
    assert torch.equal(x, x2) and torch.equal(y, y2)


def test_paired_mask_matches_bright_shapes():
    photo, mask = make_paired_shapes(16, 32, seed=0)
    assert photo.shape == mask.shape == (16, 1, 32, 32)
    assert set(mask.unique().tolist()) == {-1.0, 1.0}
    # pixels inside the mask are brighter on average than the background
    assert photo[mask > 0].mean() > photo[mask < 0].mean() + 0.5


def test_outline_is_inside_the_filled_shape_and_thinner():
    a, b = make_filled_and_outline(16, 32, seed=0)
    assert ((b > 0) & ~(a > 0)).sum() == 0
    assert (b > 0).sum() < 0.7 * (a > 0).sum()


def test_load_data_kinds_and_subset():
    d = load_data("shapes", n_train=100)
    assert d.info.kind == "labeled" and d.info.n_classes == 4 and len(d.train[0]) == 100
    assert load_data("paired-shapes", n_train=50).info.kind == "paired"
    u = load_data("unpaired-shapes", n_train=50)
    assert u.info.kind == "unpaired" and len(u.test[0]) == len(u.test[1])
    x, y = next(iter(d.loader(16)))
    assert x.shape == (16, 1, 28, 28) and y.shape == (16,)
    with pytest.raises(ValueError):
        load_data("imagenet")
