import pytest
import torch

from gans.metrics import FeatureClassifier, frechet_distance, inception_score, iou, mode_coverage


def test_frechet_distance_known_values():
    torch.manual_seed(0)
    a = torch.randn(4000, 3)
    assert frechet_distance(a, a) == pytest.approx(0, abs=1e-6)
    # shift by 2 in every dim: ||mu diff||^2 = 3 * 4 = 12, covariances equal
    assert frechet_distance(a, a + 2) == pytest.approx(12, rel=1e-4)
    # N(0, I) vs N(0, 4I): Tr(I + 4I - 2 * 2I) = 3 * (1 + 4 - 4) = 3
    b = torch.randn(200_000, 3)
    assert frechet_distance(b, 2 * torch.randn(200_000, 3)) == pytest.approx(3, rel=0.03)


def test_inception_score_and_mode_coverage():
    eye = torch.eye(10).repeat(50, 1)               # confident and balanced: IS = 10, all modes
    assert inception_score(eye) == pytest.approx(10, rel=1e-4)
    assert mode_coverage(eye) == (10, pytest.approx(0, abs=1e-6))
    collapsed = torch.zeros(500, 10)
    collapsed[:, 3] = 1                             # mode collapse: IS = 1, one mode
    assert inception_score(collapsed) == pytest.approx(1, rel=1e-4)
    assert mode_coverage(collapsed)[0] == 1
    assert inception_score(torch.full((100, 10), 0.1)) == pytest.approx(1)   # unconfident


def test_iou_and_classifier_shapes():
    a = torch.tensor([[1.0, 1.0, -1.0, -1.0]])
    b = torch.tensor([[1.0, -1.0, -1.0, -1.0]])
    assert iou(a, b) == 0.5
    clf = FeatureClassifier(1, 4)
    f, p = clf.features_and_probs(torch.randn(7, 1, 28, 28))
    assert f.shape == (7, 128) and p.shape == (7, 4) and torch.allclose(p.sum(1), torch.ones(7))
