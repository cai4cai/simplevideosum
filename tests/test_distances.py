import numpy as np
import pytest
import torch

from simplevideosum.device import get_device
from simplevideosum.distances import distance_matrix


def test_distance_matrix_is_normalised_squared_l2():
    x = np.random.default_rng(0).normal(size=(12, 5))
    dist = distance_matrix(x)
    expected = ((x[:, None] - x[None, :]) ** 2).sum(-1)
    assert dist.max() == pytest.approx(1.0)
    assert dist.numpy() == pytest.approx(expected / expected.max(), abs=1e-5)
    assert torch.equal(dist, dist.T) or torch.allclose(dist, dist.T, atol=1e-6)
    assert dist.diagonal().abs().max() == 0


def test_time_smoothing_blends_in_the_temporal_distance():
    x = np.random.default_rng(1).normal(size=(6, 3))
    temporal = (np.arange(6)[:, None] - np.arange(6)[None, :]).__abs__() / 5
    assert distance_matrix(x, 1.0).numpy() == pytest.approx(temporal, abs=1e-6)
    half = 0.5 * distance_matrix(x).numpy() + 0.5 * temporal
    assert distance_matrix(x, 0.5).numpy() == pytest.approx(half, abs=1e-6)
    with pytest.raises(ValueError):
        distance_matrix(x, 1.5)


def test_cosine_distance_ignores_the_norm_and_is_normalised():
    x = np.array([[1.0, 0.0], [5.0, 0.0], [0.0, 2.0], [-3.0, 0.0]])
    dist = distance_matrix(x, metric="cosine").numpy()
    assert dist[0, 1] == pytest.approx(0, abs=1e-6)  # same direction
    assert dist[0, 2] == pytest.approx(0.5, abs=1e-6)  # orthogonal: 1 of the peak 2 (opposite)
    assert dist.max() == pytest.approx(1.0) and dist[0, 3] == pytest.approx(1.0)
    assert np.allclose(dist, dist.T) and np.all(np.diag(dist) == 0)


def test_unknown_metric():
    with pytest.raises(ValueError, match="unknown distance"):
        distance_matrix(np.zeros((3, 2)), metric="bogus")


@pytest.mark.parametrize("metric", ["l2sq", "cosine"])
def test_tensors_and_arrays_give_the_same_matrix(metric):
    x = np.random.default_rng(3).normal(size=(10, 4)).astype(np.float32)
    from_array = distance_matrix(x, 0.3, metric)
    assert from_array.dtype == torch.float32
    assert torch.equal(from_array, distance_matrix(torch.from_numpy(x), 0.3, metric))
    assert torch.allclose(from_array, distance_matrix(x.astype(np.float64), 0.3, metric))


@pytest.mark.parametrize("metric", ["l2sq", "cosine"])
def test_matrix_stays_on_the_device_of_the_signatures(metric):
    device = get_device()
    x = torch.randn(10, 4)
    on_device = distance_matrix(x.to(device), 0.3, metric)
    assert on_device.device.type == device.type
    assert torch.allclose(on_device.cpu(), distance_matrix(x, 0.3, metric), atol=1e-5)
