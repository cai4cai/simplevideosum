import numpy as np
import pytest
import torch

from simplevideosum.segmentation import CumulativeStats, dissect, is_power_of_two


def brute_force_cost(x, start, end):
    segment = x[start:end]
    return ((segment - segment.mean(axis=0)) ** 2).sum()


def test_cumulative_cost_matches_brute_force():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 6)) + 100  # large offset: checks numerical conditioning
    stats = CumulativeStats.from_signatures(x)
    for start, end in [(0, 50), (3, 4), (0, 1), (10, 37), (49, 50)]:
        assert float(stats.cost(start, end)) == pytest.approx(
            brute_force_cost(x, start, end), rel=1e-4, abs=1e-3
        )


def test_cost_is_vectorised_over_segments():
    x = np.random.default_rng(1).normal(size=(30, 4))
    stats = CumulativeStats.from_signatures(x)
    ends = torch.arange(1, 31)
    expected = [brute_force_cost(x, 0, e) for e in ends.tolist()]
    assert stats.cost(0, ends).tolist() == pytest.approx(expected, rel=1e-4, abs=1e-3)


def test_cost_dtype_follows_requested_precision():
    x = np.random.default_rng(5).normal(size=(10, 3))
    assert CumulativeStats.from_signatures(x).cost(0, 10).dtype == torch.float32
    assert CumulativeStats.from_signatures(x, torch.float64).cost(0, 10).dtype == torch.float64


def test_float32_is_accurate_enough_on_a_long_noisy_video():
    rng = np.random.default_rng(6)
    x = np.concatenate(
        [rng.normal(size=(500, 128)) * 0.3 + rng.normal(size=128) * 3 for _ in range(8)]
    )

    def objective(bounds):  # evaluated in float64
        stats = CumulativeStats.from_signatures(x, torch.float64)
        return float(stats.cost(bounds[:-1], bounds[1:]).sum())

    b32, b64 = dissect(x, 16), dissect(x, 16, dtype=torch.float64)
    assert objective(b32) == pytest.approx(objective(b64), rel=1e-4)


def test_dissect_matches_exhaustive_search_for_two_segments():
    x = np.random.default_rng(2).normal(size=(40, 3))
    stats_costs = [brute_force_cost(x, 0, m) + brute_force_cost(x, m, 40) for m in range(1, 40)]
    assert dissect(x, 2).tolist() == [0, 1 + int(np.argmin(stats_costs)), 40]


def test_dissect_boundaries_cover_the_video_with_non_empty_segments():
    x = np.random.default_rng(3).normal(size=(37, 5))
    bounds = dissect(x, 8)
    assert bounds[0] == 0 and bounds[-1] == 37 and len(bounds) == 9
    assert bool((bounds.diff() >= 1).all())


def test_dissect_with_as_many_segments_as_frames():
    assert dissect(np.random.default_rng(4).normal(size=(8, 2)), 8).tolist() == list(range(9))


def test_dissect_validation():
    x = np.zeros((10, 2))
    with pytest.raises(ValueError, match="power of 2"):
        dissect(x, 3)
    with pytest.raises(ValueError, match="cannot split"):
        dissect(x, 16)


def test_is_power_of_two():
    assert [n for n in range(0, 20) if is_power_of_two(n)] == [1, 2, 4, 8, 16]


def test_dissect_runs_on_the_device_of_the_signatures():
    from simplevideosum.device import get_device

    device = get_device()
    x = torch.randn(40, 3)
    on_device = dissect(x.to(device), 4)
    assert on_device.device.type == device.type
    assert on_device.tolist() == dissect(x, 4).tolist()
