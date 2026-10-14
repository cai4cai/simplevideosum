import itertools

import numpy as np
import pytest
import torch

from simplevideosum.distances import distance_matrix
from simplevideosum.kmedoids import kmedoids, kmedoids_pp_init, pam


def blobs(sizes, dim=4, spread=0.05, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(len(sizes), dim)) * 10
    x = np.concatenate(
        [c + spread * rng.normal(size=(n, dim)) for c, n in zip(centres, sizes, strict=True)]
    )
    return x, np.repeat(np.arange(len(sizes)), sizes)


def cost(dist, medoids):
    return float(dist[:, list(medoids)].min(dim=1).values.sum())


def test_init_returns_distinct_medoids_and_is_deterministic():
    dist = distance_matrix(blobs([20, 20, 20, 20])[0])
    first = kmedoids_pp_init(dist, 4, np.random.default_rng(0))
    assert len(set(first.tolist())) == 4
    assert first.tolist() == kmedoids_pp_init(dist, 4, np.random.default_rng(0)).tolist()


def test_init_handles_duplicate_frames():
    dist = distance_matrix(np.zeros((6, 3)))
    assert len(set(kmedoids_pp_init(dist, 3, np.random.default_rng(0)).tolist())) == 3


@pytest.mark.parametrize("seed", range(5))
def test_pam_result_is_a_local_optimum_of_the_swap_neighbourhood(seed):
    x = np.random.default_rng(seed).normal(size=(30, 4))
    dist = distance_matrix(x)
    medoids = pam(dist, kmedoids_pp_init(dist, 4, np.random.default_rng(seed)))
    base = cost(dist, medoids)
    for i, h in itertools.product(range(4), set(range(30)) - set(medoids.tolist())):
        swapped = medoids.copy()
        swapped[i] = h
        assert cost(dist, swapped) >= base - 1e-4


def test_pam_never_increases_the_cost_and_chunking_does_not_change_it():
    dist = distance_matrix(np.random.default_rng(7).normal(size=(40, 6)))
    init = kmedoids_pp_init(dist, 5, np.random.default_rng(7))
    refined = pam(dist, init)
    assert cost(dist, refined) <= cost(dist, init) + 1e-6
    assert sorted(pam(dist, init, chunk=7).tolist()) == sorted(refined.tolist())


def test_pam_with_a_single_medoid_finds_the_medoid():
    dist = distance_matrix(np.random.default_rng(8).normal(size=(25, 3)))
    (medoid,) = pam(dist, np.array([0]))
    assert cost(dist, [medoid]) == pytest.approx(float(dist.sum(dim=1).min()), rel=1e-5)


def test_kmedoids_recovers_well_separated_clusters():
    x, truth = blobs([30, 10, 40, 20])
    indices, labels = kmedoids(x, 4)
    assert indices.tolist() == sorted(indices.tolist())
    assert len(set(truth[indices].tolist())) == 4  # one medoid per blob
    assert [labels[i] for i in indices] == [0, 1, 2, 3]  # label k belongs to medoid k
    # frames of a blob share a label
    assert all(len(set(labels[truth == b].tolist())) == 1 for b in range(4))


def test_kmedoids_is_deterministic_per_seed():
    x = np.random.default_rng(3).normal(size=(50, 8))
    assert kmedoids(x, 4)[0].tolist() == kmedoids(x, 4)[0].tolist()


def test_full_time_smoothing_gives_contiguous_clusters():
    x = np.random.default_rng(4).normal(size=(60, 8))
    _, labels = kmedoids(x, 5, time_smoothing=1.0)
    assert np.all(np.diff(labels) >= 0)
    _, unsmoothed = kmedoids(x, 5)
    assert np.any(np.diff(unsmoothed) < 0)  # random signatures interleave without smoothing


def test_fewer_frames_than_clusters_keeps_every_frame():
    indices, labels = kmedoids(np.random.default_rng(5).normal(size=(3, 4)), 8)
    assert indices.tolist() == [0, 1, 2] and labels.tolist() == [0, 1, 2]
    with pytest.raises(ValueError):
        kmedoids(np.zeros((3, 4)), 0)


def test_unknown_metric():
    with pytest.raises(ValueError, match="unknown distance"):
        kmedoids(np.zeros((9, 2)), 3, metric="bogus")


def test_clustering_runs_on_the_device_of_the_signatures():
    from simplevideosum.device import get_device

    x, truth = blobs([15, 15, 15])
    indices, labels = kmedoids(torch.as_tensor(x, dtype=torch.float32, device=get_device()), 3)
    assert isinstance(indices, np.ndarray) and isinstance(labels, np.ndarray)
    assert sorted(set(labels[indices])) == [0, 1, 2]
    assert [len(set(truth[labels == k])) for k in range(3)] == [1, 1, 1]  # pure clusters
