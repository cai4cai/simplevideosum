import itertools

import numpy as np
import pytest
import torch

from simplevideosum.device import get_device
from simplevideosum.distances import distance_matrix
from simplevideosum.kmedoids import kmedoids, kmedoids_pp_init, pam


def blobs(sizes, dim=4, spread=0.05, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(len(sizes), dim)) * 10
    x = np.concatenate(
        [c + spread * rng.normal(size=(n, dim)) for c, n in zip(centres, sizes, strict=True)]
    )
    return x, np.repeat(np.arange(len(sizes)), sizes)


def generator(seed):
    return torch.Generator().manual_seed(seed)


def cluster(x, k, **kwargs):
    """kmedoids as NumPy arrays."""
    return tuple(t.cpu().numpy() for t in kmedoids(x, k, **kwargs))


def cost(dist, medoids):
    return float(dist[:, list(medoids)].min(dim=1).values.sum())


def test_init_returns_distinct_medoids_and_is_deterministic():
    dist = distance_matrix(blobs([20, 20, 20, 20])[0])
    first = kmedoids_pp_init(dist, 4, generator(0))
    assert len(set(first.tolist())) == 4
    assert first.tolist() == kmedoids_pp_init(dist, 4, generator(0)).tolist()


def test_init_is_on_the_device_of_the_distances():
    device = get_device()
    dist = distance_matrix(blobs([10, 10, 10])[0]).to(device)
    first = kmedoids_pp_init(dist, 3, torch.Generator(device=device).manual_seed(0))
    assert first.device.type == device.type and len(set(first.tolist())) == 3


def test_init_handles_duplicate_frames():
    dist = distance_matrix(np.zeros((6, 3)))
    assert len(set(kmedoids_pp_init(dist, 3, generator(0)).tolist())) == 3


@pytest.mark.parametrize("seed", range(5))
def test_pam_result_is_a_local_optimum_of_the_swap_neighbourhood(seed):
    x = np.random.default_rng(seed).normal(size=(30, 4))
    dist = distance_matrix(x)
    medoids = pam(dist, kmedoids_pp_init(dist, 4, generator(seed)))
    base = cost(dist, medoids)
    for i, h in itertools.product(range(4), set(range(30)) - set(medoids.tolist())):
        swapped = medoids.clone()
        swapped[i] = h
        assert cost(dist, swapped) >= base - 1e-4


def test_pam_never_increases_the_cost_and_chunking_does_not_change_it():
    dist = distance_matrix(np.random.default_rng(7).normal(size=(40, 6)))
    init = kmedoids_pp_init(dist, 5, generator(7))
    refined = pam(dist, init)
    assert cost(dist, refined) <= cost(dist, init) + 1e-6
    assert sorted(pam(dist, init, chunk=7).tolist()) == sorted(refined.tolist())


def test_pam_with_a_single_medoid_finds_the_medoid():
    dist = distance_matrix(np.random.default_rng(8).normal(size=(25, 3)))
    (medoid,) = pam(dist, torch.tensor([0]))
    assert cost(dist, [int(medoid)]) == pytest.approx(float(dist.sum(dim=1).min()), rel=1e-5)


def test_kmedoids_recovers_well_separated_clusters():
    x, truth = blobs([30, 10, 40, 20])
    indices, labels = cluster(x, 4)
    assert indices.tolist() == sorted(indices.tolist())
    assert len(set(truth[indices].tolist())) == 4  # one medoid per blob
    assert [labels[i] for i in indices] == [0, 1, 2, 3]  # label k belongs to medoid k
    # frames of a blob share a label
    assert all(len(set(labels[truth == b].tolist())) == 1 for b in range(4))


def test_kmedoids_is_deterministic_per_seed():
    x = np.random.default_rng(3).normal(size=(50, 8))
    assert cluster(x, 4)[0].tolist() == cluster(x, 4)[0].tolist()


def test_full_time_smoothing_gives_contiguous_clusters():
    x = np.random.default_rng(4).normal(size=(60, 8))
    _, labels = cluster(x, 5, time_smoothing=1.0)
    assert np.all(np.diff(labels) >= 0)
    _, unsmoothed = cluster(x, 5)
    assert np.any(np.diff(unsmoothed) < 0)  # random signatures interleave without smoothing


def test_fewer_frames_than_clusters_keeps_every_frame():
    indices, labels = cluster(np.random.default_rng(5).normal(size=(3, 4)), 8)
    assert indices.tolist() == [0, 1, 2] and labels.tolist() == [0, 1, 2]
    with pytest.raises(ValueError):
        kmedoids(np.zeros((3, 4)), 0)


def test_unknown_metric():
    with pytest.raises(ValueError, match="unknown distance"):
        kmedoids(np.zeros((9, 2)), 3, metric="bogus")


def test_clustering_runs_on_the_device_of_the_signatures():
    device = get_device()
    x, truth = blobs([15, 15, 15])
    indices, labels = kmedoids(torch.as_tensor(x, dtype=torch.float32, device=device), 3)
    assert indices.device.type == labels.device.type == device.type
    indices, labels = indices.cpu().numpy(), labels.cpu().numpy()
    assert sorted(set(labels[indices])) == [0, 1, 2]
    assert [len(set(truth[labels == k])) for k in range(3)] == [1, 1, 1]  # pure clusters


def test_identical_frames_do_not_steal_the_label_of_their_medoid():
    x = np.repeat(np.array([[0.0, 0.0], [10.0, 0.0]]), 4, axis=0)  # two distinct signatures
    indices, labels = cluster(x, 3)  # k exceeds the number of distinct signatures
    assert labels[indices].tolist() == [0, 1, 2]
    assert sorted(set(labels.tolist())) == [0, 1, 2]


def test_too_many_frames_are_refused_before_allocating_the_distances():
    with pytest.raises(ValueError, match="more than the 10 supported"):
        kmedoids(np.zeros((11, 2)), 3, max_frames=10)
    kmedoids(np.random.default_rng(0).normal(size=(10, 2)), 3, max_frames=10)
