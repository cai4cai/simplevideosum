import numpy as np
import pytest

from simplevideosum.algorithms import (
    ALGORITHMS,
    CLUSTERINGS,
    DEFAULT_CLUSTERING,
    resolve,
    select_kmedoids,
    select_segments,
    select_time,
)


def test_time_indices_are_evenly_spaced():
    indices, labels = select_time(101, 5)
    assert indices.tolist() == [0, 25, 50, 75, 100]
    assert len(labels) == 101


def test_time_labels_point_to_nearest_key_frame():
    indices, labels = select_time(101, 5)
    assert labels[0] == 0 and labels[100] == 4
    assert np.all(np.abs(np.arange(101) - indices[labels]) <= 12)


def test_time_with_fewer_frames_than_requested_deduplicates():
    indices, labels = select_time(3, 10)
    assert indices.tolist() == [0, 1, 2]
    assert labels.tolist() == [0, 1, 2]


def test_time_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        select_time(0, 4)
    with pytest.raises(ValueError):
        select_time(10, 0)


def test_registry_contains_time():
    assert ALGORITHMS["time"].select is select_time
    assert not ALGORITHMS["time"].uses_signatures
    assert ALGORITHMS["inception"].uses_signatures


def piecewise_signatures(lengths, centres, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    parts = [
        np.array(c) + noise * rng.normal(size=(n, len(c)))
        for c, n in zip(centres, lengths, strict=True)
    ]
    return np.concatenate(parts)


def test_segments_recover_obvious_scenes():
    # Two far apart groups of two scenes each, so that the recursive bisection can recover them
    centres = [(0, 0), (10, 0), (100, 0), (110, 5)]
    signatures = piecewise_signatures([10, 30, 20, 40], centres)
    indices, labels = select_segments(signatures, 4)
    assert np.bincount(labels).tolist() == [10, 30, 20, 40]
    assert np.all(np.diff(labels) >= 0)  # temporally contiguous
    assert indices.tolist() == sorted(indices.tolist())
    assert [labels[i] for i in indices] == [0, 1, 2, 3]  # key frame k lies in segment k


def test_segments_key_frame_is_closest_to_segment_mean():
    signatures = np.array([[0.0], [1.0], [10.0], [11.0], [12.0], [20.0], [30.0], [31.0]])
    indices, _ = select_segments(signatures, 2)
    for k, (start, end) in enumerate([(0, 5), (5, 8)]):
        seg = signatures[start:end, 0]
        assert indices[k] == start + np.argmin(np.abs(seg - seg.mean()))


def test_clustering_registry():
    assert CLUSTERINGS["kmedoids"].select is select_kmedoids
    assert CLUSTERINGS["dissection"].select is select_segments
    assert DEFAULT_CLUSTERING == "kmedoids"


def test_resolve_picks_the_default_clustering_for_signature_algorithms():
    algorithm, clustering = resolve("inception", None, 12)
    assert algorithm.uses_signatures and clustering is CLUSTERINGS["kmedoids"]
    assert resolve("inception", "dissection", 16)[1] is CLUSTERINGS["dissection"]


def test_resolve_ignores_the_clustering_for_the_time_algorithm():
    algorithm, clustering = resolve("time", "dissection", 12)
    assert not algorithm.uses_signatures and clustering is None


@pytest.mark.parametrize(
    "args, message",
    [
        (("frechet", None, 16), "not implemented"),
        (("bogus", None, 16), "unknown algorithm"),
        (("inception", "bogus", 16), "unknown clustering"),
        (("inception", "dissection", 12), "power of 2"),
        (("inception", "dissection", 16, 0.5), "cannot use time smoothing"),
        (("time", None, 16, 0.5), "cannot use time smoothing"),
        (("inception", None, 16, 1.5), "time smoothing must be in"),
    ],
)
def test_resolve_rejects_invalid_combinations(args, message):
    with pytest.raises(ValueError, match=message):
        resolve(*args)


def test_resolve_accepts_any_number_of_frames_for_kmedoids():
    resolve("inception", "kmedoids", 12, 0.3)


def test_kmedoids_selection_matches_segments_on_obvious_scenes():
    centres = [(0, 0), (10, 0), (100, 0), (110, 5)]
    signatures = piecewise_signatures([10, 30, 20, 40], centres)
    indices, labels = select_kmedoids(signatures, 4)
    assert np.bincount(labels).tolist() == [10, 30, 20, 40]
    assert [labels[i] for i in indices] == [0, 1, 2, 3]
