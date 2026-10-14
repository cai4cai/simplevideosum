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
    assert ALGORITHMS["videosum_inception"].uses_signatures

    default = ALGORITHMS["videosum_inception"]
    assert (default.backbone, default.aggregation, default.normalisation, default.distance) == (
        "inceptionv3", "avgpool", "none", "l2sq")  # fmt: skip
    assert default.clustering == DEFAULT_CLUSTERING


def test_videosum_scda_preset():
    preset = ALGORITHMS["videosum_scda"]
    assert (preset.aggregation, preset.normalisation, preset.distance) == ("scda", "none", "cosine")
    assert (preset.backbone, preset.clustering) == ("inceptionv3", "kmedoids")
    assert "scda" not in ALGORITHMS and "inception" not in ALGORITHMS


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
    algorithm, clustering = resolve("videosum_inception", None, 12)
    assert algorithm.uses_signatures and clustering is CLUSTERINGS["kmedoids"]
    assert resolve("videosum_inception", "dissection", 16)[1] is CLUSTERINGS["dissection"]


def test_resolve_applies_component_overrides():
    algorithm, clustering = resolve(
        "videosum_scda", "dissection", 16, normalisation="l2", distance="l2sq"
    )
    assert (algorithm.aggregation, algorithm.normalisation) == ("scda", "l2")
    assert (algorithm.distance, algorithm.clustering) == ("l2sq", "dissection")
    assert clustering is CLUSTERINGS["dissection"]
    assert (
        resolve("videosum_scda", None, 8, distance=None)[0] == ALGORITHMS["videosum_scda"]
    )  # None keeps preset


def test_resolve_ignores_the_clustering_for_the_time_algorithm():
    algorithm, clustering = resolve("time", "dissection", 12)
    assert not algorithm.uses_signatures and clustering is None


@pytest.mark.parametrize(
    "args, message",
    [
        (("bogus", None, 16), "unknown algorithm"),
        (("videosum_inception", "bogus", 16), "unknown clustering"),
        (("videosum_inception", "dissection", 12), "power of 2"),
        (("videosum_inception", "dissection", 16, 0.5), "cannot use time smoothing"),
        (("time", None, 16, 0.5), "cannot use time smoothing"),
        (("videosum_inception", None, 16, 1.5), "time smoothing must be in"),
        (("videosum_scda", "dissection", 16), "only supports the l2sq distance"),
        (("videosum_inception", None, 16, 0.0, "backbone", "bogus"), "unknown backbone"),
    ],
)
def test_resolve_rejects_invalid_combinations(args, message):
    if len(args) == 6:  # a component override: (*positional, name, value)
        *args, name, value = args
        kwargs = {name: value}
    else:
        kwargs = {}
    with pytest.raises(ValueError, match=message):
        resolve(*args, **kwargs)


def test_resolve_rejects_unknown_component_names():
    with pytest.raises(TypeError):
        resolve("videosum_inception", None, 16, colour="red")


def test_resolve_accepts_any_number_of_frames_for_kmedoids():
    resolve("videosum_inception", "kmedoids", 12, 0.3)


def test_kmedoids_selection_matches_segments_on_obvious_scenes():
    centres = [(0, 0), (10, 0), (100, 0), (110, 5)]
    signatures = piecewise_signatures([10, 30, 20, 40], centres)
    indices, labels = select_kmedoids(signatures, 4)
    assert np.bincount(labels).tolist() == [10, 30, 20, 40]
    assert [labels[i] for i in indices] == [0, 1, 2, 3]
