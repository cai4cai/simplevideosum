import numpy as np
import pytest

from simplevideosum.algorithms import ALGORITHMS, select_time


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
    assert ALGORITHMS["time"] is select_time
