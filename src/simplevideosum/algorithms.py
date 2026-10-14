"""Key frame selection algorithms.

Each algorithm maps the number of (sampled) frames of a video and the number of requested
key frames to ``(indices, labels)``: the key frame indices, and for every frame the position
in ``indices`` of the key frame it is represented by.
"""

from collections.abc import Callable

import numpy as np

SelectFn = Callable[[int, int], tuple[np.ndarray, np.ndarray]]


def select_time(num_frames: int, n_key_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Pick key frames evenly spaced in time; each frame is labelled by its nearest key frame."""
    if num_frames < 1 or n_key_frames < 1:
        raise ValueError("num_frames and n_key_frames must be >= 1")
    indices = np.unique(np.round(np.linspace(0, num_frames - 1, n_key_frames)).astype(int))
    distances = np.abs(np.arange(num_frames)[:, None] - indices[None, :])
    return indices, distances.argmin(axis=1)


# Algorithms available from the command line.
ALGORITHMS: dict[str, SelectFn] = {"time": select_time}

# Algorithms that inspect every frame and therefore honour the sampling frequency (--fps).
# The others select key frames from the video length alone and ignore it.
FPS_ALGORITHMS: frozenset[str] = frozenset()

# Algorithms of the original videosum package that are not implemented here yet.
PLANNED_ALGORITHMS = ("frechet",)
