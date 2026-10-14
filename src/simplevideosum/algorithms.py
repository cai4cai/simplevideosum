"""Key frame selection: which frame representation to use, and how to cluster the frames.

Every selection returns ``(indices, labels)``: the key frame indices, and for every frame the
position in ``indices`` of the key frame that represents it.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import torch

from simplevideosum.aggregation import AGGREGATIONS, NORMALISATIONS
from simplevideosum.distances import DISTANCES
from simplevideosum.features import BACKBONES
from simplevideosum.kmedoids import kmedoids
from simplevideosum.segmentation import dissect, is_power_of_two

Selection = tuple[np.ndarray, np.ndarray]


def select_time(num_frames: int, n_key_frames: int) -> Selection:
    """Pick key frames evenly spaced in time; each frame is labelled by its nearest key frame."""
    if num_frames < 1 or n_key_frames < 1:
        raise ValueError("num_frames and n_key_frames must be >= 1")
    indices = np.unique(np.round(np.linspace(0, num_frames - 1, n_key_frames)).astype(int))
    distances = np.abs(np.arange(num_frames)[:, None] - indices[None, :])
    return indices, distances.argmin(axis=1)


def select_segments(
    signatures: torch.Tensor, n_key_frames: int, time_smoothing: float = 0.0, distance: str = "l2sq"
):
    """Cut the video into temporally contiguous segments of similar signatures.

    The key frame of each segment is the one whose signature is closest to the segment mean.
    The cost of a segment is its within-segment squared L2 distance, whatever ``distance`` says
    (the clustering is only offered with ``l2sq``).
    """
    signatures = torch.as_tensor(signatures, dtype=torch.float32).cpu().numpy()
    bounds = dissect(signatures, n_key_frames)
    labels = np.repeat(np.arange(n_key_frames), np.diff(bounds))
    indices = np.empty(n_key_frames, dtype=int)
    for k, (start, end) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        segment = signatures[start:end]
        indices[k] = start + np.argmin(((segment - segment.mean(axis=0)) ** 2).sum(axis=1))
    return indices, labels


def select_kmedoids(
    signatures: torch.Tensor, n_key_frames: int, time_smoothing: float = 0.0, distance: str = "l2sq"
):
    """Cluster the signatures with k-medoids; the key frames are the medoids."""
    return kmedoids(signatures, n_key_frames, time_smoothing, metric=distance)


def require_power_of_two(n_key_frames: int) -> None:
    if not is_power_of_two(n_key_frames):
        raise ValueError(f"this clustering needs a power of 2 number of frames, got {n_key_frames}")


@dataclass(frozen=True)
class Clustering:
    """How the signatures are grouped into key frames:
    ``select(signatures, n, time_smoothing, distance)``."""

    select: Callable[..., Selection]
    validate: Callable[[int], None] = lambda n_key_frames: None
    supports_time_smoothing: bool = False
    distances: tuple[str, ...] = DISTANCES


# Clustering methods for the algorithms with signatures (--clustering).
CLUSTERINGS: dict[str, Clustering] = {
    "kmedoids": Clustering(select_kmedoids, supports_time_smoothing=True),
    "dissection": Clustering(select_segments, validate=require_power_of_two, distances=("l2sq",)),
}
DEFAULT_CLUSTERING = "kmedoids"


@dataclass(frozen=True)
class Algorithm:
    """A named configuration of the components of frame selection.

    Algorithms with signatures inspect the video resampled at ``--fps``. A frame goes through the
    ``backbone`` (convolutional activations), the ``aggregation`` (pooled descriptors) and the
    ``normalisation`` of each of them, which gives its signature; signatures are compared with
    the ``distance`` and grouped by the ``clustering`` method. Each component can be overridden
    on the command line. The others select from the video length alone (``select``) and ignore
    all of this.
    """

    select: Callable[[int, int], Selection] | None = None
    backbone: str = "inceptionv3"
    aggregation: str = "avgpool"
    normalisation: str = "none"
    distance: str = "l2sq"
    clustering: str = DEFAULT_CLUSTERING

    @property
    def uses_signatures(self) -> bool:
        return self.select is None


# Choices of every component that can be overridden from the command line (--<name>).
COMPONENTS: dict[str, tuple[str, ...]] = {
    "backbone": tuple(BACKBONES),
    "aggregation": tuple(AGGREGATIONS),
    "normalisation": tuple(NORMALISATIONS),
    "distance": DISTANCES,
    "clustering": tuple(CLUSTERINGS),
}

# Algorithms available from the command line (-a): named configurations of the components.
ALGORITHMS: dict[str, Algorithm] = {
    # evenly spaced frames, no features
    "time": Algorithm(select=select_time),
    # videosum "inception": globally pooled activations, k-medoids on squared L2 distances
    "videosum_inception": Algorithm(),
    # videosum "scda": SCDA (Wei et al., 2017), k-medoids on cosine distances. Unlike the paper,
    # the holes of the selection mask are filled and the pooled descriptors are not L2-normalised
    # (the paper's version is --aggregation scda --normalisation l2)
    "videosum_scda": Algorithm(aggregation="scda_fill", distance="cosine"),
}


def resolve(
    algo: str,
    clustering: str | None,
    n_key_frames: int,
    time_smoothing: float = 0.0,
    **components: str | None,
) -> tuple[Algorithm, Clustering | None]:
    """Look up an algorithm, apply the component overrides and validate it (raises ValueError).

    ``clustering`` and the keyword ``components`` (``backbone``, ``aggregation``,
    ``normalisation``, ``distance``) replace those of the algorithm unless None. Returns the
    algorithm with its final components, and its clustering method (None for algorithms without
    signatures, which ignore all the components).
    """
    if algo not in ALGORITHMS:
        raise ValueError(f"unknown algorithm '{algo}' (available: {', '.join(ALGORITHMS)})")
    overrides = {"clustering": clustering, **components}
    for name, value in overrides.items():
        if name not in COMPONENTS:
            raise TypeError(f"unknown component '{name}'")
        if value is not None and value not in COMPONENTS[name]:
            raise ValueError(f"unknown {name} '{value}' (available: {', '.join(COMPONENTS[name])})")
    if not 0.0 <= time_smoothing <= 1.0:
        raise ValueError("time smoothing must be in [0, 1]")
    algorithm = ALGORITHMS[algo]
    if not algorithm.uses_signatures:
        if time_smoothing > 0:
            raise ValueError(f"the '{algo}' algorithm cannot use time smoothing")
        return algorithm, None
    algorithm = replace(algorithm, **{k: v for k, v in overrides.items() if v is not None})
    method = CLUSTERINGS[algorithm.clustering]
    if time_smoothing > 0 and not method.supports_time_smoothing:
        raise ValueError(f"the '{algorithm.clustering}' clustering cannot use time smoothing")
    if algorithm.distance not in method.distances:
        raise ValueError(
            f"the '{algorithm.clustering}' clustering only supports the "
            f"{', '.join(method.distances)} distance, not '{algorithm.distance}'"
        )
    method.validate(n_key_frames)
    return algorithm, method
