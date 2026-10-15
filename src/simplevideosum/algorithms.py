"""Key frame selection: which frame representation to use, and how to cluster the frames.

Every selection returns ``(indices, labels)``: the key frame indices, and for every frame the
position in ``indices`` of the key frame that represents it.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

import torch

from simplevideosum.aggregation import (
    AGGREGATIONS,
    DEFAULT_AGGREGATION,
    DEFAULT_NORMALISATION,
    NORMALISATIONS,
)
from simplevideosum.backbones import ALIAS_NAMES, DEFAULT_BACKBONE, validate_backbone
from simplevideosum.distances import DISTANCES
from simplevideosum.kmedoids import MAX_FRAMES, kmedoids
from simplevideosum.segmentation import dissect, fisher_segments, is_power_of_two

Selection = tuple[torch.Tensor, torch.Tensor]

MAX_FISHER_FRAMES = 10_000  # its N x N cost matrix is 400 MB in float32


def select_time(num_frames: int, n_key_frames: int) -> Selection:
    """Pick key frames evenly spaced in time; each frame is labelled by its nearest key frame."""
    if num_frames < 1 or n_key_frames < 1:
        raise ValueError("num_frames and n_key_frames must be >= 1")
    positions = torch.linspace(0, num_frames - 1, n_key_frames, dtype=torch.float64)
    indices = positions.round().long().unique()
    distances = (torch.arange(num_frames)[:, None] - indices[None, :]).abs()
    return indices, distances.argmin(dim=1)


def key_frames_of_segments(signatures: torch.Tensor, bounds: torch.Tensor) -> Selection:
    """The key frame of each segment ``[bounds[k], bounds[k + 1])`` is the one whose signature is
    closest to the segment mean. The result is on the signatures' device."""
    n_key_frames = len(bounds) - 1
    labels = torch.arange(n_key_frames, device=signatures.device).repeat_interleave(bounds.diff())
    indices = torch.empty(n_key_frames, dtype=torch.long, device=signatures.device)
    edges = bounds.tolist()
    for k, (start, end) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        segment = signatures[start:end]
        indices[k] = start + ((segment - segment.mean(dim=0)) ** 2).sum(dim=1).argmin()
    return indices, labels


def select_segments(
    signatures: torch.Tensor, n_key_frames: int, time_smoothing: float = 0.0, distance: str = "l2sq"
) -> Selection:
    """Cut the video into temporally contiguous segments by recursive bisection.

    The cost of a segment is its within-segment squared L2 distance, whatever ``distance`` says
    (the clustering is only offered with ``l2sq``).
    """
    signatures = torch.as_tensor(signatures, dtype=torch.float32)
    return key_frames_of_segments(signatures, dissect(signatures, n_key_frames))


def select_optimal_segments(
    signatures: torch.Tensor, n_key_frames: int, time_smoothing: float = 0.0, distance: str = "l2sq"
) -> Selection:
    """Cut the video into the temporally contiguous segments of least total cost (Fisher's
    exact segmentation); the cost is the same as for ``select_segments``."""
    signatures = torch.as_tensor(signatures, dtype=torch.float32)
    return key_frames_of_segments(signatures, fisher_segments(signatures, n_key_frames))


def select_kmedoids(
    signatures: torch.Tensor, n_key_frames: int, time_smoothing: float = 0.0, distance: str = "l2sq"
) -> Selection:
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
    max_frames: int | None = None  # None: no limit on the number of frames


# Clustering methods for the algorithms with signatures (--clustering).
CLUSTERINGS: dict[str, Clustering] = {
    "kmedoids": Clustering(select_kmedoids, supports_time_smoothing=True, max_frames=MAX_FRAMES),
    "dissection": Clustering(select_segments, validate=require_power_of_two, distances=("l2sq",)),
    "fisher": Clustering(
        select_optimal_segments, distances=("l2sq",), max_frames=MAX_FISHER_FRAMES
    ),
}
DEFAULT_CLUSTERING = "fisher"


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
    backbone: str = DEFAULT_BACKBONE
    aggregation: str = DEFAULT_AGGREGATION
    normalisation: str = DEFAULT_NORMALISATION
    distance: str = "l2sq"
    clustering: str = DEFAULT_CLUSTERING

    @property
    def uses_signatures(self) -> bool:
        return self.select is None


# Choices of every component that can be overridden from the command line (--<name>).
COMPONENTS: dict[str, tuple[str, ...]] = {
    "backbone": ALIAS_NAMES,  # and the weights of any torchvision classification model
    "aggregation": tuple(AGGREGATIONS),
    "normalisation": tuple(NORMALISATIONS),
    "distance": DISTANCES,
    "clustering": tuple(CLUSTERINGS),
}

# The components of the signatures and their comparison, which have an option of their own.
PIPELINE_COMPONENTS = tuple(name for name in COMPONENTS if name != "clustering")

# Algorithms available from the command line (-a): named configurations of the components.
DEFAULT_ALGORITHM = "default"
ALGORITHMS: dict[str, Algorithm] = {
    # the default components: temporally contiguous segments of least total cost of the average
    # and max pooled activations of MobileNetV3-Large, L2-normalised
    DEFAULT_ALGORITHM: Algorithm(),
    # evenly spaced frames, no features
    "time": Algorithm(select=select_time),
    # videosum "inception": globally pooled InceptionV3 activations, k-medoids on squared L2
    # distances
    "videosum_inception": Algorithm(
        backbone="inceptionv3", aggregation="avgpool", normalisation="none", clustering="kmedoids"
    ),
    # videosum "scda": SCDA (Wei et al., 2017) of InceptionV3, k-medoids on cosine distances.
    # Unlike the paper, the holes of the selection mask are filled and the pooled descriptors are
    # not L2-normalised (the paper's version is --aggregation scda --normalisation l2)
    "videosum_scda": Algorithm(
        backbone="inceptionv3",
        aggregation="scda_fill",
        normalisation="none",
        distance="cosine",
        clustering="kmedoids",
    ),
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
        if name == "backbone" and value is not None:
            validate_backbone(value)
        elif value is not None and value not in COMPONENTS[name]:
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
