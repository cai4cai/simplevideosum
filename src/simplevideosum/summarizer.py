"""Video to storyboard summarisation."""

import logging
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from simplevideosum.algorithms import resolve
from simplevideosum.collage import Layout, make_collage, make_segbar
from simplevideosum.features import build_feature_extractor, compute_signatures
from simplevideosum.io import VideoReader

logger = logging.getLogger(__name__)


class VideoSummarizer:
    """Summarise a video into a single-image storyboard of key frames.

    ``algo`` names a configuration of the components of frame selection (see ``algorithms``);
    ``backbone``, ``aggregation``, ``normalisation``, ``distance`` and ``clustering`` override its
    components, and ``time_smoothing`` tunes the clustering, for the algorithms that compare frame
    signatures. ``feature_extractor`` replaces the pre-trained model that would be built from the
    backbone, aggregation and normalisation.
    """

    def __init__(
        self,
        algo: str,
        n_frames: int,
        width: int,
        height: int,
        fps: float | None = None,
        time_segmentation: bool = False,
        segbar_height: int = 32,
        feature_extractor: nn.Module | None = None,
        clustering: str | None = None,
        time_smoothing: float = 0.0,
        backbone: str | None = None,
        aggregation: str | None = None,
        normalisation: str | None = None,
        distance: str | None = None,
    ) -> None:
        components = {
            "backbone": backbone,
            "aggregation": aggregation,
            "normalisation": normalisation,
            "distance": distance,
        }
        self.algorithm, self.clustering = resolve(
            algo, clustering, n_frames, time_smoothing, **components
        )
        self.algo = algo
        self.n_frames = n_frames
        self.fps = fps
        self.time_smoothing = time_smoothing
        if not self.algorithm.uses_signatures:
            given = {"fps": fps, "clustering": clustering, **components}
            ignored = [f"--{name}" for name, value in given.items() if value]
            if ignored:
                logger.warning("The '%s' algorithm ignores %s.", algo, " and ".join(ignored))
        self.time_segmentation = time_segmentation
        self.segbar_height = segbar_height
        self.feature_extractor = feature_extractor
        self.layout = Layout.fit(n_frames, width, height)

    def _select(self, reader: VideoReader) -> tuple[Sequence[int], torch.Tensor, torch.Tensor]:
        """Returns the candidate frame indices, the key frame positions in them and the labels."""
        if not self.algorithm.uses_signatures:
            candidates = range(reader.num_frames)
            return (candidates, *self.algorithm.select(len(candidates), self.n_frames))
        candidates = reader.sampled_indices(self.fps)
        limit = self.clustering.max_frames
        if limit is not None and len(candidates) > limit:  # fail before the costly signatures
            raise ValueError(
                f"the '{self.algorithm.clustering}' clustering supports at most {limit} frames "
                f"but the video has {len(candidates)} to compare: lower --fps "
                "or use the 'dissection' clustering"
            )
        if self.feature_extractor is None:
            self.feature_extractor = build_feature_extractor(
                backbone=self.algorithm.backbone,
                aggregation=self.algorithm.aggregation,
                normalisation=self.algorithm.normalisation,
            )
        signatures = compute_signatures(reader, candidates, self.feature_extractor)
        return (
            candidates,
            *self.clustering.select(
                signatures, self.n_frames, self.time_smoothing, self.algorithm.distance
            ),
        )

    def summarize(self, video_path: str | Path) -> torch.Tensor:
        """Return the storyboard as a (3, H, W) uint8 RGB tensor."""
        reader = VideoReader(video_path)
        candidates, indices, labels = self._select(reader)
        if len(indices) < self.n_frames:
            logger.warning(
                "Only %d of the %d requested frames available.", len(indices), self.n_frames
            )

        frames = reader.read([candidates[i] for i in indices.tolist()])
        collage = make_collage(frames, self.layout)
        if self.time_segmentation:
            bar = make_segbar(labels, indices, collage.shape[2], self.segbar_height)
            collage = torch.cat([collage, bar], dim=1)
        return collage
