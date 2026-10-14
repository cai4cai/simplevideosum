"""Video to storyboard summarisation."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from simplevideosum.algorithms import resolve
from simplevideosum.collage import Layout, make_collage, make_segbar
from simplevideosum.features import build_feature_extractor, compute_signatures
from simplevideosum.io import VideoReader


class VideoSummarizer:
    """Summarise a video into a single-image storyboard of key frames.

    ``feature_extractor`` overrides the default pre-trained InceptionV3 used by the algorithms
    that compare frame signatures. ``clustering`` (default: k-medoids) and ``time_smoothing`` only
    apply to those algorithms.
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
    ) -> None:
        self.algorithm, self.clustering = resolve(algo, clustering, n_frames, time_smoothing)
        self.algo = algo
        self.n_frames = n_frames
        self.fps = fps
        self.time_smoothing = time_smoothing
        if not self.algorithm.uses_signatures:
            ignored = [
                name for name, value in (("--fps", fps), ("--clustering", clustering)) if value
            ]
            if ignored:
                print(f"[WARN] The '{algo}' algorithm ignores {' and '.join(ignored)}.")
        self.time_segmentation = time_segmentation
        self.segbar_height = segbar_height
        self.feature_extractor = feature_extractor
        self.layout = Layout.fit(n_frames, width, height)

    def _select(self, reader: VideoReader) -> tuple[Sequence[int], np.ndarray, np.ndarray]:
        """Returns the candidate frame indices, the key frame positions in them and the labels."""
        if not self.algorithm.uses_signatures:
            candidates = range(reader.num_frames)
            return (candidates, *self.algorithm.select(len(candidates), self.n_frames))
        candidates = reader.sampled_indices(self.fps)
        if self.feature_extractor is None:
            self.feature_extractor = build_feature_extractor()
        signatures = compute_signatures(reader, candidates, self.feature_extractor)
        return (
            candidates,
            *self.clustering.select(signatures.cpu().numpy(), self.n_frames, self.time_smoothing),
        )

    def summarize(self, video_path: str | Path) -> torch.Tensor:
        """Return the storyboard as a (3, H, W) uint8 RGB tensor."""
        reader = VideoReader(video_path)
        candidates, indices, labels = self._select(reader)
        if len(indices) < self.n_frames:
            print(f"[WARN] Only {len(indices)} of the {self.n_frames} requested frames available.")

        frames = reader.read([candidates[i] for i in indices])
        collage = make_collage(frames, self.layout)
        if self.time_segmentation:
            bar = make_segbar(labels, indices, collage.shape[2], self.segbar_height)
            collage = torch.cat([collage, bar], dim=1)
        return collage
