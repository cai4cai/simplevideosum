"""Video to storyboard summarisation."""

from pathlib import Path

import torch

from simplevideosum.algorithms import ALGORITHMS, FPS_ALGORITHMS, PLANNED_ALGORITHMS
from simplevideosum.collage import Layout, make_collage, make_segbar
from simplevideosum.io import VideoReader


class VideoSummarizer:
    """Summarise a video into a single-image storyboard of key frames."""

    def __init__(
        self,
        algo: str,
        n_frames: int,
        width: int,
        height: int,
        fps: float | None = None,
        time_segmentation: bool = False,
        segbar_height: int = 32,
    ) -> None:
        if algo not in ALGORITHMS:
            if algo in PLANNED_ALGORITHMS:
                raise ValueError(f"algorithm '{algo}' is not implemented yet")
            raise ValueError(f"unknown algorithm '{algo}', available: {', '.join(ALGORITHMS)}")
        self.algo = algo
        self.n_frames = n_frames
        self.fps = fps
        if fps is not None and algo not in FPS_ALGORITHMS:
            print(f"[WARN] The '{algo}' algorithm ignores the sampling frequency (--fps).")
        self.time_segmentation = time_segmentation
        self.segbar_height = segbar_height
        self.layout = Layout.fit(n_frames, width, height)

    def summarize(self, video_path: str | Path) -> torch.Tensor:
        """Return the storyboard as a (3, H, W) uint8 RGB tensor."""
        reader = VideoReader(video_path)
        # Algorithms that do not look at the frames select from the whole video: no resampling
        sampled = (
            reader.sampled_indices(self.fps)
            if self.algo in FPS_ALGORITHMS
            else range(reader.num_frames)
        )
        indices, labels = ALGORITHMS[self.algo](len(sampled), self.n_frames)
        if len(indices) < self.n_frames:
            print(f"[WARN] Only {len(indices)} of the {self.n_frames} requested frames available.")

        frames = reader.read([sampled[i] for i in indices])
        collage = make_collage(frames, self.layout)
        if self.time_segmentation:
            bar = make_segbar(labels, indices, collage.shape[2], self.segbar_height)
            collage = torch.cat([collage, bar], dim=1)
        return collage
