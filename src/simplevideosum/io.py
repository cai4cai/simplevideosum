"""Video reading built on TorchCodec."""

from collections.abc import Iterator
from pathlib import Path

import torch
from torchcodec.decoders import VideoDecoder


class VideoReader:
    """Decode video frames as uint8 (N, 3, H, W) tensors.

    With ``device="cuda"`` decoding runs on NVDEC and frames stay on the GPU.
    """

    def __init__(self, path: str | Path, device: str | torch.device = "cpu") -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._decoder = VideoDecoder(self.path, device=str(device))

    @property
    def num_frames(self) -> int:
        return len(self._decoder)

    @property
    def fps(self) -> float:
        return self._decoder.metadata.average_fps

    @property
    def duration(self) -> float:
        return self._decoder.metadata.duration_seconds

    @property
    def frame_size(self) -> tuple[int, int]:
        """(height, width) of the decoded frames."""
        meta = self._decoder.metadata
        return meta.height, meta.width

    def read(self, indices: list[int]) -> torch.Tensor:
        """Return the frames at ``indices`` as a (len(indices), 3, H, W) uint8 tensor."""
        return self._decoder.get_frames_at(indices).data

    def sample(self, n: int) -> torch.Tensor:
        """Return ``n`` frames spread uniformly over the video."""
        if n < 1:
            raise ValueError("n must be >= 1")
        n = min(n, self.num_frames)
        indices = torch.linspace(0, self.num_frames - 1, n).round().long().tolist()
        return self.read(indices)

    def batches(self, batch_size: int, stride: int = 1) -> Iterator[torch.Tensor]:
        """Yield every ``stride``-th frame in batches of up to ``batch_size``."""
        if batch_size < 1 or stride < 1:
            raise ValueError("batch_size and stride must be >= 1")
        indices = list(range(0, self.num_frames, stride))
        for i in range(0, len(indices), batch_size):
            yield self.read(indices[i : i + batch_size])
