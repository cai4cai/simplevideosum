"""Video reading built on TorchCodec."""

from pathlib import Path

import torch
from torchcodec.decoders import VideoDecoder


class VideoReader:
    """Decode video frames as uint8 (N, 3, H, W) tensors.

    With ``device="cuda"`` decoding runs on NVDEC and frames stay on the GPU. On the CPU,
    FFmpeg picks the number of decoding threads (TorchCodec's default of one thread is several
    times slower, as each frame needs the previous ones, so seeking does not avoid decoding).
    """

    def __init__(self, path: str | Path, device: str | torch.device = "cpu") -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._decoder = VideoDecoder(self.path, device=str(device), num_ffmpeg_threads=0)

    @property
    def num_frames(self) -> int:
        return len(self._decoder)

    @property
    def fps(self) -> float:
        """Average frame rate; some containers do not store it, so fall back on the duration."""
        meta = self._decoder.metadata
        if meta.average_fps:
            return meta.average_fps
        if meta.duration_seconds:
            return self.num_frames / meta.duration_seconds
        raise ValueError(f"cannot determine the frame rate of {self.path}")

    @property
    def duration(self) -> float:
        return self._decoder.metadata.duration_seconds

    def sampled_indices(self, fps: float | None = None) -> list[int]:
        """Frame indices of the video resampled at ``fps`` (all frames if None or >= native)."""
        if fps is not None and fps <= 0:
            raise ValueError("fps must be > 0")
        if fps is None or fps >= self.fps:
            return list(range(self.num_frames))
        step = self.fps / fps
        count = max(1, int(self.num_frames / step))
        return [min(round(i * step), self.num_frames - 1) for i in range(count)]

    def read(self, indices: list[int]) -> torch.Tensor:
        """Return the frames at ``indices`` as a (len(indices), 3, H, W) uint8 tensor."""
        return self._decoder.get_frames_at(indices).data
