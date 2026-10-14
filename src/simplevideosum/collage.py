"""Storyboard layout and rendering."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

# ColorBrewer "Set3" qualitative palette, one colour per key frame (cycled if needed).
SET3 = np.array(
    [
        (141, 211, 199), (255, 255, 179), (190, 186, 218), (251, 128, 114),
        (128, 177, 211), (253, 180, 98), (179, 222, 105), (252, 205, 229),
        (217, 217, 217), (188, 128, 189), (204, 235, 197), (255, 237, 111),
    ],
    dtype=np.uint8,
)  # fmt: skip


@dataclass(frozen=True)
class Layout:
    """Grid of equally sized tiles, with the same aspect ratio as the collage."""

    width: int
    height: int
    tile_width: int
    tile_height: int
    tiles_per_row: int
    tiles_per_col: int

    @classmethod
    def fit(cls, n_frames: int, width: int, height: int) -> "Layout":
        """Largest tiles such that ``n_frames`` of them fit in a ``width`` x ``height`` collage."""
        if min(n_frames, width, height) < 1:
            raise ValueError("n_frames, width and height must be >= 1")
        ratio = width / height
        for tile_height in range(height, 0, -1):
            tile_width = max(1, round(tile_height * ratio))
            per_row, per_col = width // tile_width, height // tile_height
            if per_row * per_col >= n_frames:
                return cls(width, height, tile_width, tile_height, per_row, per_col)
        raise ValueError("cannot fit the requested number of frames")  # pragma: no cover

    @property
    def capacity(self) -> int:
        return self.tiles_per_row * self.tiles_per_col


def make_collage(frames: torch.Tensor, layout: Layout) -> torch.Tensor:
    """Tile (N, 3, H, W) uint8 frames row by row into a (3, h, w) uint8 collage."""
    if frames.shape[0] > layout.capacity:
        raise ValueError("more frames than tiles in the layout")
    collage = torch.zeros(
        3, layout.tiles_per_col * layout.tile_height, layout.tiles_per_row * layout.tile_width,
        dtype=torch.uint8,
    )  # fmt: skip
    if frames.shape[0] == 0:
        return collage
    tiles = F.interpolate(
        frames.float(), size=(layout.tile_height, layout.tile_width), mode="bilinear",
        antialias=True, align_corners=False,
    )  # fmt: skip
    tiles = tiles.round().clamp(0, 255).to(torch.uint8)
    for k, tile in enumerate(tiles):
        row, col = divmod(k, layout.tiles_per_row)
        y, x = row * layout.tile_height, col * layout.tile_width
        collage[:, y : y + layout.tile_height, x : x + layout.tile_width] = tile
    return collage


def make_segbar(
    labels: np.ndarray, indices: np.ndarray, width: int, height: int = 32
) -> torch.Tensor:
    """Colour bar showing which key frame represents each moment of the video.

    Returns a (3, height, width) uint8 tensor; key frame positions are marked in black.
    """
    n = len(labels)
    columns = np.arange(width)
    frame_of_column = np.clip(np.round(columns * n / width).astype(int), 0, n - 1)
    bar = SET3[labels[frame_of_column] % len(SET3)]  # (width, 3)
    bar[np.clip(np.round(indices * width / n).astype(int), 0, width - 1)] = 0
    return torch.from_numpy(bar).T[:, None, :].expand(3, height, width).contiguous()
