"""Storyboard layout and rendering."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

# ColorBrewer "Set3" qualitative palette, one colour per key frame (cycled if needed).
SET3 = torch.tensor(
    [
        (141, 211, 199), (255, 255, 179), (190, 186, 218), (251, 128, 114),
        (128, 177, 211), (253, 180, 98), (179, 222, 105), (252, 205, 229),
        (217, 217, 217), (188, 128, 189), (204, 235, 197), (255, 237, 111),
    ],
    dtype=torch.uint8,
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


def border_width(layout: Layout) -> int:
    """Thickness in pixels of the coloured border of a tile: thin, but always visible."""
    return max(2, layout.tile_height // 90)


def label_colours(labels: torch.Tensor) -> torch.Tensor:
    """The (N, 3) uint8 colours of the labels, cycling through the palette."""
    return SET3[torch.as_tensor(labels).cpu() % len(SET3)]


def make_collage(
    frames: torch.Tensor, layout: Layout, colours: torch.Tensor | None = None
) -> torch.Tensor:
    """Tile (N, 3, H, W) uint8 frames row by row into a (3, h, w) uint8 collage.

    With ``colours`` (N, 3), each tile gets a border of its colour, drawn over the edge of the
    tile so that the layout does not change.
    """
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
        if colours is not None:
            collage_tile = collage[:, y : y + layout.tile_height, x : x + layout.tile_width]
            b = border_width(layout)
            colour = colours[k].reshape(3, 1, 1)
            collage_tile[:, :b, :] = colour
            collage_tile[:, -b:, :] = colour
            collage_tile[:, :, :b] = colour
            collage_tile[:, :, -b:] = colour
    return collage


def make_segbar(
    labels: torch.Tensor, indices: torch.Tensor, width: int, height: int = 32
) -> torch.Tensor:
    """Colour bar showing which key frame represents each moment of the video.

    ``labels`` gives for every frame the position in ``indices`` of the key frame that
    represents it. Returns a (3, height, width) uint8 CPU tensor, like the collage it goes under;
    key frame positions are marked in black.
    """
    labels = torch.as_tensor(labels).cpu()
    indices = torch.as_tensor(indices).cpu()
    n = len(labels)
    columns = torch.arange(width, dtype=torch.float64)
    frame_of_column = (columns * n / width).round().long().clamp(0, n - 1)
    bar = label_colours(labels[frame_of_column])  # (width, 3)
    bar[(indices.double() * width / n).round().long().clamp(0, width - 1)] = 0
    return bar.T[:, None, :].expand(3, height, width).contiguous()
