import pytest
import torch

from simplevideosum.collage import SET3, Layout, border_width, make_collage, make_segbar


def test_layout_matches_reference_example():
    layout = Layout.fit(16, 1920, 1080)
    assert (layout.tile_width, layout.tile_height) == (480, 270)
    assert (layout.tiles_per_row, layout.tiles_per_col) == (4, 4)


@pytest.mark.parametrize("n", [1, 7, 16, 100])
def test_layout_fits_requested_frames(n):
    layout = Layout.fit(n, 1920, 1080)
    assert layout.capacity >= n
    assert layout.tiles_per_row * layout.tile_width <= 1920
    assert layout.tiles_per_col * layout.tile_height <= 1080


def test_layout_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        Layout.fit(0, 100, 100)


def test_collage_places_frames_row_by_row():
    layout = Layout.fit(4, 40, 20)
    frames = torch.stack([torch.full((3, 8, 8), v, dtype=torch.uint8) for v in (10, 20, 30, 40)])
    collage = make_collage(frames, layout)
    th, tw = layout.tile_height, layout.tile_width
    assert collage.dtype == torch.uint8
    assert collage[0, 0, 0] == 10
    assert collage[0, 0, tw] == 20
    assert collage[0, th, 0] == 30
    assert collage[0, -1, -1] == 40


def test_collage_rejects_too_many_frames():
    layout = Layout.fit(1, 40, 20)
    with pytest.raises(ValueError):
        make_collage(torch.zeros(2, 3, 8, 8, dtype=torch.uint8), layout)


def test_segbar_shape_and_key_frame_markers():
    indices = torch.tensor([0, 50, 99])
    labels = torch.tensor([0, 1, 2]).repeat_interleave(torch.tensor([25, 50, 25]))
    bar = make_segbar(labels, indices, width=200, height=8)
    assert bar.shape == (3, 8, 200) and bar.dtype == torch.uint8
    assert bar[:, 0, 0].tolist() == [0, 0, 0]  # key frame 0 marker
    assert bar[:, 0, 100].tolist() == [0, 0, 0]  # key frame 50 marker
    assert bar[:, 0, 10].tolist() != [0, 0, 0]


def test_collage_draws_a_coloured_border_over_the_edge_of_each_tile():
    layout = Layout.fit(4, 400, 200)
    frames = torch.full((4, 3, 20, 20), 7, dtype=torch.uint8)
    colours = SET3[:4]
    collage = make_collage(frames, layout, colours)
    th, tw, b = layout.tile_height, layout.tile_width, border_width(layout)
    assert collage.shape == make_collage(frames, layout).shape
    for k in range(4):
        y, x = divmod(k, layout.tiles_per_row)
        y, x = y * th, x * tw
        colour = colours[k].tolist()
        assert collage[:, y, x].tolist() == colour  # corner
        assert collage[:, y + th - 1, x + tw // 2].tolist() == colour  # bottom edge
        assert collage[:, y + th // 2, x + b - 1].tolist() == colour  # inside the border
        assert collage[:, y + th // 2, x + b].tolist() == [7, 7, 7]  # picture starts after it
        assert collage[:, y + th // 2, x + tw // 2].tolist() == [7, 7, 7]
