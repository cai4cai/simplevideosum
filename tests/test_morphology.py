import pytest
import torch

from simplevideosum import morphology
from simplevideosum.morphology import dilate, fill_holes, largest_connected_component


def grid(*rows: str) -> torch.Tensor:
    """(1, 1, H, W) boolean mask from rows of '#' (foreground) and '.'."""
    return torch.tensor([[[[c == "#" for c in row] for row in rows]]])


def reference_largest(mask: torch.Tensor, connectivity: int) -> torch.Tensor:
    """Flood fill in plain Python of every map; the first component in raster order wins ties."""
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    out = torch.zeros_like(mask)
    for b in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            m = mask[b, c]
            h, w = m.shape
            seen, best = set(), []
            for start in ((y, x) for y in range(h) for x in range(w)):
                if not m[start] or start in seen:
                    continue
                component, stack = [], [start]
                seen.add(start)
                while stack:
                    y, x = stack.pop()
                    component.append((y, x))
                    for dy, dx in steps:
                        n = (y + dy, x + dx)
                        if 0 <= n[0] < h and 0 <= n[1] < w and m[n] and n not in seen:
                            seen.add(n)
                            stack.append(n)
                if len(component) > len(best):
                    best = component
            for y, x in best:
                out[b, c, y, x] = True
    return out


@pytest.mark.parametrize("connectivity, expected", [(4, 5), (8, 9)])
def test_dilate_neighbourhood(connectivity, expected):
    mask = torch.zeros(1, 1, 5, 5, dtype=torch.bool)
    mask[0, 0, 2, 2] = True
    out = dilate(mask, connectivity)
    assert out.sum() == expected and out[0, 0, 1, 1] == (connectivity == 8)


def test_largest_component_follows_the_connectivity():
    diagonal = grid("#....", ".#...", "..#..", "...#.", "....#")
    assert largest_connected_component(diagonal, 8).equal(diagonal)
    assert largest_connected_component(diagonal, 4).sum() == 1  # only a single pixel is kept


@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("method", ["closure", "propagation"])
@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (8, 8), (5, 13)])
def test_largest_component_matches_a_flood_fill(connectivity, method, shape):
    torch.manual_seed(0)
    mask = torch.rand(6, 2, *shape) > 0.45
    mask[0] = False  # an empty map has an empty result
    mask[1] = True  # a full map is its own largest component
    limit = 10**6 if method == "closure" else 0
    out = largest_connected_component(mask, connectivity, closure_max_pixels=limit)
    assert out.dtype == torch.bool and out.shape == mask.shape
    assert out.equal(reference_largest(mask, connectivity))


def test_maps_of_the_batch_and_channels_are_independent():
    torch.manual_seed(1)
    mask = torch.rand(3, 4, 8, 8) > 0.5
    together = largest_connected_component(mask)
    for b in range(3):
        for c in range(4):
            alone = largest_connected_component(mask[b : b + 1, c : c + 1])
            assert together[b : b + 1, c : c + 1].equal(alone)


def test_small_maps_use_the_closure_and_large_ones_the_propagation(monkeypatch):
    used = []
    for name in ("_largest_by_closure", "_largest_by_propagation"):
        original = getattr(morphology, name)
        monkeypatch.setattr(
            morphology,
            name,
            lambda m, c, name=name, original=original: used.append(name) or original(m, c),
        )
    largest_connected_component(torch.ones(1, 1, 16, 16, dtype=torch.bool))
    largest_connected_component(torch.ones(1, 1, 17, 16, dtype=torch.bool))
    assert used == ["_largest_by_closure", "_largest_by_propagation"]


def test_long_snake_is_one_component():
    snake = grid("#######", "......#", "#######", "#......", "#######", "......#", "#######")
    assert largest_connected_component(snake, 4).equal(snake)


def test_fill_holes_fills_enclosed_background_only():
    ring = grid("#####", "#...#", "#.#.#", "#...#", "#####")
    assert fill_holes(ring).all()
    diagonal_gap = grid("####.", "#..#.", "#..#.", "#...#", "#####")  # 4-connected background
    assert fill_holes(diagonal_gap)[0, 0, 1, 1]
    leaky = grid("####.", "#..#.", "#....", "#..#.", "####.")
    assert not fill_holes(leaky)[0, 0, 1, 1]
    assert fill_holes(leaky).sum() == leaky.sum()


def test_fill_holes_background_uses_the_complementary_connectivity():
    # the background of the centre only reaches the border through a diagonal step
    mask = grid("####.", "#..#.", "#..#.", "#...#", "#####")
    assert fill_holes(mask, connectivity=8)[0, 0, 1, 1]  # 4-connected background: enclosed
    assert not fill_holes(mask, connectivity=4)[0, 0, 1, 1]  # 8-connected background: leaks out


@pytest.mark.parametrize(
    "bad",
    [torch.zeros(4, 4, dtype=torch.bool), torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4).long()],
)
def test_rejects_bad_masks_and_connectivity(bad):
    for function in (dilate, largest_connected_component, fill_holes):
        with pytest.raises(ValueError):
            function(bad)
    with pytest.raises(ValueError, match="connectivity"):
        dilate(torch.zeros(1, 1, 4, 4, dtype=torch.bool), connectivity=6)
