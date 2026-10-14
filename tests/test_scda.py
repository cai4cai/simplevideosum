import pytest
import torch

from simplevideosum.scda import SCDA, fill_holes, largest_component


def grid(*rows: str) -> torch.Tensor:
    return torch.tensor([[c == "#" for c in row] for row in rows])


def test_largest_component_is_8_connected_and_picks_the_biggest():
    mask = torch.stack(
        [
            grid("##...", "..#..", "...#.", ".....", "#...."),  # diagonal chain of 4 beats 1
            grid("#.#.#", ".....", "#.#.#", ".....", "#...."),  # isolated pixels: one of them
        ]
    )
    out = largest_component(mask)
    assert out[0].equal(grid("##...", "..#..", "...#.", ".....", "....."))
    assert out[1].sum() == 1 and (out[1] & mask[1]).sum() == 1


def test_largest_component_of_empty_mask_is_empty():
    assert not largest_component(torch.zeros(1, 4, 4, dtype=torch.bool)).any()


def test_fill_holes_fills_enclosed_background_only():
    mask = grid("#####", "#...#", "#.#.#", "#...#", "#####")
    assert fill_holes(mask[None]).all()
    open_ring = grid("####.", "#..#.", "#..#.", "#...#", "#####")  # diagonal leak: 4-connected
    assert fill_holes(open_ring[None])[0, 1, 1]
    leaky = grid("####.", "#..#.", "#....", "#..#.", "####.")
    assert not fill_holes(leaky[None])[0, 1, 1]


def test_descriptor_pools_only_the_selected_object():
    activations = torch.zeros(1, 3, 6, 6)
    activations[0, :, 1:4, 1:4] = torch.tensor([1.0, 2.0, 3.0]).view(3, 1, 1)
    activations[0, 0, 1, 1] = 5.0  # raises the max of the first channel inside the object
    activations[0, :, 5, 5] = 0.4  # weak background blob, below the mean of the aggregation map
    model = SCDA()
    assert model.selection_mask(activations)[0].sum() == 9
    descriptor = model(activations)
    assert descriptor.shape == (1, 2, 3)
    average = torch.tensor([13 / 9, 2.0, 3.0])  # first channel: eight 1s and one 5
    maximum = torch.tensor([5.0, 2.0, 3.0])
    assert torch.allclose(descriptor[0], torch.stack([average, maximum]), atol=1e-6)


def test_batch_is_processed_independently():
    torch.manual_seed(0)
    batch = torch.rand(5, 16, 8, 8)
    together = SCDA()(batch)
    alone = torch.cat([SCDA()(frame[None]) for frame in batch])
    assert torch.allclose(together, alone, atol=1e-6)


def test_flat_activations_keep_every_position():
    flat = torch.ones(1, 4, 8, 8)
    assert SCDA().selection_mask(flat).all()
    assert torch.isfinite(SCDA()(flat)).all()


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        SCDA()(torch.rand(4, 8, 8))
