import pytest
import torch

from simplevideosum.scda import SCDA


def object_with_noise() -> torch.Tensor:
    """(1, 3, 6, 6) activations: a 3x3 object, plus a weak blob far from it."""
    activations = torch.zeros(1, 3, 6, 6)
    activations[0, :, 1:4, 1:4] = torch.tensor([1.0, 2.0, 3.0]).view(3, 1, 1)
    activations[0, 0, 1, 1] = 5.0  # raises the max of the first channel inside the object
    activations[0, :, 5, 5] = 2.0  # a small noisy blob, above the mean of the aggregation map
    return activations


def test_descriptor_pools_only_the_largest_component():
    activations = object_with_noise()
    model = SCDA()
    mask = model.selection_mask(activations)
    assert mask.shape == (1, 1, 6, 6) and mask.dtype == torch.bool
    assert mask.sum() == 9 and not mask[0, 0, 5, 5]
    descriptor = model(activations)
    assert descriptor.shape == (1, 2, 3)
    average = torch.tensor([13 / 9, 2.0, 3.0])  # first channel: eight 1s and one 5
    maximum = torch.tensor([5.0, 2.0, 3.0])
    assert torch.allclose(descriptor[0], torch.stack([average, maximum]), atol=1e-6)


def test_all_positions_above_the_mean_can_be_kept():
    activations = object_with_noise()
    mask = SCDA(largest_component=False).selection_mask(activations)
    assert mask.sum() == 10 and mask[0, 0, 5, 5]
    average = SCDA(largest_component=False)(activations)[0, 0]
    assert average[0] == pytest.approx((8 * 1.0 + 5.0 + 2.0) / 10)  # the blob is pooled too


def test_holes_are_filled_on_request():
    activations = torch.zeros(1, 1, 5, 5)
    activations[0, 0, 1:4, 1:4] = 3.0
    activations[0, 0, 2, 2] = 0.5  # a dip inside the object, below the mean of the map
    assert SCDA().selection_mask(activations).sum() == 8
    assert SCDA(fill_holes=True).selection_mask(activations).sum() == 9


def test_batch_is_processed_independently():
    torch.manual_seed(0)
    batch = torch.rand(5, 16, 8, 8)
    together = SCDA()(batch)
    alone = torch.cat([SCDA()(frame[None]) for frame in batch])
    assert torch.allclose(together, alone, atol=1e-6)


@pytest.mark.parametrize("largest", [True, False])
def test_flat_activations_keep_every_position(largest):
    flat = torch.ones(1, 4, 8, 8)
    model = SCDA(largest_component=largest)
    assert model.selection_mask(flat).all()
    assert torch.isfinite(model(flat)).all()


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        SCDA()(torch.rand(4, 8, 8))
