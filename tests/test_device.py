import torch

from simplevideosum.device import get_device


def test_get_device_returns_usable_device():
    device = get_device()
    assert isinstance(device, torch.device)
    assert torch.zeros(1, device=device).device.type == device.type
