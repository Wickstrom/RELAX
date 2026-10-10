import pytest
import torch

from relax_xai.utils import (
    IMAGENET_MEAN,
    ToDevice,
    imagenet_image_transforms,
    unsqeeze_image,
    unsqueeze_image,
)


def test_unsqueeze_image_adds_batch_dim():
    x = torch.rand(3, 4, 5)
    out = unsqueeze_image(x)
    assert out.shape == (1, 3, 4, 5)
    assert torch.equal(out, x.unsqueeze(0))


def test_unsqeeze_image_is_deprecated_alias():
    x = torch.rand(3, 4, 5)
    with pytest.warns(DeprecationWarning, match="unsqueeze_image"):
        out = unsqeeze_image(x)
    assert torch.equal(out, unsqueeze_image(x))


def test_to_device_moves_tensor_and_repr():
    x = torch.rand(1, 3, 4, 5)
    out = ToDevice("cpu")(x)
    assert out.device.type == "cpu"
    assert torch.equal(out, x)
    assert "device=cpu" in repr(ToDevice("cpu"))


def test_imagenet_transforms_resize_normalize_and_batch():
    x = torch.rand(3, 100, 80)
    out = imagenet_image_transforms(device="cpu", new_shape_of_image=64)(x)
    assert out.shape == (1, 3, 64, 64)
    assert out.device.type == "cpu"


def test_imagenet_transforms_normalize_mean_image_to_zero():
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).repeat(1, 32, 32)
    out = imagenet_image_transforms(device="cpu", new_shape_of_image=32)(mean)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)
