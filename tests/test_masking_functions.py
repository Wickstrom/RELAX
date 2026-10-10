import pytest
import torch

from relax_xai.masking_functions import mask_generator

SHAPE = (32, 32)


def test_mask_shape_and_dtype():
    masks = mask_generator(batch_size=8, shape=SHAPE, device="cpu")
    assert masks.shape == (8, 1, *SHAPE)
    assert masks.dtype == torch.float32


def test_mask_values_are_in_unit_interval():
    masks = mask_generator(batch_size=16, shape=SHAPE, device="cpu")
    assert masks.min() >= 0.0
    assert masks.max() <= 1.0


def test_masks_are_spatially_smooth():
    masks = mask_generator(batch_size=4, shape=SHAPE, device="cpu")
    dx = masks[..., :, 1:] - masks[..., :, :-1]
    dy = masks[..., 1:, :] - masks[..., :-1, :]
    assert dx.abs().max() <= 0.5
    assert dy.abs().max() <= 0.5


def test_mask_generator_is_deterministic_given_seed():
    torch.manual_seed(0)
    masks_a = mask_generator(batch_size=8, shape=SHAPE, device="cpu")
    torch.manual_seed(0)
    masks_b = mask_generator(batch_size=8, shape=SHAPE, device="cpu")
    assert torch.equal(masks_a, masks_b)


def test_probability_of_drop_sets_fraction_of_ones():
    torch.manual_seed(0)
    masks_low = mask_generator(batch_size=64, shape=SHAPE, device="cpu", probability_of_drop=0.1)
    torch.manual_seed(0)
    masks_high = mask_generator(batch_size=64, shape=SHAPE, device="cpu", probability_of_drop=0.9)
    assert 0.02 < masks_low.mean() < 0.2
    assert 0.8 < masks_high.mean() < 0.98
    assert masks_high.mean() > masks_low.mean()


def test_deprecated_misspelled_kwarg_still_works():
    with pytest.warns(DeprecationWarning, match="probability_of_drop"):
        masks = mask_generator(batch_size=4, shape=SHAPE, device="cpu", probablity_of_drop=0.3)
    assert masks.shape == (4, 1, *SHAPE)


def test_num_cells_affects_mask_smoothness():
    torch.manual_seed(0)
    coarse = mask_generator(batch_size=4, shape=SHAPE, device="cpu", num_cells=15)
    torch.manual_seed(0)
    smooth = mask_generator(batch_size=4, shape=SHAPE, device="cpu", num_cells=3)
    dx_coarse = (coarse[..., :, 1:] - coarse[..., :, :-1]).abs().mean()
    dx_smooth = (smooth[..., :, 1:] - smooth[..., :, :-1]).abs().mean()
    assert dx_coarse > dx_smooth


def test_num_spatial_dims_must_be_2():
    with pytest.raises(ValueError, match="num_spatial_dims"):
        mask_generator(batch_size=2, shape=SHAPE, device="cpu", num_spatial_dims=3)


def test_shape_must_be_2d():
    with pytest.raises(ValueError, match="2-tuple"):
        mask_generator(batch_size=2, shape=(8, 8, 3), device="cpu")


def test_num_cells_must_be_positive():
    with pytest.raises(ValueError, match="num_cells"):
        mask_generator(batch_size=2, shape=SHAPE, device="cpu", num_cells=0)


def test_shape_must_be_large_enough():
    with pytest.raises(ValueError, match="too small"):
        mask_generator(batch_size=2, shape=(2, 2), device="cpu", num_cells=7)
