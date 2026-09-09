import pytest
import torch
import torch.nn as nn

from relax_xai.relax import RELAX

SHAPE = (32, 32)


class ConstantEncoder(nn.Module):
    def forward(self, x):
        return torch.ones(x.shape[0], 8, device=x.device)


class NegativeEuclideanSimilarity(nn.Module):
    def forward(self, a, b):
        return -torch.norm(a - b, dim=1)


def make_encoder(seed=0, out_dim=16):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * SHAPE[0] * SHAPE[1], out_dim))


def make_image(seed=1):
    torch.manual_seed(seed)
    return torch.rand(1, 3, *SHAPE)


def make_relax(**kwargs):
    return RELAX(
        make_image(),
        make_encoder(),
        batch_size=kwargs.pop("batch_size", 8),
        num_batches=kwargs.pop("num_batches", 3),
        **kwargs,
    )


def test_output_shapes():
    relax = make_relax()
    relax.forward()
    assert relax.importance.shape == SHAPE
    assert relax.uncertainty.shape == SHAPE
    assert relax.sum_of_weights.shape == SHAPE


def test_importance_and_uncertainty_are_finite():
    relax = make_relax()
    relax.forward()
    assert torch.isfinite(relax.importance).all()
    assert torch.isfinite(relax.uncertainty).all()
    assert relax.importance.abs().max() <= 1.0 + 1e-6


def test_deterministic_given_seed():
    torch.manual_seed(42)
    relax_a = make_relax()
    relax_a.forward()

    torch.manual_seed(42)
    relax_b = make_relax()
    relax_b.forward()

    assert torch.equal(relax_a.importance, relax_b.importance)
    assert torch.equal(relax_a.uncertainty, relax_b.uncertainty)


def test_no_autograd_graph_is_tracked():
    image = make_image()
    image.requires_grad_(True)
    relax = RELAX(image, make_encoder(), batch_size=4, num_batches=2)
    relax.forward()
    assert not relax.importance.requires_grad
    assert not relax.uncertainty.requires_grad


def test_input_image_must_be_batched():
    with pytest.raises(ValueError, match=r"\(1, C, H, W\)"):
        RELAX(torch.rand(3, *SHAPE), make_encoder())
    with pytest.raises(ValueError, match=r"\(1, C, H, W\)"):
        RELAX(torch.rand(2, 3, *SHAPE), make_encoder())


def test_constant_encoder_gives_zero_uncertainty():
    relax = RELAX(
        make_image(),
        ConstantEncoder(),
        batch_size=16,
        num_batches=5,
    )
    relax.forward()
    assert torch.allclose(relax.uncertainty, torch.zeros(SHAPE), atol=1e-6)
    assert torch.allclose(relax.importance, torch.ones(SHAPE), atol=1e-6)


def test_vectorized_updates_match_sequential_reference():
    torch.manual_seed(0)
    relax = make_relax(batch_size=8, num_batches=1)
    masks = torch.rand(8, *SHAPE)
    scores = torch.rand(8)

    importance = torch.zeros(SHAPE)
    uncertainty = torch.zeros(SHAPE)
    sum_of_weights = 1e-10 * torch.ones(SHAPE)

    for score_i, mask_i in zip(scores, masks):
        sum_of_weights += mask_i
        importance_previous_step = importance.clone()
        importance += mask_i * (score_i - importance) / sum_of_weights
        uncertainty += (score_i - importance) * (score_i - importance_previous_step) * mask_i

    relax._update_statistics(masks, scores)

    assert torch.allclose(relax.importance, importance, atol=1e-6)
    assert torch.allclose(relax.uncertainty, uncertainty, atol=1e-6)
    assert torch.allclose(relax.sum_of_weights, sum_of_weights, atol=1e-6)


def test_u_relax_filters_by_median_uncertainty():
    relax = make_relax()
    relax.forward()
    u_relax = relax.u_relax
    kept = relax.uncertainty <= relax.uncertainty.median()
    assert u_relax.shape == SHAPE
    assert torch.equal(u_relax, relax.importance * kept)
    assert 0 < kept.sum() < kept.numel()


def test_custom_similarity_measure():
    relax = RELAX(
        make_image(),
        make_encoder(),
        batch_size=4,
        num_batches=2,
        similarity_measure=NegativeEuclideanSimilarity(),
    )
    relax.forward()
    assert torch.isfinite(relax.importance).all()
    assert torch.isfinite(relax.uncertainty).all()


def test_masking_kwargs_are_passed_to_mask_generator():
    torch.manual_seed(0)
    relax = make_relax()
    relax.forward(num_cells=3, probability_of_drop=0.3)
    assert torch.isfinite(relax.importance).all()


def test_deprecated_misspelled_kwarg_still_works():
    relax = make_relax()
    with pytest.warns(DeprecationWarning, match="probability_of_drop"):
        relax.forward(probablity_of_drop=0.3)
    assert torch.isfinite(relax.importance).all()


def test_package_exposes_relax_class():
    from relax_xai import RELAX as TopLevelRELAX

    assert TopLevelRELAX is RELAX
