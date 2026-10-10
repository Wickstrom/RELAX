import warnings

import torch
import torch.nn.functional as F


def mask_generator(
    batch_size: int,
    shape: tuple,
    device: str,
    num_cells: int = 7,
    probability_of_drop: float = 0.5,
    num_spatial_dims: int = 2,
    probablity_of_drop: float | None = None,
) -> torch.Tensor:
    """
    Generates a batch of masks by sampling Bernoulli random variables
    (probability_of_drop) in a lower dimensional grid (num_cells) and upsampling
    the discrete masks using bilinear interpolation to obtain smooth continuous
    masks in (0, 1).

    Parameters
    ----------
    batch_size
        Number of masks to generate.
    shape
        Spatial shape (H, W) of the masks.
    device
        Device on which the masks are created.
    num_cells
        Number of cells in the low resolution grid used to sample masks.
    probability_of_drop
        Probability parameter of the Bernoulli distribution used to sample
        each cell of the low resolution grid. A cell takes value 1 with this
        probability, and value 0 otherwise.
    num_spatial_dims
        Number of spatial dimensions of the masks.
    probablity_of_drop
        Deprecated, misspelled alias for probability_of_drop. Will be removed
        in a future release.

    Returns
    -------
    torch.Tensor
        Tensor of masks with shape (batch_size, 1, H, W).
    """

    if probablity_of_drop is not None:
        warnings.warn(
            "The 'probablity_of_drop' argument is deprecated and will be removed"
            " in a future release. Use 'probability_of_drop' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        probability_of_drop = probablity_of_drop

    if num_spatial_dims != 2:
        raise ValueError("Only num_spatial_dims=2 is supported.")
    if len(shape) != 2:
        raise ValueError("shape must be a 2-tuple (H, W).")
    if num_cells < 1:
        raise ValueError("num_cells must be at least 1.")
    if min(shape) <= num_cells // 2:
        raise ValueError(f"shape {tuple(shape)} is too small for num_cells={num_cells}.")

    pad_size = (num_cells // 2, num_cells // 2, num_cells // 2, num_cells // 2)

    grid = (
        torch.rand(batch_size, 1, *((num_cells,) * num_spatial_dims), device=device)
        < probability_of_drop
    ).float()
    grid_up = F.interpolate(grid, size=shape, mode="bilinear", align_corners=False)
    grid_up = F.pad(grid_up, pad_size, mode="reflect")

    shift_x = torch.randint(0, num_cells, (batch_size,), device=device)
    shift_y = torch.randint(0, num_cells, (batch_size,), device=device)

    masks = torch.empty((batch_size, 1, shape[0], shape[1]), device=device)

    for mask_i in range(batch_size):
        masks[mask_i] = grid_up[
            mask_i,
            :,
            shift_x[mask_i] : shift_x[mask_i] + shape[0],
            shift_y[mask_i] : shift_y[mask_i] + shape[1],
        ]

    return masks
