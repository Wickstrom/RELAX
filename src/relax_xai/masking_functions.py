import torch
import torch.nn.functional as F


def mask_generator(
    batch_size: int,
    shape: tuple,
    device: str,
    num_cells: int = 7,
    probablity_of_drop: float = 0.5,
    num_spatial_dims: int = 2) -> torch.Tensor:
    """
    Generates a batch of masks by sampling Bernoulli random variables (probablity_of_drop) in a lower dimensional grid (num_cells)
    and upsamples the discrete masks using bilinear interpolation to obtain smooth continious mask in (0, 1).
    """

    pad_size = (num_cells // 2, num_cells // 2, num_cells // 2, num_cells // 2)

    grid = (torch.rand(batch_size, 1, *((num_cells,) * num_spatial_dims), device=device) < probablity_of_drop).float()
    grid_up = F.interpolate(grid, size=(shape), mode='bilinear', align_corners=False)
    grid_up = F.pad(grid_up, pad_size, mode='reflect')

    shift_x = torch.randint(0, num_cells, (batch_size,), device=device)
    shift_y = torch.randint(0, num_cells, (batch_size,), device=device)

    masks = torch.empty((batch_size, 1, shape[0], shape[1]), device=device)

    for mask_i in range(batch_size):
        masks[mask_i] = grid_up[
            mask_i,
            :,
            shift_x[mask_i]:shift_x[mask_i] + shape[0],
            shift_y[mask_i]:shift_y[mask_i] + shape[1]
        ]

    yield masks
