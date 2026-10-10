import torch
import torch.nn as nn

from relax_xai.masking_functions import mask_generator


class RELAX(nn.Module):
    """
    Class for RELAX, a framework for explaining representations of data through
    masking and latent space similarity measurements.

    Link to ArXiv paper: https://arxiv.org/abs/2112.10161
    Link to International Journal of Computer Vision version:
    https://link.springer.com/article/10.1007/s11263-023-01773-2

    The importance and uncertainty estimates are computed with a running
    weighted mean/variance over all masks, updated batch-wise. All encoder
    evaluations are wrapped in torch.no_grad(), so no gradients are tracked.

    Parameters
    ----------
    input_image
        Input image to be explained, with shape (1, C, H, W).
    encoder
        Encoder that transforms the input image into a new representation.
        Must accept batched inputs with shape (N, C, H, W).
    batch_size
        The size of each batch of masks.
    num_batches
        Number of batches with masks to generate.
    similarity_measure
        Function for measuring similarity between masked and unmasked
        representations. Called as similarity_measure(unmasked, masked), where
        both tensors have shape (batch_size, D), and should return a tensor of
        similarity scores with shape (batch_size,). Defaults to cosine
        similarity.
    sum_of_weights_initial_value
        Initial value for the running weight sum, to avoid division by zero.
    """

    def __init__(
        self,
        input_image: torch.Tensor,
        encoder: nn.Module,
        batch_size: int = 100,
        num_batches: int = 30,
        similarity_measure: nn.Module | None = None,
        sum_of_weights_initial_value: float = 1e-10,
    ):
        super().__init__()

        if input_image.ndim != 4 or input_image.shape[0] != 1:
            raise ValueError(
                f"input_image must have shape (1, C, H, W), got {tuple(input_image.shape)}."
            )

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if num_batches < 1:
            raise ValueError("num_batches must be at least 1.")

        self.batch_size = batch_size
        self.input_image = input_image
        self.num_batches = num_batches
        self.device = input_image.device
        self.encoder = encoder.to(self.device)
        self.encoder.eval()
        self.similarity_measure = similarity_measure or nn.CosineSimilarity(dim=1)
        self.shape = tuple(input_image.shape[2:])
        self.sum_of_weights_initial_value = sum_of_weights_initial_value

        with torch.no_grad():
            self.unmasked_representations = self.encoder(self.input_image).expand(batch_size, -1)

        self.importance = torch.zeros(self.shape, device=self.device)
        self.uncertainty = torch.zeros(self.shape, device=self.device)
        self.sum_of_weights = sum_of_weights_initial_value * torch.ones(
            self.shape, device=self.device
        )

    def forward(self, **kwargs) -> None:
        """
        Runs RELAX. Keyword arguments are passed on to mask_generator
        (e.g. num_cells and probability_of_drop). Calling forward() more than
        once accumulates the statistics across calls.
        """

        with torch.no_grad():
            for _ in range(self.num_batches):
                masks = mask_generator(self.batch_size, self.shape, self.device, **kwargs)

                x_mask = self.input_image * masks

                masked_representations = self.encoder(x_mask)

                similarity_scores = self.similarity_measure(
                    self.unmasked_representations,
                    masked_representations,
                )

                self._update_statistics(masks.squeeze(1), similarity_scores)

        return None

    def _update_statistics(self, masks: torch.Tensor, similarity_scores: torch.Tensor) -> None:
        """
        Updates the running weighted mean (importance) and variance (uncertainty)
        estimates with a batch of masks and similarity scores, using the parallel
        variant of Welford's online algorithm (Chan et al., 1979). Equivalent to
        processing the masks one at a time, but computed with batched tensor
        operations.

        Parameters
        ----------
        masks
            Tensor of masks with shape (batch_size, H, W).
        similarity_scores
            Tensor of similarity scores with shape (batch_size,).
        """

        score_shift = similarity_scores.mean()

        shifted_scores = (similarity_scores - score_shift).reshape(-1, 1, 1)

        batch_weights = masks.sum(dim=0)
        batch_weighted_shifted_scores = (masks * shifted_scores).sum(dim=0)
        batch_weighted_shifted_scores_sq = (masks * shifted_scores.square()).sum(dim=0)

        batch_weights_safe = batch_weights.clamp_min(self.sum_of_weights_initial_value)
        batch_mean = score_shift + batch_weighted_shifted_scores / batch_weights_safe
        batch_m2 = (
            batch_weighted_shifted_scores_sq
            - batch_weighted_shifted_scores.square() / batch_weights_safe
        ).clamp_min(0)

        old_weights = self.sum_of_weights
        old_mean = self.importance
        total_weights = old_weights + batch_weights

        self.importance = (
            old_weights * old_mean + batch_weights * score_shift + batch_weighted_shifted_scores
        ) / total_weights
        self.uncertainty = (
            self.uncertainty
            + batch_m2
            + (old_weights * batch_weights / total_weights) * (old_mean - batch_mean).square()
        )
        self.sum_of_weights = total_weights

    @property
    def u_relax(self) -> torch.Tensor:
        """
        Uncertainty-filtered importance, keeping only the pixels with an
        uncertainty below the median uncertainty.
        """

        return self.importance * (self.uncertainty <= self.uncertainty.median())
