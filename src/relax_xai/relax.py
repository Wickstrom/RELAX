import torch
import torch.nn as nn

from relax_xai.masking_functions import mask_generator

class RELAX(nn.Module):
    """
    Class for RELAX, a framework for explaining representations of data through masking and latent space similairty
    measurments.
    
    Link to ArXiv paper: https://arxiv.org/abs/2112.10161
    Link to International Journal of Computer Vision version: https://link.springer.com/article/10.1007/s11263-023-01773-2

    Parameters
    ----------
    input_image
        Input image to be explained.
    encoder
        Encoder that transforms the input image into a new representation
    batch_size
        The size of each batch of masks
    num_batches
        Number of batches with masks to generate
    similarity_measure
        Function for measuring similarity between masked and unmasked representation
    sum_of_weights_initial_value
        Initial values for running mean and variance estimator to avoid division by zero
    """
    def __init__(self,
                 input_image: torch.Tensor,
                 encoder: nn.Module,
                 batch_size: int = 100,
                 num_batches: int = 30,
                 similarity_measure: nn.Module = nn.CosineSimilarity(dim=1),
                 sum_of_weights_initial_value: float = 1e-10):

        super().__init__()

        self.batch_size = batch_size
        self.input_image = input_image
        self.num_batches = num_batches
        self.device = input_image.device
        self.encoder = encoder.to(self.device)
        self.similarity_measure = similarity_measure
        self.shape = tuple(input_image.shape[2:])
        self.unmasked_representations = encoder(self.input_image).expand(batch_size, -1)

        self.importance = torch.zeros(self.shape, device=self.device)
        self.uncertainty = torch.zeros(self.shape, device=self.device)

        self.sum_of_weights = sum_of_weights_initial_value*torch.ones(self.shape, device=self.device)

    def forward(self, **kwargs) -> None:

        for _ in range(self.num_batches):
            for masks in mask_generator(self.batch_size, self.shape, self.device, **kwargs):

                x_mask = self.input_image * masks

                masked_representations = self.encoder(x_mask)

                similarity_scores = self.similarity_measure(
                    self.unmasked_representations,
                    masked_representations
                )

                for similarity_i, mask_i in zip(similarity_scores, masks.squeeze()):

                    self.sum_of_weights += mask_i

                    importance_previous_step = self.importance.clone()
                    self.importance += mask_i * (similarity_i - self.importance) / self.sum_of_weights
                    self.uncertainty += (similarity_i - self.importance) * (similarity_i - importance_previous_step) * mask_i

        return None

    @property
    def u_relax(self) -> torch.Tensor:
        return self.importance * (self.uncertainty <= self.uncertainty.median())
