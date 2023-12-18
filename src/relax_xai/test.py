import torch
import torchvision
import numpy as np
import torch.nn as nn
from PIL import Image

from relax import RELAX


def load_alexnet_encoder() -> nn.Module:

    alexnet = torchvision.models.alexnet()
    encoder = nn.Sequential(
            alexnet.features,
            alexnet.avgpool,
            nn.Flatten()
        )
    encoder.eval()

    return encoder

encoder = load_alexnet_encoder()

random_numpy_array = np.random.randn(3, 345, 128)
img = Image.fromarray(random_numpy_array.astype('uint8'), 'RGB')
relax = RELAX(img, encoder, 10, 5)

with torch.no_grad(): relax.forward()

print(relax.importance.shape)
print(relax.uncertainty.shape)
print(relax.u_relax.shape)
raise