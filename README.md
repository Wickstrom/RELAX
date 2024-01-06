# RELAX: Representation Learning Explainability

</p>
<p align="center">
  <img width="600" src="https://github.com/Wickstrom/RELAX/blob/main/relax-ramework.png">
</p>

This repository contains code for RELAX, a framework for representation learning explainability. RELAX is based on perturbation-based explainability and work by measuring the change in the representation space as parts of the input are masked out.

When should you use RELAX? If your output is a vector representation and you have no label information.

More information can be found in the paper: <b>Representation Learning Explainability </b><a href="https://link.springer.com/article/10.1007/s11263-023-01773-2#citeas">(Wickstrøm et al., 2023)</a>, <a href="https://arxiv.org/abs/2112.10161">(Wickstrøm et al., 2022)</a>

You can see RELAX used in practice in medical image retrieval in the paper: <b>A clinically motivated self-supervised approach for content-based image retrieval of CT liver images </b> <a href="https://www.sciencedirect.com/science/article/pii/S0895611123000575">(Wickstrøm et al., 2023)</a>, <a href="https://arxiv.org/abs/2207.04812">(Wickstrøm et al., 2022)</a>

## Installation

RELAX can be installed using pip as follows:

```setup
pip install relax-xai
```

and requires torch and torchvision installed.

## Toy example

Here is a very simple example showing the basic structure for how to use RELAX. The input is assumed to be organized in (channel, height, width) format, and preprocessed using the Imagenet normalization (for encoders pretrained on Imagenet). The "imagenet_iamge_transforms"-function also reshapes the image into a square image (224, 224 by default) and places the image on the desired device.

```python
import torch
import torchvision
import torch.nn as nn
from relax_xai.relax import RELAX
from relax_xai.utils import imagenet_image_transforms

x = torch.rand(3, 313, 210)  # Generate some random data.
x = imagenet_image_transforms(device='cpu', new_shape_of_image=224)(x) # Resize image and apply Imagenet normalization.

alexnet = torchvision.models.alexnet() # Load Alexnet model
encoder = nn.Sequential(
            alexnet.features,
            alexnet.avgpool,
            nn.Flatten()
        ) # Remove classification head and only keep encoder part.
encoder.eval() # Put encoder in evaluation mode.


relax = RELAX(x, encoder) # Initialize RELAX
with torch.no_grad(): relax.forward() # Run RELAX (with torch.no_grad() avoid memory issues).

print(relax.importance) # Explanation for representation.
print(relax.uncertainty) # Uncertainty in explanation.
```

## Citation

If you find RELAX interesting and use it in your research, use the following Bibtex annotation to cite:

```bibtex
@article{wickstrom2023relax,
  author  = {Wickstr\o{}m, Kristoffer K. and Trosten, Daniel J. and L\o{}kse, Sigurd and Boubekki, Ahc\`{e}ne and Mikalsen, Karl \o{}yvind and Kampffmeyer, Michael C. and Jenssen, Robert},
  title   = {RELAX: Representation Learning Explainability},
  journal = {International Journal of Computer Vision},
  year    = {2023},
  volume  = {131},
  number  = {6},
  pages   = {1584–1610},
  url     = {https://doi.org/10.1007/s11263-023-01773-2}
}
```
