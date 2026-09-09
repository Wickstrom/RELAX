# RELAX: Representation Learning Explainability

[![PyPI version](https://img.shields.io/pypi/v/relax-xai.svg)](https://pypi.org/project/relax-xai/)
[![Python versions](https://img.shields.io/pypi/pyversions/relax-xai.svg)](https://pypi.org/project/relax-xai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/Wickstrom/RELAX/actions/workflows/ci.yml/badge.svg)](https://github.com/Wickstrom/RELAX/actions/workflows/ci.yml)

[![arXiv](https://img.shields.io/badge/arXiv-2112.10161-b31b1b.svg)](https://arxiv.org/abs/2112.10161)
[![IJCV](https://img.shields.io/badge/IJCV-2023-007ec7.svg)](https://doi.org/10.1007/s11263-023-01773-2)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Wickstrom/RELAX/blob/main/notebooks/getting_started_with_relax.ipynb)

<p align="center">
  <img width="600" src="https://raw.githubusercontent.com/Wickstrom/RELAX/main/relax-framework.png">
</p>

This repository contains code for RELAX, a framework for representation learning explainability. RELAX is based on perturbation-based explainability and works by measuring the change in the representation space as parts of the input are masked out.

When should you use RELAX? If your output is a vector representation and you have no label information.

More information can be found in the paper: **RELAX: Representation Learning Explainability** [Wickstrøm et al., 2023, IJCV](https://doi.org/10.1007/s11263-023-01773-2), [arXiv](https://arxiv.org/abs/2112.10161).

You can see RELAX used in practice in medical image retrieval in the paper: **A clinically motivated self-supervised approach for content-based image retrieval of CT liver images** [Wickstrøm et al., 2023](https://www.sciencedirect.com/science/article/pii/S0895611123000575), [arXiv](https://arxiv.org/abs/2207.04812).

## Installation

RELAX can be installed using pip:

```setup
pip install relax-xai
```

This also installs the required dependencies (`torch` and `torchvision`). To install the latest version from source:

```setup
pip install git+https://github.com/Wickstrom/RELAX.git
```

## Toy example

Here is a very simple example showing the basic structure for how to use RELAX. The input is assumed to be organized in (channel, height, width)-format, and preprocessed using the Imagenet normalization (for encoders pretrained on Imagenet). The `imagenet_image_transforms` function also reshapes the image into a square image ((224 x 224) by default) and places the image on the desired device.

```python
import torch
import torchvision
import torch.nn as nn
from relax_xai import RELAX
from relax_xai.utils import imagenet_image_transforms

x = torch.rand(3, 313, 210)  # Generate some random data.
x = imagenet_image_transforms(device="cpu", new_shape_of_image=224)(
    x
)  # Resize image and apply Imagenet normalization.

alexnet = torchvision.models.alexnet()  # Load Alexnet model
encoder = nn.Sequential(
    alexnet.features, alexnet.avgpool, nn.Flatten()
)  # Remove classification head and only keep encoder part.
encoder.eval()  # Put encoder in evaluation mode.


relax = RELAX(x, encoder)  # Initialize RELAX
relax.forward()  # Run RELAX (gradient tracking is disabled internally).

# The importance-heatmap produced by RELAX, and its associated uncertainty, can be
# accessed in relax.importance and relax.uncertainty. Also, an uncertainty-filtered
# version of relax.importance (U-RELAX) can be accessed in relax.u_relax.
```

## Getting started with RELAX

The `notebooks`-folder contains a notebook called `getting_started_with_relax`, where you can test out RELAX. We recommend to use Google Colab with GPU-support enabled to speed up computation: [open the notebook in Colab](https://colab.research.google.com/github/Wickstrom/RELAX/blob/main/notebooks/getting_started_with_relax.ipynb).

## Important hyperparameters

There are several hyperparameters that can affect the performance of RELAX. Masking-related hyperparameters are passed as keyword arguments to `relax.forward(...)`.

- **`batch_size` and `num_batches`:** The number of masks used is governed by these two parameters. The total number of masks is batch_size*num_batches. Since the default number of masks is high (3000), we need to perform the masking+encoding in a batch-wise manner to avoid out-of-memory issues. The default number of masks is determined using a bound on the estimator of importance (see paper for more details). Reducing either `batch_size` or `num_batches` will make RELAX faster, but could decrease the quality of the explanations.
- **`num_cells` and `probability_of_drop`:** A mask in RELAX is generated following the same procedure as in [RISE](https://arxiv.org/abs/1806.07421). In this procedure, an image (`num_cells` x `num_cells`) smaller than the original image, with each pixel following a Bernoulli distribution with `probability_of_drop`, is randomly sampled. The default value for `num_cells` is 7 and `probability_of_drop` is 0.5. This is selected with images of size (224 x 224) in mind. This selection can also work okay for images of smaller sizes (112 x 112) or larger size (224 x 224), but for smaller or bigger than this it is likely necessary to tune these hyperparameters.

## Citation

If you find RELAX interesting and use it in your research, cite it using the BibTeX annotation below (also available as [CITATION.cff](CITATION.cff)):

```bibtex
@article{wickstrom2023relax,
  author  = {Wickstr\o{}m, Kristoffer K. and Trosten, Daniel J. and L\o{}kse, Sigurd and Boubekki, Ahc\`{e}ne and Mikalsen, Karl \o{}yvind and Kampffmeyer, Michael C. and Jenssen, Robert},
  title   = {RELAX: Representation Learning Explainability},
  journal = {International Journal of Computer Vision},
  year    = {2023},
  volume  = {131},
  number  = {6},
  pages   = {1584–1610},
  doi     = {https://doi.org/10.1007/s11263-023-01773-2}
}
```

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for how to set up a development environment and the release process.

## License

RELAX is released under the [MIT License](LICENSE).
