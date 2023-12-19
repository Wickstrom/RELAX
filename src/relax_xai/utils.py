import torch
import torchvision

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def imagenet_image_transforms(device: str, new_shape_of_image: int = 224):
    """
    Returns transformations that takes a torch tensor and transforms it into a new tensor
    of size (1, C, new_shape_of_image, new_shape_of_image), normalizes the image according
    to the statistics from the Imagenet dataset, and puts the tensor on the desired device.
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((new_shape_of_image, new_shape_of_image)),
        torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        torchvision.transforms.Lambda(unsqeeze_image),
        ToDevice(device),
    ])

    return transform

class ToDevice(torch.nn.Module):
    """
    Sends the input object to the device specified in the
    object's constructor by calling .to(device) on the object.
    """
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, img):
        return img.to(self.device)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device})"

def unsqeeze_image(input_image: torch.Tensor) -> torch.Tensor:
    return input_image.unsqueeze(0)