import torch
from torchvision.transforms import v2

image_transforms = v2.Compose(
    [
        v2.RandomRotation((-7.0, 7.0)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.1307,), (0.3081,)),
    ]
)
