"""MNIST-specific multi-crop augmentation and iBOT block masking."""

from __future__ import annotations

import math
import random

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


class MultiCropMNIST:
    """Two global and several local views, adapted to grayscale digits.

    DINOv2's color augmentation becomes brightness/contrast jitter. Horizontal
    flips are intentionally omitted because mirrored digits are not reliably
    label-preserving.
    """

    def __init__(
        self,
        global_size: int = 56,
        local_size: int = 28,
        local_crops: int = 4,
        global_scale: tuple[float, float] = (0.5, 1.0),
        local_scale: tuple[float, float] = (0.2, 0.5),
    ):
        self.local_crops = local_crops
        color = transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4)], p=0.8
        )
        normalize = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)]
        )

        def geometric(size: int, scale: tuple[float, float]):
            return transforms.RandomResizedCrop(
                size, scale=scale, ratio=(0.75, 1.3333), interpolation=InterpolationMode.BICUBIC
            )

        self.global_one = transforms.Compose(
            [geometric(global_size, global_scale), color, transforms.GaussianBlur(5, (0.1, 2.0)), normalize]
        )
        self.global_two = transforms.Compose(
            [
                geometric(global_size, global_scale),
                color,
                transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 2.0))], p=0.1),
                transforms.RandomSolarize(128, p=0.2),
                normalize,
            ]
        )
        self.local = transforms.Compose(
            [
                geometric(local_size, local_scale),
                color,
                transforms.RandomApply([transforms.GaussianBlur(3, (0.1, 2.0))], p=0.5),
                normalize,
            ]
        )

    def __call__(self, image):
        return {
            "global": [self.global_one(image), self.global_two(image)],
            "local": [self.local(image) for _ in range(self.local_crops)],
        }


def _block_mask(grid: int, target: int, device: torch.device) -> torch.Tensor:
    """Generate approximately rectangular masks until exactly target cells are set."""
    mask = torch.zeros(grid, grid, dtype=torch.bool, device=device)
    attempts = 0
    while int(mask.sum()) < target and attempts < 30:
        remaining = target - int(mask.sum())
        area = random.uniform(1, max(1, remaining))
        aspect = math.exp(random.uniform(math.log(0.3), math.log(1 / 0.3)))
        height = min(grid, max(1, round(math.sqrt(area * aspect))))
        width = min(grid, max(1, round(math.sqrt(area / aspect))))
        top = random.randrange(grid - height + 1)
        left = random.randrange(grid - width + 1)
        proposal = mask.clone()
        proposal[top : top + height, left : left + width] = True
        if int(proposal.sum()) <= target:
            mask = proposal
        attempts += 1
    if int(mask.sum()) < target:
        remaining_ids = (~mask).flatten().nonzero(as_tuple=False).flatten()
        selected = remaining_ids[torch.randperm(remaining_ids.numel(), device=device)[: target - int(mask.sum())]]
        mask.flatten()[selected] = True
    return mask.flatten()


def make_masks(
    batch_size: int,
    grid: int,
    device: torch.device,
    ratio_range: tuple[float, float] = (0.1, 0.5),
    sample_probability: float = 0.5,
) -> torch.Tensor:
    """Per-image iBOT masks with the paper/repository's ratio and sampling ranges."""
    masks = torch.zeros(batch_size, grid * grid, dtype=torch.bool, device=device)
    for index in range(batch_size):
        if random.random() <= sample_probability:
            ratio = random.uniform(*ratio_range)
            target = max(1, round(grid * grid * ratio))
            masks[index] = _block_mask(grid, target, device)
    return masks
