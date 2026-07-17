"""MNIST-specific multi-crop views and iBOT block masking."""

from __future__ import annotations

import math
import random

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def upscale_bbox(image: torch.Tensor, size: int = 56) -> torch.Tensor:
    """Apply the successful custom-I-JEPA upscale/bbox/stretch preprocessing.

    The order deliberately matches :func:`mnist_ssl.baselines.mae.bbox_rescale`: resize the
    original 28x28 tensor to ``size`` first, locate the nonzero foreground on
    that upscaled image, crop to its tight bounding box, then stretch the crop
    back to ``size`` x ``size``. Aspect ratio is intentionally not preserved.
    """
    image = TF.resize(image, [size, size], antialias=True)
    foreground = image[0] > 0
    if not foreground.any():
        return image
    rows = torch.where(foreground.any(dim=1))[0]
    columns = torch.where(foreground.any(dim=0))[0]
    crop = image[:, rows[0] : rows[-1] + 1, columns[0] : columns[-1] + 1]
    return TF.resize(crop, [size, size], antialias=True)


class EvaluationTransform:
    """Deterministic teacher input with optional upscale-bbox preprocessing."""

    def __init__(self, image_size: int = 56, preprocess: bool = True):
        self.image_size = image_size
        self.preprocess = preprocess
        self.normalize = transforms.Normalize(MNIST_MEAN, MNIST_STD)

    def __call__(self, image) -> torch.Tensor:
        image = TF.to_tensor(image)
        if self.preprocess:
            image = upscale_bbox(image, self.image_size)
        else:
            image = TF.resize(image, [self.image_size, self.image_size], antialias=True)
        return self.normalize(image)


class MultiCropMNIST:
    """Two global and several local random crops of grayscale digits.

    Photometric augmentation is disabled by default for the crop-only ablation.
    It can be enabled to recover the previous DINOv2-inspired brightness,
    contrast, blur, and solarization pipeline. Horizontal flips are always
    omitted because mirrored digits are not reliably label-preserving.
    """

    def __init__(
        self,
        global_size: int = 56,
        local_size: int = 28,
        local_crops: int = 4,
        global_scale: tuple[float, float] = (0.5, 1.0),
        local_scale: tuple[float, float] = (0.2, 0.5),
        preprocess: bool = True,
        photometric_augmentations: bool = False,
    ):
        self.local_crops = local_crops
        self.global_size = global_size
        self.preprocess = preprocess
        self.photometric_augmentations = photometric_augmentations
        color = transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4)], p=0.8
        )
        normalize = transforms.Normalize(MNIST_MEAN, MNIST_STD)

        def geometric(size: int, scale: tuple[float, float]):
            return transforms.RandomResizedCrop(
                size, scale=scale, ratio=(0.75, 1.3333), interpolation=InterpolationMode.BICUBIC
            )

        if photometric_augmentations:
            global_one_photometric = [color, transforms.GaussianBlur(5, (0.1, 2.0))]
            global_two_photometric = [
                color,
                transforms.RandomApply(
                    [transforms.GaussianBlur(5, (0.1, 2.0))], p=0.1
                ),
                # Views are tensors in [0, 1] after the common preprocessing.
                transforms.RandomSolarize(0.5, p=0.2),
            ]
            local_photometric = [
                color,
                transforms.RandomApply(
                    [transforms.GaussianBlur(3, (0.1, 2.0))], p=0.5
                ),
            ]
        else:
            global_one_photometric = []
            global_two_photometric = []
            local_photometric = []

        self.global_one = transforms.Compose(
            [geometric(global_size, global_scale), *global_one_photometric, normalize]
        )
        self.global_two = transforms.Compose(
            [geometric(global_size, global_scale), *global_two_photometric, normalize]
        )
        self.local = transforms.Compose(
            [geometric(local_size, local_scale), *local_photometric, normalize]
        )

    def __call__(self, image):
        # Match custom I-JEPA before drawing DINO views:
        # 28x28 -> upscale to 56x56 -> bbox crop/stretch -> random crops -> network.
        image = TF.to_tensor(image)
        if self.preprocess:
            image = upscale_bbox(image, self.global_size)
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
