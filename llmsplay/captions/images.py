"""
Image preprocessing helpers for the captioning pipeline.

Provides 'center_crop', which crops an image to its largest centered square, and
'extract_patches', which splits a batch of image tensors into flattened, fixed-size
patches for feeding into a patch-based vision model.
"""

import torch

from PIL import Image


def center_crop(img: Image.Image) -> Image.Image:
    """
    Crop the largest centered square from 'img'.
    """

    width: int
    height: int
    width, height = img.size
    new: int = min(width, height)
    left: float = (width - new) / 2
    top: float = (height - new) / 2
    right: float = (width + new) / 2
    bottom: float = (height + new) / 2
    im: Image.Image = img.crop((left, top, right, bottom))

    # Return the cropped image
    return im

def extract_patches(image_tensor: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """
    Split a batch of images into flattened, non-overlapping square patches.

    Parameters
    ----------
    image_tensor : torch.Tensor
        Batch of images, shape (batch, channels, height, width).
    patch_size : int, default=16
        Height and width (in pixels) of each square patch.

    Returns
    -------
    torch.Tensor
        Flattened patches, shape (batch, num_patches, channels * patch_size *
        patch_size), where num_patches = (height // patch_size) * (width // patch_size).
    """

    # Get the dimensions of the image tensor
    bs, c, h, w = image_tensor.size()

    # Define the Unfold layer with appropriate parameters
    unfold = torch.nn.Unfold(kernel_size=patch_size,
                             stride=patch_size)

    # Apply Unfold to the image tensor
    unfolded = unfold(image_tensor)

    # Reshape the unfolded tensor to match the desired output shape
    # Output shape: BSxLxH, where L is the number of patches in each dimension
    unfolded = unfolded.transpose(1, 2).reshape(bs, -1, c * patch_size * patch_size)

    return unfolded