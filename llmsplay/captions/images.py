"""
Image preprocessing helpers for the captioning pipeline.

Provides 'center_crop', which crops an image to its largest centered square, and
'extract_patches', which splits a batch of image tensors into flattened, fixed-size
patches for feeding into a patch-based vision model.
"""

# Initial imports
import torch

from torch import Tensor
from PIL.Image import Image


def center_crop(img: Image) -> Image:
    """
    Crop the largest centered square from 'img'.

    Parameters
    ----------
    img : Image.Image
        Image to crop, of any width and height.

    Returns
    -------
    Image.Image
        The largest square that fits in 'img', centered on it.
    """

    # Original image dimensions
    width: int
    height: int
    width, height = img.size

    # Side length of the square: limited by the shorter dimension
    new: int = min(width, height)

    # Crop box centered on the image, 'new' pixels wide and tall
    left: float = (width - new) / 2
    top: float = (height - new) / 2
    right: float = (width + new) / 2
    bottom: float = (height + new) / 2
    im: Image = img.crop((left, top, right, bottom))

    # Return the cropped image
    return im


# Listing 4.2 Converting images to patches
def extract_patches(image_tensor: Tensor, patch_size: int = 8) -> Tensor:
    """
    Split a batch of images into flattened, non-overlapping square patches.

    Parameters
    ----------
    image_tensor : Tensor
        Batch of images, shape (batch, channels, height, width).
    patch_size : int, default=8
        Height and width (in pixels) of each square patch.

    Returns
    -------
    Tensor
        Flattened patches, shape (batch, num_patches, channels * patch_size *
        patch_size), where num_patches = (height // patch_size) * (width // patch_size).
    """

    # Get the dimensions of the image tensor (batch size, channels; height/width unused)
    bs: int
    c: int
    bs, c, _, _ = image_tensor.size()

    # Define the Unfold layer with appropriate parameters; stride == kernel_size makes
    # the patches non-overlapping
    unfold: torch.nn.Unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)

    # Apply Unfold to the image tensor: (batch, channels * patch_size^2, num_patches)
    unfolded: Tensor = unfold(image_tensor)

    # Reshape the unfolded tensor to match the desired output shape
    # Output shape: BSxLxH, where L is the number of patches in each dimension
    unfolded = unfolded.transpose(1, 2).reshape(bs, -1, c * patch_size * patch_size)

    return unfolded
