"""
Hyperparameters for the Vision Transformer defined in 'components.py'.

Holds every value the ViT modules need to size their layers (patch size, hidden size,
number of transformer blocks/heads, etc.), tuned here for training on CIFAR-10-sized
(32x32, 3-channel) images.
"""


# Listing 3.1 Setting model hyperparameters
class VisionTransformConfig:
    """
    Plain container of Vision Transformer hyperparameters, consumed by the 'nn.Module'
    classes in 'components.py'.
    """

    # Height and width (in pixels) of each square image patch
    patch_size: int = 4

    # Dimensionality every patch is projected to (the transformer's embedding size)
    hidden_size: int = 48

    # Number of stacked transformer blocks in the encoder
    num_hidden_layers: int = 4

    # Number of self-attention heads per multi-head attention layer
    num_attention_heads: int = 4

    # Width of the MLP's hidden layer inside each transformer block
    intermediate_size: int = 4 * 48

    # Height and width (in pixels) of the (square) input images
    image_size: int = 32

    # Number of output classes for the classifier head
    num_classes: int = 10

    # Number of color channels in the input images (3 for RGB)
    num_channels: int = 3

    def __repr__(self) -> str:
        """
        String representation of the configuration
        """
        return (
            f"Vision Transformer Configuration Model Hyperparameters\n"
            f"\tPatch Size: {self.patch_size}\n"
            f"\tHidden Size: {self.hidden_size}\n"
            f"\tNumber of Hidden Layers: {self.num_hidden_layers}\n"
            f"\tNumber of Attention Heads: {self.num_attention_heads}\n"
            f"\tIntermediate Size: {self.intermediate_size}\n"
            f"\tImage Size: {self.image_size}\n"
            f"\tNumber of Classes: {self.num_classes}\n"
            f"\tNumber of Channels: {self.num_channels}\n"
        )
