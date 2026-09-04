# Listing 3.1 Setting model hyperparameters
class VisionTransformConfig:

    patch_size: int = 4  # Each image patch has a height and width of 4 pixels
    hidden_size: int = 48  # Each image patch is converted to a 48-value tensor
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    intermediate_size: int = 4 * 48
    image_size: int = 32
    num_classes: int = 10
    num_channels: int = 3

    def __repr__(self) -> None:
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
