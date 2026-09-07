# Initial imports
from typing import Optional, Tuple, List

import math
import torch

from torch import nn
from torch import Tensor

# Check if CUDA is available and set the device accordingly
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


# Customized imports
from llmsplay.transformers.vision.config import VisionTransformConfig


class GELU(nn.Module):
    def forward(self, input):
        return (
            0.5
            * input
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))
                )
            )
        )


# Listing 3.2 Converting image patches into a sequence of embeddings
class PatchEmbeddings(nn.Module):
    """
    Splits an image into fixed-size patches and projects each patch into a embedding
    vector via a single Conv2d layer.
    """

    def __init__(self, config: VisionTransformConfig) -> None:
        """
        Parameters
        ----------
        config : object
            Must expose 'num_channels', 'hidden_size', and 'patch_size' attributes.
        """
        super().__init__()

        # Conv2d with kernel_size == stride == patch_size acts as a non-overlapping
        # sliding window that extracts one patch per spatial position and linearly
        # projects it to hidden_size in a single operation — equivalent to slicing the
        # image into patches and applying nn.Linear to each flattened patch, but more
        # memory-efficient.
        self.projection: nn.Conv2d = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input images of shape (batch, num_channels, height, width).

        Returns
        -------
        Tensor
            Patch embeddings of shape (batch, num_patches, hidden_size) where
            num_patches = (H // patch_size) * (W // patch_size).
        """

        # (batch, hidden_size, H/patch, W/patch)
        x = self.projection(x)

        # Flatten spatial dims into a single sequence axis, then transpose so the
        # sequence length precedes the channel dim, matching the
        # (batch, seq_len, hidden) layout expected by transformer layers downstream
        x = x.flatten(2).transpose(1, 2)

        return x


# Listing 3.3 Adding positional encoding to the image embedding
class Embeddings(nn.Module):
    """
    Builds the input sequence for the transformer encoder from an image: splits the image
    into patch embeddings, prepends a learnable cls token used later for classification,
    and adds learnable position embeddings so the model knows where each patch sits in
    the image.

    Notes
    -----
    The CLS token is also a 48-value vector that will be used in the classification layer
    in our ViT to classify the image. The CLS token isn’t associated with any specific
    part of the image but is used to aggregate information across the entire image for
    the purpose of classification. This approach allows ViT to use the strength of
    transformer models for image analysis tasks.
    """

    def __init__(self, config: VisionTransformConfig) -> None:
        """
        Parameters
        ----------
        config : object
            Must expose 'num_channels', 'hidden_size', and 'patch_size' attributes.
        """

        super().__init__()
        self.config = config

        # Turns the input image into a sequence of patch embeddings
        self.patch_embeddings = PatchEmbeddings(config=self.config)

        # Learnable token prepended to the patch sequence; its final hidden state is used
        # to represent the whole image for classification
        self.cls_token: nn.Parameter = nn.Parameter(torch.randn(1, 1, config.hidden_size))

        # Total number of patches the image is split into
        num_patches: int = (config.image_size // config.patch_size) ** 2

        # Learnable position embedding, one per patch plus one for the cls token, since
        # patch embeddings alone carry no information about patch order
        self.position_embeddings: nn.Parameter = nn.Parameter(
            torch.randn(1, num_patches + 1, config.hidden_size)
        )

    def forward(self, x: Tensor) -> Tensor:

        # (batch, num_patches, hidden_size)
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()

        # Copy the cls token across the batch dimension
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Prepend the cls token to the sequence of patch embeddings
        x = torch.cat((cls_tokens, x), dim=1)

        # Add position information to each embedding (broadcast over batch)
        x = x + self.position_embeddings

        return x


# Listing 3.4 Calculating self-attention in each attention head
class AttentionHead(nn.Module):
    """
    A single self-attention head: learns query, key, and value projections and uses them
    to let each token in the sequence attend to every other token.
    """

    def __init__(
        self, hidden_size: int, attention_head_size: int, bias: bool = True
    ) -> None:
        """
        Parameters
        ----------
        hidden_size : int
            Dimensionality of the input token embeddings.
        attention_head_size : int
            Dimensionality of this head's query, key, and value projections.
        bias : bool, default=True
            Whether the query, key, and value linear layers learn a bias term.
        """

        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size

        # Linear projections that map each token embedding into this head's
        # query, key, and value spaces
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:

        # Project the input into query, key, and value vectors
        query: Tensor = self.query(x)
        key: Tensor = self.key(x)
        value: Tensor = self.value(x)

        # Dot product of each query with every key gives a raw similarity score between
        # every pair of tokens
        attention_scores: Tensor = torch.matmul(query, key.transpose(-1, -2))

        # Scale down scores to keep gradients stable as head size grows
        attention_scores: Tensor = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize scores into probabilities over the tokens attended to
        attention_probs: Tensor = nn.functional.softmax(attention_scores, dim=-1)

        # Weighted sum of value vectors, weighted by the attention probabilities
        attention_output: Tensor = torch.matmul(attention_probs, value)

        return (attention_output, attention_probs)


class MultiHeadAttention(nn.Module):
    """
    Runs several 'AttentionHead' instances in parallel over the same input, then
    concatenates and projects their outputs back to 'hidden_size'. Splitting attention
    into multiple heads lets the model attend to different kinds of relationships
    between tokens at once.
    """

    def __init__(self, config: VisionTransformConfig) -> None:
        """
        Parameters
        ----------
        config : object
            Must expose 'hidden_size' and 'num_attention_heads' attributes.
        """

        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads

        # Size of each individual head's query/key/value projections, so that all heads
        # combined add back up to 'hidden_size'
        self.attention_head_size = self.hidden_size // self.num_attention_heads

        # Combined width of all heads' outputs once concatenated
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # One independent 'AttentionHead' per attention head
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                hidden_size=self.hidden_size, attention_head_size=self.attention_head_size
            )
            self.heads.append(head)

        # Projects the concatenated head outputs back to hidden_size
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)

    def forward(
        self, x: Tensor, output_attentions: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:

        # Run every head on the same input independently
        attention_outputs: List[Tuple[Tensor, Tensor]] = [head(x) for head in self.heads]

        # Concatenate all heads' outputs along the feature dimension
        attention_output: Tensor = torch.cat(
            [attention_output for attention_output, _ in attention_outputs], dim=-1
        )

        # Mix the concatenated heads back into a single 'hidden_size' representation
        attention_output: Tensor = self.output_projection(attention_output)

        if not output_attentions:
            return (attention_output, None)
        else:
            # Stack each head's attention probabilities into one tensor, useful for
            # inspecting or visualizing what each head attended to
            attention_probs: Tensor = torch.stack(
                [attention_probs for _, attention_probs in attention_outputs], dim=1
            )
            return (attention_output, attention_probs)


class MLP(nn.Module):
    """
    A multi-layer perceptron module.
    """

    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation = GELU()
        self.dense_2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.layernorm_1 = nn.LayerNorm(config.hidden_size)
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x, output_attentions=False):
        attention_output, attention_probs = self.attention(
            self.layernorm_1(x), output_attentions=output_attentions
        )
        x = x + attention_output
        mlp_output = self.mlp(self.layernorm_2(x))
        x = x + mlp_output
        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(config.num_hidden_layers):
            block = Block(config)
            self.blocks.append(block)

    def forward(self, x, output_attentions=False):
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)


class ViTForClassfication(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config.image_size
        self.hidden_size = config.hidden_size
        self.num_classes = config.num_classes
        self.embedding = Embeddings(config)
        self.encoder = Encoder(config)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        self.apply(self._init_weights)

    def forward(self, x, output_attentions=False):
        embedding_output = self.embedding(x)
        encoder_output, all_attentions = self.encoder(
            embedding_output, output_attentions=output_attentions
        )
        logits = self.classifier(encoder_output[:, 0, :])
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=0.02,
            ).to(module.position_embeddings.dtype)
            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=0.02,
            ).to(module.cls_token.dtype)
