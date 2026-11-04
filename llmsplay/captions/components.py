"""
Building blocks shared by the captioning models.

Provides 'SinusoidalPosEmb', which embeds scalar positions/timesteps into vectors
using fixed sine/cosine frequencies, and 'AttentionBlock', a thin wrapper around
'nn.MultiheadAttention' that optionally applies causal masking for autoregressive
decoding.
"""

# Initial imports
import math
from typing import Optional

import torch
from torch import Tensor, nn

# Customized llmsplay imports
from llmsplay.captions.images import extract_patches


class SinusoidalPosEmb(nn.Module):
    """
    Embeds a batch of scalar positions (e.g. timesteps, or sequence positions) into
    'dim'-dimensional vectors using fixed (non-learned) sine/cosine frequencies, as in
    the original Transformer paper. Unlike a learned embedding table, it generalizes to
    positions never seen during training.
    """

    def __init__(self, dim: int) -> None:
        """
        Parameters
        ----------
        dim : int
            Size of the output embedding; must be even (split evenly between the sine
            and cosine halves).
        """

        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Scalar positions, shape (batch,).

        Returns
        -------
        Tensor
            Positional embeddings, shape (batch, dim).
        """

        # x: (batch,) -- run the frequency computation on the same device as the input
        device = x.device

        # Half the dimensions encode sine, the other half cosine
        half_dim = self.dim // 2

        # Geometrically spaced frequencies, from 1 down to 1/10000 -- shape (half_dim,)
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        # Outer product: each position times every frequency
        # (batch, 1) * (1, half_dim) -> (batch, half_dim)
        emb = x[:, None] * emb[None, :]

        # Concatenate sine and cosine of the scaled positions
        # (batch, half_dim) + (batch, half_dim) -> (batch, dim)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb


# Listing 4.3 Defining the multi-head attention block
class AttentionBlock(nn.Module):
    """
    Thin wrapper around 'nn.MultiheadAttention' that optionally applies a causal mask,
    so the same block can serve as either self-attention in an autoregressive decoder
    (masking=True) or cross-attention over an encoder's output (masking=False).
    """

    def __init__(
        self, hidden_size: int = 128, num_heads: int = 4, masking: bool = True
    ) -> None:
        """
        Parameters
        ----------
        hidden_size : int, default=128
            Dimensionality of the queries, keys, and values.
        num_heads : int, default=4
            Number of attention heads.
        masking : bool, default=True
            Whether to apply a causal mask so each position can only attend to
            itself and earlier positions.
        """

        super(AttentionBlock, self).__init__()
        self.masking = masking

        # Multi-head attention mechanism
        self.multihead_attn = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, batch_first=True, dropout=0.0
        )

    def forward(
        self, x_in: Tensor, kv_in: Tensor, key_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Parameters
        ----------
        x_in : Tensor
            Query input, shape (batch, seq_len_q, hidden_size).
        kv_in : Tensor
            Key/value input, shape (batch, seq_len_kv, hidden_size). Pass the same
            tensor as 'x_in' for self-attention, or an encoder's output for
            cross-attention.
        key_mask : Optional[Tensor], default=None
            Padding mask of shape (batch, seq_len_kv); True/nonzero marks positions
            in 'kv_in' to ignore.

        Returns
        -------
        Tensor
            Attention output, shape (batch, seq_len_q, hidden_size).
        """

        # x_in: (batch, seq_len_q, hidden_size) -- apply causal masking if enabled:
        # position i may only attend to positions <= i
        if self.masking:
            _, l, _ = x_in.shape

            # (seq_len_q, seq_len_q) boolean mask, True above the diagonal (future
            # positions, to be blocked)
            mask = torch.triu(torch.ones(l, l, device=x_in.device), 1).bool()
        else:
            mask = None

        # kv_in: (batch, seq_len_kv, hidden_size) used as both key and value
        # output: (batch, seq_len_q, hidden_size)
        return self.multihead_attn(
            x_in, kv_in, kv_in, attn_mask=mask, key_padding_mask=key_mask
        )[0]


# Listing 4.4 Defining a generic transformer block
class TransformerBlock(nn.Module):
    """
    A single transformer block: self-attention followed by an optional cross-attention
    (for decoders) and an MLP, each wrapped in a residual connection and followed by
    layer normalization (post-norm). Used as-is for an encoder (decoder=False) or
    stacked as a decoder layer (decoder=True) that also attends over an encoder's
    output.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_heads: int = 4,
        decoder: bool = False,
        masking: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        hidden_size : int, default=128
            Dimensionality of the tokens flowing through this block.
        num_heads : int, default=4
            Number of attention heads used by the self-/cross-attention layers.
        decoder : bool, default=False
            Whether to add a second, cross-attention layer over an external
            'kv_cross' sequence (e.g. an encoder's output), as in a decoder layer.
        masking : bool, default=True
            Whether the first (self-attention) layer applies a causal mask.
        """

        super(TransformerBlock, self).__init__()
        self.decoder = decoder

        # Layer normalization for the input
        self.norm1 = nn.LayerNorm(hidden_size)

        # Self-attention mechanism
        self.attn1 = AttentionBlock(
            hidden_size=hidden_size, num_heads=num_heads, masking=masking
        )

        # Layer normalization for the output of the first attention layer
        if self.decoder:
            self.norm2 = nn.LayerNorm(hidden_size)
            # Self-attention mechanism for the decoder with no masking
            self.attn2 = AttentionBlock(
                hidden_size=hidden_size, num_heads=num_heads, masking=False
            )

        # Layer normalization for the output before the MLP
        self.norm_mlp = nn.LayerNorm(hidden_size)

        # Multi-layer perceptron (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(
        self,
        x: Tensor,
        input_key_mask: Optional[Tensor] = None,
        cross_key_mask: Optional[Tensor] = None,
        kv_cross: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input sequence, shape (batch, seq_len, hidden_size).
        input_key_mask : Optional[Tensor], default=None
            Padding mask for 'x' in the self-attention layer, shape (batch, seq_len).
        cross_key_mask : Optional[Tensor], default=None
            Padding mask for 'kv_cross' in the cross-attention layer, shape
            (batch, seq_len_kv_cross). Only used when 'decoder=True'.
        kv_cross : Optional[Tensor], default=None
            Key/value sequence to cross-attend to, shape (batch, seq_len_kv_cross,
            hidden_size). Required when 'decoder=True'.

        Returns
        -------
        Tensor
            Output sequence, same shape as 'x': (batch, seq_len, hidden_size).
        """

        # Self-attention over x itself: (batch, seq_len, hidden_size) -> same shape,
        # added back as a residual connection
        x = self.attn1(x, x, key_mask=input_key_mask) + x
        x = self.norm1(x)

        # If decoder, perform additional cross-attention layer over kv_cross
        # (batch, seq_len, hidden_size) attending to (batch, seq_len_kv_cross,
        # hidden_size) -> (batch, seq_len, hidden_size)
        if self.decoder:
            x = self.attn2(x, kv_cross, key_mask=cross_key_mask) + x
            x = self.norm2(x)

        # Apply MLP (expands to hidden_size * 4 and back) and layer normalization;
        # shape is unchanged throughout: (batch, seq_len, hidden_size)
        x = self.mlp(x) + x

        return self.norm_mlp(x)


# Listing 4.5 Creating a vision encoder
class VisionEncoder(nn.Module):
    """
    A Vision Transformer-style image encoder: splits an image into flattened patches,
    linearly projects and position-embeds them, then contextualizes them through a
    stack of (unmasked, encoder-only) 'TransformerBlock's. Unlike a classifier ViT,
    there is no cls token or final classification head -- the output is the full
    sequence of per-patch embeddings, meant to be cross-attended to by a decoder (e.g.
    a caption generator).
    """

    def __init__(
        self,
        image_size: int,
        channels_in: int,
        patch_size: int = 16,
        hidden_size: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
    ) -> None:
        """
        Parameters
        ----------
        image_size : int
            Height and width (in pixels) of the (square) input images.
        channels_in : int
            Number of color channels in the input images.
        patch_size : int, default=16
            Height and width (in pixels) of each square patch.
        hidden_size : int, default=128
            Dimensionality each patch is projected to.
        num_layers : int, default=3
            Number of stacked 'TransformerBlock's.
        num_heads : int, default=4
            Number of attention heads per transformer block.
        """

        super(VisionEncoder, self).__init__()

        self.patch_size = patch_size

        # Projects a flattened patch (channels_in * patch_size^2 values) to hidden_size
        self.fc_in = nn.Linear(channels_in * patch_size * patch_size, hidden_size)

        # Number of patches per image: (image_size // patch_size) ** 2
        seq_length = (image_size // patch_size) ** 2

        # Learnable position embedding, one per patch; shape (1, seq_length,
        # hidden_size) so it broadcasts over the batch dimension
        self.pos_embedding = nn.Parameter(
            torch.empty(1, seq_length, hidden_size).normal_(std=0.02)
        )

        # Create multiple transformer blocks as layers (no masking: every patch can
        # attend to every other patch)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, num_heads, decoder=False, masking=False)
                for _ in range(num_layers)
            ]
        )

    def forward(self, image: Tensor) -> Tensor:
        """
        Parameters
        ----------
        image : Tensor
            Batch of images, shape (batch, channels_in, image_size, image_size).

        Returns
        -------
        Tensor
            Contextualized per-patch embeddings, shape (batch, seq_length,
            hidden_size), where seq_length = (image_size // patch_size) ** 2.
        """

        # (batch, channels_in, image_size, image_size)
        # -> (batch, seq_length, channels_in * patch_size * patch_size)
        patch_seq = extract_patches(image, patch_size=self.patch_size)

        # (batch, seq_length, channels_in * patch_size * patch_size)
        # -> (batch, seq_length, hidden_size)
        patch_emb = self.fc_in(patch_seq)

        # Add a unique position embedding to each patch embedding (broadcast over
        # batch); shape stays (batch, seq_length, hidden_size)
        embs = patch_emb + self.pos_embedding

        # Pass the embeddings through each transformer block; shape is unchanged
        # throughout: (batch, seq_length, hidden_size)
        for block in self.blocks:
            embs = block(embs)

        return embs
