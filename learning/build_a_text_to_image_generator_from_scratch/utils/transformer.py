"""
Description.
"""

# Initial imports
import math
from copy import deepcopy
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

# Import custom modules
from .chapter2 import DEVICE, PAD


class Batch:
    """
    Container for a single training batch with masking.
    """

    def __init__(
        self,
        src: np.ndarray,
        tgt: Optional[np.ndarray] = None,
        pad: int = PAD,
    ) -> None:
        """
        Build source / target tensors and their masks.

        Parameters
        ----------
        src : np.ndarray
            Source token IDs of shape (batch_size, src_len).
        tgt : Optional[np.ndarray], default=None
            Target token IDs of shape (batch_size, tgt_len).
            When None only the source side is prepared.
        pad : int, default=PAD
            Padding token index used to build masks.
        """

        src: Tensor = torch.from_numpy(src).to(DEVICE).long()
        self.src: Tensor = src

        # Note: unsqueeze(-2) adds a broadcast dim so the mask shape becomes
        # (batch, 1, src_len), compatible with multi-head attention's
        # (batch, heads, q, k)
        self.src_mask: Tensor = (src != pad).unsqueeze(-2)

        if tgt is not None:

            tgt: Tensor = torch.from_numpy(tgt).to(DEVICE).long()

            # Teacher-forcing split: the decoder receives all tokens except the last one
            # as input (tgt) and predicts all tokens except the first one (tgt_y). This
            # one-position shift lets the model learn to predict the next token at every
            # step
            self.tgt: Tensor = tgt[:, :-1]
            self.tgt_y: Tensor = tgt[:, 1:]

            # Combines padding mask with a causal (subsequent) mask so the decoder cannot
            # attend to future positions or padding
            self.tgt_mask: Tensor = make_std_mask(tgt=self.tgt, pad=pad)

            # Token count excluding padding, used to normalise the loss per real token
            self.ntokens: Tensor = (self.tgt_y != pad).data.sum()


# Listing 2.5 an encoder-decoder transformer
class Transformer(nn.Module):
    """
    Encoder-decoder transformer.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        src_embed: nn.Module,
        tgt_embed: nn.Module,
        generator: nn.Module,
    ) -> None:
        """
        Initialize the transformer with encoder, decoder, embedding, and generator.

        Parameters
        ----------
        encoder : nn.Module
            Encoder stack.
        decoder : nn.Module
            Decoder stack.
        src_embed : nn.Module
            Source token embedding and positional encoding.
        tgt_embed : nn.Module
            Target token embedding and positional encoding.
        generator : nn.Module
            Final linear projection and softmax for token generation.
        """

        super().__init__()
        self.encoder: nn.Module = encoder
        self.decoder: nn.Module = decoder
        self.src_embed: nn.Module = src_embed
        self.tgt_embed: nn.Module = tgt_embed
        self.generator: nn.Module = generator

    def encode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Encode the source sequence.

        Parameters
        ----------
        src : Tensor
            Source token IDs of shape (batch_size, src_seq_len).
        src_mask : Tensor
            Source padding mask of shape (batch_size, 1, src_seq_len).

        Returns
        -------
        Tensor
            Encoder output (memory) of shape (batch_size, src_seq_len, d_model).
        """

        out: Tensor = self.encoder(self.src_embed(src), src_mask)

        return out

    def decode(
        self, memory: Tensor, src_mask: Tensor, tgt: Tensor, tgt_mask: Tensor
    ) -> Tensor:
        """
        Decode the target sequence conditioned on encoder memory.

        Parameters
        ----------
        memory : Tensor
            Encoder output of shape (batch_size, src_seq_len, d_model).
        src_mask : Tensor
            Source padding mask of shape (batch_size, 1, src_seq_len).
        tgt : Tensor
            Target token IDs of shape (batch_size, tgt_seq_len).
        tgt_mask : Tensor
            Combined target padding and causal mask of shape
            (batch_size, tgt_seq_len, tgt_seq_len).

        Returns
        -------
        Tensor
            Decoder output of shape (batch_size, tgt_seq_len, d_model).
        """

        out: Tensor = self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        return out

    def forward(
        self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor
    ) -> Tensor:
        """
        Run the full encoder-decoder pass.

        Parameters
        ----------
        src : Tensor
            Source token IDs of shape (batch_size, src_seq_len).
        tgt : Tensor
            Target token IDs of shape (batch_size, tgt_seq_len).
        src_mask : Tensor
            Source padding mask of shape (batch_size, 1, src_seq_len).
        tgt_mask : Tensor
            Combined target padding and causal mask of shape
            (batch_size, tgt_seq_len, tgt_seq_len).

        Returns
        -------
        Tensor
            Decoder output of shape (batch_size, tgt_seq_len, d_model).
        """

        memory: Tensor = self.encode(src, src_mask)
        output: Tensor = self.decode(memory, src_mask, tgt, tgt_mask)
        return output


class Encoder(nn.Module):
    """
    Transformer encoder stack.
    """

    def __init__(self, layer: nn.Module, N: int) -> None:
        """
        Initialize an encoder with N repeated encoder layers.

        Parameters
        ----------
        layer : nn.Module
            Encoder layer prototype to clone.
        N : int
            Number of encoder layers.
        """

        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList([deepcopy(layer) for _ in range(N)])
        self.norm: LayerNorm = LayerNorm(layer.size)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Apply all encoder layers followed by layer normalization.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, src_seq_len, d_model).
        mask : Tensor
            Source padding mask of shape (batch_size, 1, src_seq_len).

        Returns
        -------
        Tensor
            Encoded representation of shape (batch_size, src_seq_len, d_model).
        """

        for layer in self.layers:

            x = layer(x, mask)

        output: Tensor = self.norm(x)

        return output


class EncoderLayer(nn.Module):
    """
    Single transformer encoder layer.
    """

    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        feed_forward: nn.Module,
        dropout: float,
    ) -> None:
        """
        Initialize encoder layer components.

        Parameters
        ----------
        size : int
            Model dimension.
        self_attn : nn.Module
            Self-attention module.
        feed_forward : nn.Module
            Position-wise feed-forward module.
        dropout : float
            Dropout probability used in residual sublayers.
        """

        super().__init__()
        self.self_attn: nn.Module = self_attn
        self.feed_forward: nn.Module = feed_forward
        self.sublayer: nn.ModuleList = nn.ModuleList(
            [deepcopy(SublayerConnection(size, dropout)) for _ in range(2)]
        )
        self.size: int = size

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Apply self-attention and feed-forward blocks with residual connections.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, src_seq_len, d_model).
        mask : Tensor
            Source padding mask of shape (batch_size, 1, src_seq_len).

        Returns
        -------
        Tensor
            Output tensor of shape (batch_size, src_seq_len, d_model).
        """

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        output: Tensor = self.sublayer[1](x, self.feed_forward)

        return output


class SublayerConnection(nn.Module):
    """
    Residual connection followed by dropout around a pre-norm sublayer.
    """

    def __init__(self, size: int, dropout: float) -> None:
        """
        Initialize sublayer connection components.

        Parameters
        ----------
        size : int
            Model dimension.
        dropout : float
            Dropout probability for the residual branch.
        """

        super().__init__()
        self.norm: LayerNorm = LayerNorm(size)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, sublayer: Callable[[Tensor], Tensor]) -> Tensor:
        """
        Apply layer norm, sublayer transform, dropout, and residual addition.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, seq_len, d_model).
        sublayer : Callable[[Tensor], Tensor]
            Function or module mapping normalized inputs to transformed outputs.

        Returns
        -------
        Tensor
            Residual output tensor of shape (batch_size, seq_len, d_model).
        """

        output: Tensor = x + self.dropout(sublayer(self.norm(x)))

        return output


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x_zscore = (x - mean) / torch.sqrt(std**2 + self.eps)
        output = self.a_2 * x_zscore + self.b_2
        return output


# Create a decoder
class Decoder(nn.Module):
    """
    Transformer decoder stack.
    """

    def __init__(self, layer: nn.Module, N: int) -> None:
        """
        Initialize a decoder with N repeated decoder layers.

        Parameters
        ----------
        layer : nn.Module
            Decoder layer prototype to clone.
        N : int
            Number of decoder layers.
        """

        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList([deepcopy(layer) for _ in range(N)])
        self.norm: LayerNorm = LayerNorm(layer.size)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
    ) -> Tensor:
        """
        Apply all decoder layers followed by layer normalization.

        Parameters
        ----------
        x : Tensor
            Decoder input tensor of shape (batch_size, tgt_seq_len, d_model).
        memory : Tensor
            Encoder output tensor of shape (batch_size, src_seq_len, d_model).
        src_mask : Tensor
            Source padding mask of shape (batch_size, 1, src_seq_len).
        tgt_mask : Tensor
            Target causal and padding mask of shape
            (batch_size, tgt_seq_len, tgt_seq_len).

        Returns
        -------
        Tensor
            Decoded representation of shape (batch_size, tgt_seq_len, d_model).
        """

        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)

        output: Tensor = self.norm(x)

        return output


# Listing 2.6 Creating a decoder layer
class DecoderLayer(nn.Module):
    """
    Single transformer decoder layer.
    """

    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        src_attn: nn.Module,
        feed_forward: nn.Module,
        dropout: float,
    ) -> None:
        """
        Initialize decoder layer components.

        Parameters
        ----------
        size : int
            Model dimension.
        self_attn : nn.Module
            Masked self-attention module for decoder inputs.
        src_attn : nn.Module
            Cross-attention module over encoder memory.
        feed_forward : nn.Module
            Position-wise feed-forward module.
        dropout : float
            Dropout probability used in residual sublayers.
        """

        super().__init__()
        self.size: int = size
        self.self_attn: nn.Module = self_attn
        self.src_attn: nn.Module = src_attn
        self.feed_forward: nn.Module = feed_forward
        self.sublayer: nn.ModuleList = nn.ModuleList(
            [deepcopy(SublayerConnection(size, dropout)) for _ in range(3)]
        )

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
    ) -> Tensor:
        """
        Apply masked self-attention, cross-attention, and feed-forward blocks.

        Parameters
        ----------
        x : Tensor
            Decoder input tensor of shape (batch_size, tgt_seq_len, d_model).
        memory : Tensor
            Encoder output tensor of shape (batch_size, src_seq_len, d_model).
        src_mask : Tensor
            Source padding mask of shape (batch_size, 1, src_seq_len).
        tgt_mask : Tensor
            Target causal and padding mask of shape
            (batch_size, tgt_seq_len, tgt_seq_len).

        Returns
        -------
        Tensor
            Output tensor of shape (batch_size, tgt_seq_len, d_model).
        """

        # Target masked self-attention layer that prevents attending to future positions
        # in the sequence
        x = self.sublayer[0](
            x, lambda tensor: self.self_attn(tensor, tensor, tensor, tgt_mask)
        )

        # Cross-attention layer between the decoder and encoder outputs layer between
        # the two languages. The query comes from the previous decoder layer, and the
        # key and value come from the encoder outputs
        x = self.sublayer[1](
            x, lambda tensor: self.src_attn(tensor, memory, memory, src_mask)
        )

        # Feed forward network
        output = self.sublayer[2](x, self.feed_forward)

        return output


class Embeddings(nn.Module):
    """
    Token embedding layer scaled by the square root of model dimension.
    """

    def __init__(self, d_model: int, vocab: int) -> None:
        """
        Initialize embedding lookup table.

        Parameters
        ----------
        d_model : int
            Embedding dimension.
        vocab : int
            Vocabulary size.
        """

        super().__init__()
        self.lut: nn.Embedding = nn.Embedding(vocab, d_model)
        self.d_model: int = d_model

    def forward(self, x: Tensor) -> Tensor:
        """
        Embed token IDs and apply transformer scaling.

        Parameters
        ----------
        x : Tensor
            Input token IDs of shape (batch_size, sequence_length).

        Returns
        -------
        Tensor
            Embedded token representations.
        """

        out: Tensor = self.lut(x) * math.sqrt(self.d_model)

        return out


# Listing 2.3 Designing the PositionalEncoding class
class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer inputs.
    """

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000) -> None:
        """
        Initialize positional encodings and dropout layer.

        Parameters
        ----------
        d_model : int
            Embedding dimension.
        dropout : float
            Dropout probability.
        max_len : int, default=5000
            Maximum sequence length for precomputed encodings.
        """

        super().__init__()

        self.dropout: nn.Dropout = nn.Dropout(p=dropout)

        pe: Tensor = torch.zeros(max_len, d_model, device=DEVICE)
        position: Tensor = torch.arange(0.0, max_len, device=DEVICE).unsqueeze(1)
        div_term: Tensor = torch.exp(
            torch.arange(0.0, d_model, 2, device=DEVICE) * -(math.log(10000.0) / d_model)
        )
        pe_pos: Tensor = torch.mul(position, div_term)

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(pe_pos)
        pe[:, 1::2] = torch.cos(pe_pos)
        pe = pe.unsqueeze(0)

        # Not a trainable parameter, but we want it to be part of the model's state
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to token embeddings.

        Parameters
        ----------
        x : Tensor
            Input embeddings of shape (batch_size, sequence_length, d_model).

        Returns
        -------
        Tensor
            Position-aware embeddings after dropout.
        """

        # Adds positional encoding to word embedding. Note that `requires_grad_(False)`
        # means there’s no need to train these values
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)

        # Apply dropout to the combined embeddings
        out: Tensor = self.dropout(x)

        return out


class MultiHeadedAttention(nn.Module):
    """
    Multi-head attention module.
    """

    def __init__(self, h: int, d_model: int, dropout: float = 0.1) -> None:
        """
        Initialize multi-head attention module.

        Parameters
        ----------
        h : int
            Number of attention heads.
        d_model : int
            Model embedding dimension.
        dropout : float, default=0.1
            Dropout probability applied to attention weights.
        """

        super().__init__()

        if d_model % h != 0:
            raise ValueError("'d_model' must be divisible by 'h'")

        self.d_k: int = d_model // h
        self.h: int = h
        self.linears: nn.ModuleList = nn.ModuleList(
            [deepcopy(nn.Linear(d_model, d_model)) for i in range(4)]
        )
        self.attn: Optional[Tensor] = None
        self.dropout: nn.Dropout = nn.Dropout(p=dropout)

    def _reshape_projection(self, x: Tensor, batch_size: int) -> Tensor:
        """
        Reshape projected input to split embedding dimension across attention heads.

        Parameters
        ----------
        x : Tensor
            Projected input of shape (batch_size, seq_len, d_model).
        batch_size : int
            Batch size for reshaping.

        Returns
        -------
        Tensor
            Tensor of shape (batch_size, h, seq_len, d_k).
        """

        out: Tensor = x.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        return out

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply multi-head attention to query, key, and value tensors.

        Parameters
        ----------
        query : Tensor
            Query tensor of shape (batch_size, seq_len, d_model).
        key : Tensor
            Key tensor of shape (batch_size, seq_len, d_model).
        value : Tensor
            Value tensor of shape (batch_size, seq_len, d_model).
        mask : Optional[Tensor], default=None
            Attention mask broadcastable to the attention score shape.

        Returns
        -------
        Tensor
            Output tensor of shape (batch_size, seq_len, d_model).
        """

        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches: int = query.size(0)

        # linears[0], [1], [2] project Q, K, V independently
        query, key, value = [
            self._reshape_projection(l(x), nbatches)
            for l, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        # linears[3] projects concatenated heads back to 'd_model'
        output: Tensor = self.linears[-1](x)

        return output


class Generator(nn.Module):
    """
    Final linear projection and log-softmax for token generation.
    """

    def __init__(self, d_model: int, vocab: int) -> None:
        """
        Initialize the generator projection layer.

        Parameters
        ----------
        d_model : int
            Model embedding dimension.
        vocab : int
            Vocabulary size.
        """

        super().__init__()
        self.proj: nn.Linear = nn.Linear(d_model, vocab)

    def forward(self, x: Tensor) -> Tensor:
        """
        Project decoder output to vocabulary log-probabilities.

        Parameters
        ----------
        x : Tensor
            Decoder output of shape
            (batch_size, seq_len, d_model).

        Returns
        -------
        Tensor
            Log-probabilities of shape
            (batch_size, seq_len, vocab).
        """

        out: Tensor = self.proj(x)
        probs: Tensor = nn.functional.log_softmax(out, dim=-1)

        return probs


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network used in transformer blocks.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        """
        Initialize two linear projections with dropout in between.

        Parameters
        ----------
        d_model : int
            Input and output embedding dimension.
        d_ff : int
            Hidden dimension of the feed-forward layer.
        dropout : float, default=0.1
            Dropout probability applied after the first linear projection.
        """

        super().__init__()
        self.w_1: nn.Linear = nn.Linear(d_model, d_ff)
        self.w_2: nn.Linear = nn.Linear(d_ff, d_model)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply feed-forward projections to each position independently.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, seq_len, d_model).

        Returns
        -------
        Tensor
            Output tensor of shape (batch_size, seq_len, d_model).
        """

        h1: Tensor = self.w_1(x)
        h2: Tensor = self.dropout(h1)
        output: Tensor = self.w_2(h2)

        return output


class LabelSmoothing(nn.Module):
    """
    Label-smoothing loss using KL divergence.

    Instead of a hard one-hot target, probability mass is redistributed: the true class
    keeps `1 - smoothing` and the remaining mass is spread uniformly over all other
    non-padding classes.
    """

    def __init__(
        self,
        size: int,
        padding_idx: int,
        smoothing: float = 0.1,
    ) -> None:
        """
        Initialize the label-smoothing criterion.

        Parameters
        ----------
        size : int
            Vocabulary size (number of classes).
        padding_idx : int
            Index of the padding token, excluded from the smoothed distribution.
        smoothing : float, default=0.1
            Fraction of probability mass to redistribute away from the true class.
        """

        super().__init__()
        self.criterion: nn.KLDivLoss = nn.KLDivLoss(reduction="sum")
        self.padding_idx: int = padding_idx
        self.confidence: float = 1.0 - smoothing
        self.smoothing: float = smoothing
        self.size: int = size
        self.true_dist: Optional[Tensor] = None

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        """
        Compute the smoothed KL-divergence loss.

        Parameters
        ----------
        x : Tensor
            Log-probabilities of shape
            (batch_size, vocab_size).
        target : Tensor
            Ground-truth token IDs of shape
            (batch_size,).

        Returns
        -------
        Tensor
            Scalar KL-divergence loss.
        """

        assert x.size(1) == self.size

        true_dist: Tensor = x.data.clone()

        # Spread smoothing mass uniformly over all classes except the true class and
        # padding, hence (size - 2) in the denominator
        true_dist.fill_(self.smoothing / (self.size - 2))

        # Place the confidence value at the index of the true class for each sample in
        # the batch. scatter_(dim, index, value) writes `value` into positions given by
        # `index` along `dim`
        true_dist.scatter_(
            1,
            target.data.unsqueeze(1),
            self.confidence,
        )

        # Zero out the padding column so no probability mass is assigned to the padding
        # token
        true_dist[:, self.padding_idx] = 0

        # For samples whose target IS the padding token, zero out the entire row so they
        # contribute nothing to the loss
        mask: Tensor = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)

        self.true_dist = true_dist
        output: Tensor = self.criterion(x, true_dist.clone().detach())

        return output


class NoamOpt:
    """
    Learning-rate scheduler from "Attention Is All You Need".

    Implements a warmup phase where the rate increases linearly, then decays
    proportionally to the inverse square root of the step number.
    """

    def __init__(
        self,
        model_size: int,
        factor: float,
        warmup: int,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """
        Initialize the Noam learning-rate scheduler.

        Parameters
        ----------
        model_size : int
            Model embedding dimension (d_model). Controls the base magnitude of the
            learning rate.
        factor : float
            Multiplicative scaling factor for the rate.
        warmup : int
            Number of warm-up steps during which the learning rate increases linearly.
        optimizer : torch.optim.Optimizer
            Wrapped optimizer whose param-group learning rates will be overwritten each
            step.
        """

        self.optimizer: torch.optim.Optimizer = optimizer
        self._step: int = 0
        self.warmup: int = warmup
        self.factor: float = factor
        self.model_size: int = model_size
        self._rate: float = 0

    def step(self) -> None:
        """
        Advance one training step: update the learning rate for every param group, then
        call the underlying optimizer.
        """

        self._step += 1
        rate: float = self.rate()

        for p in self.optimizer.param_groups:
            p["lr"] = rate

        self._rate = rate
        self.optimizer.step()

    def rate(self, step: Optional[int] = None) -> float:
        """
        Compute the learning rate for a given step.

        Parameters
        ----------
        step : Optional[int], default=None
            Step number to evaluate. Uses the internal counter when None.

        Returns
        -------
        float
            Learning rate for the given step.
        """

        if step is None:
            step = self._step

        # The min() selects between two regimes:
        #   step ** -0.5  — inverse-sqrt decay after
        #                   warmup
        #   step * warmup ** -1.5 — linear ramp during
        #                          warmup (grows with
        #                          step)
        # The crossover happens exactly at step == warmup
        output: float = self.factor * (
            self.model_size ** (-0.5)
            * min(
                step ** (-0.5),
                step * self.warmup ** (-1.5),
            )
        )

        return output


class SimpleLossCompute:
    """
    Combines token generation, loss computation, and an optional optimizer step into a
    single callable.
    """

    def __init__(
        self,
        generator: Generator,
        criterion: nn.Module,
        opt: Optional[NoamOpt] = None,
    ) -> None:
        """
        Initialize loss-compute helper.

        Parameters
        ----------
        generator : Generator
            Projects decoder output to log-probs.
        criterion : nn.Module
            Loss function (e.g. LabelSmoothing).
        opt : Optional[NoamOpt], default=None
            Learning-rate scheduler wrapping the
            optimizer. When None, no parameter
            update is performed (evaluation mode).
        """

        self.generator: Generator = generator
        self.criterion: nn.Module = criterion
        self.opt: Optional[NoamOpt] = opt

    def __call__(self, x: Tensor, y: Tensor, norm: Tensor) -> float:
        """
        Forward + backward + optional optimizer step.

        Parameters
        ----------
        x : Tensor
            Decoder output of shape
            (batch_size, seq_len, d_model).
        y : Tensor
            Target token IDs of shape
            (batch_size, seq_len).
        norm : Tensor
            Number of non-padding tokens, used to
            normalise the loss per real token.

        Returns
        -------
        float
            Total (un-normalised) loss for this batch.
        """

        x = self.generator(x)

        # Flatten to (batch * seq_len, vocab) and (batch * seq_len,) so the criterion
        # sees a single list of per-token predictions
        loss: Tensor = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)),
                y.contiguous().view(-1),
            )
            / norm
        )

        loss.backward()

        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()

        # Re-multiply by norm to return the total (un-normalised) loss for logging, since
        # the division above was only for gradient scaling
        output: float = loss.data.item() * norm.float()

        return output


def subsequent_mask(size: int) -> Tensor:

    # Define the shape of the attention mask as (1, size, size)
    attn_shape: Tuple[int, int, int] = (1, size, size)

    # Triangular matrix of shape (1, size, size) with 1s in the upper triangle and 0s
    # elsewhere
    subsequent_mask: np.ndarray = np.triu(np.ones(attn_shape), k=1).astype("uint8")

    # Convert the numpy array to a PyTorch tensor and create a boolean mask where 1s
    # become False and 0s become True
    output: Tensor = torch.from_numpy(subsequent_mask) == 0

    return output


def make_std_mask(tgt: Tensor, pad: int) -> Tensor:

    # Create a mask for the target sequence where positions that are not padding are marked as True
    tgt_mask: Tensor = (tgt != pad).unsqueeze(-2)

    # Combine the target mask with the subsequent mask to prevent attending to future positions
    output: Tensor = tgt_mask & subsequent_mask(size=tgt.size(-1)).type_as(tgt_mask.data)

    return output


# Listing 2.4 Calculating attention based on query, key, and value
def attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Optional[Tensor] = None,
    dropout: Optional[nn.Module] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Compute scaled dot-product attention.

    Parameters
    ----------
    query : Tensor
        Query tensor of shape (batch_size, seq_len, d_k).
    key : Tensor
        Key tensor of shape (batch_size, seq_len, d_k).
    value : Tensor
        Value tensor of shape (batch_size, seq_len, d_v).
    mask : Optional[Tensor], default=None
        Attention mask where True marks valid positions and False marks masked
        positions.
    dropout : Optional[nn.Module], default=None
        Dropout module applied to attention probabilities.

    Returns
    -------
    Tuple[Tensor, Tensor]
        Attention output and attention probabilities.
    """

    d_k: int = query.size(-1)
    scale: float = math.sqrt(d_k)
    scores: Tensor = torch.matmul(query, key.transpose(-2, -1)) / scale

    if mask is not None:
        mask_bool: Tensor = mask.to(dtype=torch.bool)
        neg_inf: float = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~mask_bool, neg_inf)

    p_attn: Tensor = torch.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    context: Tensor = torch.matmul(p_attn, value)

    return context, p_attn


# Listing 2.7 Creating a model to translate German to English
def create_model(
    src_vocab: int,
    tgt_vocab: int,
    N: int,
    d_model: int,
    d_ff: int,
    h: int,
    dropout: float = 0.1,
) -> Transformer:
    """
    Build and initialize an encoder-decoder transformer model.

    Parameters
    ----------
    src_vocab : int
        Source vocabulary size.
    tgt_vocab : int
        Target vocabulary size.
    N : int
        Number of encoder and decoder layers.
    d_model : int
        Model embedding dimension.
    d_ff : int
        Hidden dimension of position-wise feed-forward layers.
    h : int
        Number of attention heads.
    dropout : float, default=0.1
        Dropout probability used in transformer blocks.

    Returns
    -------
    Transformer
        Initialized transformer model on the configured device.
    """

    # Create multi-head attention module with specified number of heads and model
    # dimension
    attn: MultiHeadedAttention = MultiHeadedAttention(h, d_model).to(DEVICE)

    # Create position-wise feed-forward network with specified dimensions and dropout
    ff: PositionwiseFeedForward = PositionwiseFeedForward(d_model, d_ff, dropout).to(
        DEVICE
    )

    # Positional encoding module that adds sinusoidal positional information to token
    # embeddings
    pos: PositionalEncoding = PositionalEncoding(d_model, dropout).to(DEVICE)

    # Create an encoder with self-attention and feed-forward layers
    encoder: Encoder = Encoder(
        EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), dropout).to(DEVICE), N
    ).to(DEVICE)

    # Create a decoder with masked self-attention, cross-attention, and feed-forward layers
    decoder: Decoder = Decoder(
        DecoderLayer(d_model, deepcopy(attn), deepcopy(attn), deepcopy(ff), dropout).to(
            DEVICE
        ),
        N,
    ).to(DEVICE)

    # Source embedding and positional encoding combined into a single sequential module
    # for the encoder in source language
    src_embed: nn.Sequential = nn.Sequential(
        Embeddings(d_model, src_vocab).to(DEVICE), deepcopy(pos)
    )

    # Target embedding and positional encoding combined into a single sequential module
    # for the decoder in target language
    tgt_embed: nn.Sequential = nn.Sequential(
        Embeddings(d_model, tgt_vocab).to(DEVICE), deepcopy(pos)
    )

    # Final linear projection and softmax for generating output tokens in the target
    # language
    generator: Generator = Generator(d_model, tgt_vocab).to(DEVICE)

    # Put all components together into the full transformer model
    model: Transformer = Transformer(
        encoder=encoder,
        decoder=decoder,
        src_embed=src_embed,
        tgt_embed=tgt_embed,
        generator=generator,
    ).to(DEVICE)

    # Initialize model parameters with Xavier uniform initialization for weights in layers
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model.to(DEVICE)
