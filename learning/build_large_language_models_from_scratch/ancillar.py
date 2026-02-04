"""
This module contains ancillary classes and functions for building and training
large language models from scratch using PyTorch. 
"""

# Load packages.
import json
import os
import urllib
from typing import Dict, List

import numpy as np
import pandas as pd
import tiktoken
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader, Dataset


class LayerNorm(nn.Module):
    """
    Layer normalization module.

    Normalizes the input tensor across the last dimension for each sample in
    the batch, then applies learnable scale and shift parameters. This helps
    stabilize and accelerate training of deep neural networks.
    """

    def __init__(self, emb_dim: int) -> None:
        """
        Initialize the LayerNorm module.

        Parameters
        ----------
        emb_dim : int
            The size of the embedding dimension to normalize over.
        """

        # Initialize the parent class.
        super().__init__()

        # Define learnable parameters for scaling and shifting.
        self.eps: float = 1e-5
        self.scale: nn.Parameter = nn.Parameter(torch.ones(emb_dim))
        self.shift: nn.Parameter = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., emb_dim), where normalization is
            performed over the last dimension.

        Returns
        -------
        torch.Tensor
            The normalized tensor with the same shape as the input, scaled and
            shifted by learnable parameters.
        """
        mean: torch.Tensor = x.mean(dim=-1, keepdim=True)
        var: torch.Tensor = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x: torch.Tensor = (x - mean) / torch.sqrt(var + self.eps)

        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """
    Listing 4.3: An implementation of the GELU activation function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        gelu = (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / 3.14))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )

        return gelu


class FeedForward(nn.Module):
    """
    Listing 4.4: A feed forward neural network module.
    """

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    """
    From chapter 3.
    Listing 3.5 An efficient multi-head attention class.
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (
            d_out % num_heads == 0
        ), f"d_out ({d_out}) must be divisible by num_heads ({num_heads})!"

        self.d_out = d_out
        self.num_heads = num_heads

        # Reduces the projection dim to match the desired output dim.
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Uses a Linear layer to combine head outputs.
        self.out_proj = nn.Linear(d_out, d_out)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            name="mask",
            tensor=torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a num_heads dimension.
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transposes:
        #   from  (b, num_tokens, num_heads, head_dim)
        #   to    (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Computes dot product for each head.
        attn_scores = queries @ keys.transpose(2, 3)

        # Masks truncated to the number of tokens.
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Uses the mask to fill attention scores.
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Tensor shape: (b, num_tokens, n_heads, head_dim).
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combines heads, where self.d_out = self.num_heads * self.head_dim.
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # Adds an optional linear projection.
        context_vec = self.out_proj(context_vec)

        return context_vec


class TransformerBlock(nn.Module):
    """
    Listing 4.6: The transformer block component of GPT.
    """

    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    """
    GPTModel implements a transformer-based language model for text generation.

    This class includes token and positional embeddings, a stack of transformer
    blocks, a final normalization layer, and an output projection head. It
    supports forward inference for generating logits over the vocabulary.
    """

    def __init__(self, cfg: dict) -> None:
        """
        Initialize the GPTModel.

        Parameters
        ----------
        cfg : dict
            Configuration dictionary containing model hyperparameters:
            - vocab_size : int
                Size of the vocabulary.
            - emb_dim : int
                Dimension of the embeddings.
            - context_length : int
                Maximum sequence length (context window).
            - drop_rate : float
                Dropout rate for regularization.
            - n_heads : int
                Number of attention heads.
            - n_layers : int
                Number of transformer blocks.
            - qkv_bias : bool
                Whether to use bias in QKV projections.
        """

        # Initialize the parent class.
        super().__init__()

        self.tok_emb: nn.Embedding = nn.Embedding(
            num_embeddings=cfg["vocab_size"], embedding_dim=cfg["emb_dim"]
        )
        self.pos_emb: nn.Embedding = nn.Embedding(
            num_embeddings=cfg["context_length"], embedding_dim=cfg["emb_dim"]
        )
        self.drop_emb: nn.Dropout = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks: nn.Sequential = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm: LayerNorm = LayerNorm(cfg["emb_dim"])

        # Output head for predicting the next token in the sequence.
        # Bias is set to False to avoid learning biases in the output layer.
        self.out_head: nn.Linear = nn.Linear(
            in_features=cfg["emb_dim"], out_features=cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the GPT model.

        Parameters
        ----------
        in_idx : torch.Tensor
            Input tensor of token indices with shape (batch_size, seq_len).

        Returns
        -------
        torch.Tensor
            Logits tensor of shape (batch_size, seq_len, vocab_size)
            representing unnormalized probabilities for each token in the
            vocabulary at each position in the sequence.
        """
        _, seq_len = in_idx.shape

        tok_embeds: torch.Tensor = self.tok_emb(in_idx)
        pos_embeds: torch.Tensor = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x: torch.Tensor = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        logits: torch.Tensor = self.out_head(x)

        return logits


# Listing 6.4 Setting up a Pytorch Dataset class.
class SpamDataset(Dataset):
    """
    A PyTorch Dataset for loading and preprocessing spam classification data.

    This dataset reads a CSV file containing text samples and their labels,
    tokenizes the text using a provided tokenizer, and pads or truncates the
    sequences to a fixed maximum length. Each item returned is a tuple of
    (input_tensor, label_tensor).
    """

    def __init__(
        self,
        csv_file: str,
        tokenizer: tiktoken.core.Encoding,
        max_length: int | None = None,
        pad_token_id: int = 50256,
    ) -> None:
        """
        Initialize the SpamDataset.

        Parameters
        ----------
        csv_file : str
            Path to the CSV file containing the data. The file must have
            columns "Text" and "Label".
        tokenizer : tiktoken.core.Encoding
            Tokenizer used to encode the text samples into token IDs.
        max_length : int or None, optional
            Maximum sequence length for each sample. If None, uses the length
            of the longest sample in the dataset.
        pad_token_id : int, optional
            Token ID used for padding shorter sequences. Default is 50256.

        Attributes
        ----------
        data : pd.DataFrame
            DataFrame containing the loaded CSV data.
        encoded_texts : list[list[int]]
            List of tokenized and padded text samples.
        max_length : int
            Maximum sequence length for all samples.
        """
        self.data: pd.DataFrame = pd.read_csv(csv_file)
        self.encoded_texts: List[List[int]] = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length: int = self._longest_encoded_length()
        else:
            self.max_length: int = max_length
            self.encoded_texts: List[List[int]] = [
                encoded_text[: self.max_length] for encoded_text in self.encoded_texts
            ]

        self.encoded_texts: List[List[int]] = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a single sample from the dataset.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - input_tensor : torch.Tensor
                Tensor of token IDs with shape (max_length,).
            - label_tensor : torch.Tensor
                Tensor containing the label for the sample.
        """
        encoded: List[int] = self.encoded_texts[index]
        label: int = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            The total number of samples.
        """
        return len(self.data)

    def _longest_encoded_length(self) -> int:
        """
        Compute the length of the longest encoded text in the dataset.

        Returns
        -------
        int
            The maximum sequence length among all encoded texts.
        """
        max_length: int = 0
        for encoded_text in self.encoded_texts:
            encoded_length: int = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


# Listing 2.5 A dataset for batched inputs and targets.
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenizes the entire text.
        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# Listing 4.8: A function for the GPT model to generate text.
def generate_text_simple(model, idx, max_new_tokens, context_size):

    # idx is (batch, n_tokens) array of indices in the current context.
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size.
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context.
        idx_cond = idx[:, -context_size:]

        # Get the predictions.
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step.
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Apply softmax to get probabilities.
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value.
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence.
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens + 1)

    return idx


# Listing 2.6 A data loader to generate batches with input-with pairs.
def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # drop_last=True:
    #     drops the last batch if it is shorter than the specified
    #     batch_size to prevent loss spikes during training.
    # num_workers:
    #     The number of CPU processes to use for preprocessing.
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


# Listing 5.1: Utility functions for text to token ID conversion.
def text_to_token_ids(text, tokenizer):

    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

    # Add batch dimension.
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    return encoded_tensor


# Listing 5.1: Utility functions for token to ID conversion.
def token_ids_to_text(token_ids, tokenizer):

    # Remove batch dimension.
    flat = token_ids.squeeze(0)

    return tokenizer.decode(flat.tolist())


# Listing 5.3: The main function for pretraining LLMs.
def train_model_simple(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer: tiktoken.core.Encoding,
    finetune_classification_task: bool = False,
) -> tuple[list[float], list[float], list[int]]:
    """
    Train a GPT model using training and validation data loaders.

    This function performs a standard training loop over multiple epochs,
    periodically evaluates the model on training and validation data, and
    tracks loss values along with the number of tokens processed.

    Parameters
    ----------
    model : nn.Module
        The GPT model to train.
    train_loader : DataLoader
        DataLoader providing training batches of input and target tokens.
    val_loader : DataLoader
        DataLoader providing validation batches of input and target tokens.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters.
    device : torch.device
        Device (CPU or CUDA) on which to perform computation.
    num_epochs : int
        Number of training epochs.
    eval_freq : int
        Frequency (in steps) to evaluate and record losses.
    eval_iter : int
        Number of batches to use for each evaluation.
    start_context : str
        Initial text prompt used to generate sample output after each epoch.
    tokenizer : tiktoken.core.Encoding
        Tokenizer used for sample text generation.
    finetune_classification_task : bool, optional
        If True, indicates that the model is being fine-tuned for a
        classification task, and only the last token's logits are used.
        Default is False.

    Returns
    -------
    tuple[list[float], list[float], list[int]]
        A tuple containing:
        - train_losses : list of float
            Recorded training losses at evaluation steps.
        - val_losses : list of float
            Recorded validation losses at evaluation steps.
        - track_tokens_seen : list of int
            Number of tokens processed at each evaluation step.

    Notes
    -----
    - Uses calc_loss_batch for loss computation.
    - Uses evaluate_model for periodic evaluation.
    - Prints a generated sample after each epoch.
    """

    # Initialize lists to track losses and tokens seen.
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop.
    for epoch in range(num_epochs):

        # Set model to training mode.
        model.train()

        # Iterate over all batches in the training data.
        for input_batch, target_batch in train_loader:

            # Reset loss gradients from previous batch iteration.
            optimizer.zero_grad()

            # Calculate loss.
            loss = calc_loss_batch(
                input_batch, target_batch, model, device, finetune_classification_task
            )

            #  Calculate loss gradients.
            loss.backward()

            # Update model weights using loss gradients.
            optimizer.step()

            # Track the number of tokens seen and global step.
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step.
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(
                    f">>> Epoch: {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.6f}, Validation loss {val_loss:.6f}"
                )

        # Print a sample text after each epoch.
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def generate_and_print_sample(
    model: nn.Module,
    tokenizer: tiktoken.core.Encoding,
    device: torch.device,
    start_context: str,
) -> None:
    """
    Generate and print a sample text from the model during training.

    This function switches the model to evaluation mode, encodes the provided
    start context, generates a short continuation, decodes it back to text, and
    prints the result in a compact format. It restores training mode afterward.

    Parameters
    ----------
    model : nn.Module
        The GPT model used to generate text samples.
    tokenizer : tiktoken.core.Encoding
        Tokenizer used to encode the prompt and decode generated tokens.
    device : torch.device
        Device (CPU or CUDA) on which to perform computation.
    start_context : str
        Initial text prompt used as the generation seed.

    Returns
    -------
    None
        This function prints the generated sample and returns nothing.

    Notes
    -----
    - Uses text_to_token_ids for prompt encoding and token_ids_to_text for decoding.
    - Calls aux.generate_text_simple for greedy decoding.
    - Temporarily sets model to eval mode and restores train mode after printing.
    """

    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():

        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size
        )

    decoded_text = token_ids_to_text(token_ids, tokenizer)

    # Compact print format.
    print("\n #-----------------------------------------------# \n")
    print(decoded_text)
    print("\n #-----------------------------------------------# \n")

    model.train()


# Listing 5.4 A modified text generation function with more diversity.
def generate(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    eos_id: int | None = None,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Generate text tokens using a GPT model with temperature and top-k sampling.

    This function performs autoregressive text generation by iteratively
    predicting the next token based on the current context. It supports
    temperature scaling for controlling randomness and top-k sampling for
    limiting the vocabulary to the most likely tokens. Generation can be
    terminated early if an end-of-sequence token is encountered.

    Parameters
    ----------
    model : nn.Module
        The GPT model used for text generation. Must output logits of shape
        (batch_size, seq_len, vocab_size) when given input token indices.
    idx : torch.Tensor
        Initial input tensor of token indices with shape (batch_size, seq_len).
        This provides the starting context for text generation.
    max_new_tokens : int
        Maximum number of new tokens to generate. The generation loop will
        run for at most this many iterations unless terminated early by an
        end-of-sequence token.
    context_size : int
        Maximum context length that the model can handle. Input sequences
        longer than this will be truncated to the most recent context_size
        tokens before being passed to the model.
    temperature : float, optional
        Temperature parameter for controlling randomness in token selection.
        - temperature = 0.0: Deterministic selection (argmax).
        - temperature < 1.0: Less random, favors high-probability tokens.
        - temperature = 1.0: Standard softmax sampling.
        - temperature > 1.0: More random, flattens probability distribution.
        Default is 0.0 (deterministic).
    top_k : int or None, optional
        If specified, restricts sampling to the top k most likely tokens at
        each step. Tokens outside the top k have their logits set to -inf.
        If None, all tokens are considered. Default is None.
    eos_id : int or None, optional
        Token ID representing the end-of-sequence marker. If specified,
        generation stops immediately when this token is sampled. If None,
        generation continues for max_new_tokens iterations. Default is None.
    verbose : bool, optional
        If True, prints diagnostic information including input tensor shapes
        and logits shapes during generation. Useful for debugging. Default
        is False.

    Returns
    -------
    torch.Tensor
        Generated token sequence of shape (batch_size, seq_len + num_generated),
        where num_generated ≤ max_new_tokens. Contains the original input
        tokens concatenated with newly generated tokens.

    Notes
    -----
    - When temperature = 0.0, the function uses deterministic argmax selection,
      equivalent to greedy decoding.
    - When temperature > 0.0, the function samples from the probability
      distribution, introducing randomness in token selection.
    - Top-k sampling filters out low-probability tokens before sampling, which
      can improve generation quality by preventing unlikely token selection.
    - The context window is managed automatically: if the sequence exceeds
      context_size, only the most recent context_size tokens are used.
    - Generation runs without gradient computation (torch.no_grad()) for
      efficiency during inference.
    - Early termination occurs only when eos_id is specified and that exact
      token is generated.
    """

    if verbose:
        print(f">>> IDX: {idx}; shape: {idx.shape}")

    # For-loop iterates up to max_new_tokens times to generate new tokens.
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size.
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context.
        idx_cond: torch.Tensor = idx[:, -context_size:]

        # Get the predictions from the model without computing gradients.
        with torch.no_grad():
            logits: torch.Tensor = model(idx_cond)

            if verbose:
                print(f">>> LOGITS shape: {logits.shape}")

        # Focus only on the last time step.
        # Extract logits for the last token position in the sequence.
        # Transform from (batch_size, seq_len, vocab_size) to (batch_size, vocab_size).
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling.
        if top_k is not None:
            # Keep only the top_k highest logit values.
            # Get the k-th largest value for each batch element.
            top_logits: torch.Tensor
            _: torch.Tensor
            top_logits, _ = torch.topk(logits, top_k)

            # Extract the minimum value among top-k logits (k-th largest).
            min_val: torch.Tensor = top_logits[:, -1]

            # Set all logits below the threshold to negative infinity.
            # This effectively removes them from consideration during sampling.
            logits = torch.where(
                condition=logits < min_val,
                input=torch.tensor(float("-inf")).to(logits.device),
                other=logits,
            )

        # New: Apply temperature scaling if temperature > 0.
        if temperature > 0.0:
            # Scale logits by temperature to control randomness.
            # Higher temperature makes distribution more uniform.
            # Lower temperature makes distribution more peaked.
            logits = logits / temperature

            # Apply softmax to convert scaled logits to probabilities.
            probs: torch.Tensor = torch.softmax(
                logits, dim=-1
            )  # (batch_size, vocab_size)

            # Sample the next token from the probability distribution.
            # This introduces stochasticity in token selection.
            idx_next: torch.Tensor = torch.multinomial(
                probs, num_samples=1
            )  # (batch_size, 1)

        # Otherwise use deterministic selection (greedy decoding).
        else:
            # Get the index of the vocabulary entry with the highest logit value.
            # This is deterministic and always selects the most likely token.
            idx_next: torch.Tensor = torch.argmax(
                logits, dim=-1, keepdim=True
            )  # (batch_size, 1)

        # Stop generating early if end-of-sequence token is encountered and eos_id is specified.
        if idx_next == eos_id:
            break

        # Append the sampled token index to the running sequence.
        # Concatenate along the sequence dimension to extend the generated text.
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, seq_len + 1)

    return idx


# Listing 5.5 Loading OpenAI weights into our GPT model code.
def load_weights_into_gpt(gpt, params):

    def assign(left, right):
        if left.shape != right.shape:
            raise ValueError(
                f"Shape mismatch. Left: {left.shape}, " "Right: {right.shape}"
            )
        return torch.nn.Parameter(torch.tensor(right))

    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1
        )
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T
        )
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T
        )
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T
        )

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1
        )
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b
        )
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b
        )

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"],
        )

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T,
        )
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"],
        )

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"]
        )
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"]
        )
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"]
        )
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"]
        )

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def plot_losses(
    epochs_seen: List[int],
    tokens_seen: List[int],
    train_losses: List[float],
    val_losses: List[float],
    save_fig: bool = False,
) -> None:
    """
    Plot training and validation losses against epochs and tokens seen.

    Parameters
    ----------
    epochs_seen : list of int
        List of epoch numbers corresponding to recorded losses.
    tokens_seen : list of int
        List of token counts corresponding to recorded losses.
    train_losses : list of float
        List of training loss values.
    val_losses : list of float
        List of validation loss values.
    save_fig : bool, optional
        If True, saves the plot as "loss-plot.pdf". Default is False.
    """

    fig, ax1 = plt.subplots(figsize=(8, 4))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Only show integer labels on x-axis.
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Create a second x-axis for tokens seen.
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis.
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks.
    ax2.set_xlabel("Tokens seen")

    # Adjust layout to make room.
    fig.tight_layout()

    # Save the figure if requested.
    if save_fig:
        plt.savefig("loss-plot.pdf")

    # Show the plot.
    plt.show()


# Listing 6.3 Splitting the dataset.
def random_split(
    df: pd.DataFrame, train_frac: float, validation_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into train, validation, and test sets by fractions.

    The function shuffles the input DataFrame and splits it into three
    disjoint subsets: training, validation, and test sets, according to the
    specified fractions. The test set receives the remaining data after
    allocating the training and validation sets.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to split.
    train_frac : float
        Fraction of the data to use for the training set. Must be in [0, 1].
    validation_frac : float
        Fraction of the data to use for the validation set. Must be in [0, 1].
        The test set will contain the remaining samples.

    Returns
    -------
    tuple of pd.DataFrame
        A tuple containing (train_df, validation_df, test_df), where each is a
        DataFrame corresponding to the respective split.

    Notes
    -----
    The function uses a fixed random seed (123) for reproducibility.
    The sum of train_frac and validation_frac should be less than or equal to 1.
    """
    # Shuffle the entire DataFrame.
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate split indices.
    train_end: int = int(len(df) * train_frac)
    validation_end: int = train_end + int(len(df) * validation_frac)

    # Split the DataFrame.
    train_df: pd.DataFrame = df[:train_end]
    validation_df: pd.DataFrame = df[train_end:validation_end]
    test_df: pd.DataFrame = df[validation_end:]

    return train_df, validation_df, test_df


# Listing 6.8 Calculating the classification accuracy.
def calc_accuracy_loader(
    data_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    num_batches: int = None,
) -> float:
    """
    Calculate the classification accuracy of a model over a data loader.

    This function evaluates the model on batches from the provided data loader,
    comparing the predicted labels to the target labels. It computes the
    proportion of correct predictions over all evaluated examples. The
    evaluation is performed without gradient computation and in evaluation mode.

    Parameters
    ----------
    data_loader : DataLoader
        A PyTorch DataLoader yielding batches of input and target tensors.
    model : nn.Module
        The neural network model to evaluate. Must output logits for each class.
    device : torch.device
        The device (CPU or CUDA) on which to perform computation.
    num_batches : int, optional
        The maximum number of batches to evaluate. If None, evaluates all
        batches in the data loader.

    Returns
    -------
    float
        The classification accuracy as a float in the range [0, 1].

    Notes
    -----
    - Assumes the model's output logits correspond to the last token in the
      sequence for each input batch.
    - The function does not update model parameters.
    """

    # Ensure the model is in evaluation mode.
    model.eval()

    # Initialize counters for correct predictions and total examples.
    correct_predictions: int = 0
    num_examples: int = 0

    # If num_batches is None, use the full length of the data loader.
    # Otherwise, limit to the specified number of batches.
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    # Iterate over the data loader.
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:

            input_batch: torch.Tensor = input_batch.to(device)
            target_batch: torch.Tensor = target_batch.to(device)

            # Forward pass through the model to get logits.
            # Assumes the model outputs logits for the last token in the sequence
            # and that the logits shape is (batch_size, seq_len, classification).
            with torch.no_grad():
                logits: torch.Tensor = model(input_batch)[:, -1, :]

            predicted_labels: torch.Tensor = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()

        else:
            break

    return correct_predictions / num_examples


# Listing 6.8b Calculating the for a single batch.
def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: nn.Module,
    device: torch.device,
    finetune_classification_task: bool = False,
) -> torch.Tensor:
    """
    Calculate cross-entropy loss for a batch of inputs and targets.

    This function moves input and target tensors to the specified device, passes
    the inputs through the model to get logits for the last position in the
    sequence, and computes the cross-entropy loss against the target labels.

    Parameters
    ----------
    input_batch : torch.Tensor
        Input tensor of shape (batch_size, seq_len) containing token IDs.
    target_batch : torch.Tensor
        Target tensor of shape (batch_size,) containing class labels.
    model : nn.Module
        Neural network model that outputs logits for each token position.
    device : torch.device
        Device (CPU or CUDA) on which to perform computation.
    finetune_classification_task : bool, optional
        If True, indicates that the model is being fine-tuned for a
        classification task, and only the last token's logits are used.
        Default is False.

    Returns
    -------
    torch.Tensor
        Scalar tensor containing the computed cross-entropy loss.

    Notes
    -----
    - Assumes the model outputs a tensor of shape (batch_size, seq_len, n_classes);
    - Only uses logits from the last token position for loss computation;
    """

    input_batch: torch.Tensor = input_batch.to(device)
    target_batch: torch.Tensor = target_batch.to(device)

    # If we are dealing with a classification fine tuning task (chapter 6), then we are
    # interested in the last token output bt the model.
    if finetune_classification_task:
        logits: torch.Tensor = model(input_batch)[:, -1, :]
        targets: torch.Tensor = target_batch[-1]

    # In instruction fine tuning (chapter 7) we use all tokens output by model.
    else:
        logits: torch.Tensor = model(input_batch).flatten(0, 1)
        targets: torch.Tensor = target_batch.flatten()

    # Calculate lodd.
    loss: torch.Tensor = torch.nn.functional.cross_entropy(logits, targets)

    return loss


# Listing 6.9 Calculating the classification loss.
def calc_loss_loader(
    data_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    num_batches: int = None,
    finetune_classification_task: bool = False,
) -> float:
    """
    Calculate the average cross-entropy loss of a model over a data loader.

    This function evaluates the model on batches from the provided data loader
    and computes the average cross-entropy loss across all processed batches.
    The evaluation is performed without gradient computation.

    Parameters
    ----------
    data_loader : DataLoader
        A PyTorch DataLoader yielding batches of input and target tensors.
    model : nn.Module
        The neural network model to evaluate. Must output logits for each class.
    device : torch.device
        The device (CPU or CUDA) on which to perform computation.
    num_batches : int, optional
        The maximum number of batches to evaluate. If None, evaluates all
        batches in the data loader.
    finetune_classification_task : bool, optional
        If True, indicates that the model is being fine-tuned for a
        classification task, and only the last token's logits are used.
        Default is False.

    Returns
    -------
    float
        The average cross-entropy loss across all processed batches. Returns NaN
        if the data loader is empty.

    Notes
    -----
    - Uses calc_loss_batch function to calculate loss for each individual batch
    - Assumes the model outputs logits with shape (batch_size, seq_len, vocab_size)
    - Only uses logits from the last token position for loss computation
    """

    # Initialize the loss accumulator.
    total_loss: float = 0.0

    # Get the number of batches in the data loader or in the specified limit.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches: int = len(data_loader)
    else:
        num_batches: int = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss: torch.Tensor = calc_loss_batch(
                input_batch, target_batch, model, device, finetune_classification_task
            )
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches


# Listing 6.10 Fine-tuning the model to classify spam.
def train_classifier_simple(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    finetune_classification_task: bool = False,
) -> tuple[list[float], list[float], list[float], list[float], int]:
    """
    Train a classification model using the provided data loaders and optimizer.

    This function performs training over several epochs, periodically evaluates
    the model on training and validation data, and tracks loss and accuracy.

    Parameters
    ----------
    model : nn.Module
        The neural network model to train.
    train_loader : DataLoader
        DataLoader for the training dataset.
    val_loader : DataLoader
        DataLoader for the validation dataset.
    optimizer : torch.optim.Optimizer
        Optimizer for updating model parameters.
    device : torch.device
        Device (CPU or CUDA) for computation.
    num_epochs : int
        Number of epochs to train the model.
    eval_freq : int
        Frequency (in steps) to evaluate and record loss.
    eval_iter : int
        Number of batches to use for evaluation.
    finetune_classification_task : bool, optional
        If True, indicates that the model is being fine-tuned for a
        classification task, and only the last token's logits are used.
        Default is False.

    Returns
    -------
    tuple of lists and int
        Tuple containing:
        - train_losses : list of float
            Recorded training losses at evaluation steps.
        - val_losses : list of float
            Recorded validation losses at evaluation steps.
        - train_accs : list of float
            Training accuracies after each epoch.
        - val_accs : list of float
            Validation accuracies after each epoch.
        - examples_seen : int
            Total number of training examples processed.

    Notes
    -----
    - Uses calc_loss_batch for loss computation.
    - Uses calc_accuracy_loader for accuracy computation.
    - Calls evaluate_model for periodic evaluation.
    """

    # Initialize lists to track losses and examples seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop.
    for epoch in range(num_epochs):

        # Set model to training mode.
        model.train()

        # Loop over batches in the training data loader.
        for input_batch, target_batch in train_loader:

            # Reset loss gradients from previous batch iteration.
            optimizer.zero_grad()
            loss: torch.Tensor = calc_loss_batch(
                input_batch, target_batch, model, device, finetune_classification_task
            )

            # Calculate loss gradients.
            loss.backward()

            # Update model weights using loss gradients.
            optimizer.step()

            # Track examples.
            examples_seen += input_batch.shape[0]

            global_step += 1

            # Optional evaluation step.
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(
                    f">>> Ep {epoch + 1} (Step {global_step: 06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        # Calculate accuracy after each epoch.
        train_accuracy: float = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy: float = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
        )

        print(f">>> Training accuracy: {train_accuracy * 100: .2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy * 100: .2f}%")

        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


# Listing 6.10b part of the previous function.
def evaluate_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    eval_iter: int,
) -> tuple[float, float]:
    """
    Evaluate the model on training and validation data loaders.

    This function computes the average loss on both the training and validation
    sets using a fixed number of batches. The model is set to evaluation mode
    during loss computation and returned to training mode after.

    Parameters
    ----------
    model : nn.Module
        The neural network model to evaluate.
    train_loader : DataLoader
        DataLoader for the training dataset.
    val_loader : DataLoader
        DataLoader for the validation dataset.
    device : torch.device
        Device (CPU or CUDA) for computation.
    eval_iter : int
        Number of batches to use for evaluation.

    Returns
    -------
    tuple of float
        Tuple containing:
        - train_loss : float
            Average loss on the training set.
        - val_loss : float
            Average loss on the validation set.

    Notes
    -----
    - Uses calc_loss_loader to compute average loss for each loader.
    - Model is set to eval mode during evaluation and restored to train mode.
    """

    model.eval()

    with torch.no_grad():
        train_loss: float = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss: float = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )

    model.train()

    return train_loss, val_loss


# Listing 6.12 Using the model to classify new texts.
def classify_review(
    text: str,
    model: nn.Module,
    tokenizer: tiktoken.core.Encoding,
    device: torch.device,
    max_length: int | None = None,
    pad_token_id: int = 50256,
) -> str:
    """
    Classify a text review as spam or not spam using a fine-tuned GPT model.

    This function tokenizes the input text, prepares it for model inference by
    truncating and padding as necessary, and uses the model's classification
    head to predict whether the text is spam. The function handles sequence
    length constraints and ensures proper formatting for the model.

    Parameters
    ----------
    text : str
        The input text to classify. Can be any length, but will be truncated
        to max_length if longer than the specified limit.
    model : nn.Module
        A fine-tuned GPT model with a classification head. Must have a
        pos_emb attribute for positional embeddings and output logits for
        binary classification (0: not spam, 1: spam).
    tokenizer : tiktoken.core.Encoding
        The tokenizer used to encode the input text into token IDs. Must be
        compatible with the model's vocabulary.
    device : torch.device
        The device (CPU or CUDA) on which the model and tensors should be
        placed for computation.
    max_length : int or None, optional
        Maximum sequence length for the input.
    pad_token_id : int, optional
        Token ID used for padding sequences shorter than max_length.
        Default is 50256 (GPT-2's end-of-text token).

    Returns
    -------
    str
        Classification result as a string: either "spam" or "not spam".

    Notes
    -----
    - The function sets the model to evaluation mode and uses torch.no_grad()
      for inference to improve performance and reduce memory usage.
    - Only the logits from the last token position are used for classification,
      following the typical approach for sequence classification with GPT.
    - Input sequences longer than max_length are truncated from the end.
    - Input sequences shorter than max_length are padded with pad_token_id.
    """

    # Ensure the model is in evaluation mode.
    model.eval()

    # Prepare inputs to the model.
    input_ids: list[int] = tokenizer.encode(text)

    # Note: In the book, this was originally written as pos_emb.weight.shape[1]
    # by mistake. It didn't break the code but would have caused unnecessary
    # truncation (to 768 instead of 1024).
    supported_context_length: int = model.pos_emb.weight.shape[0]

    # Truncate sequences if they are too long.
    input_ids = input_ids[: min(max_length, supported_context_length)]

    # Pad sequences to the specified max_length.
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor: torch.Tensor = torch.tensor(input_ids, device=device).unsqueeze(
        0
    )  # Add batch dimension.

    # Model inference.
    with torch.no_grad():
        logits: torch.Tensor = model(input_tensor)[
            :, -1, :
        ]  # Logits of the last output token.
    predicted_label: int = torch.argmax(logits, dim=-1).item()

    # Return the classified result
    return "spam" if predicted_label == 1 else "not spam"


# Listing 7.1 Downloading the dataset.
def download_and_load_file(file_path: str, url: str) -> dict:
    """
    Download a file from a URL if not present and load its JSON content.

    This function checks if the file exists locally. If not, it downloads the
    file from the specified URL and saves it. Then, it loads and returns the
    file's content as a Python dictionary.

    Parameters
    ----------
    file_path : str
        Path to the local file to check or save the downloaded content.
    url : str
        URL to download the file from if it does not exist locally.

    Returns
    -------
    dict
        The loaded JSON content as a Python dictionary.

    Notes
    -----
    - Assumes the file content is in JSON format.
    - Uses UTF-8 encoding for reading and writing files.
    - Overwrites the file if it is downloaded.
    """
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data: str = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    with open(file_path, "r", encoding="utf-8") as file:
        data: dict = json.load(file)

    return data


# Listing 7.2 Implementing the prompt formatting function (Alpaca-style input format).
def format_input(entry: Dict[str, str]) -> str:
    """
    Format a prompt entry using Alpaca-style input format.

    This function takes a dictionary containing 'instruction' and 'input'
    fields and returns a formatted string suitable for prompt-based models.

    Parameters
    ----------
    entry : dict
        Dictionary with keys:
        - 'instruction' : str
            The instruction describing the task.
        - 'input' : str
            Optional input for the instruction.

    Returns
    -------
    str
        Formatted prompt string combining instruction and input.

    Notes
    -----
    - If 'input' is empty, only the instruction is included.
    - The output follows the Alpaca prompt format.
    """

    instruction_text: str = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text: str = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


# Listing 7.4 Implementing an instruction dataset class.
class InstructionDataset(Dataset):
    """
    A PyTorch Dataset for loading and preprocessing instruction-following data.

    This dataset processes instruction-response pairs by formatting them using
    the Alpaca prompt template and tokenizing the complete text sequences. Each
    sample contains an instruction, optional input, and expected response that
    are combined into a single training sequence for instruction fine-tuning.

    Attributes
    ----------
    data : list[dict[str, str]]
        List of dictionaries containing instruction data with 'instruction',
        'input', and 'output' keys.
    encoded_texts : list[list[int]]
        List of tokenized text sequences, where each sequence represents a
        complete instruction-response pair encoded as token IDs.
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: tiktoken.core.Encoding,
    ) -> None:
        """
        Initialize the InstructionDataset.

        This method processes the input data by formatting each entry using the
        Alpaca prompt template and tokenizing the complete instruction-response
        sequences. The formatted text combines the instruction, optional input,
        and response into a single training sequence.

        Parameters
        ----------
        data : list[dict[str, str]]
            List of dictionaries where each dictionary contains:
            - 'instruction' : str
                The task instruction describing what needs to be done.
            - 'input' : str
                Optional input context for the instruction. Can be empty.
            - 'output' : str
                The expected response or completion for the instruction.
        tokenizer : tiktoken.core.Encoding
            Tokenizer used to encode the formatted text into token IDs.
            Must be compatible with the model's vocabulary.

        Notes
        -----
        - Each entry is formatted using the Alpaca prompt template via the
          format_input function.
        - The complete text sequence includes instruction, input (if present),
          and response sections.
        - All text sequences are pre-tokenized during initialization for
          efficiency during training.
        """
        # Main parameters.
        self.data: List[Dict[str, str]] = data
        self.encoded_texts: List[List[int]] = []

        # Pretokenizes all texts.
        for entry in data:
            instruction_plus_input: str = format_input(entry=entry)
            response_text: str = f"\n\n### Response:\n{entry['output']}"
            full_text: str = instruction_plus_input + response_text

            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index: int) -> List[int]:
        """
        Retrieve a single tokenized sequence from the dataset.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve. Must be in the range
            [0, len(dataset)).

        Returns
        -------
        list[int]
            A list of token IDs representing the complete instruction-response
            sequence at the specified index. The sequence includes the
            formatted instruction, optional input, and expected response.

        Raises
        ------
        IndexError
            If the index is out of bounds for the dataset.

        Notes
        -----
        - The returned sequence is already tokenized and ready for model input.
        - Each sequence may have different lengths depending on the content
          of the instruction and response.
        """
        return self.encoded_texts[index]

    def __len__(self) -> int:
        """
        Return the number of instruction-response pairs in the dataset.

        Returns
        -------
        int
            The total number of samples in the dataset, corresponding to the
            number of instruction-response pairs provided during initialization.

        Notes
        -----
        - This method is required by PyTorch's Dataset interface.
        - The length corresponds to the number of entries in the original
          data list.
        """
        return len(self.data)


def custom_collate_draft_1(
    batch: List[List[int]], pad_token_id: int = 50256, device: str = "cpu"
) -> torch.Tensor:
    """
    Custom collate function for batching variable-length token sequences.

    This function processes a batch of tokenized sequences with different
    lengths by padding them to a uniform length and preparing them for
    model training. Each sequence is padded with a specified token ID and
    converted to a tensor format suitable for PyTorch models.

    The function adds an extra padding token to each sequence, determines
    the maximum length needed for the batch, pads all sequences to this
    length, and then removes the last token to create input sequences of
    consistent length.

    Parameters
    ----------
    batch : list[list[int]]
        List of tokenized sequences where each inner list contains token IDs
        for a single sample. Sequences can have different lengths.
    pad_token_id : int, optional
        Token ID used for padding shorter sequences to match the batch's
        maximum length. Default is 50256 (GPT-2's end-of-text token).
    device : str, optional
        Target device for the output tensor. Can be "cpu", "cuda", or other
        PyTorch device specifications. Default is "cpu".

    Returns
    -------
    torch.Tensor
        A 2D tensor of shape (batch_size, max_seq_len - 1) containing the
        padded input sequences. Each row represents one sample from the
        batch, padded to the same length and ready for model input.

    Notes
    -----
    - The function adds one padding token to each sequence before determining
      the maximum length, then removes the last token from each padded
      sequence. This pattern is commonly used in language modeling where
      input and target sequences are offset by one position.
    - All sequences in the output tensor have the same length, determined by
      the longest sequence in the batch plus one padding token, minus one
      for the final truncation.
    - The output tensor is automatically moved to the specified device.

    Examples
    --------
    >>> batch = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    >>> result = custom_collate_draft_1(batch, pad_token_id=0)
    >>> print(result.shape)
    torch.Size([3, 4])
    >>> print(result)
    tensor([[1, 2, 3, 0],
            [4, 5, 0, 0],
            [6, 7, 8, 9]])
    """

    # Determine the maximum length of the batch.
    batch_max_length: int = max(len(item) + 1 for item in batch)
    inputs_lst: List[torch.Tensor] = []

    # Pad and prepares each item in the batch.
    for item in batch:

        new_item: List[int] = item.copy()
        new_item += [pad_token_id]

        padded: List[int] = new_item + [pad_token_id] * (
            batch_max_length - len(new_item)
        )

        # Removes extra padded token added earlier.
        inputs: torch.Tensor = torch.tensor(padded[:-1])

        inputs_lst.append(inputs)

    # Converts the list of inputs to a tensor and transfers it to the target
    # device.
    inputs_tensor: torch.Tensor = torch.stack(inputs_lst).to(device)

    return inputs_tensor


def custom_collate_draft_2(
    batch: List[List[int]], pad_token_id: int = 50256, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function for batching sequences with input-target pairs.

    This function processes a batch of tokenized sequences by padding them to
    uniform length and creating input-target pairs for language modeling. The
    function prepares sequences where targets are shifted one position to the
    right relative to inputs, which is the standard approach for autoregressive
    language model training.

    Each sequence is padded with an end-of-text token, then padded to the batch's
    maximum length. Input sequences exclude the final token, while target
    sequences exclude the first token, creating the necessary offset for
    next-token prediction training.

    Parameters
    ----------
    batch : list[list[int]]
        List of tokenized sequences where each inner list contains token IDs
        for a single sample. Sequences can have different lengths.
    pad_token_id : int, optional
        Token ID used for padding shorter sequences and as end-of-text marker.
        Default is 50256 (GPT-2's end-of-text token).
    device : str, optional
        Target device for the output tensors. Can be "cpu", "cuda", or other
        PyTorch device specifications. Default is "cpu".

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - inputs_tensor : torch.Tensor
            Input sequences of shape (batch_size, max_seq_len - 1) where each
            sequence is padded to uniform length and ready for model input.
        - targets_tensor : torch.Tensor
            Target sequences of shape (batch_size, max_seq_len - 1) containing
            the expected next tokens for each input position, shifted one
            position to the right relative to inputs.

    Notes
    -----
    - The function adds an end-of-text token to each sequence before padding
      to ensure proper sequence termination.
    - Input sequences are created by excluding the last token from padded
      sequences (padded[:-1]).
    - Target sequences are created by excluding the first token from padded
      sequences (padded[1:]), creating the one-position shift needed for
      next-token prediction.
    - Both output tensors have the same shape and are moved to the specified
      device for training compatibility.
    - This pattern enables the model to learn to predict the next token given
      the current context during autoregressive training.

    Examples
    --------
    >>> batch = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    >>> inputs, targets = custom_collate_draft_2(batch, pad_token_id=0)
    >>> print(inputs.shape, targets.shape)
    torch.Size([3, 4]) torch.Size([3, 4])
    >>> print("Inputs:", inputs)
    tensor([[1, 2, 3, 0],
            [4, 5, 0, 0],
            [6, 7, 8, 9]])
    >>> print("Targets:", targets)
    tensor([[2, 3, 0, 0],
            [5, 0, 0, 0],
            [7, 8, 9, 0]])
    """
    # Find the longest sequence in the batch.
    batch_max_length: int = max(len(item) + 1 for item in batch)

    # Pad and prepare inputs.
    inputs_lst: List[torch.Tensor] = []
    targets_lst: List[torch.Tensor] = []

    for item in batch:
        new_item: List[int] = item.copy()

        # Add an <|endoftext|> token.
        new_item += [pad_token_id]

        # Pad sequences to max_length.
        padded: List[int] = new_item + [pad_token_id] * (
            batch_max_length - len(new_item)
        )

        # Truncate the last token for inputs.
        inputs: torch.Tensor = torch.tensor(padded[:-1])

        # Shift +1 to the right for targets
        targets: torch.Tensor = torch.tensor(padded[1:])

        # Append results.
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs to tensor and transfer to target device.
    inputs_tensor: torch.Tensor = torch.stack(inputs_lst).to(device)
    targets_tensor: torch.Tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


# Listing 7.5 Implementing a custom batch collate function.
def custom_collate_fn(
    batch: List[List[int]],
    pad_token_id: int = 50256,
    ignore_index: int = -100,
    allowed_max_length: int | None = None,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function with ignore index masking for instruction tuning.

    This function processes a batch of tokenized sequences by padding them to
    uniform length and creating input-target pairs for language modeling with
    proper masking. The function prepares sequences where targets are shifted
    one position to the right relative to inputs, and uses ignore_index to
    mask padding tokens in targets to prevent the model from learning to
    predict padding.

    Each sequence is padded with an end-of-text token, then padded to the batch's
    maximum length. Input sequences exclude the final token, while target
    sequences exclude the first token. Crucially, all padding tokens in targets
    except the first one are replaced with ignore_index, which prevents the
    loss function from computing gradients for these positions.

    Parameters
    ----------
    batch : list[list[int]]
        List of tokenized sequences where each inner list contains token IDs
        for a single sample. Sequences can have different lengths.
    pad_token_id : int, optional
        Token ID used for padding shorter sequences and as end-of-text marker.
        Default is 50256 (GPT-2's end-of-text token).
    ignore_index : int, optional
        Special index used to mask padding tokens in target sequences. Loss
        computation will ignore positions with this value. Default is -100,
        which is the standard PyTorch ignore index for CrossEntropyLoss.
    allowed_max_length : int or None, optional
        Maximum sequence length to truncate inputs and targets. If None, uses
        the natural batch maximum length. If specified, sequences longer than
        this value will be truncated.
    device : str, optional
        Target device for the output tensors. Can be "cpu", "cuda", or other
        PyTorch device specifications. Default is "cpu".

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - inputs_tensor : torch.Tensor
            Input sequences of shape (batch_size, seq_len) where each sequence
            is padded to uniform length and ready for model input.
        - targets_tensor : torch.Tensor
            Target sequences of shape (batch_size, seq_len) containing the
            expected next tokens for each input position, with padding tokens
            (except the first) replaced by ignore_index.

    Notes
    -----
    - The function adds an end-of-text token to each sequence before padding
      to ensure proper sequence termination.
    - Input sequences are created by excluding the last token from padded
      sequences (padded[:-1]).
    - Target sequences are created by excluding the first token from padded
      sequences (padded[1:]), creating the one-position shift needed for
      next-token prediction.
    - The first padding token in targets is preserved to teach the model when
      to stop generation, while subsequent padding tokens are masked with
      ignore_index.
    - This masking strategy is crucial for instruction fine-tuning where you
      want the model to learn to generate proper end-of-sequence tokens but
      not to predict arbitrary padding.
    - Both output tensors have the same shape and are moved to the specified
      device for training compatibility.

    Examples
    --------
    >>> batch = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    >>> inputs, targets = custom_collate_fn(batch, pad_token_id=0, ignore_index=-100)
    >>> print(inputs.shape, targets.shape)
    torch.Size([3, 4]) torch.Size([3, 4])
    >>> print("Inputs:", inputs)
    tensor([[1, 2, 3, 0],
            [4, 5, 0, 0],
            [6, 7, 8, 9]])
    >>> print("Targets:", targets)
    tensor([[  2,   3,   0, -100],
            [  5,   0, -100, -100],
            [  7,   8,   9,    0]])
    """
    # Find the longest sequence in the batch
    batch_max_length: int = max(len(item) + 1 for item in batch)

    # Pad and prepare inputs and targets.
    inputs_lst: List[torch.Tensor] = []
    targets_lst: List[torch.Tensor] = []

    for item in batch:
        new_item: List[int] = item.copy()

        # Add an <|endoftext|> token.
        new_item += [pad_token_id]

        # Pad sequences to max_length.
        padded: List[int] = new_item + [pad_token_id] * (
            batch_max_length - len(new_item)
        )

        # Truncate the last token for inputs and shift +1 to the right for targets.
        inputs: torch.Tensor = torch.tensor(padded[:-1])
        targets: torch.Tensor = torch.tensor(padded[1:])

        # Replace all but the first padding tokens in targets by ignore_index.
        mask: torch.Tensor = targets == pad_token_id
        indices: torch.Tensor = torch.nonzero(mask).squeeze()

        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # Optionally truncate to maximum sequence length.
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device.
    inputs_tensor: torch.Tensor = torch.stack(inputs_lst).to(device)
    targets_tensor: torch.Tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor
