# Load packages.
import torch

import torch.nn as nn


###############################################################################
class LayerNorm(nn.Module):
    """
    Listing 4.2 A layer normalization class.
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        return self.scale * norm_x + self.shift


###############################################################################
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


###############################################################################
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


###############################################################################
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


###############################################################################
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


###############################################################################
class GPTModel(nn.Module):
    """
    Listing 4.7: The GPT model architecture implementation.
    """

    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        # `in_idx`: batch of input token indices.
        # Batch size: `batch_size, seq_len = in_idx.shape`
        _, seq_len = in_idx.shape

        # The device setting will allow us to train the model on a CPU or GPU,
        # depending on which device the input data sits on.
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        logits = self.out_head(x)

        return logits


###############################################################################
def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Listing 4.8: A function for the GPT model to generate text.
    """

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
