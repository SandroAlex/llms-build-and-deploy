---
name: pytorch
description: Expert guidance for deep learning, transformers, diffusion models, and large language model development with basic PyTorch package.
---

# Pytorch Fundamental Code

This skill covers the foundational elements of PyTorch programming including tensors, autograd, optimizers, and neural network modules.

## When to use this skill

- Write clean and efficient PyTorch code using the basic PyTorch package to build deep learning models, transformers, diffusion models, and large language models

## Best practices for writing PyTorch code

- Use brief comments that elucidate the purpose of code blocks, and highlight any non-obvious logic or decisions
- Follow PEP 8 style guidelines for Python code
- Ignore the PEP 8 recommendation of 79 columns for code and 72 columns for comments, as it is too restrictive for modern development environments and can hinder readability. Instead, all code and comments must not be longer than 90 columns to ensure readability across different devices and editors
- Write modular code by defining functions and classes to encapsulate specific functionality, which promotes code reusability and readability
- Write docstrings in numpy style for all functions and classes to provide clear documentation on their purpose, parameters, and return values. Be brief but informative in your docstrings, ensuring that they provide enough context for other developers to understand the functionality without being overly verbose
- Always use type hints for method parameters and return types to improve code readability and facilitate static type checking. Also add type hints for variable declarations, even when the type can be inferred. See more in the examples section

## Examples of PyTorch code with best practices

- Simple embedding layer implementation in PyTorch with best practices

```python
import math

from torch import Tensor, nn

class Embeddings(nn.Module):
    """
    Embedding layer for a given vocabulary size and model dimension.
    """

    def __init__(self, d_model: int, vocab: int) -> None:
		"""
		Initializes the Embeddings module.

		Parameters
		----------
		d_model : int
			The dimension of the model (embedding size)
		vocab : int
			The size of the vocabulary (number of unique tokens)
		"""
    
        super().__init__()
        
		# Layer that maps input token indices to dense vectors of size `d_model`
		self.lut: nn.Embedding = nn.Embedding(vocab, d_model)
        self.d_model: int = d_model

    def forward(self, x: Tensor) -> Tensor:
		"""
		Performs the forward pass of the Embeddings module.

		x : Tensor
			Input tensor containing token indices of shape (batch_size, sequence_length)

		Returns
		-------
		Tensor
			Output tensor containing the embedded representations of the input tokens,
			with shape (batch_size, sequence_length, d_model)
		"""

        out: Tensor = self.lut(x) * math.sqrt(self.d_model)
        
		return out
```
