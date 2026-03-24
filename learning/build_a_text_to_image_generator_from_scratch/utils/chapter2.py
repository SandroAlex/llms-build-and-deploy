# Initial imports
from typing import List

import numpy as np
import torch

# Define a constant for padding
PAD: int = 0

# Check if CUDA is available and set the device accordingly
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


def seq_padding(X: List[List[int]], padding: int = PAD) -> np.ndarray:
    """
    Pad sequences to the same length.

    Parameters
    ----------
    X : List[List[int]]
        A list of sequences, where each sequence is a list of integers.
    padding : int, optional
        The value to use for padding, by default PAD (which is 0).

    Returns
    -------
    np.ndarray
        A 2D numpy array where each sequence has been padded to the same length.
    """

    L: List[int] = [len(x) for x in X]
    ML: int = max(L)

    padded_seq: np.ndarray = np.array(
        [np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X]
    )

    return padded_seq
