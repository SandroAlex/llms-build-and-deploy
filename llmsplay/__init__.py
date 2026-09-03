# Initial imports
import torch

# Check if CUDA is available and set the device accordingly
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"