from typing import Callable, Tuple

import torch
from torch.utils.data import DataLoader


# Listing 3.8 Defining an 'EarlyStop' class
class EarlyStop:
    """
    Tracks a loss over successive calls and signals when to checkpoint the model and when
    to stop training. Training should stop once the loss has failed to improve for
    'patience' consecutive checks in a row.
    """

    def __init__(self, patience: int = 3) -> None:
        """
        Parameters
        ----------
        patience : int, default=3
            Number of consecutive non-improving checks allowed before signaling to stop
            training.
        """

        self.patience = patience

        # Number of consecutive checks since the loss last improved
        self.steps = 0

        # Best (lowest) loss seen so far
        self.min_loss = float("inf")

    def stop(self, loss: float) -> Tuple[bool, bool]:
        """
        Parameters
        ----------
        loss : float
            Latest loss value to compare against the best seen so far.

        Returns
        -------
        Tuple[bool, bool]
            (to_save, to_stop): whether this is a new best loss worth checkpointing,
            and whether training should stop.
        """

        # New best loss: reset the patience counter and flag it for saving
        if loss < self.min_loss:
            self.min_loss = loss
            self.steps = 0
            to_save = True

        # No improvement: count another non-improving check
        elif loss >= self.min_loss:
            self.steps += 1
            to_save = False

        # Stop once the loss has failed to improve for 'patience' checks in a row
        if self.steps >= self.patience:
            to_stop = True
        else:
            to_stop = False

        return to_save, to_stop


def train_batch(
    batch: Tuple[torch.Tensor, torch.Tensor],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    trainloader: DataLoader,
    device: str,
) -> float:
    """
    Runs one training step (forward, backward, and optimizer update) for a single
    batch, using automatic mixed precision.

    Parameters
    ----------
    batch : Tuple[torch.Tensor, torch.Tensor]
        A (images, labels) pair as produced by the training DataLoader.
    model : torch.nn.Module
        The model being trained.
    optimizer : torch.optim.Optimizer
        Optimizer that updates 'model' parameters.
    scaler : torch.cuda.amp.GradScaler
        Gradient scaler used for mixed-precision training.
    loss_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        Loss function applied to (predictions, labels).
    trainloader : DataLoader
        The training DataLoader that 'batch' was drawn from; only used here to
        weight this batch's loss by the full dataset size.
    device : str
        Device to run the forward/backward pass on (e.g. "cuda" or "cpu").

    Returns
    -------
    float
        This batch's loss, scaled by its share of the full training dataset, so that
        summing the return value across all batches in an epoch gives the epoch's
        average loss.
    """

    # Move every tensor in the batch onto the training device (CPU/GPU)
    batch = [t.to(device) for t in batch]
    images, labels = batch

    # Run the forward pass in mixed precision to speed up training and reduce memory
    with torch.amp.autocast(device):
        loss = loss_fn(model(images)[0], labels)

    # Clear gradients from the previous step
    optimizer.zero_grad()

    # Backward pass, with the loss scaled up to avoid vanishing gradients in fp16
    scaler.scale(loss).backward()

    # Unscale gradients and apply the optimizer step
    scaler.step(optimizer)

    # Adjust the scale factor for the next iteration
    scaler.update()

    # Weight this batch's average loss by its fraction of the full dataset
    return loss.item() * len(images) / len(trainloader.dataset)
