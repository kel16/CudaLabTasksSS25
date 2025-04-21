from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train(model: nn.Module, epochs: int, dataloader: DataLoader,
          device: torch.device, criterion, optimizer: Optimizer,
          lr_scheduler = None):
    """
    Trains a neural model.

    Returns:
    - list of batch errors
    """
    loss_list = []

    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, total=len(dataloader))
        
        for i, (imgs, labels) in enumerate(progress_bar):
            imgs, labels = imgs.to(device), labels.to(device)
        
            # forward pass
            predictions = model(imgs)
        
            # computing error
            loss = criterion(predictions, labels)
            loss_list.append(loss.item())

            # removing accumulated gradients
            optimizer.zero_grad()
        
            # backprogating error to compute gradients
            loss.backward()
        
            # updating parameters
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()
        
            if(i % 10 == 0 or i == len(dataloader) - 1):
                progress_bar.set_description(f"Epoch {epoch + 1} Iteration {i + 1}: loss {loss.item():.5f}. ")

    return loss_list
