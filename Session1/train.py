from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import List


def train(model: nn.Module, epochs: int, train_dataloader: DataLoader,
          validation_dataloader: DataLoader, device: torch.device, criterion, 
          optimizer: Optimizer, lr_scheduler = None) -> List[float]:
    """
    Trains a neural model.

    Returns:
    - List of batch errors
    """
    train_loss_list = []
    val_loss_list = []

    for epoch in range(epochs):
        progress_bar = tqdm(train_dataloader, total=len(train_dataloader))
        
        for i, (imgs, labels) in enumerate(progress_bar):
            imgs, labels = imgs.to(device), labels.to(device)
            
            model.train()
            
            # forward pass
            predictions = model(imgs)
        
            # computing error
            loss = criterion(predictions, labels)
            train_loss_list.append(loss.item())

            # removing accumulated gradients
            optimizer.zero_grad()
        
            # backprogating error to compute gradients
            loss.backward()
        
            # updating parameters
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()
            
            
            if(i % 10 == 0 or i == len(train_dataloader) - 1):
                progress_bar.set_description(f"Epoch {epoch + 1} Iteration {i + 1}: loss {loss.item():.5f}. ")
                
        model.eval()
        val_loss_total = 0.0

        with torch.no_grad():
            for val_imgs, val_labels in validation_dataloader:
                val_imgs, val_labels = val_imgs.to(device), val_labels.to(device)
                val_preds = model(val_imgs)
                val_loss = criterion(val_preds, val_labels)
                val_loss_total += val_loss.item()

        avg_val_loss = val_loss_total / len(validation_dataloader)  
        val_loss_list.append(avg_val_loss)
            

    return train_loss_list, val_loss_list
