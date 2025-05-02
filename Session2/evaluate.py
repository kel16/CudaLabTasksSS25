import torch
import numpy as np

@torch.no_grad()
def eval_model(model, eval_loader, criterion, device):
    """ Evaluating the model for either validation or test """
    correct = 0
    total = 0
    loss_list = []
    
    for images, labels in eval_loader:
        images = images.to(device)
        labels = labels.view(-1, 1).float().to(device)
        
        outputs = model(images)
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float()
        correct += len(torch.where(preds==labels)[0])
        total += len(labels)
                 
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())
                 
    accuracy = correct / total * 100
    loss = np.mean(loss_list)
    
    return accuracy, loss
