import torch
from tqdm import tqdm

@torch.no_grad()
def evaluate(model, dataloader, device, confusion_matrix):
    if confusion_matrix:
        confusion_matrix.reset()

    n_correct = 0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    
    for _, (imgs, labels) in progress_bar: 
        # everything needs to be on the same device
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        # forward pass
        preds = model(imgs)
        
        pred_labels = torch.argmax(torch.softmax(preds, dim=-1), dim=-1)
        cur_correct = len(torch.where(pred_labels == labels)[0])
        n_correct = n_correct + cur_correct
        if confusion_matrix:
            confusion_matrix.update(pred_labels, labels)
    
    metrics = {
        'accuracy': n_correct / len(dataloader.dataset) * 100
    }
    
    if confusion_matrix:
        metrics['confusion_matrix'] = confusion_matrix

    return metrics
