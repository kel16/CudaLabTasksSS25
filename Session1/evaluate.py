import torch
from tqdm import tqdm

@torch.no_grad()
def test(model, dataloader, device):
    n_correct = 0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    
    for i, (imgs, labels) in progress_bar: 
        #everything needs to be on the same device
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        # forward pass
        flattened_imgs = imgs.flatten(start_dim=1)
        preds = model(flattened_imgs)
        
        pred_labels = torch.argmax(torch.softmax(preds, dim=-1), dim=-1)
        cur_correct = len(torch.where(pred_labels == labels)[0])
        n_correct = n_correct + cur_correct

    accuracy = n_correct / len(dataloader.dataset) * 100
    print(f"Test accuracy: {round(accuracy,2)}%")

    return accuracy
