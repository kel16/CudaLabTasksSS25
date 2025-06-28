import os
import torch

SAVED_MODELS_ROOT = "checkpoints"

def save_model(model, optimizer, epoch, stats):
    """ Saving model checkpoint """
    
    if(not os.path.exists(SAVED_MODELS_ROOT)):
        os.makedirs(SAVED_MODELS_ROOT)
    savepath = f"{SAVED_MODELS_ROOT}/checkpoint_epoch_{epoch}.pth"

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stats': stats
    }, savepath)
    
    return savepath

def load_model(model, optimizer, savepath = SAVED_MODELS_ROOT):
    """ Loading pretrained checkpoint """
    checkpoint = torch.load(savepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint["epoch"]
    stats = checkpoint["stats"]
    
    return model, optimizer, epoch, stats
