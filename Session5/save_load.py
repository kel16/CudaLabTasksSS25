import os
import torch


SAVED_MODELS_ROOT = "checkpoints"
def get_save_root():
    if(not os.path.exists(SAVED_MODELS_ROOT)):
        os.makedirs(SAVED_MODELS_ROOT)
        
    return SAVED_MODELS_ROOT

def save_model(model, name, save_path = get_save_root()):
    """ Saving model checkpoint """
    
    if(not os.path.exists(save_path)):
        os.makedirs(save_path)
    savepath = f"{save_path}/checkpoint_{name}.pth"

    torch.save(model.state_dict(), savepath)

    print(f"Saved state to {savepath}")
    
    return savepath

def load_model(model, optimizer, savepath = get_save_root()):
    """ Loading pretrained checkpoint """
    checkpoint = torch.load(savepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint["epoch"]
    stats = checkpoint["stats"]
    
    return model, optimizer, epoch, stats
