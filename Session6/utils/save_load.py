import os
import torch

SAVED_MODELS_ROOT = "checkpoints"
def get_save_root():
    if(not os.path.exists(SAVED_MODELS_ROOT)):
        os.makedirs(SAVED_MODELS_ROOT)
        
    return SAVED_MODELS_ROOT

def save_model(model, name, stats = {}, save_path = get_save_root()):
    """ Saving model checkpoint """
    
    if(not os.path.exists(save_path)):
        os.makedirs(save_path)
    savepath = f"{save_path}/{name}.pth"

    torch.save({
            "model_state_dict": model.state_dict(),
            "stats": stats,
        }, savepath)

    print(f"Saved state to {savepath}")
    
    return savepath

def load_model(model, name, savedir = get_save_root()):
    """ Loading pretrained checkpoint """
    checkpoint = torch.load(f"{savedir}/{name}.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    stats = checkpoint["stats"]
    
    return model, stats
