import os
from datetime import datetime
import torch

CONFIG_ROOT = "models"

def get_save_root():
     if(not os.path.exists(CONFIG_ROOT)):
        os.makedirs(CONFIG_ROOT)

     return CONFIG_ROOT

def save_model_config(model, name = 'checkpoint'):
    savepath = f"{get_save_root()}/{f'{name}_{datetime.now().strftime("%H-%M_%d-%m-%Y")}'}.pth"

    torch.save(model.state_dict(), savepath)
    print(f"saved to {savepath}")

def load_model_config(model, file_name):
    model.load_state_dict(torch.load(f"{CONFIG_ROOT}/{file_name}"))
