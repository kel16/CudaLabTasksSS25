import shutil
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import os

from utils.save_load import save_model, get_save_root
from evaluate import eval_model

def vae_loss_function(recons, target, mu, log_var, lambda_kld=1e-3):
    """
    Combined loss function for joint optimization of 
    reconstruction and ELBO
    """
    recons_loss = F.mse_loss(recons, target)
    kld = (-0.5 * (1 + log_var - mu**2 - log_var.exp()).sum(dim=1)).mean(dim=0)  # closed-form solution of KLD in Gaussian
    loss = recons_loss + lambda_kld * kld

    return loss, (recons_loss, kld)

def train_epoch(model, train_loader, optimizer, criterion, kl_weight, device):
    """ Training a model for one epoch """
    
    loss_list = []
    recons_loss = []
    vae_loss = []

    for i, (images, _) in enumerate(train_loader):
        images = images.to(device)
        
        optimizer.zero_grad()
        
        recons, (_, mu, log_var) = model(images)

        loss, (mse, kld) = criterion(recons, images, mu, log_var, kl_weight)
        loss_list.append(loss.item())
        recons_loss.append(mse.item())
        vae_loss.append(kld.item())

        loss.backward()
         
        optimizer.step()
        
    mean_loss = np.mean(loss_list)
    
    return mean_loss, loss_list

def train_model(model, optimizer, scheduler, criterion, train_loader, valid_loader,
                num_epochs, device, save_frequency=5, vis_frequency=2, kl_weight=1e-3):
    """ Training a model for a given number of epochs"""
    timestamp = datetime.now().strftime("%H-%M_%d-%m-%Y")
    save_path_root = f"{get_save_root()}/{timestamp}"
    if(not os.path.exists(save_path_root)):
        os.makedirs(save_path_root)

    TBOARD_LOGS = os.path.join(os.getcwd(), "tboard_logs", "ConvVAE", timestamp)
    if not os.path.exists(TBOARD_LOGS):
        os.makedirs(TBOARD_LOGS)
    shutil.rmtree(TBOARD_LOGS)
    writer = SummaryWriter(TBOARD_LOGS)

    savepath = f"imgs/{timestamp}"
    if(not os.path.exists(savepath)):
        os.makedirs(savepath)
    
    train_loss = []
    val_loss =  []
    val_loss_recons =  []
    val_loss_kld =  []
    loss_iters = []
    
    for epoch in range(num_epochs): 
        # validation epoch
        model.eval() 
        log_epoch = (epoch % vis_frequency == 0 or epoch == num_epochs - 1)
        loss, recons_loss, kld_loss = eval_model(
                model=model, eval_loader=valid_loader, criterion=criterion,
                device=device, epoch=epoch, savefig=log_epoch, savepath=savepath,
                writer=writer, kl_weight=kl_weight
            )
        val_loss.append(loss)
        val_loss_recons.append(recons_loss)
        val_loss_kld.append(kld_loss)

        writer.add_scalar(f'Loss/Valid', loss, global_step=epoch)
        writer.add_scalars(f'Loss/All_Valid_Loss', {"recons": recons_loss.item(), "kld": kld_loss.item()}, global_step=epoch)
        
        # training epoch
        model.train()  # important for dropout and batch norms
        mean_loss, cur_loss_iters = train_epoch(
                model=model, train_loader=train_loader, optimizer=optimizer,
                criterion=criterion, kl_weight=kl_weight, device=device
            )
        writer.add_scalar(f'Loss/Train', mean_loss, global_step=epoch)
        writer.add_scalars(f'Loss/Comb', {"train": mean_loss.item(), "valid": loss.item()}, global_step=epoch)
        
        # PLATEAU SCHEDULER
        scheduler.step(val_loss[-1])
        train_loss.append(mean_loss)
        loss_iters = loss_iters + cur_loss_iters
        
        if((epoch + 1) % save_frequency == 0):
            stats = {
                "train_loss": train_loss,
                "valid_loss": val_loss,
                "loss_iters": loss_iters,
                "kl_weight": kl_weight,
            }
            save_model(model=model, optimizer=optimizer, epoch=(epoch+1), stats=stats, save_path=save_path_root)
        
        if(log_epoch):
            print(f"    Train loss: {round(mean_loss, 5)}")
            print(f"    Valid loss: {round(loss, 5)}")
            print(f"       Valid loss recons: {round(val_loss_recons[-1], 5)}")
            print(f"       Valid loss KL-D:   {round(val_loss_kld[-1], 5)}")
    
    print(f"Training completed")
    
    return train_loss, val_loss, loss_iters, val_loss_recons, val_loss_kld
    