import numpy as np
from tqdm import tqdm

from evaluate import eval_model

def train_epoch(model, train_loader, optimizer, criterion, epoch, device):
    """ Training a model for one epoch """
    
    loss_list = []
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.view(-1, 1).float().to(device)

        
        optimizer.zero_grad()
         
        outputs = model(images)
         
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())
         
        loss.backward()
         
        optimizer.step()
        
    mean_loss = np.mean(loss_list)
    
    return mean_loss, loss_list

def train_model(model, optimizer, scheduler, criterion, train_loader, valid_loader, num_epochs, device, logger=None, start_epoch=0):
    """ Training a model for a given number of epochs"""
    
    train_loss = []
    val_loss =  []
    loss_iters = []
    valid_acc = []
    
    for epoch in tqdm(range(num_epochs)):
        # validation epoch
        model.eval()  # important for dropout and batch norms
        accuracy, loss = eval_model(
                    model=model, eval_loader=valid_loader,
                    criterion=criterion, device=device
            )
        valid_acc.append(accuracy)
        val_loss.append(loss)
        
        # training epoch
        model.train()  # important for dropout and batch norms
        mean_loss, cur_loss_iters = train_epoch(
                model=model, train_loader=train_loader, optimizer=optimizer,
                criterion=criterion, epoch=epoch, device=device
            )
        scheduler.step()
        train_loss.append(mean_loss)

        if logger:
            logger(f'Accuracy/Valid', accuracy, global_step=epoch+start_epoch)
            logger(f'Loss/Valid', loss, global_step=epoch+start_epoch)
            logger(f'Loss/Train', mean_loss, global_step=epoch+start_epoch)

        loss_iters = loss_iters + cur_loss_iters
        
        if (epoch % 5 == 0 or epoch==num_epochs-1):
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"    Train loss: {round(mean_loss, 5)}")
            print(f"    Valid loss: {round(loss, 5)}")
            print(f"    Accuracy: {accuracy}%")
            print("\n")
    
    print(f"Training completed")
    
    return train_loss, val_loss, loss_iters, valid_acc
