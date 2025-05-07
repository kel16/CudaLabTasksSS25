import matplotlib.pyplot as plt
import numpy as np

from utils.visualize import smooth

def plot_learning_curve(stats):
    """ Plots learning curves.

        Parameter stats
        ----------
        Object of the form:
        
        "train_loss": train_loss,\n
        "val_loss": val_loss,\n
        "loss_iters": loss_iters,\n
        "valid_acc": valid_acc
    """
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(1,3)
    fig.set_size_inches(24,5)

    loss_iters = stats.get("loss_iters")
    train_loss = stats.get("train_loss")
    val_loss = stats.get("val_loss")
    valid_acc = stats.get("valid_acc")

    smooth_loss = smooth(loss_iters, 31)
    ax[0].plot(loss_iters, c="blue", label="Loss", linewidth=3, alpha=0.5)
    ax[0].plot(smooth_loss, c="red", label="Smoothed Loss", linewidth=3, alpha=1)
    ax[0].legend(loc="best")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("CE Loss")
    ax[0].set_title("Training Progress")

    epochs = np.arange(len(train_loss)) + 1
    ax[1].plot(epochs[1:], train_loss[1:], c="red", label="Train Loss", linewidth=3)
    ax[1].plot(epochs[1:], val_loss[1:], c="blue", label="Valid Loss", linewidth=3)
    ax[1].legend(loc="best")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("CE Loss")
    ax[1].set_title("Loss Curves")

    epochs = np.arange(len(val_loss)) + 1
    ax[2].plot(epochs[1:], valid_acc[1:], c="red", label="Valid accuracy", linewidth=3)
    ax[2].legend(loc="best")
    ax[2].set_xlabel("Epochs")
    ax[2].set_ylabel("Accuracy (%)")
    ax[2].set_title(f"Valdiation Accuracy (max={round(np.max(valid_acc),2)}% @ epoch {np.argmax(valid_acc)+1})")

    plt.show()
