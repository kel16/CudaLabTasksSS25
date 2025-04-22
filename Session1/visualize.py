import matplotlib.pyplot as plt
import numpy as np
from typing import List

def smooth(f, K = 5):
    """ Smoothing a function using a low-pass filter (mean) of size K """
    kernel = np.ones(K) / K
    f = np.concatenate([f[:int(K//2)], f, f[int(-K//2):]])  # to account for boundaries
    smooth_f = np.convolve(f, kernel, mode="same")
    smooth_f = smooth_f[K//2: -K//2]  # removing boundary-fixes
    
    return smooth_f

def plot_learning_curve(train_loss_list: List[int], val_loss_list: List[int], num_iterations: int):
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(18, 5)
    smooth_loss = smooth(train_loss_list, 31)
    indices = list(range(0, len(train_loss_list), num_iterations)) 
    
    ax[0].plot(train_loss_list, c="blue", label="Training Loss", linewidth=3, alpha=0.5)
    ax[0].plot(smooth_loss, c="red", label="Smoothed Training Loss", linewidth=3)
    ax[0].plot(indices, val_loss_list, c="black", label="Validation Loss", linewidth=3, alpha=0.5)
    ax[0].legend(loc="best")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("CE Loss")
    ax[0].set_title("Training Progress (linearscale)") 
    
    ax[1].plot(train_loss_list, c="blue", label="Training Loss", linewidth=3, alpha=0.5)
    ax[1].plot(smooth_loss, c="red", label="Smoothed Loss", linewidth=3)
    ax[1].plot(indices, val_loss_list, c="black", label="Validation Loss", linewidth=3, alpha=0.5)
    ax[1].legend(loc="best")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("CE Loss")
    ax[1].set_yscale("log")
    ax[1].set_title("Training Progress (logscale)")
   
    plt.show()

def plot_images(images, teacher_labels, predicted_labels, classes):
    """
    accepts list of images, corresponding true/output labels and list of label names
    """
    plt.figure(figsize=(12, 6))
    for i in range(min(8, len(images))):
        # img = images[i]
        # Unnormalize image (CIFAR-10 specific)
        img = images[i] / 2 + 0.5  # [-1, 1] -> [0, 1]
        img = img.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
        
        plt.subplot(2, 4, i + 1)
        plt.imshow(img)
        plt.title(f"Ground truth: {classes[teacher_labels[i]]}\nPrediction: {classes[predicted_labels[i]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
