import matplotlib.pyplot as plt
import numpy as np

def smooth(f, K = 5):
    """ Smoothing a function using a low-pass filter (mean) of size K """
    kernel = np.ones(K) / K
    f = np.concatenate([f[:int(K//2)], f, f[int(-K//2):]])  # to account for boundaries
    smooth_f = np.convolve(f, kernel, mode="same")
    smooth_f = smooth_f[K//2: -K//2]  # removing boundary-fixes
    
    return smooth_f

def plot_learning_curve(loss_list: list[int]):
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(18, 5)
    smooth_loss = smooth(loss_list, 31)

    ax[0].plot(loss_list, c="blue", label="Training Loss", linewidth=3, alpha=0.5)
    ax[0].plot(smooth_loss, c="red", label="Smoothed Loss", linewidth=3)
    ax[0].legend(loc="best")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("CE Loss")
    ax[0].set_title("Training Progress (linearscale)")

    ax[1].plot(loss_list, c="blue", label="Training Loss", linewidth=3, alpha=0.5)
    ax[1].plot(smooth_loss, c="red", label="Smoothed Loss", linewidth=3)
    ax[1].legend(loc="best")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("CE Loss")
    ax[1].set_yscale("log")
    ax[1].set_title("Training Progress (logscale)")

    plt.show()
    