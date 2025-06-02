import matplotlib.pyplot as plt

def visualize_vector_samples(imgs):
    """ accepts images of a shape (3, H, W) """
    _, ax = plt.subplots(1, len(imgs), figsize=(30, 3))
    
    for i,img in enumerate(imgs):
        ax[i].imshow(img.permute(1,2,0).numpy(), cmap="gray")
        ax[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.subplots_adjust(wspace=0., hspace=0)
    ax[0].set_title("Image 1")
    ax[-1].set_title("Image 2")
    plt.show()
