import cv2
import matplotlib.pyplot as plt

def visualize_attention(image, attention_maps):
    """ Overlaying the attention maps on the image """
    num_layers = len(attention_maps)
    num_heads, num_tokens = attention_maps[0].shape

    # first displaying raw image
    fig, ax = plt.subplots(1, num_layers + 1)
    fig.set_size_inches(30, 5)
    ax[0].imshow(image)
    ax[0].axis("off")
    ax[0].set_title("Image", fontsize=20)

    # displaying attention from each layer
    image_unnorm = (image * 255).astype(np.uint8)
    H, W = image.shape[:2]
    for i in range(num_layers):
        cur_attn = attention_maps[i][:, 1:]  # current attn and removing [CLS] token

        attn = cur_attn.mean(axis=0)  # average across heads 
        attn = attn / attn.max()  # renormalization
        attn_grid = attn.reshape(4, 4)  # mapping back to image

        # Resize to image resolution        
        attn_up = cv2.resize(attn_grid, (W, H), interpolation=cv2.INTER_CUBIC)
        # attn_up = cv2.resize(attn_grid, (W, H), interpolation=cv2.INTER_NEAREST)

        cmap = "coolwarm"
        # cmap = "jet"
        
        im = ax[i+1].imshow(image)
        ax[i+1].imshow(attn_up, cmap=cmap, alpha=0.1, extent=(0, W, H, 0))
        cbar = plt.colorbar(ax[i+1].imshow(attn_up, cmap=cmap, alpha=0.4, extent=(0, W, H, 0)), ax=ax[i+1], fraction=0.046, pad=0.04)
        cbar.set_label('Attention Intensity', fontsize=15)
        ax[i+1].axis('off')
        ax[i+1].set_title(f"Attention Layer {i+1}/{num_layers}", fontsize=20)
        
    plt.show()
