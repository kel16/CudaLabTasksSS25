class Patchifier:
    """ 
    Module that splits an image into patches.
    We assumen square images and patches
    """

    def __init__(self, patch_size):
        """ """
        self.patch_size = patch_size

    def __call__(self, img):
        """ """
        B, C, H, W = img.shape
        assert H % self.patch_size == 0
        assert W % self.patch_size == 0
        num_patch_H = H // self.patch_size
        num_patch_W = W // self.patch_size

        # splitting and reshaping
        patch_data = img.reshape(B, C, num_patch_H, self.patch_size, num_patch_W, self.patch_size)
        patch_data = patch_data.permute(0, 2, 4, 1, 3, 5)  # ~(B, n_p_h, n_p_w, C, p_s, p_s)
        patch_data = patch_data.reshape(B, num_patch_H * num_patch_W, C * self.patch_size * self.patch_size)
        return patch_data
    