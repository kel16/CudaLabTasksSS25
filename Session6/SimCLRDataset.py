from torchvision import transforms

class ContrastiveTransform:
    """
    Data augmentation for contrastive learning
    """
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Pad(2),
            transforms.RandomResizedCrop(32, scale=(0.5, 1.33)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor()
        ])

    def __call__(self, x):
        """ Given an Image, returning two augmented versions of the same image """
        return self.transform(x), self.transform(x)
    
class SimCLRDataset():
    """
    Wrapper to return pairs of augmented images
    """
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = ContrastiveTransform()
        return
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, _ = self.dataset[idx]  # ignore labels
        
        view1, view2 = self.transform(img)
        return view1, view2
