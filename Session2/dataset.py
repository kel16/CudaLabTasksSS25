import numpy as np
from torchvision import datasets
from torch.utils.data import Subset
from torch.utils.data import DataLoader

class PersonRobotDataset():
    """ Loads data from path, does splitting, augmentations, and creates DataLoaders """
    def __init__(self, data_dir: str = "images", train_ratio = .8,
                 train_transform = None, val_transform = None,
                 should_log = True):
        self.data_dir = data_dir
        self.train_ratio = train_ratio
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.should_log = should_log
        self._setup()

    def _setup(self, *_):
        full_dataset = datasets.ImageFolder(self.data_dir) 
        indices = list(range(len(full_dataset)))
        np.random.shuffle(indices)

        split = int(self.train_ratio * len(full_dataset))
        train_indices, val_indices = indices[:split-1], indices[split-1:]

        self.train_dataset = Subset(datasets.ImageFolder(self.data_dir, transform=self.train_transform), train_indices)
        self.val_dataset   = Subset(datasets.ImageFolder(self.data_dir, transform=self.val_transform), val_indices)

        if self.should_log:
            print(f"Training set size: {len(self.train_dataset)} images")
            print(f"Validation set size: {len(self.val_dataset)} images")

    def train_dataloader(self, collate_fn = None):
        return DataLoader(self.train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=32, shuffle=False, num_workers=4)
