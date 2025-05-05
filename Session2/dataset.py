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

        class_names = full_dataset.classes
        assert set(class_names) == {'person', 'robot'}, "Dataset must have exactly 'person' and 'robot' folders!"
        
        # group image indices by their corresponding class: { 'person': [1, 15...], ... }
        class_indices = {class_name: [] for class_name in class_names}
        for idx, (_, class_id) in enumerate(full_dataset.imgs):
            class_name = class_names[class_id]
            class_indices[class_name].append(idx)
        
        assert len(class_indices['person']) == len(class_indices['robot']), "The dataset has to be balanced with equal samples of each class!"
        
        np.random.shuffle(class_indices['person'])
        np.random.shuffle(class_indices['robot'])
        
        split_limit = int(self.train_ratio * len(class_indices['person']))
        train_indices = (class_indices['person'][:split_limit] + 
                        class_indices['robot'][:split_limit])
        val_indices = (class_indices['person'][split_limit:] + 
                    class_indices['robot'][split_limit:])
        
        self.train_dataset = Subset(
            datasets.ImageFolder(self.data_dir, transform=self.train_transform),
            train_indices
        )
        self.val_dataset = Subset(
            datasets.ImageFolder(self.data_dir, transform=self.val_transform),
            val_indices
        )

        if self.should_log:
            print(f"Training set size: {len(self.train_dataset)} images")
            print(f"Validation set size: {len(self.val_dataset)} images")


    def train_dataloader(self, collate_fn = None):
        return DataLoader(self.train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=32, shuffle=False, num_workers=4)
