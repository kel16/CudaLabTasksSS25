import torch
import random
import numpy as np

class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        """ Dataset initializer"""
        self.dataset = dataset
        self.labels = np.array([label for _, label in dataset])
        self.label_to_indices = {}

        for idx, label in enumerate(self.labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

    def __len__(self):
        """ Returning number of anchors """
        return len(self.dataset)

    def __getitem__(self, i):
        """ 
        Sampling a triplet for the dataset. Index i corresponds to anchor 
        """
        # sampling anchor
        anchor_img, anchor_lbl = self.dataset[i]

        positive_indices = self.label_to_indices[anchor_lbl]
        positive_index = i
        # if indices coincide, re-pick
        while positive_index == i:
            positive_index = random.choice(positive_indices)
        pos_img, _ = self.dataset[positive_index]

        negative_label = random.choice([lbl for lbl in self.label_to_indices if lbl != anchor_lbl])
        negative_index = random.choice(self.label_to_indices[negative_label])
        neg_img, _ = self.dataset[negative_index]

        return (anchor_img, pos_img, neg_img), (anchor_lbl, anchor_lbl, negative_label)
