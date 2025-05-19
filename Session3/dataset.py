import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch

KTH_CLASSES = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
TEST_SIZE = 0.2

class KTHDataset(Dataset):
    def __init__(self, root_dir, train=True, sequence_length=10, transform=None):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.train = train
        self.class_to_idx = {cls: i for i, cls in enumerate(KTH_CLASSES)}
        
        self.videos = []
        self.labels = []
        
        for cls in KTH_CLASSES:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.exists(cls_dir):
                continue
            for video_file in os.listdir(cls_dir):
                if video_file.endswith('.avi'):    
                    video_path = os.path.join(cls_dir, video_file)
                    self.videos.append(video_path)
                    self.labels.append(self.class_to_idx[cls])
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.videos, self.labels, test_size=TEST_SIZE,
            stratify=self.labels, shuffle=True
        )
        if self.train:
            self.videos, self.labels = X_train, y_train
        else:
            self.videos, self.labels = X_test, y_test

    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        RESIZE_SIZE = (64, 64)
        video_path = self.videos[idx]
        label = self.labels[idx]
        
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, RESIZE_SIZE)
            frames.append(frame)
        
        cap.release()
        
        if len(frames) >= self.sequence_length:
            start_idx = 0 + self.sequence_length*3
            frames = frames[start_idx : start_idx + self.sequence_length]
        else:
            while len(frames) < self.sequence_length:
                frames.append(frames[-1])
        
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        else:
            frames = [torch.tensor(frame.transpose(2, 0, 1), dtype=torch.float32) / 255.0 for frame in frames]

        frames = torch.stack(frames)  #[sequence_length, 3, 64, 64]
        
        return frames, torch.tensor(label, dtype=torch.long)
    
