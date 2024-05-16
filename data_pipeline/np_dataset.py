import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.preprocessing import OneHotEncoder
import torch

class NpDataset(Dataset):
    def __init__(self, file_paths : np.array, labels : np.array, file_reader,  transform=None,
                  augmentation_transform=None, label_encoder = None,
                  augmentation_indicator = None):
        self.file_paths = np.array(file_paths)
        self.labels = np.array(labels)
        self.file_reader = file_reader
        self.label_encoder = label_encoder
        self.transform = transform
        self.augmentation_transform = augmentation_transform
        self.augmentation_indicator = augmentation_indicator if augmentation_indicator else np.full(len(file_paths), False)
        #idea think about passing an array of augmentations to the dataset so every sample can have a different augmentation

    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        data = self.file_reader(self.file_paths[idx])
        data_labels = self.labels[idx]
        
        #transform to tensor
        label_tensor = torch.Tensor(data_labels).float()
        if self.augmentation_transform and self.augmentation_indicator[idx]:
            #todo find out how to indicate if augmentation should be done
            data = self.augmentation_transform(data)
        if self.transform:
            data = self.transform(data)
        #convert to tensor
        data = np.array(data)
        data = torch.Tensor(data).float().permute(2, 0, 1)
        return data, label_tensor
