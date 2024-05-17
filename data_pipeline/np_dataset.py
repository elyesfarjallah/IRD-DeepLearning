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
    
    def balance_augmentation(self, n_max_augmentations = 10):
        #copy the data and labels
        new_data = self.file_paths.copy()
        new_labels = self.labels.copy()
        new_augmentation_indicator = self.augmentation_indicator.copy()
        #count the number of samples for each class
        unique_labels, counts = np.unique(new_labels, return_counts=True)
        max_count = np.max(counts)
        difference = max_count - counts
        #for each class pick n_difference samples randomly and set the augmentation indicator to True
        for label, n_diff in zip(unique_labels, difference):
            indices = np.where(new_labels == label)[0]
            #shuffle the indices
            n_instances = min(n_diff, len(indices), n_max_augmentations)
            mu, std = n_instances / 2, n_instances / 4
            n_instances_to_pick = int(np.random.normal(mu, std))
            n_instances_to_pick = max(min(n_max_augmentations, n_instances_to_pick), 0)
            if n_instances_to_pick > 0:
                random_indices = np.random.choice(indices, n_instances_to_pick, replace=False)
                augmentation_indicators_to_add = augmentation_indicators_to_add = np.ones(n_instances_to_pick, dtype=bool)
                file_paths_to_add = new_data[random_indices]
                labels_to_add = new_labels[random_indices]
                new_data = np.concatenate((new_data, file_paths_to_add))
                new_labels = np.concatenate((new_labels, labels_to_add))
                new_augmentation_indicator = np.concatenate((new_augmentation_indicator, augmentation_indicators_to_add))
        #add the new data to the dataset
        self.file_paths = new_data
        self.labels = new_labels
        self.augmentation_indicator = new_augmentation_indicator
