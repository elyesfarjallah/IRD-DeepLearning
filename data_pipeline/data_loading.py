from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
from PIL import Image
import pandas as pd
import numpy as np
from copy import deepcopy



class DfDataset(Dataset):
    def __init__(self, df, data_path_col : str, label_cols : list, transform=None, augmentation=False, shuffle=True):
        self.df = df
        #add do augmentation column
        self.df['do_augmentation'] = False
        self.transform = transform
        self.data_path_col = data_path_col
        self.label_cols = label_cols
        self.augmentation = augmentation
        self.classes = label_cols
        #create a list of possible transformations
        
        if self.augmentation:
            self.augmentations = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(p=0.6),
                transforms.RandomVerticalFlip(p=0.5)
            ])
            new_data = self.augment_data(img_paths=self.df[self.data_path_col].values, img_labels=self.df[self.label_cols].values, label_cols=self.label_cols)
            self.df = pd.concat([self.df, new_data], ignore_index=True)
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        #shuffle data
        if shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row[self.data_path_col]).convert('RGB')
        label = torch.Tensor(row[self.label_cols].values.astype(np.int8)).float()
        if self.augmentation:
            img = self.augmentations(img)
        if self.transform:
            img = self.transform(img)
        return img, label
    
    def calculate_class_weights(self):
        #calculate class weights
        class_weights = []
        for label_col in self.label_cols:
            class_weights.append(1 / self.df[label_col].sum(axis = 0))
        return class_weights
    
    def augment_data(self, img_paths : np.array, img_labels : np.array, label_cols : np.array, max_augmentations : int = 10, augmentation_threshold : float = 0.2):
        #todo implemnt better augmentation, get all images with label, n random picks, augment them, add them to the dataset
        #augment data
        class_weights = self.calculate_class_weights()
        min_class_weights = min(class_weights)
        max_class_weights = max(class_weights)
        new_img_paths = []
        do_augmentation = []
        #generate map
        new_img_labels = {label : [] for label in label_cols}
        for img_path, img_label in zip(img_paths, img_labels):
            #get img class weight
            img_class_weights = [weight for weight, label in zip(class_weights, img_label) if label == 1]
            max_class_weight = max(img_class_weights)
                #make augmentation decision based on uniform distribution
            if np.random.uniform(low=min_class_weights, high=max_class_weights) < max_class_weight*(1 - augmentation_threshold):
                for i in range(max_augmentations):
                    do_augmentation.append(True)
                    new_img_paths.append(img_path)
                    for label_key, label in zip(label_cols, img_label):
                        is_labeled_with_label = label == 1
                        #convert bol to int
                        is_labeled_with_label = int(is_labeled_with_label)
                        new_img_labels[label_key].append(is_labeled_with_label)
        
        #create new dataframe
        new_data = deepcopy(new_img_labels)
        new_data.update({self.data_path_col : new_img_paths, 'do_augmentation' : do_augmentation})
        new_data = pd.DataFrame(new_data)
        
        return new_data
    

def test_dataset():
    df = pd.read_csv('datasets/2024-01-23_15-07-30/train.csv')
    #labels cols Maculopathy,Myopia,cataract,Diabetic Retinopathy,Bietti crystalline dystrophy,Best Disease,Cone Dystrophie or Cone-rod Dystrophie,Age-related Macular Degeneration,Stargardt Disease,Normal,Retinitis Pigmentosa,glaucoma
    label_cols = ['Maculopathy','Myopia','cataract','Diabetic Retinopathy','Bietti crystalline dystrophy','Best Disease','Cone Dystrophie or Cone-rod Dystrophie','Age-related Macular Degeneration',
                  'Stargardt Disease','Normal','Retinitis Pigmentosa','glaucoma']
    dataset_augmented = DfDataset(df=df, data_path_col='path_to_img', label_cols=label_cols, augmentation=True, transform=transforms.Resize((224, 224)))
    dataset_original = DfDataset(df=df, data_path_col='path_to_img', label_cols=label_cols, augmentation=False)
    #print class weights
    print(dataset_augmented.calculate_class_weights())
    print(dataset_original.calculate_class_weights())
    #count labels
    print(dataset_augmented.df[label_cols].sum(axis = 0))
    print(dataset_original.df[label_cols].sum(axis = 0))
    #create dataloader
    dataloader_augmented = DataLoader(dataset_augmented, batch_size=32, shuffle=True)
    #iterate over dataloader
