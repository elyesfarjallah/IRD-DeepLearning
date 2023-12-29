import optuna
import torch
import numpy as np
import json
from uuid import uuid4
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from image_transforms import standard_transform
import models_torch
import datetime
from trainer import train
import joblib
import random
from torchvision import transforms

#read the config file
with open('wandb_config.json') as f:
    wandb_config = json.load(f)

class Objective:
    def __init__(self, model_to_optimize, model_name, n_classes, wandb_config_dict, dataset, n_epochs : int,
                  train_sampler, validation_sampler, dataset_path : str,
                    run_tags : list = [], n_epochs_validation : int = 1, prefered_device : str = 'cuda:0'):
        # Hold this implementation specific arguments as the fields of the class.
        self.model = model_to_optimize
        self.model_name = model_name
        self.n_classes = n_classes
        self.wandb_config_dict = wandb_config_dict
        self.dataset = dataset
        self.n_epochs = n_epochs
        self.train_sampler = train_sampler
        self.validation_sampler = validation_sampler
        self.run_tags = run_tags
        self.dataset_path = dataset_path
        self.n_epochs_validation = n_epochs_validation
        self.prefered_device = prefered_device


    def __call__(self, trial):
        # Calculate an objective value by using the extra arguments.
        #suggest a value for the hyperparameter
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        batch_size = trial.suggest_categorical("    ", [4, 8, 16, 32, 64, 128, 256])
        train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.train_sampler)
        validation_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.validation_sampler)
        api_key = self.wandb_config_dict['api_key']
        project_name = self.wandb_config_dict['project_name']
        run_id = str(uuid4())
        tags = [self.model.__class__.__name__]
        tags.extend(self.run_tags)
        run_name = f'{self.model_name}_{run_id}'
        trainings_start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        best_weights_save_path = f'models/{self.model_name}/{trainings_start_time}_{self.model_name}_{run_id}'
        model_hisotory = train(model=self.model, train_loader=train_loader, validation_loader=validation_loader,
                                n_classes=self.n_classes, epochs=self.n_epochs, lr=lr, batch_size=batch_size, prefered_device=self.prefered_device,
                                early_stopping=True, patience=10, min_delta_percentage=0.05,
                                wandb_api_key= api_key, wandb_project_name= project_name, wandb_run_id= run_id, wandb_run_name=run_name, wandb_tags= tags,
                                best_weights_save_path= best_weights_save_path, dataset_path= self.dataset_path, n_epochs_validation=self.n_epochs_validation)
        return min(model_hisotory['validation']['loss'])

def optimize_model(model_key : str, n_epochs: int,
                   dataset_path : str, wandb_config: dict, alternate_image_transforms:bool,weight_train_sampler:bool, weight_validation_sampler:bool,
                     n_epochs_validation : int, n_trials:int,
                       study_save_path:str, prefered_device:str = 'cuda:0'):
    #load data
    #set random seed
    random.seed(42)
    #fix torch random seed
    torch.manual_seed(42)
    run_tags = []
    image_transform = models_torch.model_dict[model_key]['transforms']
    if alternate_image_transforms:
        image_transform = transforms.Compose([image_transform, transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),
                                         transforms.RandomRotation((-180,180))])
        run_tags.append('alternate_image_transforms')
    dataset = ImageFolder(root=dataset_path, transform=image_transform)
    targets = dataset.targets
    #split data into train, test, val
    #70-20-10
    train_val_idx, test_idx= train_test_split(np.arange(len(targets)),test_size=0.2,shuffle=True,stratify=targets, random_state=42)
    train_val_idx_list = train_val_idx.tolist()
    train_val_stratifier = np.take(targets,train_val_idx_list)
    #targets[train_val_idx_list]
    train_idx, validation_idx = train_test_split(train_val_idx,test_size=0.125,shuffle=True,stratify=train_val_stratifier, random_state=42)
    #load data into dataloader
    if weight_train_sampler:
        #calculate weights for weighted random sampler
        train_target_sequence = np.take(targets,train_idx)
        train_class_sample_count = np.unique(train_target_sequence, return_counts=True)[1]
        train_weights =  1. / train_class_sample_count
        train_sampler_weights = train_weights[train_target_sequence]
        train_sampler = torch.utils.data.WeightedRandomSampler(weights=train_sampler_weights, num_samples=len(train_idx), replacement=True)
        run_tags.append('weighted_train_sampler')
    else:
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    if weight_validation_sampler:
        validation_class_sample_count = np.unique(targets, return_counts=True)[1]
        sampler_validation_weights =  1. / validation_class_sample_count
        validation_sampler = torch.utils.data.WeightedRandomSampler(weights=sampler_validation_weights, num_samples=len(validation_idx), replacement=True)
        run_tags.append('weighted_validation_sampler')
    else:
        validation_sampler = torch.utils.data.SubsetRandomSampler(validation_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)
    n_classes = len(dataset.classes)

    start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    study_name = f'{start_time}_study_{str(uuid4())}'
    study = optuna.create_study(direction='minimize', study_name=study_name)
    model_to_optimize = models_torch.get_model(model_key, n_classes)
    model_name = model_key

    objective = Objective(model_to_optimize=model_to_optimize, model_name=model_name, n_classes=n_classes,wandb_config_dict=wandb_config, 
                             dataset=dataset, n_epochs=n_epochs, train_sampler=train_sampler, validation_sampler=validation_sampler,
                               run_tags=run_tags, dataset_path=dataset_path, n_epochs_validation=n_epochs_validation)
    study.optimize(objective,
                    n_trials=n_trials)
    joblib.dump(study, f"{study_save_path}/{study_name}.pkl")


def test_optimize_model():
    #load data
    dataset_path = 'datasets/2023-12-28_18-12-43'
    #read the config file
    with open('wandb_config.json') as f:
        wandb_config = json.load(f)
    optimize_model(model_key='resnet18', dataset_path=dataset_path, wandb_config=wandb_config,
                    alternate_image_transforms=True, weight_train_sampler=False, weight_validation_sampler=False,
                      n_epochs_validation=1 ,n_trials=5, study_save_path='studies')