import hyperparameter_optimization
import models_torch
import argparse
import logging
import json
import numpy as np
from trainer import train
from uuid import uuid4
from uuid import uuid4
import random
import os
import datetime
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from torchvision import transforms

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for a given model')
    #parse list of model keys
    parser.add_argument('--model_key', type=str, help='keys of the models to optimize')
    parser.add_argument('--dataset_path', type=str, help='path to the dataset')
    parser.add_argument('--n_epochs', type=int, help='number of epochs to train the model')
    parser.add_argument('--wandb_config_path', type=str, help='path to the wandb config file')
    parser.add_argument('weight_train_sampler', action='store_true', help='use weighted random sampler for training')
    parser.add_argument('weight_validation_sampler', action='store_true', help='use weighted random sampler for validation')
    parser.add_argument('--alternate_image_transforms', action='store_true', help='use alternate image transforms')
    parser.add_argument('--pretrained', action='store_true', help='use pretrained weights')
    parser.add_argument('--lr', type=float, help='number of trials for the hyperparameter optimization')
    parser.add_argument('--n_epochs_validation', type=int, help='number of epochs after which the validation is executed')
    #optional arguments
    parser.add_argument('--prefered_device', type=str, default='cuda:0', help='prefered device for training')
    #batch_size_options = [4, 8, 16, 32]
    parser.add_argument('--batch_size', type=int, help='batch size options for the hyperparameter optimization')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    #read the config file
    with open(args.wandb_config_path) as f:
        wandb_config = json.load(f)
    #set random seed
    random.seed(42)
    #fix torch random seed
    torch.manual_seed(42)
    run_tags = []
    image_transform = models_torch.model_dict[args.model_key]['transforms']
    if args.alternate_image_transforms:
        image_transform = transforms.Compose([image_transform, transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),
                                         transforms.RandomRotation((-180,180))])
        run_tags.append('alternate_image_transforms')
    dataset = ImageFolder(root=args.dataset_path, transform=image_transform)
    targets = dataset.targets
    #split data into train, test, val
    #70-20-10
    train_val_idx, test_idx= train_test_split(np.arange(len(targets)),test_size=0.2,shuffle=True,stratify=targets, random_state=42)
    train_val_idx_list = train_val_idx.tolist()
    train_val_stratifier = np.take(targets,train_val_idx_list)
    #targets[train_val_idx_list]
    train_idx, validation_idx = train_test_split(train_val_idx,test_size=0.125,shuffle=True,stratify=train_val_stratifier, random_state=42)
    #load data into dataloader
    if args.weight_train_sampler:
        #calculate weights for weighted random sampler
        train_target_sequence = np.take(targets,train_idx)
        train_class_sample_count = np.unique(train_target_sequence, return_counts=True)[1]
        train_weights =  1. / train_class_sample_count
        train_sampler_weights = train_weights[train_target_sequence]
        train_sampler = torch.utils.data.WeightedRandomSampler(weights=train_sampler_weights, num_samples=len(train_idx), replacement=True)
        run_tags.append('weighted_train_sampler')
    else:
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    if args.weight_validation_sampler:
        validation_class_sample_count = np.unique(targets, return_counts=True)[1]
        sampler_validation_weights =  1. / validation_class_sample_count
        validation_sampler = torch.utils.data.WeightedRandomSampler(weights=sampler_validation_weights, num_samples=len(validation_idx), replacement=True)
        run_tags.append('weighted_validation_sampler')
    else:
        validation_sampler = torch.utils.data.SubsetRandomSampler(validation_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)
    n_classes = len(dataset.classes)
    model = models_torch.get_model(args.model_key, n_classes, pretrained=args.pretrained)
    lr = args.lr
    batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=validation_sampler)
    with open(args.wandb_config_path) as f:
        wandb_config_dict = json.load(f)
    api_key = wandb_config_dict['api_key']
    project_name = wandb_config_dict['project_name']
    run_id = str(uuid4())
    tags = [model.__class__.__name__]
    tags.extend(run_tags)
    if args.pretrained:
        run_tags.append('pretrained')
    model_name = args.model_key
    run_name = f'{model_name}_{run_id}'
    trainings_start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    best_weights_save_path = f'models/{args.model_name}/{trainings_start_time}_{args.model_name}_{run_id}'
    #if path does not exist create it
    if not os.path.exists(best_weights_save_path):
        os.makedirs(best_weights_save_path)
    train(model=model, train_loader=train_loader, validation_loader=validation_loader,
                                n_classes=args.n_classes, epochs=args.n_epochs, lr=lr, batch_size=args.batch_size, prefered_device=args.prefered_device,
                                early_stopping=True, patience=10, min_delta_percentage=0.05,
                                wandb_api_key= api_key, wandb_project_name= project_name, wandb_run_id= run_id, wandb_run_name=run_name, wandb_tags= run_tags,
                                best_weights_save_path= best_weights_save_path, dataset_path= args.dataset_path,
                                n_epochs_validation=args.n_epochs_validation, model_name= model_name)    
    

#execute the hyperparameter optimization
#python train_model.py --model_key resnet18 --dataset_path datasets/2023-12-28_18-12-43 --n_epochs 5 --wandb_config_path wandb_config.json --alternate_image_transforms --n_epochs_validation 1 --prefered_device cpu --batch_size 4 --lr 0.001
#python main.py --model_keys shufflenet_v2_x1_0 shufflenet_v2_x1_5 shufflenet_v2_x2_0 mnasnet0_5 mnasnet0_75 mnasnet1_0 mnasnet1_3 resnext50_32x4d resnext101_32x8d resnext101_64x4d wide_resnet50_2 wide_resnet101_2 swin_v2_t swin_v2_s swin_v2_b vit_b_16 vit_b_32 vit_l_16 vit_l_32 vit_h_14 --dataset_path datasets/2023-12-28_18-12-43 --n_epochs 100 --wandb_config_path wandb_config.json --alternate_image_transforms --n_trials 100 --study_save_path studies --n_epochs_validation 1 --prefered_device cuda:0
#python main.py --model_keys shufflenet_v2_x1_0 shufflenet_v2_x1_5 shufflenet_v2_x2_0 mnasnet0_5 mnasnet0_75 mnasnet1_0 mnasnet1_3 resnext50_32x4d resnext101_32x8d resnext101_64x4d wide_resnet50_2 wide_resnet101_2 swin_v2_t swin_v2_s swin_v2_b vit_b_16 vit_b_32 vit_l_16 vit_l_32 vit_h_14 --dataset_path datasets/2023-12-28_18-12-43 --n_epochs 100 --wandb_config_path wandb_config.json --n_trials 100 --study_save_path studies --n_epochs_validation 1 --prefered_device cuda:1
        

#python main.py --model_keys resnext101_32x8d resnext101_64x4d --dataset_path datasets/2023-12-28_18-12-43 --n_epochs 5 --wandb_config_path wandb_config.json --alternate_image_transforms --n_trials 5 --study_save_path studies --n_epochs_validation 1 --prefered_device cuda:0

