#!/bin/bash
# This is a Slurm batch script for executing optimization

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ird_deep_learning
cd ~/IRD-DeepLearning
echo 'Start optimization'
nohup python main.py --model_keys resnet18 resnet50 --dataset_path datasets/2023-12-28_18-12-43 --n_epochs 20 --wandb_config_path wandb_config.json --n_trials 50 --n_epochs_validation 1 --prefered_device cuda:0 --batch_size_options 2 4 8 16 32 64 128 --lr_min 1e-6 --lr_max 1e-2  --pretrained >vit_h_14.out &
nohup python main.py --model_keys resnet101 --dataset_path datasets/2023-12-28_18-12-43 --n_epochs 20 --wandb_config_path wandb_config.json --n_trials 50 --n_epochs_validation 1 --prefered_device cuda:1 --batch_size_options 2 4 8 16 32 64 128 --lr_min 1e-6 --lr_max 1e-2 --pretrained >vit_l_32.out &
nohup python main.py --model_keys resnet152 --dataset_path datasets/2023-12-28_18-12-43 --n_epochs 20 --wandb_config_path wandb_config.json --n_trials 50 --n_epochs_validation 1 --prefered_device cuda:2 --batch_size_options 2 4 8 16 32 64 128 --lr_min 1e-6 --lr_max 1e-2 --pretrained >vit_b_32.out &
nohup python main.py --model_keys resnext101_64x4d --dataset_path datasets/2023-12-28_18-12-43 --n_epochs 25 --wandb_config_path wandb_config.json --n_trials 25 --n_epochs_validation 1 --prefered_device cuda:3 --batch_size_options 2 4 8 16 32 64 --lr_min 1e-6 --lr_max 1e-2 --pretrained >retfound.out &
wait
echo 'Optimization done'