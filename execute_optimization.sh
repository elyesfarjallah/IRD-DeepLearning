#!/bin/bash
# This is a Slurm batch script for executing optimization

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ird_deep_learning
cd ~/IRD-DeepLearning
nohup python main.py --model_keys vit_h_14 --dataset_path datasets/2023-12-28_18-12-43 --n_epochs 30 --wandb_config_path wandb_config.json --n_trials 20 --n_epochs_validation 1 --prefered_device cuda:0 --batch_size_options 2 4 8 16 32 & > vit_h_14 &
nohup python main.py --model_keys vit_l_32 --dataset_path datasets/2023-12-28_18-12-43 --n_epochs 30 --wandb_config_path wandb_config.json --n_trials 20 --n_epochs_validation 1 --prefered_device cuda:1 --batch_size_options 2 4 8 16 32 & > vit_l_32 &
nohup python main.py --model_keys vit_b_32 --dataset_path datasets/2023-12-28_18-12-43 --n_epochs 30 --wandb_config_path wandb_config.json --n_trials 20 --n_epochs_validation 1 --prefered_device cuda:2 --batch_size_options 2 4 8 16 32 & > vit_b_32 &
nohup python main.py --model_keys resnext50_32x4d resnext101_32x8d --dataset_path datasets/2023-12-28_18-12-43 --n_epoch --wandb_config_path wandb_config.json --n_trials 20 --n_epochs_validation 1 --prefered_device cuda:3 --batch_size_options 2 4 8 16 32 64 128 & > resnext &