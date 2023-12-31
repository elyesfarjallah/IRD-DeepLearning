#!/bin/bash
# This is a Slurm batch script for executing optimization

conda activate ird_deep_learning
python main.py --model_keys vit_h_14 vit_b_16 resnext50_32x4d --dataset_path datasets/2023-12-28_18-12-43 --n_epochs 30 --wandb_config_path wandb_config.json --n_trials 20 --study_save_path studies --n_epochs_validation 1 --prefered_device cuda:0 &
python main.py --model_keys vit_l_32 swin_v2_t resnet18 resnet34 resnet50 swin_v2_b --dataset_path datasets/2023-12-28_18-12-43 --n_epochs 30 --wandb_config_path wandb_config.json --n_trials 20 --study_save_path studies --n_epochs_validation 1 --prefered_device cuda:1 &
python main.py --model_keys vit_b_32 swin_v2_s resnet101 resnet152 --dataset_path datasets/2023-12-28_18-12-43 --n_epochs 30 --wandb_config_path wandb_config.json --n_trials 20 --study_save_path studies --n_epochs_validation 1 --prefered_device cuda:2 &
python main.py --model_keys vit_l_16 resnext50_32x4d resnext101_32x8d --dataset_path datasets/2023-12-28_18-12-43 --n_epochs 30 --wandb_config_path wandb_config.json --n_trials 20 --study_save_path studies --n_epochs_validation 1 --prefered_device cuda:3 &