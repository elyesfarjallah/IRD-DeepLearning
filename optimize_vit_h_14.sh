#!/bin/bash
# This is a Slurm batch script for executing optimization

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ird_deep_learning
nohup python main.py --model_keys vit_h_14 --dataset_path datasets/2023-12-28_18-12-43 --n_epochs 30 --wandb_config_path wandb_config.json --n_trials 20 --study_save_path studies --n_epochs_validation 1 --prefered_device cpu & > vit_h_14.out &