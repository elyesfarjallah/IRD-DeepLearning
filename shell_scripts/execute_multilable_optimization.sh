#!/bin/bash
# This is a Slurm batch script for executing optimization

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ird_deep_learning
cd ~/IRD-DeepLearning
echo 'Start optimization'
nohup python train.py --model_key resnet18 --lr_min 1e-5 --lr_max 1e-1 --batch_size_options 16 32 64 128 --device cuda:0 &
nohup python train.py --model_key vit_b_32 --lr_min 1e-7 --lr_max 1e-4 --batch_size_options 8 16 32 --device cuda:1 &
wait
echo 'Optimization done'