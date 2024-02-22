#!/bin/bash
# This is a Slurm batch script for executing k fold cross validation

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ird_deep_learning
cd ~/IRD-DeepLearning
echo 'Start optimization'
nohup python k_fold_cross_validation.py --model_key resnet18 --n_train_epochs 60 --batch_size 128 --lr 0.00009354747253832916 --dataset_path datasets_k_fold/2024-02-22_15-58-58 --device cuda:0 --transform_type standard --augmentation &
nohup python k_fold_cross_validation.py --model_key resnet18 --n_train_epochs 60 --batch_size 16 --lr 0.00004614948033730265 --dataset_path datasets_k_fold/2024-02-22_15-58-58 --device cuda:1 --transform_type ben --augmentation &
wait
echo 'K-fold cross validation done'