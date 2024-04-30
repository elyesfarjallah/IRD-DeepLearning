from data_pipeline.dataset_creation import stratified_multilabel_dataset_split
import pandas as pd
import numpy as np
import argparse
import json
import os
from datetime import datetime

#short script to generate a normal dataset from a k fold split
parser = argparse.ArgumentParser(description='Generate dataset from k fold split')
parser.add_argument('--k_fold_dataset_path', type=str, help='k fold dataset path', default='datasets_k_fold/2024-02-22_19-42-13')

date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
#parse arguments
args = parser.parse_args()
k_fold_dataset_path = args.k_fold_dataset_path
#load the dataset configuration
with open(f'{k_fold_dataset_path}/dataset_config.json', 'r') as f:
    config = json.load(f)

#load the all test and train_val datasets
train_val_data_path = f'{k_fold_dataset_path}/All_train_validation.csv'
train_validation_df = pd.read_csv(train_val_data_path)

#split the dataset into train and validation
label_columns = config['label_names']
other_encoded_columns = config['other_encoded_columns']
stratification_columns = label_columns + other_encoded_columns
#suffle the dataset
train_validation_df = train_validation_df.sample(frac=1, random_state=42)
#split the dataset
train_df,_, validation_df, _,_,_ = stratified_multilabel_dataset_split(train_validation_df, train_frac=0.7/0.8, validation_frac=(1-0.7/0.8), test_frac=0.0, label_columns=stratification_columns, random_state=42)
config.update({'generated_from': k_fold_dataset_path})

new_dataset_path = f'./datasets/{date}'
os.makedirs(new_dataset_path, exist_ok=True)

#save the datasets
train_df.to_csv(f'{new_dataset_path}/All_train.csv', index=False)
validation_df.to_csv(f'{new_dataset_path}/All_validation.csv', index=False)
with open(f'{new_dataset_path}/dataset_config.json', 'w') as f:
    json.dump(config, f)

#copy the test dataset
test_df = pd.read_csv(f'{k_fold_dataset_path}/All_test.csv')
test_df.to_csv(f'{new_dataset_path}/All_test.csv', index=False)
