from convert_user_inputs import convert_user_input, create_observer_structure, create_datasets
from ai_backend.loggers.model_logger import is_min
from input_mapping.metric_mapping import get_multilabel_metrics_by_names, get_classwise_metrics_by_names
from data_pipeline.data_extraction import extract_all_databases
from data_pipeline.dataset_creation import multi_label_k_fold_split, stratified_multilabel_dataset_split, split_database_by_source_name, convert_to_multilabel_format, balance_multilabel_df, drop_labels_with_too_few_entries, one_hot_encode_column
import pandas as pd
import numpy as np
from uuid import uuid4
import torch.nn as nn
from torch.optim import Adam, AdamW
import torch
import json
import wandb
import argparse
from datetime import datetime
import os


#parse the arguments
date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
parser = argparse.ArgumentParser(description='Generate k fold dataset')
parser.add_argument('--n_splits', type=int, default=5, help='The number of splits')
parser.add_argument('--multilabel_order', type=int, default=1, help='The multilabel order to consider for the split')
parser.add_argument('--dataset_save_path', type=str, default=f'datasets_k_fold/{date}/', help='dataset save path')
parser.add_argument('--similarity_matrix_path', type=str, default='similarity_matrices/2023-12-27_15-25-54_disease_key_matrix.xlsx', help='The path to the similarity matrix')
parser.add_argument('--diseases_of_interest', type=str, nargs='+', default=['Retinitis Pigmentosa'], help='The diseases of interest')
parser.add_argument('--source_names', type=str, nargs='+', default=['SES'], help='The source names to split the data by')
parser.add_argument('--data_path_column_name', type=str, default='path_to_img', help='The column name of the column which contains the path to the images')
parser.add_argument('--label_column_name', type=str, default='disease_key', help='The label columns to use for the model')
parser.add_argument('--label_seperator', type=str, default=', ', help='The seperator for the multilabel columns')
parser.add_argument('--other_columns_to_keep', type=str, nargs='+', default=['dataset_name'], help='The other columns to keep in the dataframe')
parser.add_argument('--other_columns_to_encode', type=str, nargs='+', default=['dataset_name'], help='The other columns to one hot encode in the dataframe')

#parse the arguments
args = parser.parse_args()

#create save path
os.makedirs(args.dataset_save_path, exist_ok=True)
#1. extract the databases
extracted_db_df = extract_all_databases()
#2. reduce the label columns to the ones that are relevant for the model
sim_table = pd.read_excel(args.similarity_matrix_path)
#set index to column named Spalte1
sim_table.set_index('Spalte1', inplace=True)
diseases_of_interest = args.diseases_of_interest
#only keep the diseases of interest columns and the index
sim_table_interest = sim_table[diseases_of_interest]
for interesting_disease in diseases_of_interest:
    sim_table_interest = sim_table_interest[sim_table_interest[interesting_disease].notna()]
#create a dataframe which only contains the diseases which are in the index column of the sim_table_interest
disease_keys = list(sim_table_interest.index)
#filter all_images_df by the disease keys
extracted_db_df_relevant_labels = extracted_db_df[extracted_db_df[args.label_column_name].isin(disease_keys)]
#convert to multilabel
multilabel_df = convert_to_multilabel_format(df = extracted_db_df_relevant_labels, column_to_unify=args.label_column_name,
                                             label_separator=args.label_seperator, data_path_column_name=args.data_path_column_name, other_columns_to_keep=args.other_columns_to_keep)
#balance the multilabel df
mulilabel_df_balnced = balance_multilabel_df(df = multilabel_df, label_names= disease_keys, median_deviation=0.3)
#drop labels with too few entries
mulilabel_df_balnced, disease_keys = drop_labels_with_too_few_entries(df = mulilabel_df_balnced, label_names = disease_keys, min_entries=3)
#3. separate the databases by source name
source_names = args.source_names
data_dfs = []

other_encoded_columns = []
for column in args.other_columns_to_encode:
    unique_values = mulilabel_df_balnced[column].unique()
    other_encoded_columns.extend(unique_values)
    mulilabel_df_balnced = one_hot_encode_column(df = mulilabel_df_balnced, column_name = column, unique_column_values = unique_values)

columns_to_stratify_split = disease_keys + other_encoded_columns
#perform a stratified split based on the columns to stratify
train_validation_df,_,_,_,test_df,_ = stratified_multilabel_dataset_split(df = mulilabel_df_balnced, train_frac = 0.8, validation_frac = 0.0,test_frac=0.2, label_columns = columns_to_stratify_split)
#c

test_df.to_csv(f'{args.dataset_save_path}/All_test.csv', index=False)
train_validation_df.to_csv(f'{args.dataset_save_path}/All_train_validation.csv', index=False)

#7. create the k fold splits on the train_validation dataframe 80/20
k_fold_splits = multi_label_k_fold_split(df = train_validation_df, label_columns = columns_to_stratify_split,
                                          n_splits = 5, multilabel_order = args.multilabel_order)

#8.save the k fold splits while iterating over the splits
for i, (train_idxs, validation_idxs) in enumerate(k_fold_splits):
    train_df = train_validation_df.iloc[train_idxs]
    validation_df = train_validation_df.iloc[validation_idxs]
    #create the folder for the fold
    train_data_savepath = f'{args.dataset_save_path}fold_{i}/train.csv'
    validation_data_savepath = f'{args.dataset_save_path}fold_{i}/validation.csv'
    os.makedirs(f'{args.dataset_save_path}fold_{i}/', exist_ok=True)
    train_df.to_csv(train_data_savepath, index=False)
    validation_df.to_csv(validation_data_savepath, index=False)
#save a json file with thhe dataset configuration
dataset_config = {'label_names' : disease_keys, 'path_to_img_column' : args.data_path_column_name, 'other_encoded_columns': other_encoded_columns}
with open(f'{args.dataset_save_path}/dataset_config.json', 'w') as json_file:
    json.dump(dataset_config, json_file)