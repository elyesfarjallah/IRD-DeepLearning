from datetime import datetime
import os
import argparse
import numpy as np
import pandas as pd
from data_pipeline.ukb_data_extractor import UkbDataExtractor
from data_pipeline.data_extraction import extract_all_databases
from data_pipeline.dataset_creation import multi_label_k_fold_split, stratified_multilabel_dataset_split, split_database_by_source_name, convert_to_multilabel_format, balance_multilabel_df, drop_labels_with_too_few_entries, one_hot_encode_column

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
parser.add_argument('--ukb_database_path', type=str, default='data/UKB', help='The path to the UKB database')
parser.add_argument('--ukb_label_path', type=str, default='data/UKB/UKB_Diagnosen.xlsx', help='The path to the UKB label file')
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
ukb_data_extractor = UkbDataExtractor(database_path = args.ukb_database_path, label_path = args.ukb_label_path)
#extract the data of the labels
labels_to_extract = ['Retinitis pigmentosa', 'Morbus Best', 'Morbus Stargardt']
ukb_data = ukb_data_extractor.extract_data_of_labels(labels_to_extract)
#replace the labels
ukb_data[:,1] = ukb_data[:,1].replace('Morbus Best', 'Best disease')
ukb_data[:,1] = ukb_data[:,1].replace('Morbus Stargardt', 'Stargardt disease')
ukb_data[:,1] = ukb_data[:,1].replace('Retinitis pigmentosa', 'Retinitis Pigmentosa')
#add a datasource column
ukb_data = np.insert(ukb_data, 2, 'UKB', axis=1)
ukb_df = pd.DataFrame(ukb_data, columns=['path_to_img', 'disease_key', 'dataset_name'])
#add the missing columns
for column in args.other_columns_to_keep:
    ukb_df[column] = None

for value in disease_keys:
    ukb_df[value] = ukb_df['disease_key'].str.contains(value).astype(int)

#concatenate the dataframes
multilabel_df = pd.concat([multilabel_df, ukb_df])
mulilabel_df_balnced = balance_multilabel_df(df = multilabel_df, label_names= disease_keys, median_deviation=0.3)
#drop labels with too few entries
mulilabel_df_balnced, disease_keys = drop_labels_with_too_few_entries(df = mulilabel_df_balnced, label_names = disease_keys, min_entries=3)
