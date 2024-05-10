from data_pipeline.ukb_data_extractor import UkbDataExtractor
from data_pipeline.data_splitting_utils import stratified_instance_split
from sklearn.preprocessing import OneHotEncoder
import json
import numpy as np

with open('datasets/2024-02-23-13-31-39/dataset_config.json') as f:
    dataset_config = json.load(f)

ukb_extractor = UkbDataExtractor(database_path='data/UKB', label_path='data/UKB/UKB_Diagnosen.xlsx')
#extract the data of the labels
labels_to_extract = ['Retinitis pigmentosa', 'Morbus Best', 'Morbus Stargardt']
ukb_data = ukb_extractor.extract_data_of_labels(labels_to_extract)
#replace the labels
ukb_data[:,1] = ukb_data[:,1].replace('Morbus Best', 'Best disease')
ukb_data[:,1] = ukb_data[:,1].replace('Morbus Stargardt', 'Stargardt disease')
ukb_data[:,1] = ukb_data[:,1].replace('Retinitis pigmentosa', 'Retinitis Pigmentosa')

#split the data by instance
split_ratios = [0.7, 0.1, 0.2]
split_instances = stratified_instance_split(ukb_data[:,0], split_ratios, stratify_column=ukb_data[:,1])

#one hot encode the labels
#import one hot encoder

#initialize the encoder
#fit the encoder to the labels
all_categories = [dataset_config['label_names']]
encoder = OneHotEncoder(categories = all_categories, sparse=False)
encoder.fit(ukb_data[:,1].reshape(-1,1))

#one hot encode the labels for each split
for split in split_instances:
    #encode the labels
    encoded_labels = encoder.transform(split[:,1].reshape(-1,1))
    #add the encoded labels to the split
    split = np.insert(split, 2, encoded_labels, axis=1)



#todo finish the script