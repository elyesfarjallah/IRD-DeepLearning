import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from data_pipeline.odir_5k_data_extractor import ODIR5KDataExtractor
from data_pipeline.rfmid_data_extractor import RFMiDDataExtractor
from data_pipeline.rfmid2_data_extractor import RFMiD2DataExtractor
from data_pipeline.ukb_data_extractor import UkbDataExtractor
from data_pipeline.rips_data_extractor import RIPSDataExtractor
from data_pipeline.ses_data_extractor import SESDataExtractor
from data_pipeline.one_thousand_images_data_extractor import OneThousandImagesDataExtractor

from data_pipeline.data_processing_utils import standardize_labels
from data_pipeline.data_processing_utils import create_one_hot_encoder
import data_pipeline.data_processing_utils as dpu
import data_pipeline.data_splitting_utils as dsu

#1 extract the data frothe data sources
def create_data_packages(labels_to_encode : np.array = None):
    odir5k_data_extractor = ODIR5KDataExtractor(database_path='databases/ODIR-5K/full_df.csv', database_test_images_path='databases/ODIR-5K/Testing Images',
                                                database_train_images_path='databases/ODIR-5K/Training Images')

    rfmid_train_data_extractor = RFMiDDataExtractor(database_path='databases/RFMiD/Training_Set/RFMiD_Training_Labels.csv',
                                            data_path='databases/RFMiD/Training_Set/Training', file_format='png')

    rfmid_validation_datae_xtractor = RFMiDDataExtractor(database_path='databases/RFMiD/Evaluation_Set/RFMiD_Validation_Labels.csv',
                                            data_path='databases/RFMiD/Evaluation_Set/Validation', file_format='png')

    rfmid_test_data_extractor = RFMiDDataExtractor(database_path='databases/RFMiD/Test_Set/RFMiD_Testing_Labels.csv',
                                                data_path='databases/RFMiD/Test_Set/Test', file_format='png')

    rfmid2_train_data_extractor = RFMiD2DataExtractor(database_path='databases/RFMiD2_0/Training_set/RFMiD_2_Training_labels.csv',
                                                    data_path='databases/RFMiD2_0/Training_set')
    rfmid2_validation_data_extractor = RFMiD2DataExtractor(database_path='databases/RFMiD2_0/Validation_set/RFMiD_2_Validation_labels.csv',
                                                            data_path='databases/RFMiD2_0/Validation_set')

    rfmid2_test_data_extractor = RFMiD2DataExtractor(database_path='databases/RFMiD2_0/Test_set/RFMiD_2_Testing_labels.csv',
                                                        data_path='databases/RFMiD2_0/Test_set')


    one_thousand_images_data_extractor = OneThousandImagesDataExtractor(database_path='databases/1000images/')

    rips_data_extractor = RIPSDataExtractor(database_path='databases/RIPS/Original')

    ses_data_extractor = SESDataExtractor(database_path='databases/SES/')

    #todo add UKB data extractor when the data is available
    #ukb_data_extractor = UkbDataExtractor(database_path=#ukb_path)

    #create the data extraction list
    data_extractors = [odir5k_data_extractor, rfmid_train_data_extractor, rfmid_validation_datae_xtractor, rfmid_test_data_extractor,
                    rfmid2_train_data_extractor, rfmid2_validation_data_extractor, rfmid2_test_data_extractor,
                    one_thousand_images_data_extractor, rips_data_extractor, ses_data_extractor]

    #extract the data
    for data_extractor in data_extractors:
        data_extractor.extract()

    #standardize the data
    #get the labels of the data
    datasets_labels = []
    for data_extractor in data_extractors:
        datasets_labels.append(data_extractor.get_labels())
    #flatten
    labels = []
    for dataset_labels in datasets_labels:
        labels.extend(dataset_labels)
    #concatenate the labels
    labels = np.concatenate(labels)
    #drop the None values
    labels = labels[labels != None]
    #create the standertizer
    not_summarize_set = set(RFMiD2DataExtractor.abbreviation_map.values())
    label_standertizer = standardize_labels(labels = labels, not_summarize_set=not_summarize_set)
    #value count the labels
    label_counts = pd.Series(labels).value_counts()
    #get the median label count
    median_label_count = label_counts.median()
    label_instance_limit = int((max(label_counts) - median_label_count) // 4)
    #balance the labels
    #find the over represented labels
    for labels,extractor in zip(datasets_labels, data_extractors):
        #find the over represented labels
        #replace none with empty string
        over_represented_labels_idxs, _, _ = dpu.find_over_represented_samples(file_paths=extractor.get_file_paths(), labels=labels,
                                                                                max_samples_per_class=label_instance_limit)
        #remove the over represented labels
        #conver the indexes to a boolean array
        over_represented_labels_series = np.isin(np.arange(len(labels)), over_represented_labels_idxs)
        extractor.extracted_data = extractor.extracted_data[~over_represented_labels_series]
    #split each data extractor into train, validation and test data
    train_portion = 0.7
    validation_portion = 0.1
    test_portion = 0.2
    split_portions = [train_portion, validation_portion, test_portion]
    stratify = True
    data_splits = []
    for data_extractor in data_extractors:
        extractor_splits = data_extractor.split_extracted_data(split_portions=split_portions, stratify=stratify)
        data_splits.append(extractor_splits)

    #standardize the labels and encode them
    #initialize the one hot encoder
    standard_labels = np.array(list(set(label_standertizer.values())))
    labels_to_encode = standard_labels if labels_to_encode is None else labels_to_encode
    one_hot_encoder = create_one_hot_encoder(unique_labels=labels_to_encode)
    for data_split in data_splits:
        for split in data_split:
            standarized_split_labels = np.vectorize(label_standertizer.get)(split.get_labels())
            encoded_labels = dpu.encode_multistring_labels(labels=standarized_split_labels, encoder=one_hot_encoder)
            split.labels = encoded_labels

    return data_splits, label_standertizer, one_hot_encoder

# ############################################################################################################
# #print the counts of the data splits
# for data_split in data_splits:
#     split_relation = ['train', 'validation', 'test']
#     buffer_string = ''
#     #add the counts of the data split and print them at the end of the outer loop
#     for split, relation in zip(data_split, split_relation):
#         buffer_string += f'{relation}: {len(split)}\n'
#     print(buffer_string)

# #create a bar plot of the label counts of the data splits and plot each data split consisting of train, validation and test as one stacked bar
# #initialize the label counts
# label_counts = []
# #iterate over the data splits
# for data_split in data_splits:
#     #iterate over the data split
#     split_label_counts = []
#     for split in data_split:
#         #create pandas dataframe from the split
#         #get the labels
#         labels = split.get_labels()
#         #flatten the labels
#         labels = labels.flatten()
#         #drop the None values
#         labels = labels[labels != None]
#         #get the label counts
#         values, counts = np.unique(labels, return_counts=True)
#         #create the label count dictionary
#         label_count_dict = dict(zip(values, counts))
#         split_label_counts.append(label_count_dict)

#     label_counts.append(split_label_counts)

# #convert data splits

# import matplotlib.pyplot as plt

# for i, data_split in enumerate(data_splits):
#     #get the labels
#     labels = data_splits[i][0].get_labels().flatten()
#     #drop the None values
#     labels = labels[labels != None]
#     #get the unique labels
#     labels = np.unique(labels)
#     #get the counts
#     train_counts = [label_counts[i][0][label] if label in label_counts[i][0] else 0 for label in labels]
#     val_counts = [label_counts[i][1][label] if label in label_counts[i][1] else 0 for label in labels]
#     test_counts = [label_counts[i][2][label] if label in label_counts[i][2] else 0 for label in labels]
#     # Plotting
#     plt.figure(figsize=(20, 6))
#     plt.bar(labels, train_counts, label='Train', color='blue')
#     plt.bar(labels, val_counts, bottom=train_counts, label='Val', color='orange')
#     plt.bar(labels, test_counts, bottom=np.array(train_counts) + np.array(val_counts), label='Test', color='green')
#     plt.xlabel('Labels')
#     plt.ylabel('Counts')
#     plt.title('Stacked Bar Chart for Train, Val, and Test Splits')
#     plt.xticks(rotation=90)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
