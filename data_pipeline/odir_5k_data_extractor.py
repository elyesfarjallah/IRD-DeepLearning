from data_pipeline.data_extraction import DataExtractor
from data_pipeline.data_splitting_utils import split_by_instance_count
from data_pipeline.data_package import DataPackage
import pandas as pd
import numpy as np
import os

class ODIR5KDataExtractor(DataExtractor):
    dataset_name = "ODIR-5K"
    def __init__(self, database_path: str, database_test_images_path: str, database_train_images_path: str):
        super().__init__(database_path=database_path)
        self.source_df = pd.read_csv(database_path)
        #Important: drop the duplicates from the dataframe else there will be duplicates in the dataset
        self.source_df = self.source_df.drop_duplicates(keep='first', subset=['Right-Fundus', 'Left-Fundus'])
        self.database_test_images_path = database_test_images_path
        self.database_train_images_path = database_train_images_path

    def extract(self):
        instance_id_column_name = 'ID'
        left_keywords = ['Left-Fundus', 'Left-Diagnostic Keywords', instance_id_column_name]
        right_keywords = ['Right-Fundus', 'Right-Diagnostic Keywords', instance_id_column_name]
        #copy
        copied_source = self.source_df.copy()
        #add the current index as a column
        
        copied_source[instance_id_column_name] = copied_source.index
        #
        right_fundus_df = copied_source[right_keywords]
        left_fundus_df = copied_source[left_keywords]
        #rename the columns
        data_column_name = 'Fundus'
        label_column_name = 'Diagnostic Keywords'
        new_column_names = [data_column_name, label_column_name, instance_id_column_name]
        right_fundus_df.columns = new_column_names
        left_fundus_df.columns = new_column_names
        #concatenate the dataframes
        rejoined_df = pd.concat([right_fundus_df, left_fundus_df])
        train_directory = os.listdir(self.database_train_images_path)
        test_directory = os.listdir(self.database_test_images_path)

        #check which datapoints are in the test set
        test_df = rejoined_df[rejoined_df[data_column_name].isin(test_directory)]
        #check which datapoints are in the train set
        train_df = rejoined_df[rejoined_df[data_column_name].isin(train_directory)]
        #add a path column to the dataframes
        path_column_name = 'Path'
        test_df[path_column_name] = test_df[data_column_name].apply(lambda x: f'{self.database_test_images_path}/{x}')
        train_df[path_column_name] = train_df[data_column_name].apply(lambda x: f'{self.database_train_images_path}/{x}')
        #rejoin the dataframes
        full_path_df = pd.concat([test_df, train_df])
        #drop the fundus column
        full_path_df = full_path_df.drop(columns=[data_column_name])
        #resort the columns instance_id, path, Diagnostic Keywords
        full_path_df = full_path_df[[instance_id_column_name, path_column_name, label_column_name]]
        #convert to np array
        full_path_np = full_path_df.values
        #get the labels
        labels = full_path_np[:,-1]
        #split the labels by the ", " separator
        #todo split on the special character only and then remove trailing and leading whitespaces
        labels = [label.split('，') for label in labels]
        #make the label list all the same length
        max_label_length = max([len(label) for label in labels])
        labels = [label + [None]*(max_label_length-len(label)) for label in labels]
        #delete the old label column
        no_label_np = np.delete(full_path_np, -1, -1)
        #add the labels to the np array
        result_np = np.hstack((no_label_np, np.array(labels)))
        self.extracted_data = result_np
        self.remove_not_existing_file_paths()
        return self.extracted_data
    
    def get_labels(self, data_truth_series : np.ndarray = None):
        return self.extracted_data[:,2:] if data_truth_series is None else self.extracted_data[data_truth_series][:,2:]
    
    def get_file_paths(self, data_truth_series : np.ndarray = None):
        return self.extracted_data[:,1] if data_truth_series is None else self.extracted_data[data_truth_series][:,1]
    
    def get_instance_ids(self, data_truth_series : np.ndarray = None):
        return self.extracted_data[:,0] if data_truth_series is None else self.extracted_data[data_truth_series][:,0]
    
    def split_extracted_data(self, split_portions, stratify):
        if stratify:
            #todo add warning
            pass
        #get the ids assigned to splits
        instance_split = split_by_instance_count(instance_list=self.get_instance_ids(), split_ratios=split_portions)
        data_splits = []
        for split in instance_split:
            extraction_series = np.isin(self.get_instance_ids(), split)
            split_file_paths = self.get_file_paths(data_truth_series=extraction_series)
            split_labels = self.get_labels(data_truth_series=extraction_series)
            split_instance_ids = self.get_instance_ids(data_truth_series=extraction_series)
            data_splits.append(DataPackage(data=split_file_paths, labels=split_labels, instance_ids= split_instance_ids,
                                           data_source_name=self.dataset_name))
        self.current_split = data_splits
        return data_splits


#test
def test_extract():
    base_path = 'databases/ODIR-5K/full_df.csv'
    train_images_path = 'databases/ODIR-5K/Training Images'
    test_images_path = 'databases/ODIR-5K/Testing Images'
    odir5k_data_extractor = ODIR5KDataExtractor(database_path=base_path, database_test_images_path=test_images_path, database_train_images_path=train_images_path)
    data = odir5k_data_extractor.extract()
    #save the data as test.csv
    pd.DataFrame(data).to_csv('test_save_odir5k_converted.csv',header=False, index=False)
    #split the data
    split_portions = [0.7, 0.1, 0.2]
    split_data = odir5k_data_extractor.split_extracted_data(split_portions, stratify=False)
    print(split_data)
