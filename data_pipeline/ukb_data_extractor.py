from data_pipeline.data_extraction import DataExtractor
from data_pipeline import data_extraction_utils
from data_pipeline.data_splitting_utils import split_by_instance_count, stratified_instance_split
from data_pipeline.data_package import DataPackage
import numpy as np
import pandas as pd
import pydicom
#todo data splitting still needs to be adjusted
class UkbDataExtractor(DataExtractor):
    dataset_name = "UKB"
    def __init__(self, database_path: str, label_path: str,
                  label_column_name: str = 'Diagnose', key_column_name: str = 'Index', cfp_column_name: str = 'CFP'):
        super().__init__(database_path=database_path)
        self.label_source_df = pd.read_excel(label_path)
        self.labels = self.label_source_df[label_column_name].values
        keys_unformatted = self.label_source_df[key_column_name].values
        self.keys = np.vectorize(lambda x: str(x).zfill(4))(keys_unformatted)
        cfps = self.label_source_df[cfp_column_name].values
        self.contain_cfp = np.vectorize(lambda x: 'True' == x)(cfps)
        #an array of indexes to extract with all values True
        self.indexes_to_extract = np.full(len(self.keys), True)

    def extract(self):
        #filter the data to only contain the indexes to extract
        keys_to_extract = self.keys[self.indexes_to_extract]
        labels_to_extract = self.labels[self.indexes_to_extract]
        #get all the keys and labels which contain a cfp
        keys_with_cfp = keys_to_extract[self.contain_cfp]
        labels_with_cfp = labels_to_extract[self.contain_cfp]
        #match the keys with the labels
        matched_data = data_extraction_utils.match_keys_labels(keys=keys_with_cfp, labels=labels_with_cfp, data_storage_path=self.database_path)
        #check if the files have pixel data
        pixel_checker = lambda x: data_extraction_utils.dicom_detect_pixels(pydicom.dcmread(x))
        have_pixels = np.vectorize(pixel_checker)(matched_data[:,1])
        #filter the data to only contain pixel data
        pixel_data_paths_labels = matched_data[have_pixels]
        self.extracted_data = pixel_data_paths_labels
        self.remove_not_existing_file_paths()
        #strip the labels from trailing and leading whitespaces
        self.extracted_data[:,2:] = np.vectorize(lambda x: x.strip())(self.extracted_data[:,2:])
        return self.extracted_data
    
    def get_labels(self):
        return self.extracted_data[:,2:]
    
    def get_file_paths(self):
        return self.extracted_data[:,1]
    
    def get_instance_ids(self):
        return self.extracted_data[:,0]
    
    def extract_data_of_label(self, label):
        self.indexes_to_extract = self.labels == label
        extracted_data = self.extract()
        #reset the indexes to extract
        self.indexes_to_extract = np.full(len(self.keys), True)
        return extracted_data
    
    def extract_data_of_labels(self, labels):
        return np.vectorize(self.extract_data_of_label)(labels)
    
    def split_extracted_data(self, split_portions, stratify):
        instance_list = self.get_instance_ids()
        if stratify:
            instance_split = stratified_instance_split(instance_list=instance_list, split_ratios=split_portions, stratify_column=self.get_labels().flatten())
        else:
            instance_split = split_by_instance_count(instance_list=instance_list, split_ratios=split_portions)
        data_splits = []
        for split in instance_split:
            extraction_series = np.isin(self.get_instance_ids(), split)
            file_paths_split = self.get_file_paths()[extraction_series]
            labels_split = self.get_labels()[extraction_series]
            instance_ids_split = self.get_instance_ids()[extraction_series]
            data_splits.append(DataPackage(data=file_paths_split, labels=labels_split, instance_ids=instance_ids_split, data_source_name=self.dataset_name))
        self.current_split = data_splits
        return data_splits

def test_data_split():
    ukb_database_path = 'databases/ird_dataset/IRD-Dataset-Complete-03-anonymized.xlsx'
    ukb_data_path ='databases/ird_dataset/export_heyex_original_dataset_03/DICOM'
    ukb_extractor = UkbDataExtractor(database_path=ukb_data_path, label_path=ukb_database_path)
    ukb_extractor.extract()
    ukb_extractor.split_extracted_data([0.7, 0.3, 0.2], stratify='Diagnose')
