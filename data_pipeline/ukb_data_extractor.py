from data_pipeline.data_extraction import DataExtractor
from data_pipeline import data_extraction_utils
import numpy as np
import pandas as pd
import pydicom

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
        have_pixels = np.vectorize(pixel_checker)(matched_data[:,0])
        #filter the data to only contain pixel data
        pixel_data_paths_labels = matched_data[have_pixels]
        return pixel_data_paths_labels
    
    def extract_data_of_label(self, label):
        self.indexes_to_extract = self.labels == label
        extracted_data = self.extract()
        #reset the indexes to extract
        self.indexes_to_extract = np.full(len(self.keys), True)
        return extracted_data
    
    def extract_data_of_labels(self, labels):
        return np.vectorize(self.extract_data_of_label)(labels)
    

    
