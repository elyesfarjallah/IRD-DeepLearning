from data_pipeline.data_extraction import DataExtractor
from data_pipeline.data_extraction_utils import find_files
from data_pipeline.data_extraction_utils import insert_instance_id_dimension
from data_pipeline.data_splitting_utils import split_by_ratios
from data_pipeline.data_package import DataPackage
import pandas as pd
import numpy as np
import os

class SESDataExtractor(DataExtractor):
    dataset_name = "SES"
    abbreviations_dict = {'Best': 'Best Disease', 'CD-CRD':'Cone Dystrophie or Cone-rod Dystrophie',
                          'LCA': 'Leber congenital amaurosis', 'RP': 'Retinitis Pigmentosa', 'STGD': 'Stargardt Disease'}
    
    def __init__(self, database_path: str):
        super().__init__(database_path=database_path)
        self.database = os.listdir(database_path)

    def extract(self):
        result = []
        for directory in self.database:
            #get the directorrie path
            directory_path = self.database_path + directory
            #get the disease key
            disease_key = self.abbreviations_dict[directory]
            #get all the images in the directory
            file_paths = find_files(directory_path)
            # get the instance ids
            labels = [disease_key] * len(file_paths)
            #create an array of the instance ids, file paths
            instance_result = np.array([file_paths, labels]).T
            #insert the instance id dimension
            result.extend(insert_instance_id_dimension(instance_result))
        self.extracted_data = np.array(result)
        self.remove_not_existing_file_paths()
        return self.extracted_data
    
    def get_labels(self, data_truth_series : np.ndarray = None):
        return self.extracted_data[:,2:] if data_truth_series is None else self.extracted_data[data_truth_series][:,2:]
    
    def get_file_paths(self, data_truth_series : np.ndarray = None):
        return self.extracted_data[:,1] if data_truth_series is None else self.extracted_data[data_truth_series][:,1]
    
    def get_instance_ids(self, data_truth_series : np.ndarray = None):
        return self.extracted_data[:,0] if data_truth_series is None else self.extracted_data[data_truth_series][:,0]
       
    def split_extracted_data(self, split_portions, stratify):
        data_to_split = np.concatenate((self.get_instance_ids().reshape(-1,1), self.get_file_paths().reshape(-1,1)), axis=1)
        labels = self.get_labels()
        split_data = split_by_ratios(data=data_to_split, labels=labels, split_ratios=split_portions, stratify=labels)
        return_packages = []
        for split in split_data:
            file_paths = split.get_data()[:,1]
            labels = split.get_labels()
            instance_ids = split.get_data()[:,0]
            return_packages.append(DataPackage(data=file_paths, labels=labels, instance_ids=instance_ids, data_source_name=self.dataset_name))
        return return_packages
    

#test
def test_extract():
    base_path = 'databases/SES/'
    ses_data_extractor = SESDataExtractor(database_path=base_path)
    data = ses_data_extractor.extract()
    #save the data as test.csv
    pd.DataFrame(data).to_csv('test_save_ses_converted.csv',header=False, index=False)