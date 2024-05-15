from data_pipeline.data_extraction import DataExtractor
from data_pipeline.data_extraction_utils import find_files
from data_pipeline.data_splitting_utils import split_by_instance_count
from data_pipeline.data_package import DataPackage
import os
import numpy as np
import pandas as pd

class RIPSDataExtractor(DataExtractor):
    dataset_name = "RIPS"
    def __init__(self, database_path: str):
        super().__init__(database_path=database_path)
        self.database = os.listdir(database_path)
        self.label = 'Retinitis Pigmentosa'
    def extract(self):
        result = []
        for instance in self.database:
            #get the path to the image
            path_to_instance_dir = f'{self.database_path}/{instance}'
            #find all the files in the instance directory
            instance_file_paths = np.array(find_files(path_to_instance_dir))
            n_file_paths = len(instance_file_paths)
            matching__instance_ids = [instance] * n_file_paths
            labels = [self.label] * n_file_paths
            #create an array of the instance ids, file paths and labels then Transpose it
            instance_result = np.array([matching__instance_ids, instance_file_paths, labels]).T
            result.extend(instance_result)
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
        if stratify:
            #todo add warning
            pass
        instance_split = split_by_instance_count(instance_list=self.get_instance_ids(), split_ratios=split_portions)
        data_splits = []
        for split in instance_split:
            extraction_series = np.isin(self.get_instance_ids(), split)
            split_data = self.extracted_data[extraction_series]
            split_labels = self.extracted_data[extraction_series][:,2:]
            data_splits.append(DataPackage(data=split_data, labels=split_labels, data_source_name=self.dataset_name))
        return data_splits
def test_extract():
    base_path = 'databases/RIPS/Original'
    rips_data_extractor = RIPSDataExtractor(database_path=base_path)
    data = rips_data_extractor.extract()
    #save the data as test.csv
    pd.DataFrame(data).to_csv('test_save_rips_converted.csv',header=False, index=False)
