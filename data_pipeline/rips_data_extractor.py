from data_pipeline.data_extraction import DataExtractor
from data_pipeline.data_extraction_utils import find_files
from data_pipeline.data_splitting_utils import split_by_instance_count
import os
import numpy as np
import pandas as pd

class RIPSDataExtractor(DataExtractor):
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
        return self.extracted_data
    
    def get_labels(self):
        return self.extracted_data[:,2:]
    
    def get_data(self):
        return self.extracted_data[:,1]
    
    def get_instance_ids(self):
        return self.extracted_data[:,0]
    
    def split_extracted_data(self, split_portions, stratify):
        if stratify:
            #todo add warning
            pass
        #todo check what exactly here is returned
        instance_split = split_by_instance_count(instance_list=self.get_instance_ids(), split_ratios=split_portions)
        data_splits = [[] * len(instance_split)]
        for split, data_split in zip(instance_split, data_splits):
            extraction_series = np.isin(self.get_instance_ids(), split)
            data_split.extend(self.extracted_data[extraction_series])
        return data_splits
def test_extract():
    base_path = 'databases/RIPS/Original'
    rips_data_extractor = RIPSDataExtractor(database_path=base_path)
    data = rips_data_extractor.extract()
    #save the data as test.csv
    pd.DataFrame(data).to_csv('test_save_rips_converted.csv',header=False, index=False)
