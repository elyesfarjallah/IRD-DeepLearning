from data_pipeline.data_extraction import DataExtractor
from data_pipeline.data_extraction_utils import find_files
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
        return np.array(result)
#test
base_path = 'databases/RIPS/Original'
rips_data_extractor = RIPSDataExtractor(database_path=base_path)
data = rips_data_extractor.extract()
#save the data as test.csv
pd.DataFrame(data).to_csv('test_save_rips_converted.csv',header=False, index=False)