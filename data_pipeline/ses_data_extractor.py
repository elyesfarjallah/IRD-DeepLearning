from data_pipeline.data_extraction import DataExtractor
from data_pipeline.data_extraction_utils import find_files
from data_pipeline.data_extraction_utils import insert_instance_id_dimension

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
        return np.array(result)

#test
def test_extract():
    base_path = 'databases/SES/'
    ses_data_extractor = SESDataExtractor(database_path=base_path)
    data = ses_data_extractor.extract()
    #save the data as test.csv
    pd.DataFrame(data).to_csv('test_save_ses_converted.csv',header=False, index=False)