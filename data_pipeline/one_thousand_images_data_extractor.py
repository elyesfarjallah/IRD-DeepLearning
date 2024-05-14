import numpy as np
import pandas as pd
import os
import re

from data_pipeline.data_extraction import DataExtractor
from data_pipeline.data_extraction_utils import insert_instance_id_dimension
from data_pipeline.data_splitting_utils import split_by_ratios
class OneThousandImagesDataExtractor(DataExtractor):
    dataset_name = "1000images"
    regex_1000_images_disease_key = r'\d+\.(\d+\.)?(.+)'
    regex_1000_images_disease_key_compiled = re.compile(regex_1000_images_disease_key)

    def __init__(self, database_path: str):
        super().__init__(database_path=database_path)
        self.database = os.listdir(database_path)

    def extract(self):
        extracted_data = []
        for directory in self.database:
            #get the disease key
            disease_key = self.regex_1000_images_disease_key_compiled.findall(directory)[0][1]
            #get the path to the image
            image_names = os.listdir(self.database_path + directory)
            for image_name in image_names:
                path_to_img = self.database_path + directory + '/' + image_name
                #add the entry to the dataset
                extracted_data.append([path_to_img, disease_key])
        #convert the list to a numpy array
        extracted_data = np.array(extracted_data)
        #insert the instance id dimension
        extracted_data = insert_instance_id_dimension(extracted_data)
        self.extracted_data = extracted_data
        return self.extracted_data
    
    def get_labels(self):
        return self.extracted_data[:,2:]
    
    def get_data(self):
        return self.extracted_data[:,1]
    
    def get_instance_ids(self):
        return self.extracted_data[:,0]
    
    def split_extracted_data(self, split_portions, stratify):
        if stratify:
            return split_by_ratios(data=self.extracted_data, split_ratios=split_portions, stratify=self.get_labels())
        else:
            return split_by_ratios(data=self.extracted_data, split_ratios=split_portions)
    

#test the data extraction
def test_extraction():
    extractor = OneThousandImagesDataExtractor(database_path='databases/1000images/')
    data = extractor.extract()
    #save the data
    pd.DataFrame(data).to_csv('test_extraction_one_thousand_images.csv', index=False, header=False)

            
