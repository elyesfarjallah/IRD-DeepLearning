from data_pipeline.data_extraction import DataExtractor
from data_pipeline.data_extraction_utils import reverse_one_hot_encode
from data_pipeline.data_extraction_utils import rename_columns_from_dict
from data_pipeline.data_extraction_utils import insert_instance_id_dimension
from data_pipeline.rfmid_data_extractor import RFMiDDataExtractor
import pandas as pd
import numpy as np


class RFMiD2DataExtractor(DataExtractor):
    abbreviation_map = {
    "WNL": "Normal",
    "ARMD": "Age-related Macular Degeneration",
    "BRVO": "Branch Retinal Vein Occlusion",
    "CB": "Coloboma",
    "CF": "Choroidal Folds",
    "CL": "Collateral",
    "CSR": "Central Serous Retinopathy",
    "ME": "Macular Edema",
    "NV": "Neovascularization",
    "CRAO": "Central Retinal Artery Occlusion",
    "CRS": "Chorioretinitis",
    "CRVO": "Central Retinal Vein Occlusion",
    "CSC": "Cysticercosis",
    "CWS": "Cotton Wool Spots",
    "DN": "Drusens",
    "DR": "Diabetic Retinopathy",
    "EX": "Exudation",
    "ERM": "Epiretinal Membrane",
    "GRT": "Giant Retinal Tear",
    "HTN": "Hypertension",
    "HPED": "Hemorrhagic Pigment Epithelial Detachment",
    "IIH": "Idiopathic Intracranial Hypertension",
    "HTR": "Hypertensive Retinopathy",
    "HR": "Haemorrhagic Retinopathy",
    "LS": "Laser Scar",
    "MCA": "Microaneurysm",
    "MH": "Media Haze",
    "MHL": "Macular Hole",
    "MS": "Macular Scar",
    "MYA": "Myopia",
    "ODC": "Optic Disc Cupping",
    "ODE": "Optic Disc Edema",
    "ODP": "Optic Disc Pallor",
    "ON": "Optic Neuritis",
    "ODPM": "Optic Disc Pit Maculopathy",
    "PRH": "Preretinal Hemorrhage",
    "RD": "Retinal Detachment",
    "RHL": "Retinal Holes",
    "RTR": "Retinal Tears",
    "RP": "Retinitis Pigmentosa",
    "RPEC": "Retinal Pigment Epithelium Changes",
    "RS": "Retinitis",
    "RT": "Retinal Traction",
    "SOFE": "Silicone Oil-Filled Eye",
    "ST": "Optociliary Shunt",
    "TD": "Tilted Disc",
    "TSLN": "Tessellation",
    "TV": "Tortuous Vessels",
    "VS": "Vasculitis"
}
    #add the abbreviation map to the RFMiD2DataExtractor abbreviation map
    abbreviation_map.update(RFMiDDataExtractor.abbreviation_map)
    dataset_name = "RFMiD2"
    csv_encoding = 'ISO-8859-1'
    def __init__(self, database_path: str, data_path : str, file_format: str):
        super().__init__(database_path=database_path)
        self.source_df = pd.read_csv(database_path, encoding=self.csv_encoding)
        self.data_path = data_path
        self.file_format = file_format
    
    def extract(self):
        # result_df = pd.DataFrame(columns=['disease_key','path_to_img','dataset_name'])
        # for abbreviation in self.abbreviation_map.keys():
        #     #get all the images for the abberation
        #     try:
        #         images = self.source_df[self.source_df[abbreviation] > 0]
        #     except:
        #         continue
        #     for index, row in images.iterrows():
        #         #get the path to the image
        #         image_id = str(int(row['ID']))
        #         path_to_img = f'{self.data_path}/{image_id}.{self.file_format}'
        #         #add the entry to the dataset
        #         result_df = self.add_entry_to_dataset(result_df, self.abbreviation_map[abbreviation], path_to_img, self.dataset_name)
        # return result_df
        datapoint_id_column_name = 'ID'
        full_path_column_name = 'Path'
        #copy the source_df
        copied_source = self.source_df.copy()
        #rename the columns
        full_name_df = rename_columns_from_dict(copied_source, self.abbreviation_map)
        path_index_df = self.convert_to_full_path_index_df(df = full_name_df, datapoint_id_column_name = datapoint_id_column_name, full_path_column_name = full_path_column_name)
        #drop the ID column
        path_index_df = path_index_df.drop(columns=[datapoint_id_column_name])
        #map binary encoding to text encoding
        reverse_encoded_df = reverse_one_hot_encode(path_index_df)
        #return the np array of the dataframe including the path
        #index needs to be reset to get the path as a column
        #return the np array of the dataframe including the path
        #index needs to be reset to get the path as a column
        result_np = reverse_encoded_df.reset_index().values
        result_np_with_instance_id = insert_instance_id_dimension(data = result_np)
        return result_np_with_instance_id

    def convert_to_full_path_index_df(self, df: pd.DataFrame, datapoint_id_column_name : str,
                                       full_path_column_name: str) -> pd.DataFrame:
        df[full_path_column_name] = df[datapoint_id_column_name].apply(lambda x: f'{self.data_path}/{str(int(x))}.{self.file_format}')
        return df.set_index(full_path_column_name)

#test
def test_extract():
    base_path = 'databases/RFMiD2_0/Validation_set/RFMiD_2_Validation_labels.csv'
    data_path = f'databases/RFMiD2_0/Validation_set'
    rfmid2_data_extractor = RFMiD2DataExtractor(database_path=base_path, data_path=data_path, file_format='jpg')
    data = rfmid2_data_extractor.extract()
    #save the data as test.csv
    pd.DataFrame(data).to_csv('test_save_rfmid2_converted.csv',header=False, index=False)



