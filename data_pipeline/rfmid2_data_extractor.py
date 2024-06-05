from data_pipeline.data_extraction import DataExtractor
from data_pipeline.data_extraction_utils import reverse_one_hot_encode
from data_pipeline.data_extraction_utils import rename_columns_from_dict
from data_pipeline.data_extraction_utils import insert_instance_id_dimension
from data_pipeline.data_splitting_utils import stratified_multilabel_split, split_by_ratios
from data_pipeline.data_processing_utils import create_one_hot_encoder
from data_pipeline.data_processing_utils import encode_multistring_labels
from data_pipeline.rfmid_data_extractor import RFMiDDataExtractor
from data_pipeline.data_package import DataPackage
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
    def __init__(self, database_path: str, data_path : str, file_format: str = 'jpg'):
        super().__init__(database_path=database_path)
        self.source_df = pd.read_csv(database_path, encoding=self.csv_encoding)
        self.data_path = data_path
        self.file_format = file_format
    
    def extract(self):
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
        self.extracted_data = result_np_with_instance_id
        self.remove_not_existing_file_paths()
        return result_np_with_instance_id

    def convert_to_full_path_index_df(self, df: pd.DataFrame, datapoint_id_column_name : str,
                                       full_path_column_name: str) -> pd.DataFrame:
        df[full_path_column_name] = df[datapoint_id_column_name].apply(lambda x: f'{self.data_path}/{str(int(x))}.{self.file_format}')
        return df.set_index(full_path_column_name)
    
    def get_labels(self, data_truth_series : np.ndarray = None):
        return self.extracted_data[:,2:] if data_truth_series is None else self.extracted_data[data_truth_series][:,2:]
    
    def get_file_paths(self, data_truth_series : np.ndarray = None):
        return self.extracted_data[:,1] if data_truth_series is None else self.extracted_data[data_truth_series][:,1]
    
    def get_instance_ids(self, data_truth_series : np.ndarray = None):
        return self.extracted_data[:,0] if data_truth_series is None else self.extracted_data[data_truth_series][:,0]
      
    def split_extracted_data(self, split_portions, stratify):
        data_to_split = np.concatenate((self.get_instance_ids().reshape(-1,1), self.get_file_paths().reshape(-1,1), self.get_labels()), axis=1)
        if stratify:
            #get all the labels
            labels = self.get_labels()
            #unique labels
            flat_labels = labels.flatten()
            #filter out none values
            flat_labels = flat_labels[flat_labels != None]
            unique_labels = np.unique(flat_labels)
            #filter out none values
            #encode the labels
            encoder = create_one_hot_encoder(unique_labels=unique_labels)
            labels_encoded = encode_multistring_labels(labels=labels, encoder=encoder)
            splits_packaged = stratified_multilabel_split(data=self.extracted_data, labels=labels_encoded, split_ratios=split_portions)
        else:
            splits_data = split_by_ratios(data=data_to_split, labels=labels, split_ratios=split_portions)
        
        splits_data = [split.get_data() for split in splits_packaged]
        return_packages = []
        for split in splits_data:
            file_paths = split[:,1]
            labels = split[:,2:]
            instance_ids = split[:,0]
            return_packages.append(DataPackage(data=file_paths, labels=labels, instance_ids=instance_ids, data_source_name=self.dataset_name))
        return return_packages    

#test
def test_extract():
    base_path = 'databases/RFMiD2_0/Validation_set/RFMiD_2_Validation_labels.csv'
    data_path = f'databases/RFMiD2_0/Validation_set'
    rfmid2_data_extractor = RFMiD2DataExtractor(database_path=base_path, data_path=data_path, file_format='jpg')
    data = rfmid2_data_extractor.extract()
    #save the data as test.csv
    pd.DataFrame(data).to_csv('test_save_rfmid2_converted.csv',header=False, index=False)



