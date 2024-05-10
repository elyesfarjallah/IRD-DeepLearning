from data_pipeline.data_extraction import DataExtractor
from data_pipeline.data_extraction_utils import reverse_one_hot_encode
from data_pipeline.data_extraction_utils import rename_columns_from_dict
from data_pipeline.data_extraction_utils import insert_instance_id_dimension
import pandas as pd
import numpy as np

class RFMiDDataExtractor(DataExtractor):
    abbreviation_map = {
    "Normal": "Normal",
    "DR": "Diabetic Retinopathy",
    "ARMD": "Age-related Macular Degeneration",
    "MH": "Media Haze",
    "DN": "Drusens",
    "MYA": "Myopia",
    "BRVO": "Branch Retinal Vein Occlusion",
    "TSLN": "Tessellation",
    "ERM": "Epiretinal Membrane",
    "LS": "Laser Scars",
    "MS": "Macular Scar",
    "CSR": "Central Serous Retinopathy",
    "ODC": "Optic Disc Cupping",
    "CRVO": "Central Retinal Vein Occlusion",
    "TV": "Tortuous Vessels",
    "AH": "Asteroid Hyalosis",
    "ODP": "Optic Disc Pallor",
    "ODE": "Optic Disc Edema",
    "ST": "Optociliary Shunt",
    "AION": "Anterior Ischemic Optic Neuropathy",
    "PT": "Parafoveal Telangiectasia",
    "RT": "Retinal Traction",
    "RS": "Retinitis",
    "CRS": "Chorioretinitis",
    "EDN": "Exudation",
    "RPEC": "Retinal Pigment Epithelium Changes",
    "MHL": "Macular Hole",
    "RP": "Retinitis Pigmentosa",
    "CWS": "Cotton-Wool Spots",
    "CB": "Coloboma",
    "ODPM": "Optic Disc Pit Maculopathy",
    "PRH": "Preretinal Hemorrhage",
    "MNF": "Myelinated Nerve Fibers",
    "HR": "Hemorrhagic Retinopathy",
    "CRAO": "Central Retinal Artery Occlusion",
    "TD": "Tilted Disc",
    "CME": "Cystoid Macular Edema",
    "PTCR": "Post-Traumatic Choroidal Rupture",
    "CF": "Choroidal Folds",
    "VH": "Vitreous Hemorrhage",
    "MCA": "Macroaneurysm",
    "VS": "Vasculitis",
    "BRAO": "Branch Retinal Artery Occlusion",
    "PLQ": "Plaque",
    "HPED": "Hemorrhagic Pigment Epithelial Detachment",
    "CL": "Collateral"
}
    dataset_name = "RFMiD"
    def __init__(self, database_path: str, data_path: str, file_format: str, diseas_risk_column : str = 'Disease_Risk'):
        super().__init__(database_path=database_path)
        self.diseas_risk_column = diseas_risk_column
        self.source_df = pd.read_csv(database_path)
        self.file_format = file_format
        self.data_path = data_path

    def extract(self) -> np.ndarray:
        full_path_column_name = 'Path'
        no_leasion_column_name = 'Normal'
        datapoint_id_column_name = 'ID'
        #copy
        copied_source = self.source_df.copy()

        #encode the disease risk column as binary
        copied_source[no_leasion_column_name] = copied_source[self.diseas_risk_column].apply(lambda x: 1 if x == 0 else 0)
        copied_source = copied_source.drop(columns=[self.diseas_risk_column])
        #rename the columns from the abbreviation to the full name
        
        full_name_df = rename_columns_from_dict(df = copied_source, renamer_dict = self.abbreviation_map)
        #convert the dataframe to a dataframe with the full path as index
        path_index_df = self.convert_to_full_path_index_df(df = full_name_df, datapoint_id_column_name = datapoint_id_column_name, full_path_column_name = full_path_column_name)
        #drop the ID column
        path_index_df = path_index_df.drop(columns=[datapoint_id_column_name])
        #map binary encoding to text encoding
        reverse_encoded_df = reverse_one_hot_encode(path_index_df)
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
    database_path = 'C:/Users/elyes/IRD-DeepLearning/databases/RFMiD/Evaluation_Set/RFMiD_Validation_Labels.csv'
    data_path = 'C:/Users/elyes/IRD-DeepLearning/databases/RFMiD/Evaluation_Set/Validation'
    data_extractor = RFMiDDataExtractor(database_path=database_path, data_path=data_path, file_format='png')
    data = data_extractor.extract()
    #save the data as test.csv
    pd.DataFrame(data).to_csv('test_save_rfmid_converted.csv',header=False, index=False)

test_extract()