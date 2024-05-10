from abc import ABC, abstractmethod
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import data_extraction_utils
import pydicom
class DataExtractor(ABC):
    def __init__(self, database_path: str):
        self.database_path = database_path

    @abstractmethod
    def extract(self):
        pass


    #method that adds an entry to an existing dataset
    def add_entry_to_dataset(self, dataset: pd.DataFrame, disease_key : str, path_to_img : str, dataset_name : str):
        #create a new entry
        new_entry = pd.DataFrame([[disease_key, path_to_img, dataset_name]], columns = dataset.columns)
        #append the new entry to the dataset
        dataset_concatinated = pd.concat([dataset, new_entry])
        return dataset_concatinated

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
    def __init__(self, database_path: str, data_path: str, file_format: str):
        super().__init__(database_path=database_path)
        self.source_df = pd.read_csv(database_path)
        #generate a column Named Normal that is 1 if the Disease_Risk is 0 and 0 otherwise
        self.source_df['Normal'] = self.source_df['Disease_Risk'].apply(lambda x: 1 if x == 0 else 0)
        self.source_df = self.source_df.drop(columns=['Disease_Risk'])
        self.file_format = file_format
        self.data_path = data_path

    def extract(self):
        result_df = pd.DataFrame(columns=['disease_key','path_to_img','dataset_name'])
        for abbreviation in self.abbreviation_map.keys():
            #get all the images for the abberation
            try:
                images = self.source_df[self.source_df[abbreviation] > 0]
            except:
                continue
            for index, row in images.iterrows():
                #get the path to the image
                path_to_img = self.data_path + str(int(row['ID'])) + '.' + self.file_format
                #add the entry to the dataset
                result_df = self.add_entry_to_dataset(result_df, self.abbreviation_map[abbreviation], path_to_img, self.dataset_name)
        return result_df

class RFMiD2DataExtractor(DataExtractor):
    abbreviation_map = {
    "WNL": "Normal",
    "BRVO": "Branch Retinal Vein Occlusion",
    "CB": "Coloboma",
    "CF": "Choroidal Folds",
    "CL": "Collateral",
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
    
    dataset_name = "RFMiD2"
    csv_encoding = 'ISO-8859-1'
    def __init__(self, database_path: str, file_format: str):
        super().__init__(database_path=database_path)
        self.source_df = pd.read_csv(database_path, encoding=self.csv_encoding)
        
        
        self.file_format = file_format
    
    def extract(self):
        result_df = pd.DataFrame(columns=['disease_key','path_to_img','dataset_name'])
        for abbreviation in self.abbreviation_map.keys():
            #get all the images for the abberation
            try:
                images = self.source_df[self.source_df[abbreviation] > 0]
            except:
                continue
            for index, row in images.iterrows():
                #get the path to the image
                path_to_img = self.database_path.rsplit('/', 1)[0] + '/' + str(int(row['ID'])) + '.' + self.file_format
                #add the entry to the dataset
                result_df = self.add_entry_to_dataset(result_df, self.abbreviation_map[abbreviation], path_to_img, self.dataset_name)
        return result_df 

class ODIR5KDataExtractor(DataExtractor):
    dataset_name = "ODIR-5K"
    def __init__(self, database_path: str, database_test_images_path: str, database_train_images_path: str):
        super().__init__(database_path=database_path)
        self.source_df = pd.read_csv(database_path)
        #Important: drop the duplicates from the dataframe else there will be duplicates in the dataset
        self.source_df = self.source_df.drop_duplicates(keep='first', subset=['Right-Fundus', 'Left-Fundus'])
        self.database_test_images_path = database_test_images_path
        self.database_train_images_path = database_train_images_path

    def extract_eye_keywords(self, keywords : str, splitter : str):
        keywords_splitted = keywords.split(splitter)
        return set(keywords_splitted)
    
    def extract(self):
        odir5k_df = pd.DataFrame(columns=['disease_key','path_to_img','dataset_name'])
        for index, row in self.source_df.iterrows():
            #get the disease keys
            keyword_set_left = self.extract_eye_keywords(row['Left-Diagnostic Keywords'], '，')
            keyword_set_right = self.extract_eye_keywords(row['Right-Diagnostic Keywords'], '，')
            trainings_images = os.listdir(self.database_train_images_path)
            testing_images = os.listdir(self.database_test_images_path)
            for left_keyword in keyword_set_left:
                path_to_img_left = ""
                if row['Left-Fundus'] in trainings_images:
                    path_to_img_left = self.database_train_images_path + row['Left-Fundus']
                elif row['Left-Fundus'] in testing_images:
                    path_to_img_left = self.database_test_images_path + row['Left-Fundus']
                odir5k_df = self.add_entry_to_dataset(odir5k_df, left_keyword, path_to_img_left, self.dataset_name)
            #do the same for the right side
            for right_keyword in keyword_set_right:
                path_to_img_right = ""
                if row['Right-Fundus'] in trainings_images:
                    path_to_img_right = self.database_train_images_path + row['Right-Fundus']
                elif row['Right-Fundus'] in testing_images:
                    path_to_img_right = self.database_test_images_path + row['Right-Fundus']
                odir5k_df = self.add_entry_to_dataset(odir5k_df, right_keyword, path_to_img_right, self.dataset_name)
        return odir5k_df

class OneThousandImagesDataExtractor(DataExtractor):
    dataset_name = "1000images"
    regex_1000_images_disease_key = r'\d+\.(\d+\.)?(.+)'
    regex_1000_images_disease_key_compiled = re.compile(regex_1000_images_disease_key)

    def __init__(self, database_path: str):
        super().__init__(database_path=database_path)
        self.database = os.listdir(database_path)

    def extract(self):
        one_thousand_images_df = pd.DataFrame(columns=['disease_key','path_to_img','dataset_name'])
        for directory in self.database:
            #get the disease key
            disease_key = self.regex_1000_images_disease_key_compiled.findall(directory)[0][1]
            #get the path to the image
            image_names = os.listdir(self.database_path + directory)
            for image_name in image_names:
                path_to_img = self.database_path + directory + '/' + image_name
                #add the entry to the dataset
                one_thousand_images_df = self.add_entry_to_dataset(one_thousand_images_df, disease_key, path_to_img, '1000images')
        return one_thousand_images_df


class SESDataExtractor(DataExtractor):
    dataset_name = "SES"
    abbreviations_dict = {'Best': 'Best Disease', 'CD-CRD':'Cone Dystrophie or Cone-rod Dystrophie',
                          'LCA': 'Leber congenital amaurosis', 'RP': 'Retinitis Pigmentosa', 'STGD': 'Stargardt Disease'}
    
    def __init__(self, database_path: str):
        super().__init__(database_path=database_path)
        self.database = os.listdir(database_path)

    def extract(self):
        ses_df = pd.DataFrame(columns=['disease_key','path_to_img','dataset_name'])
        for directory in self.database:
            #get the directorrie path
            directory_path = self.database_path + directory
            #get the disease key
            disease_key = self.abbreviations_dict[directory]
            for image in os.listdir(directory_path):
                #get the image path
                image_path = directory_path + '/' + image
                #add the entry to the dataset
                ses_df = self.add_entry_to_dataset(ses_df, disease_key, image_path, 'SES')
        return ses_df

class RIPSDataExtractor(DataExtractor):
    def __init__(self, database_path: str):
        super().__init__(database_path=database_path)
        self.database = os.listdir(database_path)
    
    def extract(self):
        rips_df = pd.DataFrame(columns=['disease_key','path_to_img','dataset_name'])
        for name in self.database:
            #get the disease key
            disease_key = 'retinitis pigmentosa'
            #get the path to the image
            img_original_relative_path = name.replace('_', '/', 3)
            path_to_img = 'databases/RIPS/Original/' + img_original_relative_path
            #add the entry to the dataset
            rips_df = self.add_entry_to_dataset(rips_df, disease_key, path_to_img, 'RIPS')
        return rips_df

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

    def extract(self):
        #get all the keys and labels which contain a cfp
        keys_with_cfp = self.keys[self.contain_cfp]
        labels_with_cfp = self.labels[self.contain_cfp]
        #match the keys with the labels
        matched_data = data_extraction_utils.match_keys_labels(keys=keys_with_cfp, labels=labels_with_cfp, data_storage_path=self.database_path)
        #check if the files have pixel data
        pixel_checker = lambda x: data_extraction_utils.dicom_detect_pixels(pydicom.dcmread(x))
        have_pixels = np.vectorize(pixel_checker)(matched_data[:,0])
        #filter the data to only contain pixel data
        pixel_data_paths_labels = matched_data[have_pixels]
        print(f'Found {len(pixel_data_paths_labels)} images with pixel data')
        return pixel_data_paths_labels




    
def extract_all_databases():
    rfmid_train_extractor = RFMiDDataExtractor(database_path='databases/RFMiD/Training_Set/RFMiD_Training_Labels.csv',data_path='databases/RFMiD/Training_Set/Training/', file_format='png')
    rfmid_validation_extractor = RFMiDDataExtractor(database_path='databases/RFMiD/Evaluation_Set/RFMiD_Validation_Labels.csv', data_path='databases/RFMiD/Test_Set/Test/', file_format='png')
    rfmid_test_extractor = RFMiDDataExtractor(database_path='databases/RFMiD/Test_Set/RFMiD_Testing_Labels.csv', data_path='databases/RFMiD/Test_Set/Test/', file_format='png')

    rfmid2_train_extractor = RFMiD2DataExtractor(database_path='databases/RFMiD2_0/Training_set/RFMiD_2_Training_labels.csv', file_format='jpg')
    rfmid2_validation_extractor = RFMiD2DataExtractor(database_path='databases/RFMiD2_0/Validation_set/RFMiD_2_Validation_labels.csv', file_format='jpg')
    rfmid2_test_extractor = RFMiD2DataExtractor(database_path='databases/RFMiD2_0/Test_set/RFMiD_2_Testing_labels.csv', file_format='jpg')

    odir5k_extractor = ODIR5KDataExtractor(database_path='databases/ODIR-5k/full_df.csv', 
                                           database_test_images_path='databases/ODIR-5k/Testing Images/', 
                                           database_train_images_path='databases/ODIR-5K/Training Images/')
    
    one_thousand_images_extractor = OneThousandImagesDataExtractor(database_path='databases/1000images/')

    ses_extractor = SESDataExtractor(database_path='databases/SES/')

    rips_extractor = RIPSDataExtractor(database_path='databases/RIPS/RP/')
    #add all extractors to a list
    extractors = [rfmid_train_extractor, rfmid_validation_extractor, rfmid_test_extractor, rfmid2_train_extractor,
                   rfmid2_validation_extractor, rfmid2_test_extractor, odir5k_extractor,
                     one_thousand_images_extractor, ses_extractor, rips_extractor]
    #extract the data into a list
    data = []
    for extractor in extractors:
        data.append(extractor.extract())
    #concatinate the data
    full_data = pd.concat(data)
    #map the abbreviations
    full_data_no_abbreviations = map_abbreviations(full_data)
    not_summarize_set = set(RFMiDDataExtractor.abbreviation_map.values()).union(set(RFMiD2DataExtractor.abbreviation_map.values()))
    #summarize the common disease keys
    standardization_dict = generate_disease_key_standardization(full_data_no_abbreviations, not_summarize_set)
    #map the disease keys
    full_data_standardized = full_data_no_abbreviations.copy()
    full_data_standardized['disease_key'] = full_data_standardized['disease_key'].map(lambda x: standardization_dict.get(x, x))
    return full_data_standardized


def filter_labels_of_interest(df: pd.DataFrame, relevant_disease_keys: list):
    #filter the dataframe to only contain the relevant disease keys
    df_filtered = df[df['disease_key'].isin(relevant_disease_keys)]
    return df_filtered

def map_abbreviations(df: pd.DataFrame):
    #standardize the disease keys
    disease_abbreviations = {
    'BRVO': 'Branch Retinal Vein Occlusion',
    'CRVO': 'Central Retinal Vein Occlusion',
    'CSCR': 'Central Serous Chorioretinopathy',
    'DR1': 'Diabetic Retinopathy',
    'DR2': 'Diabetic Retinopathy',
    'DR3': 'Diabetic Retinopathy',
    'ERM': 'Epiretinal Membrane',
    'MH': 'Macular Hole',
    'RAO': 'Retinal Artery Occlusion',
    'normal fundus': 'Normal',
}
    #if the disease key is in the disease_abbreviations dict, replace the disease key with the value of the dict
    df['disease_key'] = df['disease_key'].map(lambda x: disease_abbreviations.get(x, x))
    return df

def generate_disease_key_standardization(df: pd.DataFrame, not_summarize_set: set):
    #create a set of all the disease keys
    disease_keys = set(df['disease_key'])
    disease_keys = list(disease_keys)
    #sort disease keys by length
    disease_keys.sort(key=lambda x: (len(x), x))
    skip_words = ['suspected', 'possible', 'suspicious', 'abnormal']
    #create a map of disease keys in which a disease key contains a different disease key is mapped to the disease key ignoring special characters spaces and capital letters
    disease_key_map = {}
    for disease_key in disease_keys:
        if disease_key not in disease_key_map.keys():
            for disease_key2 in disease_keys:
                disease_key_modified = disease_key.lower().replace(' ', '').replace('-', ' ').replace('_', ' ')
                disease_key2_modified = disease_key2.lower().replace(' ', '').replace('-', ' ').replace('_', ' ')
                summarize = not (disease_key2 in not_summarize_set and disease_key in not_summarize_set)
                skip = any(word in disease_key2_modified for word in skip_words)
                if disease_key != disease_key2 and summarize and not skip:
                    #check if disease_key2 is in disease_key
                    if disease_key_modified in disease_key2_modified:
                        disease_key_map[disease_key2] = disease_key
    return disease_key_map
    

def test_extract_all_databases():
    df = extract_all_databases()
    #count Retinitis Pigmentosa
    print(df['disease_key'].value_counts().head(15))

def test_rfmid():
    rfmid_train_extractor = RFMiDDataExtractor(database_path='databases/RFMiD/Training_Set/RFMiD_Training_Labels.csv', file_format='png')
    rfmid_validation_extractor = RFMiDDataExtractor(database_path='databases/RFMiD/Evaluation_Set/RFMiD_Validation_Labels.csv', file_format='png')
    rfmid_test_extractor = RFMiDDataExtractor(database_path='databases/RFMiD/Test_Set/RFMiD_Testing_Labels.csv', file_format='png')
    concat = pd.concat([rfmid_train_extractor.source_df, rfmid_validation_extractor.source_df, rfmid_test_extractor.source_df])
    print(concat['Normal'].sum())

