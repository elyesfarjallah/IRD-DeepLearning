import numpy as np
import pandas as pd
import os
from pydicom.dataset import FileDataset


def find_files(folder_path):
    """
    Recursively find all files in a folder and its subfolders.
    
    Args:
    - folder_path (str): Path to the folder to search.
    
    Returns:
    - file_paths (list): List of paths to all files found.
    """
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def match_key_label(key : str, label, data_storage_path : str):
    """
    Find all files in a folder with a specific key in their path and match them with a label.
    
    Args:
    - key (str): Key to search for in the file path.
    - label: Label to match with the files found.
    - data_storage_path (str): Path to the folder to search.
    
    Returns:
    - matched_data (np.array): Array of matched data, where each row is [file_path, label]."""
    file_paths = find_files(f'{data_storage_path}/{key}')
    matched_data = [[file_path, label] for file_path in file_paths]
    return np.array(matched_data)

def match_keys_labels(keys : list, labels : list, data_storage_path : str):
    """
    Find all files in a folder with specific keys in their path and match them with labels.
    
    Args:
    - keys (list): List of keys to search for in the file paths.
    - labels (list): List of labels to match with the files found.
    - data_storage_path (str): Path to the folder to search.
    
    Returns:
    - matched_data (np.array): Array of matched data, where each row is [file_path, label]."""
    matched_data = []
    for key, label in zip(keys, labels):
        matched_data.extend(match_key_label(key, label, data_storage_path))
    return np.array(matched_data)

def dicom_detect_pixels(dcm_file :FileDataset) -> bool:
    """
    Check if a DICOM file has pixel data.
    
    Args:
    - dcm_file (FileDataset): DICOM file to check.
    
    Returns:
    - has_pixels (bool): True if the DICOM file has pixel data, False otherwise.
    """
    try:
        dcm_file.pixel_array
        return True
    except:
        return False
    
def reverse_one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reverse one-hot encoding in a DataFrame.

    Args:
    - df (pd.DataFrame): DataFrame to reverse one-hot encoding in.

    Returns:
    - df (pd.DataFrame): DataFrame with one-hot encoding reversed.
    """
    for column in df.columns:
        df[column] = df[column].apply(lambda x: column if x > 0 else None)
    return df

def rename_columns_from_dict(df: pd.DataFrame, renamer_dict : dict) -> pd.DataFrame:
    """
    Rename columns in a DataFrame using a dictionary.

    Args:
    - df (pd.DataFrame): DataFrame to rename columns in.
    - renamer_dict (dict): Dictionary mapping old column names to new column names.

    Returns:
    - df (pd.DataFrame): DataFrame with columns renamed.
    """
    renamer = lambda x: renamer_dict.get(x, x)
    return df.rename(columns=renamer)

def insert_instance_id_dimension(data: np.ndarray) -> np.ndarray:
    """
    Insert an instance ID dimension in a 2D array.

    Args:
    - data (np.ndarray): 2D array to insert an instance ID dimension in.

    Returns:
    - data (np.ndarray): 3D array with an instance ID dimension inserted.
    """
    ids = np.arange(data.shape[0]).reshape(-1, 1)
    return np.hstack((ids, data))