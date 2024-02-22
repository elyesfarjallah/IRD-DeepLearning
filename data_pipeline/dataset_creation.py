import pandas as pd
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.model_selection import IterativeStratification
import os


def convert_to_multilabel_format(df : pd.DataFrame, column_to_unify : str, label_separator : str, data_path_column_name : str):
    """
    Convert a dataframe to a multilabel format
    """
    df = df.copy()
    #get the unique labels
    unique_labels = df[column_to_unify].unique()
    df_grouped = df.groupby(data_path_column_name)
    #join the labels
    df_labels_jonined = df_grouped[column_to_unify].apply(label_separator.join).reset_index()
    #drop duplicates
    df_labels_jonined_unique = df_labels_jonined.drop_duplicates(subset = data_path_column_name, keep='first')
    #generate a new column for each label
    for label in unique_labels:
        df_labels_jonined_unique[label] = df_labels_jonined_unique[column_to_unify].str.contains(label).astype(int)
    return df_labels_jonined_unique

def balance_multilabel_df(df : pd.DataFrame, label_names : list, median_deviation : float):
    """
    Balance a multilabel dataframe
    """
    df = df.copy()
    #drop the images
    for label_name in label_names:
        possible_drops_df, n_images_to_drop = get_droppable_entries(df = df, label_names = label_names, median_deviation = median_deviation)
        df = drop_n_images_by_key(df_to_drop_from = df, df_droppable_entries = possible_drops_df, n_images_to_drop = n_images_to_drop[label_name], label_name = label_name)
    return df

def get_droppable_entries(df : pd.DataFrame, label_names : list, median_deviation : float):
    #get label counts
    label_counts = df[label_names].sum(axis=0)
    #get median
    label_counts_median = label_counts.median()
    max_number_of_images = label_counts_median * (1 + median_deviation)
    #cehck which labels have more than the median
    #get labels which should be kept
    labels_to_keep = label_counts[label_counts <= max_number_of_images].index
    #filter the df by the labels to keep by summing the labels and checking if the sum is greater than 0
    possible_drops_df = df.apply(lambda x: x[labels_to_keep].sum() == 0, axis=1)
    n_images_to_drop = label_counts - max_number_of_images
    #round the number of images to drop
    n_images_to_drop = n_images_to_drop.apply(lambda x: int(x))
    return possible_drops_df, n_images_to_drop

def drop_n_images_by_key(df_to_drop_from : pd.DataFrame, df_droppable_entries : pd.DataFrame, n_images_to_drop : int, label_name : str):
    #get all the entries which have the label
    possible_drops_df = df_to_drop_from[df_droppable_entries]
    df_with_label = possible_drops_df[possible_drops_df[label_name] == 1]
    #get the indices of the images to drop
    if n_images_to_drop <= 0:
        return df_to_drop_from
    indices_to_drop = df_with_label.sample(n=n_images_to_drop).index
    #drop the images
    df_dropped_images = df_to_drop_from.drop(indices_to_drop, inplace=False)
    return df_dropped_images

def drop_labels_with_too_few_entries(df : pd.DataFrame, label_names : list, min_entries : int):
    """
    Drop labels which have too few entries
    """
    df = df.copy()
    #get label counts
    label_counts = df[label_names].sum(axis=0)
    #get labels which should be kept
    labels_to_keep = label_counts[label_counts >= min_entries].index
    #filter the df by the labels to keep by summing the labels and checking if the sum is greater than 0
    df = df[df[labels_to_keep].sum(axis=1) > 0]
    #remove the labels which are not in the labels to keep
    labels_to_drop = label_counts[label_counts < min_entries].index
    df = df.drop(columns = labels_to_drop)
    return df, labels_to_keep.to_list()
    
def split_database_by_source_name(df: pd.DataFrame, source_column : str, source_name_to_split: str):
    """
    Split a dataframe into two dataframes based on a column value
    """
    df = df.copy()
    mask = df[source_column].str.contains(source_name_to_split)
    data_from_source = df[mask]
    rest_data = df[~mask]
    return data_from_source, rest_data

def stratified_multilabel_dataset_split(df : pd.DataFrame, train_frac : float, validation_frac : float, test_frac : float, label_columns : list, random_state : int = 42):
    '''
    Split a dataframe into train, validation and test sets
    df : pandas dataframe to split
    train_frac : float [0, 1]
    validation_frac : float [0, 1]
    test_frac : float [0, 1]
    label_columns : list[str] list of column names to use as labels
    random_state : int
    '''
    labels = df[label_columns].values
    #convert dataframe to numpy array
    df_np = df.values
    X_train, y_train, X_val_test, y_val_test = iterative_train_test_split(X = df_np, y = labels, test_size=validation_frac + test_frac)
    X_val, y_val, X_test, y_test = iterative_train_test_split(X = X_val_test, y = y_val_test, test_size=test_frac / (validation_frac + test_frac))
    #return the splits
    return X_train, y_train, X_val, y_val, X_test, y_test

def multi_label_k_fold_split(df : pd.DataFrame, label_columns : list, n_splits : int, multilabel_order : int = 1):
    """"
    Split a dataframe into k folds
    df : pandas dataframe to split
    label_columns : list[str] list of column names to use as labels
    n_splits : int number of splits
    random_state : int
    multilabel_order : int magnitude order to consider for multilabel split
    """

    labels = np.array(df[label_columns].values, dtype=int)
    #convert dataframe to numpy array
    df_np = df.values
    #create the stratified k fold object
    k_fold = IterativeStratification(n_splits = n_splits, order = multilabel_order)
    #return the splits
    return k_fold.split(X = df_np, y = labels)

#todo test the functions, create a k fold dataet, run a k fold cross validation and save the results


def test_k_fold_split():
    #create a random dataset
    inputs_and_outputs = np.random.randint(low=0, high=2, size=(500, 7))
    input_columns = ["input1", "input2", "input3", "input4", "input5"]
    label_columns = ["output1", "output2"]
    df = pd.DataFrame(data = inputs_and_outputs, columns = input_columns + label_columns)
    #initialize the iterative stratification object
    splits = multi_label_k_fold_split(df = df, label_columns = label_columns, n_splits = 10, multilabel_order = 5)
    for train, test in splits:
        #print("Train indices: ", train.shape, "Test indices: ", test.shape)
        #print("Train labels: ", df.iloc[train][label_columns].value_counts())
        print("Test labels: ", df.iloc[test][label_columns].value_counts())
    
    
def test_stratified_multilabel_dataset_split():
    #create a random dataset
    inputs_and_outputs = np.random.randint(low=0, high=2, size=(500, 7))
    input_columns = ["input1", "input2", "input3", "input4", "input5"]
    label_columns = ["output1", "output2"]
    df = pd.DataFrame(data = inputs_and_outputs, columns = input_columns + label_columns)
    X_train, y_train, X_val, y_val, X_test, y_test = stratified_multilabel_dataset_split(df = df, train_frac = 0.8, validation_frac = 0.0,
                                                                                          test_frac = 0.2, label_columns = label_columns)
    print("Train labels: ", pd.DataFrame(data = y_train, columns = label_columns).value_counts())
    print("Validation labels: ", pd.DataFrame(data = y_val, columns = label_columns).value_counts())
    print("Test labels: ", pd.DataFrame(data = y_test, columns = label_columns).value_counts())
