import numpy as np
from sklearn.preprocessing import OneHotEncoder


def standardize_labels(labels: np.array, not_summarize_set: set):
    #create a set of all the disease keys
    unique_labels = set(labels)
    unique_labels = list(unique_labels)
    #sort disease keys by length
    unique_labels.sort(key=lambda x: (len(x), x))
    skip_words = ['suspected', 'possible', 'suspicious', 'abnormal']
    #create a map of disease keys in which a disease key contains a different disease key is mapped to the disease key ignoring special characters spaces and capital letters
    label_standardization_dict = {}
    for label in unique_labels:
        if label not in label_standardization_dict.keys():
            for comparing_label in unique_labels:
                disease_key_modified = label.lower().replace(' ', '').replace('-', ' ').replace('_', ' ')
                disease_key2_modified = comparing_label.lower().replace(' ', '').replace('-', ' ').replace('_', ' ')
                summarize = not (comparing_label in not_summarize_set and label in not_summarize_set)
                skip = any(word in disease_key2_modified for word in skip_words)
                if label != comparing_label and summarize and not skip:
                    #check if disease_key2 is in disease_key
                    if disease_key_modified in disease_key2_modified:
                        label_standardization_dict[comparing_label] = label
    return label_standardization_dict

def create_one_hot_encoder(unique_labels: np.array, handle_unknown : str = 'ignore'):
    #create a set of all the disease keys
    #create a one hot encoder
    one_hot_encoder = OneHotEncoder(handle_unknown=handle_unknown, categories=[unique_labels])
    #reshape the labels to be a column vector
    unique_labels_shaped = unique_labels.reshape(-1,1)
    #fit the one hot encoder
    #one_hot_encoder.categories_ = [unique_labels]
    encoder_fitted = one_hot_encoder.fit(unique_labels_shaped)
    return encoder_fitted

def find_over_represented_samples(file_paths: np.array, labels: np.array, max_samples_per_class : int):
    flat_labels = np.concatenate(labels)
    #remove the None values
    flat_labels = flat_labels[flat_labels != None]
    #count the number of samples per class
    unique_labels, counts = np.unique(flat_labels, return_counts=True)
    #calculate the number of samples to remove
    samples_to_remove = counts - max_samples_per_class
    #initialize the indices to remove
    indices_to_remove = []
    #iterate over the unique labels and the samples to remove
    for label, samples in zip(unique_labels, samples_to_remove):
        if samples > 0:
            #get the indices of the samples with the current label
            label_indices = np.where(labels == label)[0]
            #randomly select the samples to remove
            label_indices = np.random.permutation(label_indices)
            indices_to_remove.extend(label_indices[:samples])
    return indices_to_remove, file_paths[indices_to_remove], labels[indices_to_remove]

def encode_multistring_labels(labels: np.array, encoder: OneHotEncoder):
    #initialize the encoded labels
    encoded_labels = []
    #iterate over the labels
    for label in labels:
        #iterate over the strings in the label
        encoded_label_sum = encode_multi_string_label(label = label, encoder = encoder)
        #append the encoded label to the encoded labels
        encoded_labels.append(encoded_label_sum)
    return np.vstack(encoded_labels)

def encode_multi_string_label(label: str, encoder: OneHotEncoder):
    #initialize the encoded label
    encoded_label = []
    #iterate over the strings in the label
    for string in label:
        #encode the string
        encoded_string = encoder.transform(np.array(string).reshape(1,-1)).toarray()
        #append the encoded string to the encoded label
        encoded_label.append(encoded_string)
    #sum the encoded label
    #convert the encoded label to a numpy array
    encoded_label = np.array(encoded_label)
    encoded_label_sum = encoded_label.sum(axis=0)
    return encoded_label_sum

