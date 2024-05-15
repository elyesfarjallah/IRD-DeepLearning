import numpy as np

class DataPackage:
    def __init__(self, data : np.ndarray, labels : np.ndarray, data_source_name : str = None):
        self.data = data
        self.labels = labels
        self.data_source_name = data_source_name
    def get_data(self):
        return self.data
    def get_labels(self):
        return self.labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    def set_data_source_name(self, data_source_name):
        self.data_source_name = data_source_name