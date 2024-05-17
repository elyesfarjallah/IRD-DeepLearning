import numpy as np
import json

class DataPackage:
    def __init__(self, data : np.ndarray, labels : np.ndarray, data_source_name : str = None, instance_ids : np.ndarray = None):
        self.data = data
        self.labels = labels
        self.instance_ids = instance_ids
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
    def save(self, file_path):
        save_data = {
            'data': self.data.tolist(),
            'labels': self.labels.tolist(),
            'data_source_name': self.data_source_name,
            'instance_ids': self.instance_ids.tolist() if self.instance_ids is not None else None
        }
        with open(file_path, 'w') as f:
            json.dump(save_data, f)
    @staticmethod
    def load(file_path):
        with open(file_path, 'r') as f:
            load_data = json.load(f)
            data = np.array(load_data['data'])
            labels = np.array(load_data['labels'])
            data_source_name = load_data['data_source_name']
            instance_ids = np.array(load_data['instance_ids']) if load_data['instance_ids'] is not None else None
            return DataPackage(data=data, labels=labels, data_source_name=data_source_name, instance_ids=instance_ids)