from ai_backend.loggers.wandb_logger import WandbObserver
from ai_backend.dummy_subject import DummySubject
import torch
import wandb
from uuid import uuid4
import json
from convert_user_inputs import create_subjects_and_observers

from data_pipeline.data_loading_utils import data_packages_to_datasets, filter_data_packages_by_labels
from data_pipeline.data_package import DataPackage
from data_pipeline.image_transforms import get_transforms
import input_mapping.models_torch as models_torch
from input_mapping import metric_mapping
from ai_backend.loggers.model_logger import is_min
from ai_backend.utils import calc_best_thresholds
import argparse
from torch.utils.data import ConcatDataset, DataLoader
import json
import os
import torch
import wandb
import numpy as np
from uuid import uuid4
# #create a model
# model = torch.nn.Linear(10, 1)
# project_name = 'test_project'
# run_name = 'test_run'
# run_id = str(uuid4())

# config = {'learning_rate' : 0.01, 'batch_size' : 32}
# tags = ['test']
# #create dict with some dummy score results like accuracy, f1, etc., which goes like train -> epoch, score_name
# train_results = {'train' : {'loss' : 0.9}, 'val' : {'loss' : 0.8}, 'epoch' : 1}
# val_test_score_results = {'val' : {'accuracy' : 0.9, 'f1' : 0.8}, 'test' : {'accuracy' : 0.9, 'f1' : 0.8}}
# dummy_subject = DummySubject()
# is_watching = False
# watch_log_freq = 0

# #get the api key from the json file
# with open('wandb_config.json') as f:
#     api_key = json.load(f)['api_key']
# wandb.login(key=api_key)
# wandb_observer = WandbObserver(model, project_name, run_name, run_id, config, tags, is_watching, watch_log_freq)


# #attach the observer
# dummy_subject.attach(wandb_observer)
# for i in range(10):
#     #update the epoch
#     train_results['epoch'] = i
#     dummy_subject.update(train_results)

# dummy_subject.update(val_test_score_results)
# wandb.finish()


train_dataset_paths = ['datasets/2024-05-17_12-13-57/train/ODIR-5K.json', 'datasets/2024-05-17_12-13-57/train/RFMiD2_13c5.json', 'datasets/2024-05-17_12-13-57/train/RIPS.json']
validation_dataset_paths = ['datasets/2024-05-17_12-13-57/val/ODIR-5K.json', 'datasets/2024-05-17_12-13-57/val/RFMiD2_7290.json', 'datasets/2024-05-17_12-13-57/val/RIPS.json']
test_dataset_paths = ['datasets/2024-05-17_12-13-57/test/UKB.json']

# Load data packages
train_packages = [DataPackage.load(path) for path in train_dataset_paths]
validation_packages = [DataPackage.load(path) for path in validation_dataset_paths]
test_packages = [DataPackage.load(path) for path in test_dataset_paths]
file_reader_matcher = lambda x: 'dicom' if 'UKB' in x.split('/')[-1] else 'default'

unique_train_labels = np.unique(np.concatenate([package.get_labels() for package in train_packages], axis=0), axis=0)
#print the unique train labels
print(unique_train_labels)
print('-----------------')
print(np.unique(np.concatenate([package.get_labels() for package in test_packages], axis=0), axis=0))
#filter the test packages by the unique train labels
test_packages = filter_data_packages_by_labels(test_packages, unique_train_labels)
print('-----------------')
#print the unique test labels
print(np.unique(np.concatenate([package.get_labels() for package in test_packages], axis=0), axis=0))

train_file_readers = [file_reader_matcher(path) for path in train_dataset_paths]
validation_file_readers = [file_reader_matcher(path) for path in validation_dataset_paths]
test_file_readers = [file_reader_matcher(path) for path in test_dataset_paths]


#get the transform for the model
transform_type = 'standard'
transform_config = models_torch.model_dict['resnet18']['transforms_config']
transform = get_transforms(transform_name = transform_type, transforms_config = transform_config)

#create the datasets
train_datasets = data_packages_to_datasets(train_packages, train_file_readers, [transform]*len(train_packages))
validation_datasets = data_packages_to_datasets(validation_packages, validation_file_readers, [transform]*len(validation_packages))
test_datasets = data_packages_to_datasets(test_packages, test_file_readers, [transform]*len(test_packages))
