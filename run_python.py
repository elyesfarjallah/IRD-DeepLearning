from run_wandb_tracked_training import main
import os
from uuid import uuid4
from pydicom import dcmread
from PIL import Image
import wandb
from data_pipeline.data_loading_utils import data_packages_to_datasets, filter_data_packages_by_labels
from data_pipeline.data_package import DataPackage
from data_pipeline.image_transforms import get_transforms
from torch.utils.data import ConcatDataset, DataLoader
import input_mapping.models_torch as models_torch
import torch
from itertools import chain, combinations

def power_set(s):
    """Generate the power set of the set s, excluding the empty set, and return a list of lists."""
    s = list(s)  # Ensure s is a list
    # Generate all subsets
    all_subsets = list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))
    # Convert each tuple to a list and exclude the empty set
    non_empty_subsets = [list(subset) for subset in all_subsets if subset]
    return non_empty_subsets

read_dicom = lambda x: dcmread(x).pixel_array
dicom_file_reader = lambda x: Image.fromarray(read_dicom(x)).convert('RGB')
default_file_reader = lambda x: Image.open(x).convert('RGB')
file_readers = {'dicom' : dicom_file_reader, 'default' : default_file_reader}

dataset_path = 'datasets/2024-06-05_16-22-01'
train_data = os.listdir(f'{dataset_path}/train')
print(train_data)
#create the power set of the train datasets
train_dataset_groups = power_set(train_data)
n_times = 3
for _ in range(n_times):
    for train_data in train_dataset_groups:
        train_paths = [f'{dataset_path}/train/{path}' for path in train_data]
        val_paths = [f'{dataset_path}/val/{path}' for path in train_data]
        dataset_config_path = f"{dataset_path}/dataset_config.json"
        model_key = 'resnet18'
        transform_type = 'standard'
        wandb_run_tags = '_'.join(train_data).replace('.json', '')
        arguments = ['--model_key', model_key, '--transform_type', transform_type, '--train_dataset_paths', *train_paths, '--validation_dataset_paths', *val_paths, '--wandb_project_name', 'zero_shot_performance_augmented_train_data',
                    '--dataset_config_path', dataset_config_path, '--lr', '0.00009354747253832916', '--batch_size', '128',
                    '--pretrained', 'True', '--augmentation', 'True', '--shuffle', 'True', '--epochs', '50', '--wandb_run_id', str(uuid4()), '--wandb_run_tags', wandb_run_tags]

        trainer, model, unique_train_labels = main(arguments)
        test_data = os.listdir(f'{dataset_path}/test')
        test_dataset_groups = power_set(test_data)
        for test_group in test_dataset_groups:
            #create the paths for the test datasets
            test_paths = [f'{dataset_path}/test/{dataset}' for dataset in test_group]
            test_packages = [DataPackage.load(path) for path in test_paths]
            file_reader_matcher = lambda x: 'dicom' if 'UKB' in x.split('/')[-1] else 'default'
            test_file_readers = [file_readers[file_reader_matcher(path)] for path in test_paths]
            transform_config = models_torch.model_dict[model_key]['transforms_config']
            transform = get_transforms(transform_name = transform_type, transforms_config = transform_config)
            #filter the test packages by the unique train labels
            test_packages = filter_data_packages_by_labels(test_packages, unique_train_labels)
            test_datasets = data_packages_to_datasets(test_packages, test_file_readers, [transform]*len(test_packages))
            test_dataset = ConcatDataset(test_datasets)
            num_workers = 96
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=num_workers)
            #do the same for the test data
            trainer.clear_results()
            test_data = trainer.test(model=model, test_loader=test_loader, device=device)
            joined_test_names = '_'.join(test_group).replace('.json', '')
            test_results = test_data['test']
            restructured_test_results = {'test' :{joined_test_names : test_results}}
            trainer.update_results(restructured_test_results)
            trainer.notify()
        wandb.finish()