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
from pydicom import dcmread
from PIL import Image
read_dicom = lambda x: dcmread(x).pixel_array
dicom_file_reader = lambda x: Image.fromarray(read_dicom(x)).convert('RGB')
default_file_reader = lambda x: Image.open(x).convert('RGB')
file_readers = {'dicom' : dicom_file_reader, 'default' : default_file_reader}
def main(raw_args = None):
    argparser = argparse.ArgumentParser(description='Convert user inputs to model, datasets and observers')
    argparser.add_argument('--transform_type', type=str, help='Transform type')
    argparser.add_argument('--train_dataset_paths', type=str, help='Path to train datasets', nargs='+')
    argparser.add_argument('--validation_dataset_paths', type=str, help='Path to validation datasets', nargs='+')
    argparser.add_argument('--wandb_project_name', type=str, help='Wandb project name')
    argparser.add_argument('--dataset_config_path', type=str, help='Path to dataset config')
    argparser.add_argument('--lr', type=float, help='Learning rate')
    argparser.add_argument('--batch_size', type=int, help='Batch size')
    #arguments with default values
    argparser.add_argument('--model_key', type=str, help='Model key', default='resnet18')
    argparser.add_argument('--pretrained', type=bool, help='Use pretrained model', default=True)
    argparser.add_argument('--augmentation', type=bool, help='Use data augmentation', default=True)
    argparser.add_argument('--shuffle', type=bool, help='Shuffle data', default=True)
    argparser.add_argument('--epochs', type=int, help='Number of epochs', default=30)
    argparser.add_argument('--wandb_run_id', type=str, help='Wandb run id', default=str(uuid4()))


    args = argparser.parse_args(raw_args)

    train_dataset_paths = args.train_dataset_paths
    validation_dataset_paths = args.validation_dataset_paths

    # Load data packages
    train_packages = [DataPackage.load(path) for path in train_dataset_paths]
    validation_packages = [DataPackage.load(path) for path in validation_dataset_paths]
    file_reader_matcher = lambda x: 'dicom' if 'UKB' in x.split('/')[-1] else 'default'

    unique_train_labels = np.unique(np.concatenate([package.get_labels() for package in train_packages], axis=0), axis=0)

    

    train_file_readers = [file_readers[file_reader_matcher(path)] for path in train_dataset_paths]
    validation_file_readers = [file_readers[file_reader_matcher(path)] for path in validation_dataset_paths]


    #get the transform for the model
    transform_type = args.transform_type
    transform_config = models_torch.model_dict[args.model_key]['transforms_config']
    transform = get_transforms(transform_name = transform_type, transforms_config = transform_config)

    #create the datasets
    train_datasets = data_packages_to_datasets(train_packages, train_file_readers, [transform]*len(train_packages))
    validation_datasets = data_packages_to_datasets(validation_packages, validation_file_readers, [transform]*len(validation_packages))
    

    #exlude the labels that are not in the train labels from the test datasets

    for dataset in train_datasets:
        dataset.balance_augmentation()
    #concatenate the datasets
    train_dataset = ConcatDataset(train_datasets)
    validation_dataset = ConcatDataset(validation_datasets)
    

    #load the dataset config
    with open(args.dataset_config_path, 'r') as f:
        dataset_config = json.load(f)

    #get the labels
    label_names = dataset_config['labels_to_encode']

    #create the model, dataloaders
    model = models_torch.get_model(model_name=args.model_key, num_classes=len(label_names), pretrained=args.pretrained)

    num_workers = 111
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=num_workers)
    validation_loader = DataLoader(validation_dataset, batch_size=512, shuffle=False, num_workers=num_workers)
    

    #create a run config
    run_config = {'model_key' : args.model_key, 'transform_type' : transform_type, 'train_dataset_paths' : train_dataset_paths,
                'validation_dataset_paths' : validation_dataset_paths,
                'train_dataset_size' : len(train_dataset),
                    'validation_dataset_size' : len(validation_dataset),
                    'lr' : args.lr, 'transform_type' : transform_type, 'pretrained' : args.pretrained,
                    'augmentation' : args.augmentation, 'shuffle' : args.shuffle}
    #create the model save path
    model_save_folder = f'models/{args.model_key}/{args.wandb_run_id}'
    model_save_path = f'{model_save_folder}/weights.pth'
    os.makedirs(model_save_folder, exist_ok=True)
    #save the run config
    with open(f'{model_save_folder}/run_config.json', 'w') as f:
        json.dump(run_config, f)
    #get the metrics
    averages = ['micro', 'macro']
    metric_names = ['accuracy', 'f1', 'precision', 'recall']
    averag_metric_names = [f'{metric}_{average}' for metric in metric_names for average in averages]
    averaging_metrics = metric_mapping.get_multilabel_metrics_by_names(metric_names=averag_metric_names)
    #append cm metric
    metric_names.append('confusion_matrix')
    classwise_metrics = [metric_mapping.get_classwise_metrics_by_names(metric_names=metric_names)]
    model_logger_criterrion = metric_mapping.get_multilabel_metric_by_name('bce_with_logits_loss')
    averaging_metrics = [averaging_metrics, [model_logger_criterrion]]
    #create observers
    trainer, model_logger, averaging_evaluators, classwise_evaluators, wandb_observer = create_subjects_and_observers(is_logging_to_wandb = True, project_name = args.wandb_project_name, run_name = args.wandb_run_id,
                                run_id = args.wandb_run_id, config = run_config, tags=[], watch_gradients = False, gradients_log_freq = 0,
                                model = model, averaging_metrics = averaging_metrics, model_logger_evaluation_function = is_min, model_logger_criterion = model_logger_criterrion,
                                    model_save_path = model_save_path, classwise_metrics = classwise_metrics, label_names = label_names)

    score_evaluator = averaging_evaluators[0]
    loss_evaluator = averaging_evaluators[1]
    #attach the logger and wandb observer to the trainer
    trainer.attach(model_logger)
    trainer.attach(loss_evaluator)
    loss_evaluator.attach(wandb_observer)

    #get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #train the model
    trainer.train(model=model, criterion=criterion, optimizer=optimizer, train_loader=train_loader, validation_loader=validation_loader,
                early_stopping=True, patience=15, min_delta_percentage=0.0, device=device, epochs=args.epochs)

    #detach the loss evaluator from thhe trainer
    trainer.detach(loss_evaluator)
    #detach the model logger from the trainer
    trainer.detach(model_logger)
    #load the best model#todo fix the model save path
    model.load_state_dict(torch.load(model_save_path))
    #validate the model
    validation_data = trainer.validate(model=model, validation_loader=validation_loader, device=device)
    y_true = validation_data['validation'][trainer.LABEL_KEY]
    y_pred = validation_data['validation'][trainer.PREDICTION_KEY]
    best_thresholds = calc_best_thresholds(y_true, y_pred, beta=1.0, step_size=0.01)

    #save the best thresholds
    thresholds_save_path = f'{model_save_folder}/best_thresholds.json'
    with open(thresholds_save_path, 'w') as f:
        json.dump(best_thresholds.tolist(), f)

    #attach the score and classwise evaluators to the trainer
    trainer.attach(score_evaluator)
    #todo fix that somehow the classwise_evaluators are metrics and not evaluators
    for classwise_evaluator in classwise_evaluators:
        for metric in classwise_evaluator.metrics:
                metric.set_thresholds(best_thresholds)
        trainer.attach(classwise_evaluator)
        classwise_evaluator.attach(wandb_observer)
    score_evaluator.attach(wandb_observer)
    #todo set the best thresholds as the threshold for the metrics in the evaluators
    #evaluate the model
    trainer.update_results(validation_data)
    trainer.notify()

    return trainer, model, unique_train_labels

if __name__ == '__main__':
     main()