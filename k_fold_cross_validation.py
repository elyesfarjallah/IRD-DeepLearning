import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from ai_backend.model_executors.trainer import Trainer
from ai_backend.loggers.model_logger import is_min, ModelLogger
from ai_backend.loggers.wandb_logger import WandbObserver
from ai_backend.evaluators.evaluator import Evaluator
from ai_backend.evaluators.metrics.loss_metrics import LossMetricWrapper
from ai_backend.evaluators.multilabel_evaluator import MultiLabelEvaluator
from ai_backend.evaluators.metrics.multi_label_metrics import multi_label_f_beta

from input_mapping.metric_mapping import get_classwise_metrics_by_names, get_multilabel_metrics_by_names

import argparse
import re
import os
from convert_user_inputs import create_datasets,  convert_user_input, create_subjects_and_observers
from uuid import uuid4
import json
import wandb
import logging



def single_fold_validation(model: nn.Module, criterion: nn.Module, train_dataloader: DataLoader, validation_dataloader: DataLoader, trainer: Trainer, n_train_epochs: int, n_epochs_validation: int,
                               model_logger: ModelLogger, train_evaluator: Evaluator, validation_evaluator: Evaluator, validation_evaluator_multilabel: MultiLabelEvaluator, device: torch.device,
                               is_logging_to_wandb: bool, wandb_logger: WandbObserver):
    
    #attach the observers
    train_evaluator = Evaluator(metrics=loss_metrics)
    trainer.attach(model_logger)
    trainer.attach(train_evaluator)
    
    #initialize the wandb logger
    if is_logging_to_wandb:
        train_evaluator.attach(wandb_logger)
        validation_evaluator_multilabel.attach(wandb_logger)
        validation_evaluator.attach(wandb_logger)
        
    trainer.train(model=model, criterion=criterion, optimizer=optimizer, train_loader=train_dataloader,
                    validation_loader=validation_dataloader, epochs=n_train_epochs, n_epochs_validation=n_epochs_validation,
                    device=device, early_stopping=True, patience=15, min_delta_percentage=0.1)
    
    #load the best model
    model.load_state_dict(torch.load(best_model_save_path))
    #find the best threshold
    #detach the evaluators
    trainer.detach(train_evaluator)
    trainer.detach(model_logger)
    best_thresholds = find_operating_point(model=model, validation_dataloader=validation_dataloader, device=device)
    #todo: find a way to attach the evaluators while passing the best thresholds to the metrics they contain
    
    #attach the validation evaluators
    trainer.attach(validation_evaluator)
    trainer.attach(validation_evaluator_multilabel)
    for metric in validation_evaluator_multilabel.metrics:
        metric.set_thresholds(best_thresholds)
    for metric in validation_evaluator.metrics:
        metric.set_thresholds(best_thresholds)
    #evaluate the model
    process_key = 'validation_best_thresholds'
    trainer.clear_results()
    results = trainer.no_grad_evaluate(model=model,process_key=process_key, data_loader=validation_dataloader, device=device)
    trainer.update_results(results)
    trainer.notify()
    return model, results
    
def find_operating_point(model: nn.Module, validation_dataloader: DataLoader, device: torch.device):
    trainer = Trainer()
    results = trainer.no_grad_evaluate(model=model, data_loader=validation_dataloader, device=device, process_key=None)
    y_true, y_pred = results[trainer.LABEL_KEY], results[trainer.PREDICTION_KEY]
    #find the best threshold for each class
    possible_thresholds = np.linspace(0, 1, 100)
    threshold_results = []
    for possible_threshold in possible_thresholds:
        threshold_results.append(multi_label_f_beta(y_true, y_pred, threshold=possible_threshold))
    threshold_results = np.array(threshold_results)
    #find the best threshold for each class
    best_thresholds = []
    for i in range(threshold_results.shape[-1]):
        scores = threshold_results[:, i]
        best_thresholds.append(possible_thresholds[np.argmax(scores)])
    return best_thresholds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--model_key', type=str, help='The model key', default='resnet18')
    parser.add_argument('--n_train_epochs', type=int, help='The number of training epochs', default=2)
    parser.add_argument('--batch_size', type=int, help='The batch size', default=128)
    parser.add_argument('--lr', type=float, help='The learning rate', default=0.001)
    parser.add_argument('--dataset_path', type=str, help='The path to the dataset', default= 'datasets_k_fold/2024-02-20_23-50-58')
    parser.add_argument('--device', type=str, help='The device to use for training', default= 'cuda:1')
    parser.add_argument('--transform_type', type=str, help='The type of transform to use', default= 'standard')
    parser.add_argument('--augmentation', action='store_true', help='Whether to use augmentation')
    #python k_fold_cross_validation.py --model_key resnet18 --n_train_epochs 60 --batch_size 128 --lr 0.00009354747253832916 --dataset_path datasets_k_fold/2024-02-20_23-50-58 --device cuda --transform_type standard --augmentation
    #python k_fold_cross_validation.py --model_key resnet18 --n_train_epochs 60 --batch_size 16 --lr 0.00004614948033730265 --dataset_path datasets_k_fold/2024-02-20_23-50-58 --device cuda --transform_type ben --augmentation
    args = parser.parse_args()
    device = torch.device(args.device)
    #read the wandb config
    with open('wandb_config.json', 'r') as f:
        wandb_config = json.load(f)
    #login to wandb
    wandb.login(key=wandb_config['api_key'])
    #check how many folders are in the dataset path that match the name patterrn ^fold_[0-9]+$
    fold_folder_name_pattern = re.compile(r'^fold_[0-9]+$')
    fold_folders = [folder for folder in os.listdir(args.dataset_path) if fold_folder_name_pattern.match(folder)]
    run_id = str(uuid4())
    #read in the dataset config
    with open(f'{args.dataset_path}/dataset_config.json', 'r') as f:
        dataset_config = json.load(f)
    label_names = dataset_config['label_names']
    data_column_name = dataset_config['path_to_img_column']
    #create the datasets
    for folder in fold_folders:
        train_dataset_path = f'{args.dataset_path}/{folder}/train.csv'
        validation_dataset_path = f'{args.dataset_path}/{folder}/validation.csv'
        test_dataset_path = f'{args.dataset_path}/All_test.csv'
        train_dataset, validation_datsaset, test_dataset = create_datasets(model_key=args.model_key, transform_type=args.transform_type,
                                                                           train_dataset_path=train_dataset_path, validation_dataset_path=validation_dataset_path,
                                                                          test_dataset_path=test_dataset_path, path_to_img_column=data_column_name, label_cols=label_names, augmentation=args.augmentation, shuffle=True)
        #create the dataloaders
        model, train_loader, validation_loader, test_dataloader, n_classes, model_dataset_config = convert_user_input(model_key=args.model_key, train_dataset=train_dataset, validation_dataset=validation_datsaset, test_dataset=test_dataset,
                                                                                                                        dataset_path=args.dataset_path, augmentation=args.augmentation, pretrained=True, shuffle=True, lr=args.lr,
                                                                                                                        batch_size=args.batch_size, transform_type=args.transform_type)
        #run a fold validation
        fold_number = int(folder.split('_')[1])
        #add a fold number to the model dataset config and the run_id of the fold validation
        model_dataset_config['fold_number'] = fold_number
        model_dataset_config['run_id'] = run_id
        model_dataset_config['patience'] = 15
        #create the model save path
        train_id = str(uuid4())
        run_name = f'{args.model_key}_{train_id}'
        model_data_path = f'models/{args.model_key}/{train_id}/'
        best_model_save_path = f'{model_data_path}best_model.pt'
        #run the fold validation
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        #todo: check how to perform k fold validation
        criterion = torch.nn.BCEWithLogitsLoss()
        model_logger_criterion = torch.nn.BCEWithLogitsLoss()
        validation_metrics = get_classwise_metrics_by_names(['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix'])
        all_classwise_metrics = [validation_metrics]
        validation_averaging_metrics = get_multilabel_metrics_by_names(['accuracy_micro', 'precision_micro', 'recall_micro', 'f1_micro', 'accuracy_macro', 'precision_macro', 'recall_macro', 'f1_macro'])
        loss_metrics = get_multilabel_metrics_by_names(['bce_with_logits_loss'])
        averaging_metrics =[validation_averaging_metrics, loss_metrics]

        trainer, model_logger, averaging_evaluators, classwise_evaluators, wandb_observer = create_subjects_and_observers(model=model,run_id=train_id, run_name=run_name, project_name='k_fold_cross_validation',config=model_dataset_config, tags=[],
                                                                                                                            is_logging_to_wandb=True,watch_gradients=True, gradients_log_freq=10,model_logger_criterion=model_logger_criterion,
                                                                                                                              model_logger_evaluation_function=is_min,model_save_path=best_model_save_path,averaging_metrics=averaging_metrics,
                                                                                                                                classwise_metrics=all_classwise_metrics,label_names=train_dataset.classes)
        
        train_evaluator = averaging_evaluators[1]
        validation_evaluator = classwise_evaluators[0]
        validation_evaluator_multilabel = averaging_evaluators[0]
        #add forward hooks to the model
        for name, module in model.named_modules():
            re_pattern = re.compile(r'^layer\d+$')
            if re_pattern.match(name) is not None:
                print('Adding forward hook for:', name)
                module.register_forward_hook(lambda module, input,
                                            output: torch.nn.functional.dropout2d(output, p=0.2, training=module.training))
        #set logging level
        logging.basicConfig(level=logging.INFO)
        single_fold_validation(model=model, criterion=criterion, train_dataloader=train_loader, validation_dataloader=validation_loader, trainer=trainer, n_train_epochs=args.n_train_epochs, n_epochs_validation=1,
                        model_logger=model_logger, train_evaluator=train_evaluator, validation_evaluator=validation_evaluator, validation_evaluator_multilabel=validation_evaluator_multilabel, device=device,
                        is_logging_to_wandb=True, wandb_logger=wandb_observer)
        trainer.clear_results()
        test_results = trainer.test(model=model, test_loader=test_dataloader, device=device)
        
        trainer.update_results(test_results)
        trainer.notify()
        
        wandb.finish()

