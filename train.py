from convert_user_inputs import convert_user_input, create_observer_structure, create_datasets
from ai_backend.loggers.model_logger import is_min
from input_mapping.metric_mapping import get_multilabel_metrics_by_names, get_classwise_metrics_by_names
from uuid import uuid4
import torch.nn as nn
from torch.optim import Adam, AdamW
import torch
import json
import wandb
import optuna
import joblib
import argparse
from datetime import datetime
import os


#parse the arguments
date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--model_key', type=str, help='The model key')
parser.add_argument('--lr_min', type=float, help='The minimum learning rate')
parser.add_argument('--lr_max', type=float, help='The maximum learning rate')
parser.add_argument('--batch_size_options', type=int, nargs='+', help='The batch size options')
parser.add_argument('--device', type=str, help='The device to train on')
#parse the arguments
args = parser.parse_args()
model_key = args.model_key
#read in json file with the label columns
label_names_path  = 'datasets/2024-01-31_11-09-02/dataset_config.json'
with open(label_names_path) as json_file:
    dataset_config = json.load(json_file)
label_cols = dataset_config['label_names']
path_to_img_column = dataset_config['path_to_img_coulumn']
#read wandb config
wandb_config_path = 'wandb_config.json'
with open(wandb_config_path) as json_file:
    wandb_config = json.load(json_file)
api_key = wandb_config['api_key']
project_name = f'{model_key}_{date}_multilabel_classification'#wandb_config['project_name']
#initialize wand
wandb.login(key=api_key)
train_dataset, validation_dataset, test_dataset = create_datasets(model_key = model_key, transform_type = 'ben',train_dataset_path = 'datasets/2024-01-31_11-09-02/All_train.csv', validation_dataset_path = 'datasets/2024-01-31_11-09-02/All_validation.csv',
                                                                   test_dataset_path = 'datasets/2024-01-31_11-09-02/All_test.csv', path_to_img_column = path_to_img_column, label_cols = label_cols, augmentation = True, shuffle = True)



def  objective(trial : optuna.Trial):
    #get the hyperparameters
    batch_size = trial.suggest_categorical('batch_size', args.batch_size_options)
    transform_type = trial.suggest_categorical('transform_type', ['ben', 'standard'])
    lr = trial.suggest_float('lr', args.lr_min, args.lr_max, log=True)
    
    model, train_loader, validation_loader, test_dataloader, n_classes, model_dataset_config = convert_user_input(model_key = model_key, train_dataset=train_dataset, validation_dataset=validation_dataset, test_dataset=test_dataset,
                                                                                                                   dataset_path = 'datasets/2024-01-31_11-09-02/', augmentation = True,  pretrained = True, shuffle=True, lr=lr,
                                                                                                                   batch_size = batch_size, transform_type = transform_type)
    criterion = nn.BCEWithLogitsLoss()
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])#torch.optim.Adam(model.parameters(), lr=lr)
    if optimizer_name == 'Adam':
        optimizer = Adam(model.parameters(), lr=lr)
    else:
        optimizer = AdamW(model.parameters(), lr=lr)
    model_dataset_config['optimizer'] = optimizer_name
    #generate a run id for the current run
    run_id = str(uuid4())
    run_name = f'{model_key}_{run_id}'
    model_data_path = f'models/{model_key}/{run_id}/'
    model_save_path = f'{model_data_path}best_model.pt'
    train_dataset_save_path = f'{model_data_path}train_dataset.pt'

    logger_criterion = nn.BCEWithLogitsLoss()
    #get the metrics
    metric_keys = ['accuracy_micro', 'accuracy_macro', 'f1_micro', 'f1_macro', 'precision_micro', 'precision_macro', 'recall_micro', 'recall_macro', 'bce_with_logits_loss']
    metrics = get_multilabel_metrics_by_names(metric_keys)
    classwise_metric_keys = ['accuracy', 'f1', 'precision', 'recall', 'confusion_matrix']
    classwise_metrics = get_classwise_metrics_by_names(classwise_metric_keys)
    run_tags = [str(model.__class__.__name__)]
    trainer, model_logger, evaluator, wandb_observer = create_observer_structure(is_logging_to_wandb = True, project_name = project_name, run_name = run_name, run_id = run_id,
                                                                                config = model_dataset_config, tags = run_tags, watch_gradients = True, gradients_log_freq = 5,
                                                                                model = model, metrics = metrics, model_logger_evaluation_function = is_min, model_logger_criterrion = logger_criterion,
                                                                                model_save_path = model_save_path, classwise_metrics = classwise_metrics, label_names = label_cols)

    
    
    trainer.train(model=model, criterion=criterion, optimizer=optimizer, train_loader=train_loader, validation_loader=validation_loader,
                   n_epochs_validation=1, epochs=30, prefered_device=args.device, early_stopping=True, patience=10, min_delta_percentage=0.1)
    #load the best model
    best_state_dict = torch.load(model_save_path)
    model.load_state_dict(best_state_dict)
    trainer.test(model=model,test_loader=test_dataloader, prefered_device=args.device)

    #save the datasets in the dataloaders as a file
    torch.save(train_loader.dataset, train_dataset_save_path)
    train_loader.dataset.df.to_csv(f'{model_data_path}train_dataset.csv', index=False)
    wandb.finish()
    return model_logger.get_best_value()


class Save_Study_Callback:
    def __init__(self, study_save_path):
        self.study_save_path = study_save_path

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        #check if the study save path exists
        if not os.path.exists(self.study_save_path):
            os.makedirs(self.study_save_path)
        joblib.dump(study, f"{self.study_save_path}{study.study_name}.pkl")

#create a study

#optimize the study
#save the study
study_id = str(uuid4())
study_name = f'{date}_{study_id}'
study_path = f'models/{model_key}/studies'

study = optuna.create_study(direction='minimize', study_name=study_name)
#load the study
study_save_path = 'models/resnet18/studies/'
#study = joblib.load('models/resnet18/studies/ce90cdd0-374e-47af-a13d-e97063661b14.pkl')
save_callback = Save_Study_Callback(study_save_path)
study.optimize(objective, n_trials=100, show_progress_bar=True, callbacks=[save_callback])