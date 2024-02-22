import optuna
import torch
import numpy as np
import pandas as pd
from data_pipeline.data_loading import DfDataset
import json
from uuid import uuid4
import datetime
import joblib
from convert_user_inputs import convert_user_input, create_observer_structure
from ai_backend.loggers.model_logger import is_min
from input_mapping.criterion_mapping import get_criterion_by_name
from input_mapping.metric_mapping import get_metrics_by_names, get_multilabel_metrics_by_names
import os
import wandb
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class Objective:
    def __init__(self, model_key : str, n_epochs : int, criterion_key : str,
                  train_dataset_path : str, validation_dataset_path : str, path_to_img_column : str, label_cols : list, metric_keys : list,
                  wandb_project_name : str,
                    run_tags : list = [], n_epochs_validation : int = 1, prefered_device : str = 'cuda:0',
                      batch_size_options : list = [4, 8, 16, 32], lr_min : float = 1e-5, lr_max : float = 1e-1,
                      early_stopping : bool = True, min_delta_percentage : float = 0.0, patience : int = 10,
                        augmentation : bool = False, pretrained : bool = False, is_logging_to_wandb : bool = True, watch_gradients : bool = True):
        # Hold this implementation specific arguments as the fields of the class.
        self.model_key = model_key
        self.n_epochs = n_epochs
        self.run_tags = run_tags
        self.train_dataset_path = train_dataset_path
        self.validation_dataset_path = validation_dataset_path
        self.path_to_img_column = path_to_img_column
        self.label_cols = label_cols
        self.metric_keys = metric_keys
        self.n_epochs_validation = n_epochs_validation
        self.prefered_device = prefered_device
        self.batch_size_options = batch_size_options
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.min_delta_percentage = min_delta_percentage
        self.patience = patience
        self.augmentation = augmentation
        self.pretrained = pretrained
        self.early_stopping = early_stopping
        self.criterion_name = criterion_key
        self.is_logging_to_wandb = is_logging_to_wandb
        self.watch_gradients = watch_gradients
        self.wandb_project_name = wandb_project_name


    def __call__(self, trial):
        # Calculate an objective value by using the extra arguments.
        #suggest a value for the hyperparameter
        lr = trial.suggest_float("lr", self.lr_min, self.lr_max, log=True)
        batch_size = trial.suggest_categorical("batch_size", self.batch_size_options)
        

        run_id = str(uuid4())
        tags = []
        tags.extend(self.run_tags)
        run_name = f'{self.model_key}_{run_id}'
        trainings_start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        best_weights_save_path = f'models/{self.model_key}/{trainings_start_time}_{self.model_key}_{run_id}.pt'
        #create user inputs
        model, train_loader, validation_loader, n_classes, config, test_data_loader = convert_user_input(model_key=self.model_key, train_dataset_path=self.train_dataset_path,
                                                                                        path_to_img_column=self.path_to_img_column, label_cols=self.label_cols,
                                                                                            validation_dataset_path=self.validation_dataset_path,
                                                                                              augmentation=self.augmentation, batch_size=batch_size, pretrained=self.pretrained)
        tags = [model.__class__.__name__, self.model_key]
        #create observer structure
        model_logger_criterrion = get_criterion_by_name('CrossEntropyLoss')
        #prepare training
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = get_criterion_by_name(self.criterion_name)
        optimizer_class_name = optimizer.__class__.__name__
        criterion_class_name = criterion.__class__.__name__
        config.update({'batch_size' : batch_size, 'lr' : lr, 'epochs' : self.n_epochs, 'n_epochs_validation' : self.n_epochs_validation,
                       'criterion' : criterion_class_name, 'optimizer' : optimizer_class_name, 'device' : self.prefered_device, 'n_classes' : n_classes,
                         'model_name' : self.model_key,
                        'early_stopping' : self.early_stopping, 'patience' : self.patience, 'min_delta_percentage' : self.min_delta_percentage, 'model_name' : self.model_key,
                          'dataset_path' : self.train_dataset_path, 'best_weights_save_path' : best_weights_save_path})
        
        #create list with metrics
        metrics = get_multilabel_metrics_by_names(self.metric_keys)
        trainer, model_logger, evaluator, wandb_observer  = create_observer_structure(is_logging_to_wandb=self.is_logging_to_wandb, project_name=self.wandb_project_name,
                                                                                      run_name=run_name, run_id=run_id, config=config,
                                                                                        tags=tags, watch_gradients=self.watch_gradients, gradients_log_freq=10,
                                                                                          model=model, metrics=metrics, model_logger_evaluation_function=is_min,
                                                                                            model_logger_criterrion=model_logger_criterrion,
                                                                                              model_save_path=best_weights_save_path)
        #train the model
        trainer.train(model=model, epochs=self.n_epochs, train_loader=train_loader, validation_loader=validation_loader,
                       criterion=criterion, optimizer=optimizer,
                         n_epochs_validation=self.n_epochs_validation, early_stopping=self.early_stopping, patience=self.patience, min_delta_percentage=self.min_delta_percentage,
                         device=self.prefered_device)
        
        #sload test data
        
        #get all test predictions
        y_true_all = []
        y_pred_all = []
        device = torch.device('cpu')
        #move model to cpu
        model.to(device)
        for batch in test_data_loader:
            x, y_true = batch
            x = x.to(device)
            y_true = y_true.to(device)
            y_true_all.append(y_true)
            y_pred = model(x)
            y_pred_all.append(y_pred)
        y_true_all = torch.cat(y_true_all, dim=0)
        y_pred_all = torch.cat(y_pred_all, dim=0)
        #convert to numpy
        #move to cpu
        device = torch.device('cpu')
        y_true_np = y_true_all.to(device)
        y_true_np = y_true_np.int().detach().numpy()
        y_pred_np = y_pred_all.to(device)
        y_pred_np = torch.sigmoid(y_pred_np)
        y_pred_np = y_pred_np.int().detach().numpy() > 0.5
        #get confusion matrix
        confusion_matrix = multilabel_confusion_matrix(y_true_np, y_pred_np)
        #save all confusion matrices in model_key/run_id/confusion_matrices/label_name.png
        confusion_matrices_save_path = f'models/{self.model_key}/{run_id}/confusion_matrices/'
        if not os.path.exists(confusion_matrices_save_path):
            os.makedirs(confusion_matrices_save_path)
        for label_name, cm in zip(self.label_cols, confusion_matrix):
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
            disp.plot(include_values=True, cmap='Blues', ax=None, xticks_rotation='horizontal', values_format=None)
            plt.title(f'Confusion Matrix for {label_name}')
            plt.savefig(f'{confusion_matrices_save_path}/{label_name}.png')
            plt.close()
        wandb.finish()
        return model_logger.get_best_value()
class Save_Study_Callback:
    def __init__(self, study_save_path):
        self.study_save_path = study_save_path

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        joblib.dump(study, f"{self.study_save_path}/{study.study_name}.pkl")


def optimize_model(model_key : str, n_epochs: int,
                   train_dataset_path : str, validation_dataset_path, path_to_img_column : str, label_cols : list, criterion_key : str, metric_keys : list,
                   wandb_project_name : str,
                     n_epochs_validation : int, n_trials:int, prefered_device:str = 'cuda:0',
                       batch_size_options : list = [4, 8, 16, 32], lr_min : float = 1e-5, lr_max : float = 1e-1,
                         early_stopping : bool = True, min_delta_percentage : float = 0.0, patience : int = 10,
                         pretrained : bool = False, augmentation : bool = False, run_tags : list = [], is_logging_to_wandb : bool = True, project_name : str = None, watch_gradients : bool = True):
    
    run_tags = []
    start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    study_name = f'{start_time}_study_{str(uuid4())}'
    study = optuna.create_study(direction='minimize', study_name=study_name)
    if pretrained:
        run_tags.append('pretrained')
    study_save_path = f'models/{model_key}/studies/'
    if not os.path.exists(study_save_path):
        os.makedirs(study_save_path)
    objective = Objective(model_key=model_key, n_epochs=n_epochs, criterion_key=criterion_key,
                           train_dataset_path=train_dataset_path, validation_dataset_path=validation_dataset_path,
                              path_to_img_column=path_to_img_column, label_cols=label_cols,
                                metric_keys=metric_keys, wandb_project_name=wandb_project_name, run_tags=run_tags,
                                  n_epochs_validation=n_epochs_validation, prefered_device=prefered_device,
                                    batch_size_options=batch_size_options, lr_min=lr_min, lr_max=lr_max,
                                      early_stopping=early_stopping, min_delta_percentage=min_delta_percentage, patience=patience,
                                        augmentation=augmentation, pretrained=pretrained, is_logging_to_wandb=is_logging_to_wandb, watch_gradients=watch_gradients)
    callback_save_study = Save_Study_Callback(study_save_path)
    study.optimize(objective,
                    n_trials=n_trials, callbacks=[callback_save_study], n_jobs=1)


def test_optimize_model():
    #test optimize model
    with open('wandb_config.json', 'r') as f:
        config = json.load(f)
    wandb.login(key=config['api_key'])
    project_name = config['project_name']
    label_cols = ['Maculopathy','Myopia','cataract','Diabetic Retinopathy','Bietti crystalline dystrophy','Best Disease','Cone Dystrophie or Cone-rod Dystrophie','Age-related Macular Degeneration',
                  'Stargardt Disease','Normal','Retinitis Pigmentosa','glaucoma']
    path_to_img_column = 'path_to_img'
    optimize_model(model_key='resnet152', n_epochs=15, criterion_key='BCEWithLogitsLoss',
                    train_dataset_path='datasets/2024-01-23_15-07-30/train.csv', validation_dataset_path='datasets/2024-01-23_15-07-30/validation.csv',
                      path_to_img_column=path_to_img_column, label_cols=label_cols,
                      metric_keys=['precision_micro', 'precision_macro', 'recall_micro', 'recall_macro', 'f1_micro', 'f1_macro', 'bce_with_logits_loss'],
                    wandb_project_name='test', n_epochs_validation=1, n_trials=5, prefered_device='cuda:0',
                      batch_size_options=[32], lr_min=1e-4, lr_max=1e-1, early_stopping=True, min_delta_percentage=0.1,
                      patience=10, pretrained=True, augmentation=True, run_tags=['test'], is_logging_to_wandb=True, project_name=None, watch_gradients=True)
    wandb.finish()
