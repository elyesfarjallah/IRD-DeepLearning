from ai_backend.model_executors.trainer import Trainer
from ai_backend.evaluators.evaluator import Evaluator
from ai_backend.evaluators.multilabel_evaluator import MultiLabelEvaluator
from ai_backend.loggers.wandb_logger import WandbObserver
from ai_backend.dummy_observer import DummyObserver
import input_mapping.models_torch as models_torch
from data_pipeline.image_transforms import get_transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from data_pipeline.data_loading import DfDataset
from torch import nn
from ai_backend.loggers.wandb_utils import count_parameters, count_learnable_parameters
from ai_backend.loggers.model_logger import ModelLogger
import pandas as pd

def create_datasets(model_key : str, transform_type : str,train_dataset_path : str, validation_dataset_path : str, test_dataset_path : str, path_to_img_column : str, label_cols : list, augmentation: bool, shuffle : bool = False):
    #load dataset
    transforms_config = models_torch.model_dict[model_key]['transforms_config']
    transform = get_transforms(transform_name = transform_type, transforms_config = transforms_config)
    train_df = pd.read_csv(train_dataset_path)
    train_dataset = DfDataset(df=train_df, data_path_col=path_to_img_column, label_cols=label_cols, transform=transform, augmentation=augmentation, shuffle=shuffle)
    validation_df = pd.read_csv(validation_dataset_path)
    validation_dataset = DfDataset(df=validation_df, data_path_col=path_to_img_column, label_cols=label_cols, transform=transform, augmentation=False)
    test_df = pd.read_csv(test_dataset_path)
    test_dataset = DfDataset(df=test_df, data_path_col=path_to_img_column, label_cols=label_cols, transform=transform, augmentation=False)
    return train_dataset, validation_dataset, test_dataset

def convert_user_input(model_key : str, train_dataset : Dataset, validation_dataset : Dataset, test_dataset : Dataset, dataset_path : str,
                        augmentation: bool, batch_size : int, pretrained : bool, transform_type : str, lr : float, shuffle : bool = False):
    #get transforms
    transforms_config = models_torch.model_dict[model_key]['transforms_config']
    transform = get_transforms(transform_name = transform_type, transforms_config = transforms_config)
    train_dataset.transform = transform
    validation_dataset.transform = transform
    test_dataset.transform = transform
    #create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    n_classes = len(train_dataset.classes)
    #load model
    model = models_torch.get_model(model_key, n_classes, pretrained= pretrained)
    #count parameters
    n_parameters = count_parameters(model)
    n_learnable_parameters = count_learnable_parameters(model)
    #create config
    model_dataset_config = {'model_name' : model_key, 'n_classes' : n_classes, 'batch_size' : batch_size,
                             'dataset_path' : dataset_path,
                               'train_dataset_size' : len(train_dataset), 'validation_dataset_size' : len(validation_dataset),'test_dataset_size': len(test_dataset),
                               'n_parameters' : n_parameters, 'n_learnable_parameters' : n_learnable_parameters, 'lr' : lr, 'transform_type' : transform_type, 'pretrained' : pretrained, 'augmentation' : augmentation, 'shuffle' : shuffle}
    
    return model, train_loader, validation_loader, test_dataloader, n_classes, model_dataset_config.copy()


def create_observer_structure(is_logging_to_wandb : bool, project_name : str, run_name : str,
                               run_id : str, config : dict, tags : list, watch_gradients : bool,gradients_log_freq : int,
                               model, metrics : list, model_logger_evaluation_function : callable, model_logger_criterrion : nn.Module,
                                 model_save_path : str, classwise_metrics : list = None, label_names : list = None):
    evaluators = []
    average_evaluator = Evaluator(metrics=metrics)
    evaluators.append(average_evaluator)
    model_logger = ModelLogger(model=model, criterion=model_logger_criterrion, evaluation_function=model_logger_evaluation_function,
                                save_path=model_save_path)
    trainer = Trainer()
    trainer.attach(model_logger)
    if classwise_metrics is not None and label_names is not None:
        classwise_evaluator = MultiLabelEvaluator(metrics=classwise_metrics, label_names=label_names)
        evaluators.append(classwise_evaluator)
    
    wandb_observer = None
    if is_logging_to_wandb:
        wandb_observer = WandbObserver(project_name=project_name, model=model,
                                    run_id=run_id, run_name = run_name, config=config, tags=tags,
                                    is_watching=watch_gradients, watch_log_freq=gradients_log_freq)
        #attach the wandb observer to every evaluator
        for evaluator in evaluators:
            evaluator.attach(wandb_observer)
    #attach the evaluators to the trainer
    for evaluator in evaluators:
        trainer.attach(evaluator)
    
    return trainer, model_logger, evaluators, wandb_observer

def create_subjects_and_observers(is_logging_to_wandb : bool, project_name : str, run_name : str,
                               run_id : str, config : dict, tags : list, watch_gradients : bool,gradients_log_freq : int,
                               model, averaging_metrics : list, model_logger_evaluation_function : callable, model_logger_criterion : nn.Module,
                                 model_save_path : str, classwise_metrics : list = None, label_names : list = None):
    averaging_evaluators = [Evaluator(metrics=metric_group) for metric_group in averaging_metrics]
    classwise_evaluators = [MultiLabelEvaluator(metrics=classwise_metric_group, label_names=label_names) for classwise_metric_group in classwise_metrics]
    
    wandb_observer = None
    if is_logging_to_wandb:
        wandb_observer = WandbObserver(project_name=project_name, model=model,
                                    run_id=run_id, run_name = run_name, config=config, tags=tags,
                                    is_watching=watch_gradients, watch_log_freq=gradients_log_freq)
    
    model_logger = ModelLogger(model=model, criterion=model_logger_criterion, evaluation_function=model_logger_evaluation_function,save_path=model_save_path)

    trainer = Trainer()
    return trainer, model_logger, averaging_evaluators, classwise_evaluators, wandb_observer



