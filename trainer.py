import torch
import torch.nn as nn
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score
from torcheval.metrics import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import json
import datetime
import logging


def generate_score_calculator_dict(n_classes : int):
    return {'accuracy': MulticlassAccuracy(num_classes = n_classes),
             'precision_micro': MulticlassPrecision(num_classes = n_classes),'recall_micro': MulticlassRecall(num_classes = n_classes), 'f1_micro': MulticlassF1Score(num_classes = n_classes),
               'precision_macro': MulticlassPrecision(num_classes = n_classes, average = 'macro'), 'recall_macro': MulticlassRecall(num_classes = n_classes, average = 'macro'), 'f1_macro': MulticlassF1Score(num_classes = n_classes, average = 'macro')}

class EmptyDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise IndexError("Empty dataset cannot be indexed")

def generate_score_dict(n_classes : int):
    return {score : [] for score in generate_score_calculator_dict(n_classes).keys()}

def generate_performance_documentation_dict(n_classes : int):
    performance_dict = generate_score_dict(n_classes)
    performance_dict.update({'dataset_size' : [], 'loss' : []})
    return performance_dict
#function to generate a map in which all scores can be stored
def generate_model_history_dict(n_classes : int):
    train_dict = generate_performance_documentation_dict(n_classes)
    validation_dict = generate_performance_documentation_dict(n_classes)
    #add dataset size
    train_dict['dataset_size'] = []
    validation_dict['dataset_size'] = []
    model_history_dict = {'train' : train_dict, 'validation' : validation_dict,
                           'batch_size' : [], 'lr' : [], 'dataset_path' : [], 'epochs_trained' : [], 'validated_after_n_epochs' : [],
                           'best_validation_loss_weights' : []}
    return model_history_dict

def update_calculators(calculator_dict, y_true, y_pred):
    for key in calculator_dict.keys():
        calculator_dict[key].update(y_pred, y_true)

#function to compute and store scores
def compute_scores(calculator_dict, model_history_dict):
    #remove batch dimension
   for key in calculator_dict.keys():
       model_history_dict[key].append(calculator_dict[key].compute().item())

def reset_calculators(calculator_dict):
    for key in calculator_dict.keys():
        calculator_dict[key].reset()

def train(model : nn.Module, n_classes : int, train_loader: DataLoader, batch_size : int,  lr : float, dataset_path : str, best_weights_save_path : str, epochs : int = 50,
           validation_loader: DataLoader = EmptyDataset(),
           model_history_dict : dict = None,
             n_epochs_validation : int = 5, shuffle = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model_name = model.__class__.__name__

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    calculators = generate_score_calculator_dict(n_classes =  n_classes)
    #check if history dicts are given
    if model_history_dict is None:
       model_history_dict = generate_model_history_dict(n_classes = n_classes)

    #check if there are weights in the model_history_dict
    if len(model_history_dict['best_validation_loss_weights']) > 0:
        model.load_state_dict(torch.load(model_history_dict['best_validation_loss_weights']))
        logging.info(f"Load weights from {model_history_dict['best_validation_loss_weights']}")
    

    #store in model_history_dict
    model_history_dict['train']['dataset_size'].append(len(train_loader.sampler))
    model_history_dict['validation']['dataset_size'].append(len(validation_loader.sampler))
    #store batch_size
    model_history_dict['batch_size'].append(batch_size)
    #store lr
    model_history_dict['lr'].append(lr)
    #store dataset path
    model_history_dict['dataset_path'].append(dataset_path)

    logging.info(f"Start training {model_name} with {epochs} epochs, batch size {batch_size}, learning rate {lr} and {model_history_dict['train']['dataset_size'][-1]} training images")
    logging.info(f"Save path: {best_weights_save_path}")

    epoch_progress_bar = tqdm(range(epochs), desc=f"Epochs")

    #training loop
    model.train()
    for epoch in epoch_progress_bar:
        #reset calculators
        reset_calculators(calculators)
        epoch_losses = []
        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)
            #forward
            outputs = model(data)
            loss = criterion(outputs, labels)
            #store loss
            epoch_losses.append(loss.item())
            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #update calculators
            update_calculators(calculator_dict= calculators, y_true=labels, y_pred=outputs)
        #compute scores
        compute_scores(calculator_dict= calculators, model_history_dict= model_history_dict['train'])
        #calculate epoch loss
        model_history_dict['train']['loss'].append(np.mean(epoch_losses))
        #update progress bar, average scores for the epoch
        epoch_progress_bar.set_postfix({'train loss': model_history_dict['train']['loss'][-1], 'train accuracy': model_history_dict['train']['accuracy'][-1]})
        #validate after n epochs
        if epoch % n_epochs_validation == 0 and len(validation_loader) > 0:
            #reset calculators
            reset_calculators(calculators)
            #reset loss
            epoch_val_losses = []
            logging.info(f"Validate after {epoch} epochs")
            #validate
            model.eval()
            with torch.no_grad():
                for data, labels in validation_loader:
                    data = data.to(device)
                    labels = labels.to(device)
                    #forward
                    outputs = model(data)
                    loss = criterion(outputs, labels)
                    #store loss
                    epoch_val_losses.append(loss.item())
                    #update calculators
                    update_calculators(calculators, labels, outputs)
            #compute scores
            compute_scores(calculators, model_history_dict['validation'])
            #save loss
            epoch_val_loss = np.mean(epoch_val_losses)
            #update progress bar
            epoch_progress_bar.set_postfix({'validation loss': epoch_val_loss, 'validation accuracy': model_history_dict['validation']['accuracy'][-1]})
            #save model if validation loss is the best loss
            if len(model_history_dict['validation']['loss']) == 0 or epoch_val_loss < min(model_history_dict['validation']['loss']):
                logging.info(f"Validation loss is the best loss. Save model.")
                model_history_dict['best_validation_loss_weights'].append(best_weights_save_path)
                torch.save(model.state_dict(), best_weights_save_path)
            model_history_dict['validation']['loss'].append(epoch_val_loss)
    
    #store number of trained epochs
    model_history_dict['epochs_trained'].append(epochs)
    #store validation after n epochs
    model_history_dict['validated_after_n_epochs'].append(n_epochs_validation)
    return model_history_dict

def filter_model_history_by_training_run(model_history_dict : dict, training_run_idx : int):
    #filter model history dict by training run
    #sum up epoch counts of previous training runs
    previous_training_runs = sum(model_history_dict['epochs_trained'][:training_run_idx])
    epochs_trained = model_history_dict['epochs_trained'][training_run_idx]
    filtered_model_history_dict = {}
    for key in model_history_dict.keys():
        if key != 'validation' and key  != 'train':
            filtered_model_history_dict[key] = model_history_dict[key][training_run_idx]
        else:
            filtered_model_history_dict[key] = {}
            for sub_key in model_history_dict[key].keys():
                filtered_model_history_dict[key][sub_key] = model_history_dict[key][sub_key][previous_training_runs:previous_training_runs + epochs_trained]
    return filtered_model_history_dict

def plot_single_training_run_evaluation(model_history_dict : dict, metric_to_plot : str,fig = None, ax = None):
    #create figure
    fig, ax = plt.subplots() if fig is None and ax is None else (fig, ax)
    #plot train and validation
    ax.plot(model_history_dict['train'][metric_to_plot], label='train')
    #check validated after n epochs
    spacing = int(model_history_dict['validated_after_n_epochs'])
    epochs_trained = model_history_dict['epochs_trained']
    ax.plot([i * spacing for i in range(epochs_trained//spacing)],model_history_dict['validation'][metric_to_plot] , label='validation')
    #set labels
    label_metric = metric_to_plot.replace('_', ' ')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(label_metric)
    #set title
    ax.set_title(f"{label_metric} over epochs")
    #set legend
    ax.legend()
    #return figure
    return fig, ax

def plot_all_single_training_run_evaluations(model_history_dict : dict, title : str = None):
    #create subplots
    eval_keys = [key for key in model_history_dict['validation'].keys() if key not in ['dataset_size']]
    n_evaluations = len(eval_keys)
    epochs_trained = model_history_dict['epochs_trained']
    n_rows = math.ceil(n_evaluations / 2)
    fig, axs = plt.subplots(n_rows, 2, figsize=(10, 5 * n_rows))
    #plot all evaluations
    for i, evaluation in enumerate(eval_keys):
        if i < n_rows:
            plot_single_training_run_evaluation(model_history_dict, evaluation, fig=fig, ax=axs[i][0])
        else:
            plot_single_training_run_evaluation(model_history_dict, evaluation, fig=fig, ax=axs[i - n_rows][1])
    #return figure
    # adjust spacing between subplots and leave space for the title
    fig.tight_layout()
    fig.subplots_adjust(top = 0.95)
    if title is not None:
        #make title bold
        fig.suptitle(title, fontweight='bold')
    return fig, axs


#unittest for train function with dummy data and model
def test_train():
    #generate dummy data
    n_classes = 3
    #generate dummy data random tensor
    train_data = torch.rand((100, 3, 224, 224))
    train_labels = torch.randint(0, n_classes, (100,))
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    #create dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
    #generate dummy data random tensor
    validation_data = torch.rand((100, 3, 224, 224))
    validation_labels = torch.randint(0, n_classes, (100,))
    validation_dataset = torch.utils.data.TensorDataset(validation_data, validation_labels)
    #create dataloader
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=10, shuffle=True)
    #generate dummy model
    model = torchvision.models.resnet18(weights=None)
    #swap last layer
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    #train model
    model_history_dict = train(model=model, n_classes=n_classes, train_loader=train_loader, batch_size=10, lr=0.001, dataset_path="dummy_path", best_weights_save_path="dummy_path", epochs=10, validation_loader=validation_loader, n_epochs_validation=1)
    #check if model_history_dict values are not empty lists
    for key in model_history_dict.keys():
        try:
            assert len(model_history_dict[key]) > 0
        except AssertionError:
            logging.error(f"Key {key} is empty")
            print(model_history_dict[key])
    


def test_evaluation():
    n_classes = 3
    #generate dummy data random tensor
    train_data = torch.rand((100, 3, 224, 224))
    train_labels = torch.randint(0, n_classes, (100,))
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    #generate dummy data random tensor
    validation_data = torch.rand((100, 3, 224, 224))
    validation_labels = torch.randint(0, n_classes, (100,))
    validation_dataset = torch.utils.data.TensorDataset(validation_data, validation_labels)
    #generate dummy model
    model = torchvision.models.resnet18(weights=None)
    #swap last layer
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    #predict
    model.train()
    #move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_data = train_data.to(device)
    #train_labels = train_labels.to(device)
    y_pred = model(train_data)
    #compute scores
    calculators = generate_score_calculator_dict(n_classes)
    update_calculators(calculators, train_labels, y_pred)
    for calc in calculators:
        #compute scores and check if item is a float
        assert isinstance(calculators[calc].compute().item(), float)
    logging.info("Test evaluation passed")

def test_plot_evaluation():
    #read in json file
    with open("models/2023-12-12_23-42_ResNet152.pth.json", "r") as f:
        model_history_dict = json.load(f)
    #plot evaluation
    dict_eval = filter_model_history_by_training_run(model_history_dict, 0)
    fig = plot_all_single_training_run_evaluations(dict_eval, title="Test Title")
    #plt.show()
    #plot precision macro
    #fig_2, ax_2 = plot_single_training_run_evaluation(dict_eval, 'precision_macro')
    plt.show()
    logging.info("Test plot evaluation passed")
#test_plot_evaluation()