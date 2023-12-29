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
from image_transforms import standard_transform
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from uuid import uuid4
import matplotlib.pyplot as plt
import wandb
import math
import json
import logging
import os


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
    performance_dict.update({'loss' : []})
    return performance_dict
#function to generate a map in which all scores can be stored
def generate_model_history_dict(n_classes : int):
    train_dict = generate_performance_documentation_dict(n_classes)
    validation_dict = generate_performance_documentation_dict(n_classes)
    #add dataset size
    model_history_dict = {'train' : train_dict, 'validation' : validation_dict}
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def generate_run_id():
    return str(uuid4())

def generate_config(model_name : str, n_parameters : int, n_learnable_parameters : int, n_classes : int, batch_size : int, lr : float, dataset_path : str, train_dataset_size : int, validation_dataset_size : int,
                           epochs : int, n_epochs_validation : int, early_stopping : bool, patience : int, min_delta_percentage : float, optimizer : str, criterion : str, device : str, previous_run_id : str = ''):
    
    return {'model_name' : model_name, 'n_classes' : n_classes, 'batch_size' : batch_size, 'lr' : lr, 'dataset_path' : dataset_path, 'train_dataset_size' : train_dataset_size, 'validation_dataset_size' : validation_dataset_size,
             'epochs' : epochs, 'n_epochs_validation' : n_epochs_validation, 'early_stopping' : early_stopping, 'patience' : patience, 'min_delta_percentage' : min_delta_percentage,
               'optimizer' : optimizer, 'criterion' : criterion, 'device' : device, 'n_learnable_parameters' : n_learnable_parameters, 'n_parameters' : n_parameters, 'previous_run_id' : previous_run_id}

def init_model_history_dict(config : dict):
    #generate model history dict
    model_history_dict = generate_model_history_dict(config['n_classes'])
    #init wandb run
    #store config in model history dict
    model_history_dict.update(config)
    return model_history_dict

def save_model_history_dict(model_history_dict : dict, save_path : str):
    #save model history dict as json
    with open(save_path, "w") as f:
        json.dump(model_history_dict, f)

def train(model : nn.Module, n_classes : int, train_loader: DataLoader, validation_loader: DataLoader, batch_size : int,  lr : float, dataset_path : str,
            best_weights_save_path : str,
             wandb_api_key: str, wandb_project_name : str, wandb_run_name : str , wandb_tags : list , wandb_run_id : str = None,
              epochs : int = 50,
             n_epochs_validation : int = 5, prefered_device = None,
               early_stopping : bool = False, patience : int = 10, min_delta_percentage : float = 0.0, previous_run_id : str = ''):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if prefered_device is None else torch.device(prefered_device)
    
    model.to(device)
    model_name = model.__class__.__name__

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    calculators = generate_score_calculator_dict(n_classes =  n_classes)
    

    config = generate_config(model_name=model_name, n_parameters=count_parameters(model), n_learnable_parameters=count_learnable_parameters(model),
                              n_classes=n_classes, batch_size=batch_size, lr=lr, dataset_path=dataset_path, train_dataset_size=len(train_loader.dataset), validation_dataset_size=len(validation_loader.dataset),
                              epochs=epochs, n_epochs_validation=n_epochs_validation, early_stopping=early_stopping, patience=patience, min_delta_percentage=min_delta_percentage,
                              optimizer=optimizer.__class__.__name__, criterion=criterion.__class__.__name__, device= str(device), previous_run_id=previous_run_id)
    #generate run id
    run_id = generate_run_id() if wandb_run_id is None else wandb_run_id
    #login to wandb
    wandb.login(key=wandb_api_key, relogin=True)
    #init wandb run
    wandb.init(project=wandb_project_name, name=wandb_run_name, tags=wandb_tags, id=run_id, config=config)
    #init model history dict
    model_history_dict = init_model_history_dict(config=config)

    logging.info(f"Start training {model_name} with {epochs} epochs, batch size {batch_size}, learning rate {lr} and {model_history_dict['train_dataset_size']} training images")
    logging.info(f"Save path: {best_weights_save_path}")

    epoch_progress_bar = tqdm(range(epochs), desc=f"Epochs")
    patience_counter = 0
    #training loop
    model.train()
    wandb.watch(model, log='gradients', log_freq=100)
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
            #check if early stopping is enabled
            if early_stopping and epoch > 0:
                best_stored_val_loss = min(model_history_dict['validation']['loss'])
                #check if validation loss is smaller than best stored validation loss
                if epoch_val_loss < best_stored_val_loss * (1 - min_delta_percentage):
                    #reset patience
                    patience_counter = 0
                    logging.info(f"Validation loss is smaller than best stored validation loss. Reset patience.")
                else:
                    #increase patience
                    patience_counter += 1 * n_epochs_validation
                    logging.info(f"Validation loss is not smaller than best stored validation loss. Increase patience to {patience_counter}")
                #check if patience is reached
                if patience_counter >= patience:
                    logging.info(f"Patience is reached. Stop training.")
                    break
            
            #save model if validation loss is the best loss
            if len(model_history_dict['validation']['loss']) == 0 or epoch_val_loss < min(model_history_dict['validation']['loss']):
                logging.info(f"Validation loss is the best loss. Save model.")
                model_history_dict['best_validation_loss_weights'] = best_weights_save_path
                #if save does not exist, create it
                best_weights_folder_path = best_weights_save_path.rsplit('/', 1)[0]
                if not os.path.exists(best_weights_folder_path):
                    os.makedirs(best_weights_folder_path)
                torch.save(model.state_dict(), f'{best_weights_save_path}.pt')
            model_history_dict['validation']['loss'].append(epoch_val_loss)
    
        #if save does not exist, create it
        best_weights_folder_path = best_weights_save_path.rsplit('/', 1)[0]
        if not os.path.exists(best_weights_folder_path):
            os.makedirs(best_weights_folder_path)
        #save model history dict as json
        save_model_history_dict(model_history_dict, f'{best_weights_save_path}.json')
        #log all metrics to wandb
        if epoch % n_epochs_validation == 0:
            wandb.log({'epoch': epoch, 'train': { 'loss' : model_history_dict['train']['loss'][-1], 'accuracy' : model_history_dict['train']['accuracy'][-1],
                        'precision_micro' : model_history_dict['train']['precision_micro'][-1], 'recall_micro' : model_history_dict['train']['recall_micro'][-1],
                        'f1_micro' : model_history_dict['train']['f1_micro'][-1], 'precision_macro' : model_history_dict['train']['precision_macro'][-1],
                            'recall_macro' : model_history_dict['train']['recall_macro'][-1], 'f1_macro' : model_history_dict['train']['f1_macro'][-1]},
                            'validation':{'epoch': epoch, 'loss' : epoch_val_loss, 'accuracy' : model_history_dict['validation']['accuracy'][-1], 'precision_micro' : model_history_dict['validation']['precision_micro'][-1],
                            'recall_micro' : model_history_dict['validation']['recall_micro'][-1],
                            'f1_micro' : model_history_dict['validation']['f1_micro'][-1], 'precision_macro' : model_history_dict['validation']['precision_macro'][-1],
                            'recall_macro' : model_history_dict['validation']['recall_macro'][-1], 'f1_macro' : model_history_dict['validation']['f1_macro'][-1]}})
        else:
            wandb.log({'epoch': epoch, 'train': {'epoch': epoch, 'loss' : model_history_dict['train']['loss'][-1], 'accuracy' : model_history_dict['train']['accuracy'][-1],
                        'precision_micro' : model_history_dict['train']['precision_micro'][-1], 'recall_micro' : model_history_dict['train']['recall_micro'][-1],
                        'f1_micro' : model_history_dict['train']['f1_micro'][-1], 'precision_macro' : model_history_dict['train']['precision_macro'][-1],
                            'recall_macro' : model_history_dict['train']['recall_macro'][-1], 'f1_macro' : model_history_dict['train']['f1_macro'][-1]}})
    #store number of trained epochs
    model_history_dict['epochs_trained'] = (epoch + 1)
    #store validation after n epochs
    model_history_dict['validated_after_n_epochs'] = (n_epochs_validation)
    #save model history dict as json
    save_model_history_dict(model_history_dict, f'{best_weights_save_path}.json')
    #finish wandb run
    wandb.finish()
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


def generate_image_dataloaders(dataset_path : str, batch_size : int, train_portion : float = 0.7, validation_portion : float = 0.1, test_portion : float = 0.2):
    #load data
    transform = standard_transform()
    dataset = ImageFolder(root=dataset_path, transform=transform)
    targets = dataset.targets
    #split data into train, test, val
    first_split = test_portion
    train_val_idx, test_idx= train_test_split(np.arange(len(targets)),test_size=first_split,shuffle=True,stratify=targets, random_state=42)
    train_val_idx_list = train_val_idx.tolist()
    train_val_stratifier = np.take(targets,train_val_idx_list)
    second_split = validation_portion / (1 - first_split)
    train_idx, validation_idx = train_test_split(train_val_idx,test_size=second_split,shuffle=True,stratify=train_val_stratifier, random_state=42)
    
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    validation_sampler = torch.utils.data.SubsetRandomSampler(validation_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=validation_sampler)
    n_classes = len(dataset.classes)
    return train_loader, validation_loader, test_loader, n_classes

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

def test_early_stopping():
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
    model_history_dict = train(model=model, n_classes=n_classes, train_loader=train_loader, batch_size=10, lr=0.001, dataset_path="dummy_path", best_weights_save_path="dummy_path", epochs=100, validation_loader=validation_loader, n_epochs_validation=1, early_stopping=True, patience=13)
    #check if model_history_dict values are not empty lists
    assert model_history_dict['epochs_trained'][0] < 100
    print(model_history_dict['epochs_trained'])
    logging.info("Test early stopping passed")

def test_wandb_implementation():
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
    model_history_dict = train(model=model, n_classes=n_classes, train_loader=train_loader, batch_size=10, lr=0.001, dataset_path="dummy_path", best_weights_save_path="dummy_path",
                                epochs=10, validation_loader=validation_loader, n_epochs_validation=4, early_stopping=True, patience=13,
                                  wandb_project_name="dummy_project", wandb_run_name="dummy_run", wandb_tags=["dummy_tag"], wandb_api_key='28206af99a544e605ea5edf81110c79035d408bc', prefered_device='cuda')

