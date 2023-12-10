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


#function to generate a map in which all scores can be stored
def generate_model_history_dict(n_classes : int):
    model_history_dict = {score : [] for score in generate_score_calculator_dict(n_classes).keys()}
    #add batch size, lr, n_batches, dataset_path, best_loss_weights
    model_history_dict['batch_size'] = []
    model_history_dict['lr'] = []
    model_history_dict['n_batches'] = []
    model_history_dict['dataset_path'] = []
    model_history_dict['best_loss_weights'] = ''
    return model_history_dict

def update_calculators(calculator_dict, y_true, y_pred):
    for key in calculator_dict.keys():
        calculator_dict[key].update(y_pred, y_true)

#function to compute and store scores
def compute_scores(calculator_dict, model_history_dict, y_true, y_pred):
    #remove batch dimension
   for key in calculator_dict.keys():
       model_history_dict[key].append(calculator_dict[key].compute().item())

def reset_calculators(calculator_dict):
    for key in calculator_dict.keys():
        calculator_dict[key].reset()

def train(model : nn.Module, train_dataset: Dataset, batch_size : int,  lr : float, dataset_path : str, epochs : int = 50,
           validation_dataset: Dataset = EmptyDataset(),
           train_history_dict : dict = None, validation_history_dict : dict = None,
             n_epochs_validation : int = 5, shuffle = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model_name = model.__class__.__name__

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    n_classes = len(set([label for _, label in train_loader]))
    calculators = generate_score_calculator_dict(n_classes =  n_classes)
    #check if history dicts are given
    if train_history_dict is None:
        train_history_dict = generate_model_history_dict(n_classes = n_classes)
    if validation_history_dict is None:
        validation_history_dict = generate_model_history_dict(n_classes = n_classes)

    #check if there are weights in the model_history_dict
    if validation_history_dict['best_loss_weights'] != '':
        model.load_state_dict(torch.load(validation_history_dict['best_loss_weights']))
        logging.info(f"Load weights from {validation_history_dict['best_loss_weights']}")
    
    batches_per_epoch = len(train_loader)
    train_history_dict['n_batches'] = batches_per_epoch
    train_history_dict['batches_per_epoch'] = batches_per_epoch
    logging.info(f"Start training {model_name} with {epochs} epochs, {batches_per_epoch} batches per epoch, batch size {batch_size}, learning rate {lr} and {len(train_dataset)} training images")
    save_path = f"models/{model_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth"
    validation_history_dict['best_loss_weights'] = save_path
    logging.info(f"Save path: {save_path}")
    epoch_progress_bar = tqdm(range(epochs), desc=f"Epochs")
    for epoch in epoch_progress_bar:
        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)
            #forward
            outputs = model(data)
            loss = criterion(outputs, labels)
            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #update calculators
            update_calculators(calculator_dict= calculators, y_true=labels, y_pred=outputs)
            #compute scores
            compute_scores(calculator_dict= calculators, model_history_dict= train_history_dict, y_true=labels, y_pred=outputs)
            #reset calculators
            reset_calculators(calculators) 
            #save lr, batch size
            train_history_dict['lr'].append(lr)
            train_history_dict['batch_size'].append(batch_size)
        #update progress bar, average scores for the epoch
        epoch_progress_bar.set_postfix({'train loss': np.mean(train_history_dict['loss'][-batches_per_epoch:]), 'train accuracy': np.mean(train_history_dict['accuracy'][-batches_per_epoch:])})
        #validate after n epochs
        if epoch % n_epochs_validation == 0 and len(validation_dataset) > 0:
            #validate
            logging.info(f"Validate after {epoch} epochs")
            with torch.no_grad():
                for data, labels in validation_loader:
                    data = data.to(device)
                    labels = labels.to(device)
                    #forward
                    outputs = model(data)
                    loss = criterion(outputs, labels)
                    #compute scores
                    compute_scores(calculators, validation_history_dict, labels, outputs)
                    #reset calculators
                    reset_calculators(calculators)

            #update progress bar
            validation_epoch_loss = np.mean(validation_history_dict['loss'][-batches_per_epoch:])
            epoch_progress_bar.set_postfix({'validation loss': validation_epoch_loss, 'validation accuracy': np.mean(validation_history_dict['accuracy'][-batches_per_epoch:])})
            #save model if validation loss is the best loss
            if 'best_epoch_loss' not in validation_history_dict.keys() or validation_epoch_loss < validation_history_dict['best_epoch_loss']:
                logging.info(f"Validation loss is the best loss. Save model.")
                validation_history_dict['best_epoch_loss'] = validation_epoch_loss
                validation_history_dict['best_loss_weights'] = save_path
                torch.save(model.state_dict(), save_path)
    return train_history_dict, validation_history_dict

#unittest for train function with dummy data and model
def test_train():
    #generate dummy data
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
    #train model
    train_history_dict, validation_history_dict = train(model, train_dataset, batch_size = 10, lr = 0.001, epochs = 3, validation_dataset = validation_dataset, dataset_path='dummy_path')
    
    #check if every value in dict len > 0
    for key in train_history_dict.keys():
        assert len(train_history_dict[key]) > 0
    for key in validation_history_dict.keys():
        assert len(validation_history_dict[key]) > 0
    

#run unittest
if __name__ == '__main__':
    #dummy test
    test_train()
    #test_train()
