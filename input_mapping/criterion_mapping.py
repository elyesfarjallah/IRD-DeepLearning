from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss

criterion_dict = {'CrossEntropyLoss' : CrossEntropyLoss, 'BCELoss' : BCELoss, 'BCEWithLogitsLoss' : BCEWithLogitsLoss}

def get_criterion_by_name(criterion_name : str):
    if criterion_name in criterion_dict:
        return criterion_dict[criterion_name]()
    raise ValueError(f'Criterion with name {criterion_name} not found')