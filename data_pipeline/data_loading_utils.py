from data_pipeline.np_dataset import NpDataset
from data_pipeline.data_package import DataPackage
from PIL import Image
from pydicom import dcmread

def data_package_to_dataset(package, file_reader, transform, augmentations=None):
    dataset = NpDataset(file_paths=package.get_data(), labels=package.get_labels(),
                         file_reader=file_reader, transform=transform, augmentation_transform=augmentations)
    return dataset

def data_packages_to_datasets(package_list : list, file_readers :list, transforms : list, augmentations : list =None):
    datasets = []
    if augmentations is None:
        augmentations = [None] * len(package_list)
    for package, file_reader, transform, augmentation in zip(package_list, file_readers, transforms, augmentations):
        dataset = data_package_to_dataset(package=package, file_reader=file_reader, transform=transform, augmentations=augmentation)
        datasets.append(dataset)
    return datasets

def get_file_reader(file_reader_name : str):
    #create np datasets for training, validation and testing
    read_dicom = lambda x: dcmread(x).pixel_array
    dicom_file_reader = lambda x: Image.fromarray(read_dicom(x)).convert('RGB')
    default_file_reader = lambda x: Image.open(x).convert('RGB')
    file_readers = {'dicom' : dicom_file_reader, 'default' : default_file_reader}
    return file_readers[file_reader_name]

def filter_data_package_by_labels(data_package : DataPackage, labels_to_keep : list):
    labels = data_package.get_labels()
    indices = [i for i, label in enumerate(labels) if label in labels_to_keep]
    data = data_package.get_data()[indices]
    labels = labels[indices]
    instance_ids = data_package.instance_ids[indices]
    data_source_name = data_package.data_source_name
    return DataPackage(data=data, labels=labels, instance_ids=instance_ids, data_source_name=data_source_name)

def filter_data_packages_by_labels(data_packages : list, labels_to_keep : list):
    return [filter_data_package_by_labels(data_package, labels_to_keep) for data_package in data_packages]
  