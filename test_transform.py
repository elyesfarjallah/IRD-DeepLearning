from data_pipeline.fundus_segment_crop import FundusSegmentCrop
from data_pipeline.np_dataset import NpDataset
from data_pipeline.image_transforms import RectAngularPadTransform, ben_transform
import torchvision
import numpy as np
import os
from PIL import Image
import torch
import matplotlib.pyplot as plt

def find_files(folder_path):
    """
    Recursively find all files in a folder and its subfolders.
    
    Args:
    - folder_path (str): Path to the folder to search.
    
    Returns:
    - file_paths (list): List of paths to all files found.
    """
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths
ses_dir ='databases/SES' #'databases/RFMiD/Test_Set/Test'
#get filepaths for all files in that directory and subdirectories
ses_paths = find_files(ses_dir)
rfmid_dir = 'databases/RFMiD/Test_Set/Test'
rfmid_paths = find_files(rfmid_dir)
#pick the first 15 from each and combine them
ses_paths = ses_paths[:15]
rfmid_paths = rfmid_paths[:15]
all_paths = ses_paths + rfmid_paths
#shuffle the list
np.random.shuffle(all_paths)


img_reader_lambda = lambda x: Image.open(x).convert('RGB')
transform = torchvision.transforms.Compose([
    FundusSegmentCrop(),
    RectAngularPadTransform(),
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor()
])

transform_with_out_crop = torchvision.transforms.Compose([
    RectAngularPadTransform(),
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor()
])

ben_transform_T = ben_transform(224)
#create an np dataset
random_labels = np.random.randint(0, 2, size=(len(all_paths), 10))
np_dataset_crop = NpDataset(all_paths, random_labels, img_reader_lambda, transform=transform)
np_dataset_crop_no_crop = NpDataset(all_paths, random_labels, img_reader_lambda, transform=transform_with_out_crop)
np_dataset_ben = NpDataset(all_paths, random_labels, img_reader_lambda, transform=ben_transform_T)

#create a dataloader
#iterate over the dataloader
i = 0
for data_crop, data_no_crop, data_ben in zip(np_dataset_crop, np_dataset_crop_no_crop, np_dataset_ben):
    
    image, label = data_crop
    image_no_crop, labels_no_crop = data_no_crop
    image_ben, label_ben = data_ben
    #plot the images from the dataset and the original images
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 3, 1)
    plt.imshow(image.permute(1, 2, 0).numpy())
    #print(np.unique(images.permute(1, 2, 0).numpy().flatten()))
    plt.title("Cropped Image")
    plt.subplot(1, 3, 2)
    plt.imshow(image_no_crop.permute(1, 2, 0).numpy())
    plt.title("Image without cropping")
    
    plt.subplot(1, 3, 3)
    plt.imshow(image_ben.permute(1, 2, 0).numpy())
    plt.title("Image without cropping")
    plt.savefig(f"test_images/test_transform_{i}.png")
    plt.clf()
    if i == 10:
        break
    i += 1
