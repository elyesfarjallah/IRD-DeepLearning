import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image


class RectAngularPadTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, img):
        padding = (
            max(0, (img.size[1] - img.size[0]) // 2),
            max(0, (img.size[0] - img.size[1]) // 2)
        )
        new_img = new_img = F.pad(img, padding)
        return new_img
    
    def __repr__(self):
        return self.__class__.__name__

class OuterEdgeCrop(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fail_cont = 0
    
    def find_outer_edges(self, image):
        # Read the image in grayscale
        image_np = np.array(image)
        #restructure array if image is not in RGB format
        if image_np.shape[2] != 3:
            image_np = np.transpose(image_np, (1, 2, 0))
        image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        # Apply Gaussian blur to reduce noise and improve edge detection
        blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)

        # Convert to 8-bit unsigned integer format
        blurred = cv2.convertScaleAbs(blurred)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 100, 100)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get the bounding box of the outer contour (assumes the outer contour is the largest)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            _, _, w, h = cv2.boundingRect(largest_contour)
            return w, h
        else:
            return None
    def forward(self, img):
        try:
            w, h = self.find_outer_edges(img)
            #crop image
            crop_size = max(w, h)
            crop = transforms.CenterCrop(crop_size)
            return crop(img)
        except:
            return img
    
    def __repr__(self):
        return self.__class__.__name__
    

def standard_transform(height : int = 224, width : int = 224, mean : list = [0.485, 0.456, 0.406], std : list = [0.229, 0.224, 0.225]):
    return transforms.Compose([
        #OuterEdgeCrop(),
        RectAngularPadTransform(),
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def test_standard_transform():
    transforms_test = transforms.Compose([OuterEdgeCrop()])
    transform = OuterEdgeCrop()
    img = Image.open("datasets/2023-12-08-16-54/Age-related Macular Degeneration/ODIR-5k_Training Images_1889_left.jpg")
    img_transformed = transforms_test(img)
    print(type(img_transformed))
    print(img_transformed.size)
    plt.imshow(img_transformed)
    plt.show()
