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

class BenTransform(torch.nn.Module):
    #todo switch to a pytorch implementation
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size

    def forward(self, img):
        #convert to numpy array
        img_np = np.array(img)
        #convert to RGB
        #img_cv2 = np.transpose(img_np, (1, 2, 0))
        img_transformed = self.load_ben_color(img_np)
        #convert to PIL image
        img_transformed = Image.fromarray(img_transformed)
        #plot image
        #return transformed image
        return img_transformed

    
    def crop_image1(img,tol=7):
        # img is image data
        # tol  is tolerance
            
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]

    def crop_image_from_gray(self, img,tol=7):
        if img.ndim ==2:
            mask = img>tol
            return img[np.ix_(mask.any(1),mask.any(0))]
        elif img.ndim==3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask = gray_img>tol
            
            check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
            if (check_shape == 0): # image is too dark so that we crop out everything,
                return img # return original image
            else:
                img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
                img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
                img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
        #         print(img1.shape,img2.shape,img3.shape)
                img = np.stack([img1,img2,img3],axis=-1)
        #         print(img.shape)
            return img
    
    def load_ben_color(self, image, sigmaX=10):
        image_cropped = self.crop_image_from_gray(image)
        image_resized = cv2.resize(image_cropped, (224, 224))
        image_blurred =cv2.addWeighted(image_resized,4, cv2.GaussianBlur( image_resized , (0,0) , sigmaX) ,-4 ,128)
        return image_blurred
    
def ben_transform(img_size):
    return transforms.Compose([
        BenTransform(img_size),
    ])

def standard_transform(height : int = 224, width : int = 224, mean : list = [0.485, 0.456, 0.406], std : list = [0.229, 0.224, 0.225]):
    return transforms.Compose([
        #OuterEdgeCrop(),
        RectAngularPadTransform(),
        transforms.Resize((height, width)),
    ])

def get_transforms(transform_name : str, transforms_config : dict):
    img_size = transforms_config['img_size']

    if transform_name == 'standard':
       return standard_transform(img_size, img_size)
    elif transform_name == 'ben':
        return ben_transform(img_size)
    else:
        raise ValueError(f'Transform {transform_name} not supported')

def test_standard_transform():
    transforms_test = transforms.Compose([OuterEdgeCrop()])
    transform = OuterEdgeCrop()
    img = Image.open("databases/ODIR-5k/Testing Images/937_right.jpg")
    img_transformed = transforms_test(img)
    print(type(img_transformed))
    print(img_transformed.size)
    plt.imshow(img_transformed)
    plt.show()

def test_ben_transform():
    transform = BenTransform(224)
    img = Image.open("databases/ODIR-5k/Testing Images/937_right.jpg")
    #show original image
    img.show()
    img_transformed = transform(img)
    print(type(img_transformed))
    print(img_transformed.size)
    plt.imshow(img_transformed)
    plt.show()