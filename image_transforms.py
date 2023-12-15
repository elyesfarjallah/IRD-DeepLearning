import torch
import torchvision.transforms as transforms


class RectAngularPadTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, img):
        padding = (
            max(0, (img.size[1] - img.size[0]) // 2),
            max(0, (img.size[0] - img.size[1]) // 2)
        )
        #show image
        new_img = new_img = F.pad(img, padding)
        return new_img
    
    def __repr__(self):
        return self.__class__.__name__

def standard_transforms(h_size : int = 224, w_size : int = 224, mean : list = [0.485, 0.456, 0.406], std : list = [0.229, 0.224, 0.225]):
    return transforms.Compose([
        RectAngularPadTransform(),
        transforms.Resize((h_size, w_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])