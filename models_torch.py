import torchvision.models as models
from image_transforms import standard_transform
import torch.nn as nn

#resnet18 resnet34 resnet101 resnet152 mobilenet_v3_large mobilenet_v3_small shufflenet_v2_x0_5 
#shufflenet_v2_x1_0 shufflenet_v2_x1_5 shufflenet_v2_x2_0 mnasnet0_5 mnasnet0_75 mnasnet1_0 mnasnet1_3 resnext50_32x4d resnext101_32x8d
#resnext101_64x4d wide_resnet50_2 wide_resnet101_2 swin_v2_t swin_v2_s swin_v2_b vit_b_16 vit_b_32 vit_l_16 vit_l_32 vit_h_14
#a dict for the models and their weights
model_dict = {
    "resnet18" :{ 'model': models.resnet18,'weights' : models.ResNet18_Weights.DEFAULT, 'transforms' : standard_transform()},
    "resnet34" : { 'model': models.resnet34,'weights' : models.ResNet34_Weights.DEFAULT, 'transforms' : standard_transform()},
    "resnet50" : { 'model': models.resnet50,'weights' : models.ResNet50_Weights.DEFAULT, 'transforms' : standard_transform()},
    "resnet101" : { 'model': models.resnet101,'weights' : models.ResNet101_Weights.DEFAULT, 'transforms' : standard_transform()},
    "resnet152" : { 'model': models.resnet152,'weights' : models.ResNet152_Weights.DEFAULT, 'transforms' : standard_transform()},
    'mobilenet_v2' : { 'model': models.mobilenet_v2,'weights' : models.MobileNet_V2_Weights.DEFAULT, 'transforms' : standard_transform()},
    'mobilenet_v3_large' : { 'model': models.mobilenet_v3_large,'weights' : models.MobileNet_V3_Large_Weights.DEFAULT, 'transforms' : standard_transform()},
    'mobilenet_v3_small' : { 'model': models.mobilenet_v3_small,'weights' : models.MobileNet_V3_Small_Weights.DEFAULT, 'transforms' : standard_transform()},
    'shufflenet_v2_x0_5' : { 'model': models.shufflenet_v2_x0_5,'weights' : models.ShuffleNet_V2_X0_5_Weights.DEFAULT, 'transforms' : standard_transform()},
    'shufflenet_v2_x1_0' : { 'model': models.shufflenet_v2_x1_0,'weights' : models.ShuffleNet_V2_X1_0_Weights.DEFAULT, 'transforms' : standard_transform()},
    'shufflenet_v2_x1_5' : { 'model': models.shufflenet_v2_x1_5,'weights' : models.ShuffleNet_V2_X1_5_Weights.DEFAULT, 'transforms' : standard_transform()},
    'shufflenet_v2_x2_0' : { 'model': models.shufflenet_v2_x2_0,'weights' : models.ShuffleNet_V2_X2_0_Weights.DEFAULT, 'transforms' : standard_transform()},
    'mnasnet0_5' : { 'model': models.mnasnet0_5,'weights' : models.MNASNet0_5_Weights.DEFAULT, 'transforms' : standard_transform()},
    'mnasnet0_75' : { 'model': models.mnasnet0_75,'weights' : models.MNASNet0_75_Weights.DEFAULT, 'transforms' : standard_transform()},
    'mnasnet1_0' : { 'model': models.mnasnet1_0,'weights' : models.MNASNet1_0_Weights.DEFAULT, 'transforms' : standard_transform()},
    'mnasnet1_3' : { 'model': models.mnasnet1_3,'weights' : models.MNASNet1_3_Weights.DEFAULT, 'transforms' : standard_transform()},
    'resnext50_32x4d' : { 'model': models.resnext50_32x4d,'weights' : models.ResNeXt50_32X4D_Weights.DEFAULT, 'transforms' : standard_transform()},
    'resnext101_32x8d' : { 'model': models.resnext101_32x8d,'weights' : models.ResNeXt101_32X8D_Weights.DEFAULT, 'transforms' : standard_transform()},
    'resnext101_64x4d' : { 'model': models.resnext101_64x4d,'weights' : models.ResNeXt101_64X4D_Weights.DEFAULT, 'transforms' : standard_transform()},
    'wide_resnet50_2' : { 'model': models.wide_resnet50_2,'weights' : models.Wide_ResNet50_2_Weights.DEFAULT, 'transforms' : standard_transform()},
    'wide_resnet101_2' : { 'model': models.wide_resnet101_2,'weights' : models.Wide_ResNet101_2_Weights.DEFAULT, 'transforms' : standard_transform()},
    'swin_v2_t' : { 'model': models.swin_v2_t,'weights' : models.Swin_V2_T_Weights.DEFAULT, 'transforms' : standard_transform(height= 256, width= 256)},
    'swin_v2_s' : { 'model': models.swin_v2_s,'weights' : models.Swin_V2_S_Weights.DEFAULT, 'transforms' : standard_transform(height= 256, width= 256)},
    'swin_v2_b' : { 'model': models.swin_v2_b,'weights' : models.Swin_V2_B_Weights.DEFAULT, 'transforms' : standard_transform(height= 256, width= 256)},
    'vit_b_16' : { 'model': models.vit_b_16,'weights' : models.ViT_B_16_Weights.DEFAULT, 'transforms' : standard_transform()},
    'vit_b_32' : { 'model': models.vit_b_32,'weights' : models.ViT_B_32_Weights.DEFAULT, 'transforms' : standard_transform()},
    'vit_l_16' : { 'model': models.vit_l_16,'weights' : models.ViT_L_16_Weights.DEFAULT, 'transforms' : standard_transform()},
    'vit_l_32' : { 'model': models.vit_l_32,'weights' : models.ViT_L_32_Weights.DEFAULT, 'transforms' : standard_transform()},
    'vit_h_14' : { 'model': models.vit_h_14,'weights' : models.ViT_H_14_Weights.DEFAULT, 'transforms' : standard_transform()},
}

#swap the last layer of every model with a new one
def get_model(model_name: str, num_classes : int, pretrained : bool = False):
    if pretrained:
        weights = model_dict[model_name]['weights']
        model = model_dict[model_name]['model'](weights=weights)
    else:
        model = model_dict[model_name]['model'](pretrained=False)
    if 'resnet' in model_name or 'shufflenet' in model_name or 'resnext' in model_name or 'wide_resnet' in model_name:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'mobilenet' in model_name or 'mnasnet' in model_name:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 10)
    elif 'swin' in model_name:
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif 'vit' in model_name:
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model

def get_available_models():
    return model_dict.keys()

def test_get_model():
    for model_name in model_dict.keys():
        model = get_model(model_name, 10)
        print(model_name, model)