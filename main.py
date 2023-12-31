import hyperparameter_optimization
import models_torch
import argparse
import logging
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for a given model')
    #parse list of model keys
    parser.add_argument('--model_keys', nargs='+', type=str, help='keys of the models to optimize')
    parser.add_argument('--dataset_path', type=str, help='path to the dataset')
    parser.add_argument('--n_epochs', type=int, help='number of epochs to train the model')
    parser.add_argument('--wandb_config_path', type=str, help='path to the wandb config file')
    parser.add_argument('--alternate_image_transforms', action='store_true', help='use alternate image transforms')
    parser.add_argument('--weight_train_sampler', action='store_true', help='use weighted random sampler for training data')
    parser.add_argument('--weight_validation_sampler', action='store_true', help='use weighted random sampler for validation data')
    parser.add_argument('--n_trials', type=int, help='number of trials for the hyperparameter optimization')
    parser.add_argument('--study_save_path', type=str, help='path to save the study')
    parser.add_argument('--n_epochs_validation', type=int, help='number of epochs after which the validation is executed')
    #optional arguments
    parser.add_argument('--prefered_device', type=str, default='cuda:0', help='prefered device for training')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    #read the config file
    with open(args.wandb_config_path) as f:
        wandb_config = json.load(f)
    for model_key in args.model_keys:
        hyperparameter_optimization.optimize_model(model_key=model_key, dataset_path=args.dataset_path,
                                                wandb_config=wandb_config, alternate_image_transforms=args.alternate_image_transforms,
                                                weight_train_sampler=args.weight_train_sampler, weight_validation_sampler=args.weight_validation_sampler,
                                                n_trials=args.n_trials, study_save_path=args.study_save_path,
                                                  n_epochs_validation= args.n_epochs_validation, n_epochs=args.n_epochs,
                                                  prefered_device=args.prefered_device)

#execute the hyperparameter optimization
#python main.py --model_keys shufflenet_v2_x1_0 shufflenet_v2_x1_5 shufflenet_v2_x2_0 mnasnet0_5 mnasnet0_75 mnasnet1_0 mnasnet1_3 resnext50_32x4d resnext101_32x8d resnext101_64x4d wide_resnet50_2 wide_resnet101_2 swin_v2_t swin_v2_s swin_v2_b vit_b_16 vit_b_32 vit_l_16 vit_l_32 vit_h_14 --dataset_path datasets/2023-12-28_18-12-43 --n_epochs 100 --wandb_config_path wandb_config.json --alternate_image_transforms --n_trials 100 --study_save_path studies --n_epochs_validation 1 --prefered_device cuda:0
#python main.py --model_keys shufflenet_v2_x1_0 shufflenet_v2_x1_5 shufflenet_v2_x2_0 mnasnet0_5 mnasnet0_75 mnasnet1_0 mnasnet1_3 resnext50_32x4d resnext101_32x8d resnext101_64x4d wide_resnet50_2 wide_resnet101_2 swin_v2_t swin_v2_s swin_v2_b vit_b_16 vit_b_32 vit_l_16 vit_l_32 vit_h_14 --dataset_path datasets/2023-12-28_18-12-43 --n_epochs 100 --wandb_config_path wandb_config.json --n_trials 100 --study_save_path studies --n_epochs_validation 1 --prefered_device cuda:1
        

#python main.py --model_keys resnext101_32x8d resnext101_64x4d --dataset_path datasets/2023-12-28_18-12-43 --n_epochs 5 --wandb_config_path wandb_config.json --alternate_image_transforms --n_trials 5 --study_save_path studies --n_epochs_validation 1 --prefered_device cuda:0

