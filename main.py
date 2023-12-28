import hyperparameter_optimization
import models_torch
import argparse
import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for a given model')
    #parse list of model keys
    parser.add
    parser.add_argument('--model_keys', nargs='+', type=str, help='keys of the models to optimize')
    parser.add_argument('--dataset_path', type=str, help='path to the dataset')
    parser.add_argument('--wandb_config_path', type=str, help='path to the wandb config file')
    parser.add_argument('--alternate_image_transforms', action='store_true', help='use alternate image transforms')
    parser.add_argument('--weight_train_sampler', type=bool, action='store_true', help='use weighted random sampler for training data')
    parser.add_argument('--weight_validation_sampler', type=bool,  action='store_true', help='use weighted random sampler for validation data')
    parser.add_argument('--n_trials', type=int, help='number of trials for the hyperparameter optimization')
    parser.add_argument('--study_save_path', type=str, help='path to save the study')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    hyperparameter_optimization.optimize_model(model_key=args.model_key, dataset_path=args.dataset_path,
                                                wandb_config_path=args.wandb_config_path, alternate_image_transforms=args.alternate_image_transforms,
                                                weight_train_sampler=args.weight_train_sampler, weight_validation_sampler=args.weight_validation_sampler,
                                                n_trials=args.n_trials, study_save_path=args.study_save_path)