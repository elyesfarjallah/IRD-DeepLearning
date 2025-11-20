# IRD-DeepLearning

Comprehensive tooling to extract, harmonize, and train deep learning models on multimodal ophthalmic datasets with a focus on inherited retinal diseases (IRD).

## Highlights

- **End-to-end dataset pipeline** – `data_pipeline/` contains per-source extractors (e.g., `rips_data_extractor.py`, `rfmid_data_extractor.py`, `ukb_data_extractor.py`), normalization utilities, stratified multi-label split helpers, and `DataPackage` abstractions for reuse.
- **Configurable dataset builders** – top-level scripts such as `create_dataset.py`, `create_ukb_dataset.py`, and `k_fold_dataset_creation.py` compose the pipeline to emit ready-to-train datasets and metadata bundles.
- **Flexible training orchestration** – `run_training.py`, `run_wandb_tracked_training.py`, `run_all_dataset_training.py`, and `train.py` enable single-run, WANDB-tracked, and batched experiments across architectures defined in `input_mapping/models_torch.py`.
- **Automated search & evaluation** – `optimization.py` (Optuna), `k_fold_cross_validation.py`, and the notebooks under `jupyter_notebooks/` streamline hyper-parameter sweeps, cross-validation, and study inspection.
- **Reproducible experimentation** – logging utilities in `ai_backend/` (e.g., `loggers/model_logger.py`) and shell templates under `shell_scripts/` help deploy runs on workstations or clusters.

## Repository Layout

| Path | Purpose |
|------|---------|
| `ai_backend/` | Training observers, evaluators, logging bridges, and WANDB helpers. |
| `data_pipeline/` | Core extraction/processing modules (data loading, transformations, splitting, dataset creation). |
| `datasets/`, `datasets_k_fold/` | Generated single- and k-fold dataset artifacts. |
| `jupyter_notebooks/` | Exploratory analyses (dataset exploration, DICOM inspection, Monte-Carlo dropout, study review). |
| `input_mapping/` | Model, loss, and metric registries used by training scripts. |
| `shell_scripts/` | Batch/HPC/Slurm launchers for cross-validation and optimization jobs. |
| `studies/`, `similarity_matrices/`, `dataset_plots/` | Stored Optuna studies, similarity heatmaps, and visualization outputs. |
| `test_transform.py`, `test_wandb_logging.py` | Quick regression tests for data transforms and logging infrastructure. |

## Prerequisites

- Python 3.10+ (align with the versions in `requirements.txt`).
- CUDA-capable GPU recommended for training.
- WANDB account (optional) for tracked experiments.

## Installation

1. Create and activate a virtual environment (venv, conda, poetry, etc.).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Creation Workflow

1. **Configure inputs** – Define diseases, modalities, and data sources through CLI flags or helper configs (see `convert_user_inputs.py`).
2. **Extract & harmonize** – Run:
   ```bash
   python create_dataset.py --dataset_save_path datasets/$(date +%Y-%m-%d_%H-%M-%S)
   ```
   Key flags include `--diseases_of_interest`, `--source_names`, `--min_images_per_study`, and output format options.
3. **Specialized builders** – Use `create_ukb_dataset.py` for UK Biobank feeds or `convert_k_fold_dataset.py` / `k_fold_dataset_creation.py` to prepare stratified folds saved under `datasets_k_fold/`.
4. **Verification** – Inspect stats and label balance with `plot_dataset_distribution.py` or notebooks like `jupyter_notebooks/dataset_analysis.ipynb`.

## Training & Optimization

- **Single-run training**
  ```bash
  python run_training.py --train_dataset_path datasets/<run>/train.json --val_dataset_path datasets/<run>/val.json
  ```
- **WANDB-tracked multi-source training**
  ```bash
  python run_wandb_tracked_training.py --train_dataset_paths datasets/<run1>/train.json datasets/<run2>/train.json \
                                       --model_key resnet18 --experiment_name "ird_multisource"
  ```
- **Batch automation** – `run_all_dataset_training.py` and `run_python.py` provide wrappers for launching multiple configurations or scripted sweeps.
- **Hyper-parameter search** – Trigger Optuna optimization via:
  ```bash
  python optimization.py --study_name ir_disease_opt --model_key resnet50
  ```
- **Cross-validation** – Combine `k_fold_dataset_creation.py` and `k_fold_cross_validation.py` to orchestrate fold-specific training/evaluation runs.

Model definitions, metrics, and losses are centrally registered in `input_mapping/models_torch.py`, `input_mapping/metric_mapping.py`, and `input_mapping/criterion_mapping.py` to keep experiments repeatable.

## Evaluation & Monitoring

- Use `dataset_evaluation/` utilities and `jupyter_notebooks/inspect_predictions.ipynb`, `inspect_study.ipynb`, or `model_evaluation.ipynb` to visualize metrics, ROC curves, and attention maps.
- Monitor experiments through WANDB dashboards or the persisted studies under `studies/`.
- Validate preprocessing by running:
  ```bash
  python test_transform.py
  python test_wandb_logging.py
  ```

## Notebooks

Each notebook under `jupyter_notebooks/` documents a stage of the workflow (data exploration, clustering, Monte Carlo dropout, segmentation, SAM experimentation, etc.). Launch them with Jupyter Lab/VS Code and ensure the same environment is activated.

## License

No explicit license file is included. Confirm permissions with the repository authors before distributing or using the code in production.
