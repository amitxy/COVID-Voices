#!/usr/bin/env python3
"""
Optuna hyperparameter optimization script for COVID-Voices.
Optimizes hyperparameters for COVID-19 tweet sentiment classification.
"""

import optuna
import logging
import os
import time
import wandb
import torch
from functools import partial
from covid_voices.config import Config, OptunaConfig
from covid_voices.data import load_and_prepare_datasets
from covid_voices.utils import init_logging, set_seed
from train import create_model_and_trainer


# Set ALL wandb directories to your writable location
os.environ["WANDB_DIR"] = "/home/yandex/MLWG2025/amitr5/tmp/wandb"
os.environ["WANDB_CACHE_DIR"] = "/home/yandex/MLWG2025/amitr5/tmp/wandb_cache"
os.environ["WANDB_ARTIFACTS_DIR"] = "/home/yandex/MLWG2025/amitr5/tmp/wandb_artifacts"
os.environ["WANDB_STAGING_DIR"] = "/home/yandex/MLWG2025/amitr5/tmp/wandb_staging"

# Create ALL directories
os.makedirs("/home/yandex/MLWG2025/amitr5/tmp/wandb", exist_ok=True)
os.makedirs("/home/yandex/MLWG2025/amitr5/tmp/wandb_cache", exist_ok=True)
os.makedirs("/home/yandex/MLWG2025/amitr5/tmp/wandb_artifacts", exist_ok=True)
os.makedirs("/home/yandex/MLWG2025/amitr5/tmp/wandb_staging", exist_ok=True)


# Guard against launching the Optuna orchestrator under torchrun/DDP
_local_rank = os.environ.get("LOCAL_RANK")
if _local_rank not in (None, "", "0"):
    # Non-zero ranks should exit immediately
    print(f"[Optuna] Non-zero LOCAL_RANK={_local_rank} exiting.")
    import sys
    sys.exit(0)

# Remove distributed env so HF Trainer doesn't enable DDP in the orchestrator process
for _k in ("LOCAL_RANK", "RANK", "WORLD_SIZE"):
    if _k in os.environ:
        os.environ.pop(_k, None)

# Setup logging
logger = init_logging()


def objective(trial: optuna.Trial, tokenized_datasets=None, tokenizer=None) -> float:
    """
    Objective function for Optuna optimization.
    
    This function will be called for each trial to evaluate a set of hyperparameters.
    For now, it returns a dummy score, but later we'll integrate actual training.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        float: Objective value (F1 score)
    """
    # Get hyperparameters for this trial
    params = OptunaConfig.get_hyperparameter_search_space(trial)
    
    logger.info(f"Trial {trial.number}: params={params}")

    # Build a per-trial config instance (no global mutation)
    config = Config()
    
    # Update trial number for this trial
    config.TRIAL_NUMBER = trial.number
    
    config.LEARNING_RATE = float(params.get("learning_rate", config.LEARNING_RATE))
    config.NUM_TRAIN_EPOCHS = int(params.get("num_train_epochs", config.NUM_TRAIN_EPOCHS))
    config.BATCH_SIZE = int(params.get("batch_size", config.BATCH_SIZE))
    config.LR_SCHEDULER_TYPE = str(params.get("lr_scheduler_type", config.LR_SCHEDULER_TYPE))
    config.WEIGHT_DECAY = float(params.get("weight_decay", config.WEIGHT_DECAY))
    config.EARLY_STOPPING_PATIENCE = int(params.get("early_stopping_patience", config.EARLY_STOPPING_PATIENCE))
    
    # New training parameters
    config.OPTIM = str(params.get("optim", config.OPTIM))
    config.ADAM_EPS = float(params.get("adam_eps", config.ADAM_EPS))
    config.ADAM_BETA1 = float(params.get("adam_beta1", config.ADAM_BETA1))
    config.ADAM_BETA2 = float(params.get("adam_beta2", config.ADAM_BETA2))
    # Per-trial metric to optimize/select best
    objective_metric = params.get("objective_metric", Config.OPTUNA_OBJECTIVE_METRIC)
    # Keep Config global used by reporting consistent
    config.OPTUNA_OBJECTIVE_METRIC = objective_metric
    config.OPTUNA_DIRECTION = OptunaConfig.get_direction(objective_metric)
    config.METRIC_FOR_BEST_MODEL = objective_metric
    config.CLASSIFIER_DROPOUT = float(params["classifier_dropout"])
    config.HIDDEN_DROPOUT_PROB = float(params["hidden_dropout_prob"])
    config.ATTENTION_PROBS_DROPOUT_PROB = float(params["attention_probs_dropout_prob"])

    # Build model and trainer using train.py
    model, trainer = create_model_and_trainer(tokenized_datasets, tokenizer, config)

    logger.info(f"Objective metric: {objective_metric}")
    logger.info(f"Optimization direction: {config.OPTUNA_DIRECTION}")

    # Initialize W&B for this trial
    run_name = f"optuna-trial-{trial.number}"
    os.environ.setdefault("WANDB_PROJECT", config.PROJECT_NAME)
    
    
    wandb.init(project=config.PROJECT_NAME, name=run_name, config=config.to_dict())
    d = config.to_dict()
    table = wandb.Table(columns=list(d.keys()), data=[list(d.values())])
    wandb.log({"config": table})

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Train and evaluate on validation set
        trainer.train()
        metrics = trainer.evaluate(tokenized_datasets["val"])  # returns eval_* keys
        score = float(metrics.get(objective_metric, float("nan")))
        logger.info(f"Trial {trial.number}: {objective_metric}={score:.6f}")
        return score
        
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        wandb.log({"error": str(e)})
        return float("nan")
    finally:
        # Finish W&B and free GPU memory between trials
        wandb.finish()
        del model, trainer
        torch.cuda.empty_cache()


def create_optuna_study() -> optuna.Study:
    """
    Create or load an Optuna study with proper configuration.
    
    Returns:
        optuna.Study: Configured study object
        params: dict: Hyperparameters
    """
    # Get Optuna configuration
    sampler_config = OptunaConfig.get_sampler_config()
    
    # Normalize and prepare SQLite storage path if provided
    storage_uri = Config.OPTUNA_STORAGE
    if storage_uri and storage_uri.startswith("sqlite"):
        prefix = "sqlite:///"
        path_part = storage_uri[len(prefix):] if storage_uri.startswith(prefix) else storage_uri.split("sqlite:///")[-1]
        abs_path = os.path.abspath(path_part)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        storage_uri = f"sqlite:////{abs_path.lstrip('/')}"

    if storage_uri:
        # Use persistent storage (SQLite database)
        study = optuna.create_study(
            study_name=Config.OPTUNA_STUDY_NAME,
            storage=storage_uri,
            load_if_exists=True,
            direction=OptunaConfig.get_direction(Config.OPTUNA_OBJECTIVE_METRIC),
            **sampler_config,
            # **pruning_config
        )
        logger.info(f"Using Optuna storage at {storage_uri}")
    else:
        # In-memory study
        study = optuna.create_study(
            study_name=Config.OPTUNA_STUDY_NAME,
            direction=OptunaConfig.get_direction(Config.OPTUNA_OBJECTIVE_METRIC),
            **sampler_config,
            # **pruning_config
        )
        logger.info("Created new in-memory study")
    
    return study


def main():
    """Main optimization function."""
    logger.info("Starting Optuna hyperparameter optimization for COVID-Voices...")
    logger.info(f"Model: {Config.MODEL_NAME}")
    logger.info(f"Task: COVID-19 Tweet Sentiment Classification")
   
    
    # Set seed for reproducibility
    set_seed(Config.SEED)
    
    # Prepare data once and reuse across trials
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    datasets = load_and_prepare_datasets()
    tokenized_datasets = datasets["tokenized_datasets"]
    tokenizer = datasets["tokenizer"]

    # Create or load study
    
    study = create_optuna_study()
    
    # Run optimization
    logger.info(f"Starting optimization with {Config.N_TRIALS} trials...")
    study.optimize(
        partial(objective, tokenized_datasets=tokenized_datasets, tokenizer=tokenizer), 
        n_trials=Config.N_TRIALS,
        show_progress_bar=False,
        callbacks=[
            # Robust logging for trials where value can be None (PRUNED/FAIL)
            lambda study, trial: logger.info(
                f"Trial {trial.number} finished with state={trial.state.name}, value={trial.value}"
            )
        ]
    )
    
    # Print final results
    logger.info("=" * 60)
    logger.info("OPTIMIZATION COMPLETED!")
    logger.info("=" * 60)
    
    # Check if any trials completed successfully
    completed_trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
    successful_trials = [t for t in completed_trials if t.value is not None and not (isinstance(t.value, float) and t.value != t.value)]  # filter out NaN
    
    if successful_trials:
        logger.info(f"Best trial number: {study.best_trial.number}")
        # Get the metric name from the best trial's user attributes or from config
        metric_name = study.best_trial.user_attrs.get('objective_metric', "eval_f1")
        logger.info(f"Best {metric_name} score: {study.best_value:.4f}")
        logger.info(f"Best hyperparameters:")
        for key, value in study.best_trial.params.items():
            logger.info(f"  {key}: {value}")
    else:
        logger.warning("No trials completed successfully!")
        logger.warning("All trials either failed or returned NaN values.")
        if completed_trials:
            logger.info(f"Found {len(completed_trials)} completed trials, but all had NaN/None values")
    
    # Save study results only if there are successful trials
    if successful_trials and Config.OPTUNA_OUTPUT_DIR:
        os.makedirs(Config.OPTUNA_OUTPUT_DIR, exist_ok=True)
        
        # Save as pickle
        study_path = os.path.join(Config.OPTUNA_OUTPUT_DIR, f"{Config.OPTUNA_STUDY_NAME}.pkl")
        try:
            study.export_models(study_path)
            logger.info(f"Study saved to {study_path}")
        except Exception as e:
            logger.warning(f"Could not save study to pickle: {e}")
        
        # Save as SQLite (for future loading)
        sqlite_path = os.path.join(Config.OPTUNA_OUTPUT_DIR, f"{Config.OPTUNA_STUDY_NAME}.db")
        try:
            study.storage = f"sqlite:///{sqlite_path}"
            logger.info(f"Study database saved to {sqlite_path}")
        except Exception as e:
            logger.warning(f"Could not save study database: {e}")
    elif not successful_trials:
        logger.warning("Skipping study save - no successful trials to save")
    
    # Print study statistics
    logger.info("\nStudy Statistics:")
    logger.info(f"  Total trials: {len(study.trials)}")
    logger.info(f"  Completed trials: {len(study.trials)}")
    logger.info(f"  Pruned trials: {len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))}")
    logger.info(f"  Complete trials: {len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))}")


if __name__ == "__main__":
    main()
