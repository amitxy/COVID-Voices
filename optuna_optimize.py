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
from functools import partial
from covid_voices.config import Config, OptunaConfig
from covid_voices.data import load_and_prepare_datasets
from covid_voices.utils import init_logging, set_seed, build_wandb_config
from train import create_model_and_trainer

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
    config.LEARNING_RATE = float(params.get("learning_rate", config.LEARNING_RATE))
    config.NUM_EPOCHS = int(params.get("num_epochs", config.NUM_EPOCHS))
    config.BATCH_SIZE = int(params.get("batch_size", config.BATCH_SIZE))
    config.LR_SCHEDULER_TYPE = str(params.get("lr_scheduler_type", config.LR_SCHEDULER_TYPE))
    config.WEIGHT_DECAY = float(params.get("weight_decay", config.WEIGHT_DECAY))
    config.EARLY_STOPPING_PATIENCE = int(params.get("early_stopping_patience", config.EARLY_STOPPING_PATIENCE))
    # Per-trial metric to optimize/select best
    objective_metric = params.get("objective_metric", Config.OPTUNA_OBJECTIVE_METRIC)
    # Keep Config global used by reporting consistent
    Config.OPTUNA_OBJECTIVE_METRIC = objective_metric
    config.METRIC_FOR_BEST_MODEL = objective_metric
    config.CLASSIFIER_DROPOUT = float(params["classifier_dropout"])
    config.HIDDEN_DROPOUT_PROB = float(params["hidden_dropout_prob"])
    config.ATTENTION_PROBS_DROPOUT_PROB = float(params["attention_probs_dropout_prob"])

    # Build model and trainer using train.py
    model, trainer = create_model_and_trainer(tokenized_datasets, tokenizer, config)

    # Initialize W&B for this trial
    run_name = f"optuna-{trial.number}-{time.strftime('%Y%m%d-%H%M%S')}-{config.MODEL_NAME}"
    os.environ.setdefault("WANDB_PROJECT", config.PROJECT_NAME)
    base_cfg = build_wandb_config(trainer, config, tokenized_datasets=None, trial_number=trial.number)
    wandb.init(project=config.PROJECT_NAME, name=run_name, config=base_cfg)

    try:
        # Train and evaluate on validation set
        trainer.train()
        metrics = trainer.evaluate(tokenized_datasets["val"])  # returns eval_* keys
        score = float(metrics.get(objective_metric, float("nan")))
        logger.info(f"Trial {trial.number}: {objective_metric}={score:.6f}")
        
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        wandb.log({"error": str(e)})
        return float("nan")
    finally:
        # Finish W&B and free GPU memory between trials
        try:
            wandb.finish()
        except Exception:
            pass
        try:
            import torch
            del model, trainer
            torch.cuda.empty_cache()
        except Exception:
            pass


def create_optuna_study() -> optuna.Study:
    """
    Create or load an Optuna study with proper configuration.
    
    Returns:
        optuna.Study: Configured study object
    """
    # Get Optuna configuration
    sampler_config = OptunaConfig.get_sampler_config()
    pruning_config = OptunaConfig.get_pruning_config()
    
    if Config.OPTUNA_STORAGE:
        # Use persistent storage (SQLite database)
        study = optuna.create_study(
            study_name=Config.OPTUNA_STUDY_NAME,
            storage=Config.OPTUNA_STORAGE,
            load_if_exists=True,
            direction=OptunaConfig.get_direction(),
            **sampler_config,
            **pruning_config
        )
        logger.info(f"Loaded existing study from {Config.OPTUNA_STORAGE}")
    else:
        # In-memory study
        study = optuna.create_study(
            study_name=Config.OPTUNA_STUDY_NAME,
            direction=OptunaConfig.get_direction(),
            **sampler_config,
            **pruning_config
        )
        logger.info("Created new in-memory study")
    
    return study


def main():
    """Main optimization function."""
    logger.info("Starting Optuna hyperparameter optimization for COVID-Voices...")
    logger.info(f"Model: {Config.MODEL_NAME}")
    logger.info(f"Task: COVID-19 Tweet Sentiment Classification")
    logger.info(f"Objective metric: {OptunaConfig.get_objective_metric()}")
    logger.info(f"Optimization direction: {OptunaConfig.get_direction()}")
    
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
        show_progress_bar=True,
        callbacks=[
            # Log intermediate results
            lambda study, trial: logger.info(
                f"Trial {trial.number} completed with value: {trial.value:.4f}"
            )
        ]
    )
    
    # Print final results
    logger.info("=" * 60)
    logger.info("OPTIMIZATION COMPLETED!")
    logger.info("=" * 60)
    logger.info(f"Best trial number: {study.best_trial.number}")
    logger.info(f"Best F1 score: {study.best_value:.4f}")
    logger.info(f"Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        logger.info(f"  {key}: {value}")
    
    # Save study results
    if Config.OPTUNA_OUTPUT_DIR:
        os.makedirs(Config.OPTUNA_OUTPUT_DIR, exist_ok=True)
        
        # Save as pickle
        study_path = os.path.join(Config.OPTUNA_OUTPUT_DIR, f"{Config.OPTUNA_STUDY_NAME}.pkl")
        study.export_models(study_path)
        logger.info(f"Study saved to {study_path}")
        
        # Save as SQLite (for future loading)
        sqlite_path = os.path.join(Config.OPTUNA_OUTPUT_DIR, f"{Config.OPTUNA_STUDY_NAME}.db")
        study.storage = f"sqlite:///{sqlite_path}"
        logger.info(f"Study database saved to {sqlite_path}")
    
    # Print study statistics
    logger.info("\nStudy Statistics:")
    logger.info(f"  Total trials: {len(study.trials)}")
    logger.info(f"  Completed trials: {len(study.trials)}")
    logger.info(f"  Pruned trials: {len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))}")
    logger.info(f"  Complete trials: {len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))}")


if __name__ == "__main__":
    main()
