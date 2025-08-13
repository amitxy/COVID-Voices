import os
import yaml
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import optuna


@dataclass
class Config:
    """Configuration class for training parameters with Optuna support."""
    
    # Project settings
    PROJECT_NAME: str = "corona-NLP-ensemble"
    MODEL_NAME: str = "huawei-noah/TinyBERT_General_4L_312D"
    
    # Optuna settings
    USE_OPTUNA: bool = True
    N_TRIALS: int = 20
    OPTUNA_STUDY_NAME: str = "covid_voices_optimization"
    OPTUNA_STORAGE: Optional[str] = None  # SQLite database path for study persistence
    
    # Data settings
    SEED: int = 42
    VAL_SIZE: float = 0.2
    MAX_LENGTH: int = 280
    
    # Training hyperparameters (will be optimized by Optuna)
    BATCH_SIZE: int = 128  # REMOVE
    NUM_EPOCHS: int = 2
    LEARNING_RATE: float = 2e-5
    LR_SCHEDULER_TYPE: str = "linear"  # one of {"linear", "cosine"}
    WEIGHT_DECAY: float = 0.01
    WARMUP_STEPS: int = 100
    GRADIENT_ACCUMULATION_STEPS: int = 1
    EARLY_STOPPING_PATIENCE: int = 3
    
    # Model hyperparameters
    DROPOUT: float = 0.1
    CLASSIFIER_DROPOUT: float = 0.1
    HIDDEN_DROPOUT_PROB: float = 0.1
    ATTENTION_PROBS_DROPOUT_PROB: float = 0.1
    ATTENTION_PROPOUT_PROB: float = 0.1  # legacy typo, not used
    
    # Device settings
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_GPUS: int = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # Output directories
    OUTPUT_BASE_DIR: str = "./checkpoints/"
    OUTPUT_DIR: str = "./test_output"
    OPTUNA_OUTPUT_DIR: str = "./optuna_studies/"

    # Metrics
    METRIC_FOR_BEST_MODEL: str = "f1"          # Trainer will look for eval_f1
    OPTUNA_OBJECTIVE_METRIC: str = "eval_f1"   # Metric name returned by compute_metrics
    OPTUNA_DIRECTION: str = "maximize"         # "maximize" or "minimize"
    GREATER_IS_BETTER: bool = True if OPTUNA_DIRECTION == "maximize" else False            


class OptunaConfig:
    """Optuna-specific configuration for COVID-19 tweet classification optimization."""
    
    @staticmethod
    def get_hyperparameter_search_space(trial: optuna.Trial) -> dict:
        """
        Define hyperparameter search space specifically for COVID-19 tweet classification.
        
        Based on the actual model architecture and training setup in train.py:
        - AutoModelForSequenceClassification (BERT-based)
        - Multi-class classification (5 sentiment classes)
        - Multi-GPU training with gradient accumulation
        
        Args:
            trial: Optuna trial object
            
        Returns:
            dict: Dictionary of suggested hyperparameters
        """
        return {
            # Core training hyperparameters
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-4, log=True),

            # Batch size (considering multi-GPU setup)
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        
            # Training duration
            "num_epochs": trial.suggest_int("num_epochs", 5, 10),

            # Regularization - May add a "zero weight decay" option
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),

            "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine"]),

            # Early stopping patience for callbacks
            "early_stopping_patience": trial.suggest_int("early_stopping_patience", 3, 5),

            # Model architecture specific (for BERT-based models)
            "classifier_dropout": trial.suggest_float("classifier_dropout", 0.0, 0.3),
            "hidden_dropout_prob": trial.suggest_float("hidden_dropout_prob", 0.0, 0.3),
            "attention_probs_dropout_prob": trial.suggest_float("attention_probs_dropout_prob", 0.0, 0.3),

          
        }
    
    @staticmethod
    def get_objective_metric() -> str:
        """Get the primary metric to optimize for COVID-19 sentiment classification."""
        return Config.OPTUNA_OBJECTIVE_METRIC
    
    @staticmethod
    def get_direction() -> str:
        """Get the optimization direction."""
        return Config.OPTUNA_DIRECTION
    
    @staticmethod
    def get_pruning_config() -> dict:
        """Get Optuna pruning configuration for early stopping unpromising trials."""
        return {
            "pruner": optuna.pruners.MedianPruner(
                n_startup_trials=5,      # Don't prune first 5 trials
                n_warmup_steps=100,      # Wait 100 steps before pruning
                interval_steps=50         # Check every 50 steps
            )
        }
    
    @staticmethod
    def get_sampler_config() -> dict:
        """Get Optuna sampler configuration for efficient search."""
        return {
            "sampler": optuna.samplers.TPESampler(
                seed=42,                 # For reproducibility
                n_startup_trials=10,     # Random search for first 10 trials
                n_ei_candidates=24       # Number of candidates for EI
            )
        }