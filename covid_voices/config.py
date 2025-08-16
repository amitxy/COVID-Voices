import os
import yaml
import torch
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Any, Iterable
import optuna


@dataclass
class Config:
    """Configuration class for training parameters with Optuna support."""
    
    # Project settings
    # PROJECT_NAME: str = "corona-NLP-ensemble"
    PROJECT_NAME: str = "TinyBERT_RUN_3-more_optimizers"
    MODEL_NAME: str = "huawei-noah/TinyBERT_General_4L_312D"
    TRIAL_NUMBER: int = 0
    # Optuna settings
    USE_OPTUNA: bool = True
    N_TRIALS: int = 20
    OPTUNA_STUDY_NAME: str = "covid_voices_optimization"
    OPTUNA_STORAGE: Optional[str] = "sqlite:///optuna_studies/covid_voices_opt_2.db"  # SQLite database path for study persistence
    
    # Data settings
    SEED: int = 42
    VAL_SIZE: float = 0.2
    MAX_LENGTH: int = 350 #280
    
    # Training hyperparameters (will be optimized by Optuna)
    BATCH_SIZE: int = 128  # REMOVE
    NUM_EPOCHS: int = 15
    LEARNING_RATE: float = 2e-5
    LR_SCHEDULER_TYPE: str = "linear"  # one of {"linear", "cosine"}
    WEIGHT_DECAY: float = 0.01
    EARLY_STOPPING_PATIENCE: int = 3
    OPTIM: str = "adamw_torch"
    ADAM_EPS: float = 1e-8
    ADAM_BETA1: float = 0.9
    ADAM_BETA2: float = 0.999

    
    # Model hyperparameters
    DROPOUT: float = 0 # legacy typo, not used
    CLASSIFIER_DROPOUT: float = 0.1
    HIDDEN_DROPOUT_PROB: float = 0.1
    ATTENTION_PROBS_DROPOUT_PROB: float = 0.1
    ATTENTION_PROPOUT_PROB: float = 0 # legacy typo, not used
    
   
    
    # Metrics
    METRIC_FOR_BEST_MODEL: str = "eval_f1"      # Trainer compares on this key
    OPTUNA_OBJECTIVE_METRIC: str = "eval_f1"   # Metric name returned by compute_metrics
    OPTUNA_DIRECTION: str =   "maximize"      # "maximize" or "minimize"
    GREATER_IS_BETTER: bool = True if OPTUNA_DIRECTION == "maximize" else False            

    # Device settings
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_GPUS: int = torch.cuda.device_count() if torch.cuda.is_available() else 0
     
     # Output directories
    OUTPUT_BASE_DIR: str = "./checkpoints/"
    OUTPUT_DIR: str = "./test_output"
    OPTUNA_OUTPUT_DIR: str = "./optuna_studies/"

    def to_dict(self) -> Dict[str, Any]:
        exclude = ["DEVICE","OUTPUT_BASE_DIR", "OUTPUT_DIR",
         "OPTUNA_OUTPUT_DIR", "OPTUNA_STUDY_NAME", "OPTUNA_STORAGE", "OPTUNA_DIRECTION", "GREATER_IS_BETTER","OPTUNA_OBJECTIVE_METRIC", "PROJECT_NAME", "MODEL_NAME"]
        d = asdict(self)
        d["MODEL_NAME"] = self.MODEL_NAME.split("/")[-1]
        d = {k: v for k, v in d.items() if k not in exclude}
        return d


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
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 4e-4, log=True),

            # Batch size (considering multi-GPU setup)
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512, 1024]),
        
            # Training duration
            "num_epochs": 20, #trial.suggest_int("num_epochs", 5, 10),

            # Regularization - May add a "zero weight decay" option
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),

            "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine", "cosine_with_restarts"]),

            # Early stopping patience for callbacks
            "early_stopping_patience": 6, #trial.suggest_int("early_stopping_patience", 3, 5),

            # Model architecture specific (for BERT-based models)
            # "classifier_dropout": trial.suggest_float("classifier_dropout", 0.0, 0.3),
            "classifier_dropout": trial.suggest_float("classifier_dropout", 0.0, 0.1),
            "hidden_dropout_prob": trial.suggest_float("hidden_dropout_prob", 0.0, 0.3),
            "attention_probs_dropout_prob": trial.suggest_float("attention_probs_dropout_prob", 0.0, 0.3),
            "optim": trial.suggest_categorical("optim", ["adamw_torch", "adamw_hf", "adafactor"]),
            "adam_eps": trial.suggest_float("adam_eps", 1e-8, 1e-4, log=True),
            "adam_beta1": trial.suggest_float("adam_beta1", 0.8, 0.999),
            "adam_beta2": trial.suggest_float("adam_beta2", 0.8, 0.999),

            # Which eval metric to optimize/select best model by
            "objective_metric": trial.suggest_categorical(
                "objective_metric",
                [
                    "eval_f1",
                    # "eval_accuracy",
                    # "eval_precision",
                    # "eval_recall",
                    "eval_roc_auc",
                    "eval_loss"  # new
                ]
            ),

          
        }
    

    @staticmethod
    def get_direction(metric: str) -> str:
        """Get the optimization direction."""
        return "minimize" if metric == "eval_loss" else "maximize"
    
        
    @staticmethod
    def get_sampler_config() -> dict:
        """Get Optuna sampler configuration for efficient search."""
        return {
            "sampler": optuna.samplers.TPESampler(
                # seed=Config.SEED,                 # For reproducibility
                n_startup_trials=5,     # Random search for first 5 trials
                n_ei_candidates=24,       # Number of candidates for Expected Improvement (EI)
            )
        }