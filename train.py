#!/usr/bin/env python3
"""
COVID-Voices Training Script

This script trains a sequence classification model on COVID-19 tweet data
using Hugging Face Transformers and PyTorch.
"""

import os
import torch
import wandb
from datasets import DatasetDict
import time

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from covid_voices.data import load_and_prepare_datasets
from covid_voices.config import Config
from covid_voices.utils import init_logging, set_seed, ensure_dir
from covid_voices.metrics import compute_metrics

logger = init_logging()

def create_model_and_trainer(tokenized_datasets: DatasetDict, tokenizer: AutoTokenizer, config: Config):
    """Create model and trainer."""
    logger.info("Creating model and trainer...")
    
    # Get number of labels from dataset
    num_labels = len(set(tokenized_datasets["train"]["label"]))
    logger.info(f"Number of labels: {num_labels}")
    
    # Create model
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=num_labels,
        hidden_dropout_prob=config.HIDDEN_DROPOUT_PROB,
        attention_probs_dropout_prob=config.ATTENTION_PROBS_DROPOUT_PROB,
        classifier_dropout=config.CLASSIFIER_DROPOUT,
        ignore_mismatched_sizes=True,  # Ignore classifier size mismatch For twitter-roberta-base-sentiment-latest
    )
    

    effective_batch_size = max(config.BATCH_SIZE // max(config.NUM_GPUS, 1), 8)
    
    # Training arguments
    training_args = TrainingArguments(

        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        lr_scheduler_type=config.LR_SCHEDULER_TYPE,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
        greater_is_better=config.GREATER_IS_BETTER,
        
        # New training parameters
        optim=config.OPTIM,
        adam_epsilon=config.ADAM_EPS,
        adam_beta1=config.ADAM_BETA1,
        adam_beta2=config.ADAM_BETA2,
        
        # Splits the batch evenly across devices
        per_device_train_batch_size=effective_batch_size,
        per_device_eval_batch_size=effective_batch_size, 

        output_dir=config.OUTPUT_DIR,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="best",
        load_best_model_at_end=True,
        logging_steps=500,
        save_total_limit=1,
        disable_tqdm=True,      # hide Trainer bars
        log_level="info",
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOPPING_PATIENCE),
        ],
    )
    
    logger.info("Model and trainer created successfully")
    return model, trainer

def train_model(trainer: Trainer, tokenizer: AutoTokenizer, tokenized_datasets: DatasetDict, config: Config):
    """Train the model."""
    logger.info("Starting model training...")
    
    wandb.init(
        project=config.PROJECT_NAME,
        name=f"{time.strftime('%Y%m%d-%H%M%S')}-{config.MODEL_NAME}",
        config=config.to_dict()
    )
    
    
    try:
        # Train the model
        trainer.train()
        
        # Save the model
        model_save_path = os.path.join(config.OUTPUT_BASE_DIR, config.PROJECT_NAME, config.MODEL_NAME, config.TRIAL_NUMBER)
        ensure_dir(model_save_path)
        
        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        
        logger.info(f"Model saved to {model_save_path}")
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(tokenized_datasets["test"])
        logger.info(f"Test results: {test_results}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        wandb.finish()

def main():
    """Main training function."""
    logger.info("Starting COVID-Voices training pipeline...")
    
    # Set seed for reproducibility
    set_seed(Config.SEED)
    
    # Check device and GPU info
    logger.info(f"Using device: {Config.DEVICE}")
    logger.info(f"Number of GPUs: {Config.NUM_GPUS}")
    if Config.NUM_GPUS > 0:
        logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"Effective batch size per device: {Config.BATCH_SIZE // Config.NUM_GPUS}")
        logger.info(f"Total effective batch size: {Config.BATCH_SIZE}")
    
    try:
        # Load and prepare datasets
        datasets = load_and_prepare_datasets()
        tokenized_datasets = datasets["tokenized_datasets"]

        tokenizer = datasets["tokenizer"]
        
        # Create model and trainer
        config = Config()
        model, trainer = create_model_and_trainer(tokenized_datasets, tokenizer, config)
        
        # Train the model
        train_model(trainer, tokenizer, tokenized_datasets, config)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
