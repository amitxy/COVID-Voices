#!/usr/bin/env python3
"""
COVID-Voices Training Script

This script trains a sequence classification model on COVID-19 tweet data
using Hugging Face Transformers and PyTorch.
"""

import os
import torch
import wandb
from datasets import Dataset, DatasetDict
import time

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from covid_voices.data import CoronaTweetDataset, load_and_prepare_datasets
from covid_voices.config import Config
from covid_voices.utils import init_logging, set_seed, ensure_dir
from covid_voices.metrics import compute_metrics

logger = init_logging()

def create_model_and_trainer(tokenized_datasets: DatasetDict, tokenizer: AutoTokenizer):
    """Create model and trainer."""
    logger.info("Creating model and trainer...")
    
    # Get number of labels from dataset
    num_labels = len(set(tokenized_datasets["train"]["label"]))
    logger.info(f"Number of labels: {num_labels}")
    
    # Create model
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=num_labels
    )
    

    effective_batch_size = Config.BATCH_SIZE // max(Config.NUM_GPUS, 1)
    
    # Training arguments
    training_args = TrainingArguments(

        num_train_epochs=Config.NUM_EPOCHS,
        lr_scheduler_type="linear",
        learning_rate=Config.LEARNING_RATE,
        # Splits the batch evenly across devices
        per_device_train_batch_size=effective_batch_size,
        per_device_eval_batch_size=effective_batch_size, 

        output_dir=Config.OUTPUT_DIR,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="best",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
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
    )
    
    logger.info("Model and trainer created successfully")
    return model, trainer

def train_model(trainer: Trainer, tokenizer: AutoTokenizer, tokenized_datasets: DatasetDict):
    """Train the model."""
    logger.info("Starting model training...")
    
    # Initialize wandb
    wandb.init(
        project=Config.PROJECT_NAME,
        name=f"{time.strftime('%Y%m%d-%H%M%S')}-{Config.MODEL_NAME}",
        config={
            "model_name": Config.MODEL_NAME,
            "batch_size": Config.BATCH_SIZE,
            "max_length": Config.MAX_LENGTH,
            "num_epochs": Config.NUM_EPOCHS,
            "learning_rate": Config.LEARNING_RATE,
        }
    )
    
    try:
        # Train the model
        trainer.train()
        
        # Save the model
        model_save_path = os.path.join(Config.OUTPUT_BASE_DIR, Config.MODEL_NAME)
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
        model, trainer = create_model_and_trainer(tokenized_datasets, tokenizer)
        
        # Train the model
        train_model(trainer, tokenizer, tokenized_datasets)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
