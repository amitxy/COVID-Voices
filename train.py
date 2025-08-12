#!/usr/bin/env python3
"""
COVID-Voices Training Script

This script trains a sequence classification model on COVID-19 tweet data
using Hugging Face Transformers and PyTorch.
"""

import os
import logging
import numpy as np
import torch
import wandb
from datasets import Dataset, DatasetDict
import evaluate

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from covid_voices.data.corona_dataset import CoronaTweetDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration class for training parameters."""
    
    # Model and training
    MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"
    OUTPUT_BASE_DIR = "./checkpoints/"
    PROJECT_NAME = "corona-NLP-ensemble"
    
    # Training hyperparameters
    BATCH_SIZE = 128
    MAX_LENGTH = 280  # max length of tweet
    NUM_EPOCHS = 5
    LEARNING_RATE = 2e-5
    
    # Data
    VAL_SIZE = 0.2
    SEED = 42
    
    # Output
    OUTPUT_DIR = "./test_output"
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def preprocess_tweet(text: str) -> str:
    """Clean and normalize tweet text."""
    text = text.lower()
    text = text.replace('#', 'hashtag_')
    text = text.replace('@', 'mention_')
    return text


def make_tokenizer(model_name: str, max_length: int = 512):
    """Factory function to create a tokenizer function."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            # padding=False,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
    
    return tokenize, tokenizer


def load_and_prepare_datasets() -> dict:
    """Load and prepare datasets for training."""
    logger.info("Loading and preparing datasets...")
    
    # Load datasets with preprocessing
    datasets = CoronaTweetDataset.load_datasets(
        preprocessing=preprocess_tweet,
        is_val_split=True,
        val_size=Config.VAL_SIZE,
        seed=Config.SEED
    )
    
    logger.info(f"Loaded datasets: {list(datasets.keys())}")
    
    # Convert to Hugging Face format
    hf_datasets = DatasetDict({
        "train": Dataset.from_pandas(datasets["train"].df, preserve_index=False),
        "val": Dataset.from_pandas(datasets["val"].df, preserve_index=False),
        "test": Dataset.from_pandas(datasets["test"].df, preserve_index=False)
    })
    
    # Tokenize datasets
    tokenize_function, tokenizer = make_tokenizer(Config.MODEL_NAME, Config.MAX_LENGTH)
    columns_to_remove = list(set(hf_datasets["train"].column_names) - {"label"})
    
    tokenized_datasets = hf_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=columns_to_remove,
        desc="Tokenizing datasets"
    )
    
    logger.info("Datasets prepared successfully")
    return {
        "raw_datasets": datasets,
        "hf_datasets": hf_datasets,
        "tokenized_datasets": tokenized_datasets,
        "tokenizer": tokenizer
    }


def compute_metrics(eval_pred: tuple) -> dict:
    """Compute accuracy, macro-F1, macro-precision, macro-recall for single-label classification."""
    metric_names = ["accuracy", "f1", "precision", "recall"]
    metrics = {name: evaluate.load(name) for name in metric_names}

    logits, labels = eval_pred
    
    # Ensure tensors are properly shaped for multi-GPU
    if hasattr(logits, 'cpu'):
        logits = logits.cpu().numpy()
    if hasattr(labels, 'cpu'):
        labels = labels.cpu().numpy()
    
    # Handle different tensor shapes from multi-GPU training
    if len(logits.shape) == 3:
        logits = logits.reshape(-1, logits.shape[-1])
    if len(labels.shape) == 2:
        labels = labels.reshape(-1)
    
    # single-label classification: pick the highest logit
    predictions = np.argmax(logits, axis=-1)

    results = {}
    # accuracy
    results.update(metrics["accuracy"].compute(predictions=predictions, references=labels))

    # macro-averaged metrics (multiclass-safe; note: in binary this is macro, not 'positive class' binary)
    for name in ["f1", "precision", "recall"]:
        res = metrics[name].compute(
            predictions=predictions,
            references=labels,
            average="macro",
        )
        results.update(res)

    return results


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
    
    # Calculate effective batch size per device
    effective_batch_size = Config.BATCH_SIZE // max(Config.NUM_GPUS, 1)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        eval_strategy="epoch",
        per_device_train_batch_size=effective_batch_size,
        per_device_eval_batch_size=effective_batch_size,
        num_train_epochs=Config.NUM_EPOCHS,
        learning_rate=Config.LEARNING_RATE,
        do_train=True,
        do_eval=True,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=3,
        # Multi-GPU settings
        dataloader_num_workers=4 * max(Config.NUM_GPUS, 1),  # Scale workers with GPUs
        dataloader_pin_memory=True,
        # Fix for multi-GPU tensor gathering issues
        remove_unused_columns=False,
        group_by_length=True,
        # Better handling of distributed training
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=25,
        # Additional fixes for tensor gathering warnings
        dataloader_drop_last=True,
        gradient_accumulation_steps=1,
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
        name=Config.MODEL_NAME,
        reinit=True,
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
        os.makedirs(model_save_path, exist_ok=True)
        
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
