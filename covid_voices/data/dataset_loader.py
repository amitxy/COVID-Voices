"""
Dataset loading and preparation utilities for COVID-19 tweet data.
"""

from datasets import Dataset, DatasetDict
from covid_voices.data import CoronaTweetDataset, preprocess_tweet, make_tokenizer
from covid_voices.config import Config
import logging

logger = logging.getLogger(__name__)


def load_and_prepare_datasets() -> dict:
    """
    Load and prepare datasets for training.
    
    Returns:
        dict: Dictionary containing raw_datasets, hf_datasets, tokenized_datasets, and tokenizer
    """
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
