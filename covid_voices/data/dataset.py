import os
from typing import Optional, Callable, Dict
import logging
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split as sk_train_test_split
from transformers import AutoTokenizer

DATA_DIR = "data/processed"
DATA_TRAIN_PATH = f"{DATA_DIR}/Corona_NLP_train.csv"
DATA_TEST_PATH = f"{DATA_DIR}/Corona_NLP_test.csv"

DEFAULT_LABEL_MAPPING: Dict[str, int] = {
    "Extremely Negative": 0,
    "Negative": 1,
    "Neutral": 2,
    "Positive": 3,
    "Extremely Positive": 4,
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


# def _validate_columns(processed_df: pd.DataFrame):
#     required_columns = ["OriginalTweet", "Sentiment", "text", "label"]
#     if not all(col in processed_df.columns for col in required_columns):
#         raise ValueError(f"Processed DataFrame must contain the following columns: {required_columns}")


class CoronaTweetDataset(Dataset):
    """ 
    This dataset handles loading and preprocessing tweet data and prepares it for training
    """
    
    def __init__(self,
                 data_path: str = DATA_TRAIN_PATH,
                 preprocessing: Optional[Callable] = None,
                 label_mapping: Optional[Dict[str, int]] = None,
                 processed_df: Optional[pd.DataFrame] = None, 
                 tokenizer: Optional[AutoTokenizer] = None,
                 max_len: int = 64):
        """
        Initialize the dataset with raw data and preprocessing options.

        Args:
            data_path: Path to the CSV file with tweet data.
            preprocessing: Callable to preprocess tweet text; defaults to identity.
            label_mapping: Optional mapping from sentiment text labels to integer IDs.
            processed_df: Optional preconstructed DataFrame with columns [OriginalTweet, Sentiment, text, label]. If provided, data_path is ignored.
        """
        # Default label mapping if none provided
        self.label_mapping = label_mapping or DEFAULT_LABEL_MAPPING
        self.num_labels = len(self.label_mapping)
        self.preprocess = preprocessing or (lambda x: x)

        if processed_df is not None:
            self.df = processed_df.reset_index(drop=True)  # Use provided processed frame as-is
        else:
            # Read and preprocess from file
            self.df = pd.read_csv(data_path, encoding="latin1")
            self.df["text"] = self.df["OriginalTweet"].apply(self.preprocess)
            self.df["label"] = self.df["Sentiment"].map(self.label_mapping)

        self.tokenizer = tokenizer
        self.max_len = max_len

        super().__init__()

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        Args:
            idx (int): Index of the sample to fetch
            
        Returns:
            dict: A dictionary with input_ids, attention_mask, and label
        """
        row = self.df.iloc[idx]
        text, label = row["text"], int(row["label"])

        if self.tokenizer is not None:
            # Tokenize the text and return the expected format
            tokenized = self.tokenizer(
                text, 
                truncation=True, 
                padding='max_length', 
                max_length=self.max_len, 
                return_tensors='pt'
            )
            # Remove the batch dimension since DataLoader will add it
            return {
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0),
                "labels": label
            }
        else:
            # Return raw text if no tokenizer
            return {"text": text, "label": label}
  
    @property
    def label2id(self) -> Dict[str, int]:
        """
        Get the label to ID mapping
        
        Returns:
            dict: Mapping from sentiment labels to integer IDs
        """
        return self.label_mapping
    
    @property
    def id2label(self) -> Dict[int, str]:
        """
        Get the ID to label mapping
        
        Returns:
            dict: Mapping from integer IDs to sentiment labels
        """
        return {v: k for k, v in self.label_mapping.items()}
    
    @classmethod
    def load_datasets(cls,
                      is_val_split: bool = False,
                      val_size: float = 0.2,
                      seed: int = 42,
                      data_dir: str = '',
                      preprocessing: Optional[Callable] = None,
                      label_mapping: Optional[Dict[str, int]] = None,
                      tokenizer: Optional[AutoTokenizer] = None,
                      max_len: int = 64):
        """
        Factory method to load train/test (and optional val) Torch datasets with consistent parameters.

        Args:
            is_val_split: Whether to split off a validation set from train.
            val_size: Fraction of train to use as validation when is_val_split is True.
            seed: Random seed for reproducibility.
            data_dir: Directory containing the CSV files; uses defaults when empty.
            preprocessing: Function to preprocess tweets.
            label_mapping: Mapping from text labels to integers; defaults to 5-class mapping.

        Returns:
            dict: {"train": Dataset, "test": Dataset} or {"train", "val", "test"}
        """
        # Define paths for train and test files
        train_path = DATA_TRAIN_PATH if not data_dir else os.path.join(data_dir, "Corona_NLP_train.csv")
        test_path = DATA_TEST_PATH if not data_dir else os.path.join(data_dir, "Corona_NLP_test.csv")
       
        # Create dataset instances
        train_dataset = cls(train_path, preprocessing, label_mapping, None, tokenizer, max_len)
        test_dataset = cls(test_path, preprocessing, label_mapping, None, tokenizer, max_len)
        if not is_val_split:
            return {"train": train_dataset, "test": test_dataset}

        # Split train dataset into train and validation
        train_ds, val_ds = train_dataset.train_test_split(test_size=val_size, seed=seed, stratify=True, tokenizer=tokenizer, max_len=max_len)
        return {"train": train_ds, "val": val_ds, "test": test_dataset}


    def train_test_split(self, test_size: float = 0.2, seed: int = 42, stratify: bool = True, tokenizer: Optional[AutoTokenizer] = None, max_len: int = 64):
        """
        Split this Torch dataset into train and test CoronaTweetDataset instances.

        Args:
            test_size: Fraction of samples to use for the test split (0 < test_size < 1).
            seed: Random seed for reproducibility.
            stratify: If True, perform stratified split based on the 'label' column.

        Returns:
            (train_dataset, test_dataset): Tuple of CoronaTweetDataset instances.
        """
        n = len(self.df)
        if not (0.0 < test_size < 1.0):
            raise ValueError("test_size must be a float in (0, 1)")

        indices = np.arange(n)
        strat = self.df["label"] if stratify else None
        train_idx, test_idx = sk_train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=strat,
        )

        train_df = self.df.iloc[train_idx].reset_index(drop=True)
        test_df = self.df.iloc[test_idx].reset_index(drop=True)

        train_ds = CoronaTweetDataset("", self.preprocess, self.label_mapping, processed_df=train_df, tokenizer=tokenizer, max_len=max_len)
        test_ds = CoronaTweetDataset("", self.preprocess, self.label_mapping, processed_df=test_df, tokenizer=tokenizer, max_len=max_len)
        return train_ds, test_ds
