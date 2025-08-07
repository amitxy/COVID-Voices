import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from typing import Optional, Callable, Dict

DATA_TRAIN_PATH = "data/processed/Corona_NLP_train.csv"
DATA_TEST_PATH = "data/processed/Corona_NLP_test.csv"

class CoronaTweetDataset(Dataset):
    """ 
    This dataset handles loading and preprocessing tweet data and prepares
    it for use with PyTorch models.
    """
    
    def __init__(self, 
                 data_path: str = DATA_TRAIN_PATH,
                 tokenizer = None,
                 max_length: int = 128,
                 preprocessing: Optional[Callable] = None,
                 label_mapping: Optional[Dict[str, int]] = None):
        """
        Initialize the dataset with raw data and preprocessing options
        
        Args:
            data_path (str): Path to the CSV file with tweet data
            tokenizer: HuggingFace tokenizer for text processing
            max_length (int): Maximum sequence length for tokenization
            preprocessing (callable, optional): Function to preprocess tweets
            label_mapping (dict, optional): Mapping from text labels to integers
        """
        # Default label mapping if none provided
        self.label_mapping = label_mapping or {
            "Extremely Negative": 0,
            "Negative": 1,
            "Neutral": 2,
            "Positive": 3,
            "Extremely Positive": 4
        }
        
        
        # Load the dataset
        self.df = pd.read_csv(data_path, encoding="latin1")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocessing = preprocessing
        
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        Args:
            idx (int): Index of the sample to fetch
            
        Returns:
            dict: A dictionary with input_ids, attention_mask and labels
        """
        tweet = self.df.iloc[idx]["OriginalTweet"]
        sentiment = self.df.iloc[idx]["Sentiment"]
        
        # Apply preprocessing if provided
        if self.preprocessing:
            tweet = self.preprocessing(tweet)
            
        # Convert label to numeric using mapping
        label = torch.tensor(self.label_mapping[sentiment], dtype=torch.long)
        
        # Return raw text and label if no tokenizer is provided
        if self.tokenizer is None:
            return {"text": tweet, "label": label}
        
        # Tokenize the tweet
        encoding = self.tokenizer(
            tweet,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Remove the batch dimension added by the tokenizer
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        return {
            "input_ids": input_ids, 
            "attention_mask": attention_mask, 
            "label": label
        }
    
  
    @classmethod
    def load_datasets(cls, 
                     data_dir: str='', 
                     tokenizer=None,
                     max_length: int = 128,
                     preprocessing: Optional[Callable] = None,
                     label_mapping: Optional[Dict[str, int]] = None,
                     transform: Optional[Callable] = None) -> Dict[str, 'CoronaTweetDataset']:
        """
        Factory method to load both train and test datasets with consistent parameters
        
        Args:
            data_dir (str): Directory containing the CSV files
            tokenizer: HuggingFace tokenizer for text processing
            max_length (int): Maximum sequence length for tokenization
            preprocessing (callable, optional): Function to preprocess tweets
            label_mapping (dict, optional): Mapping from text labels to integers
            transform (callable, optional): Additional transformations to apply to samples
            
        Returns:
            dict: Dictionary with 'train' and 'test' datasets
        """
        # Define paths for train and test files
        if not data_dir:
            train_path = DATA_TRAIN_PATH
            test_path = DATA_TEST_PATH
        else:
            train_path = os.path.join(data_dir, "Corona_NLP_train.csv")
            test_path = os.path.join(data_dir, "Corona_NLP_test.csv")
       

        # Create dataset instances
        train_dataset = cls(
            train_path,
            tokenizer=tokenizer,
            max_length=max_length,
            preprocessing=preprocessing,
            label_mapping=label_mapping
        )
        
        test_dataset = cls(
            test_path,
            tokenizer=tokenizer,
            max_length=max_length,
            preprocessing=preprocessing,
            label_mapping=label_mapping
        )
        
        return {"train": train_dataset, "test": test_dataset}
