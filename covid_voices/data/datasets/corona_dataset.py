import os
from typing import Optional, Callable, Dict

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split as sk_train_test_split

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


def _validate_columns(processed_df: pd.DataFrame):
    required_columns = ["OriginalTweet", "Sentiment", "text", "label"]
    if not all(col in processed_df.columns for col in required_columns):
        raise ValueError(f"Processed DataFrame must contain the following columns: {required_columns}")


class CoronaTweetDataset(Dataset):
    """ 
    This dataset handles loading and preprocessing tweet data and prepares it for training
    """
    
    def __init__(self,
                 data_path: str = DATA_TRAIN_PATH,
                 preprocessing: Optional[Callable] = None,
                 label_mapping: Optional[Dict[str, int]] = None,
                 processed_df: Optional[pd.DataFrame] = None):
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
            _validate_columns(processed_df)  # validate columns
            self.df = processed_df.reset_index(drop=True)  # Use provided processed frame as-is
        else:
            # Read and preprocess from file
            self.df = pd.read_csv(data_path, encoding="latin1")
            self.df["text"] = self.df["OriginalTweet"].apply(self.preprocess)
            self.df["label"] = self.df["Sentiment"].map(self.label_mapping)

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
            dict: A dictionary with text and label
        """
        row = self.df.iloc[idx]
        return {"text": row["text"], "label": int(row["label"]) }
  
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
                      label_mapping: Optional[Dict[str, int]] = None):
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
        train_dataset = cls(train_path, preprocessing, label_mapping)
        test_dataset = cls(test_path, preprocessing, label_mapping)
        if not is_val_split:
            return {"train": train_dataset, "test": test_dataset}

        # Split train dataset into train and validation
        train_ds, val_ds = train_dataset.train_test_split(test_size=val_size, seed=seed, stratify=True)
        return {"train": train_ds, "val": val_ds, "test": test_dataset}


    def train_test_split(self, test_size: float = 0.2, seed: int = 42, stratify: bool = True):
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

        train_ds = CoronaTweetDataset(preprocessing=self.preprocess, label_mapping=self.label_mapping, processed_df=train_df)
        test_ds = CoronaTweetDataset(preprocessing=self.preprocess, label_mapping=self.label_mapping, processed_df=test_df)
        return train_ds, test_ds




# class CoronaTweetConfig(BuilderConfig):
#     """Configuration for CoronaTweetDataset builder."""

#     def __init__(
#         self,
#         *,
#         data_files: Optional[Dict[str, str]] = None,
#         label_mapping: Optional[Dict[str, int]] = None,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.data_files = data_files or {}
#         self.label_mapping = label_mapping or DEFAULT_LABEL_MAPPING


# class CoronaTweetDataset(GeneratorBasedBuilder):
#     """
#     Hugging Face dataset builder for the Corona NLP tweets dataset.
#     """

#     VERSION = datasets.Version("1.0.0")
#     BUILDER_CONFIG_CLASS = CoronaTweetConfig
#     DEFAULT_CONFIG_NAME = "default"

#     def _info(self) -> datasets.DatasetInfo:
#         label_names = [label for label, _ in sorted(self.config.label_mapping.items(), key=lambda kv: kv[1])]
#         features = Features(
#             {
#                 "OriginalTweet": Value("string"),
#                 "label": ClassLabel(num_classes=len(label_names), names=label_names),
#                 "Sentiment": Value("string"),
#             }
#         )
#         return datasets.DatasetInfo(
#             description="Corona NLP tweets with sentiment labels",
#             features=features,
#             supervised_keys=("OriginalTweet", "label"),
#             version=self.VERSION,
#         )

#     def _split_generators(self, dl_manager) -> list:
#         # Resolve files from config or defaults
#         train_path = self.config.data_files.get("train", DATA_TRAIN_PATH)
#         test_path = self.config.data_files.get("test", DATA_TEST_PATH)

#         return [
#             SplitGenerator(name=Split.TRAIN, gen_kwargs={"filepath": train_path}),
#             SplitGenerator(name=Split.TEST, gen_kwargs={"filepath": test_path}),
#         ]

#     def _generate_examples(self, filepath: str):
#         """Yield (key, example) tuples from the CSV file."""
#         df = pd.read_csv(filepath, encoding="latin1")

#         # Ensure required columns exist
#         required = {"OriginalTweet", "Sentiment"}
#         missing = required - set(df.columns)
#         if missing:
#             raise ValueError(f"Missing required columns in {filepath}: {missing}")

#         # Map sentiment to label ids
#         df["label"] = df["Sentiment"].map(self.config.label_mapping)

#         for idx, row in df.iterrows():
#             text = row["OriginalTweet"]
#             sent = row["Sentiment"]
#             label = int(row["label"]) if pd.notna(row["label"]) else None
#             if label is None:
#                 # Skip rows with unmapped labels
#                 continue
#             yield idx, {
#                 "OriginalTweet": text,
#                 "label": label,
#                 "Sentiment": sent,
#             }


# def load_datasets(
#     data_dir: str = "",
#     *,
#     preprocessing: Optional[Callable[[str], str]] = None,
#     label_mapping: Optional[Dict[str, int]] = None,
# ) -> DatasetDict:
#     """
#     Convenience loader that builds a DatasetDict using the CoronaTweetDataset builder
#     and applies optional text preprocessing to the OriginalTweet column.

#     Args:
#         data_dir: Directory containing CSV files (Corona_NLP_train.csv, Corona_NLP_test.csv).
#         preprocessing: Optional callable(text) -> text applied to OriginalTweet.
#         label_mapping: Optional sentiment->id mapping; defaults to 5-class mapping.

#     Returns:
#         DatasetDict with 'train' and 'test' splits.
#     """
#     train_path = DATA_TRAIN_PATH if not data_dir else os.path.join(data_dir, "Corona_NLP_train.csv")
#     test_path = DATA_TEST_PATH if not data_dir else os.path.join(data_dir, "Corona_NLP_test.csv")

#     builder = CoronaTweetDataset(
#         config=CoronaTweetConfig(
#             name="default",
#             data_files={"train": train_path, "test": test_path},
#             label_mapping=label_mapping or DEFAULT_LABEL_MAPPING,
#         )
#     )
#     builder.download_and_prepare()
#     ds: DatasetDict = builder.as_dataset()

#     if preprocessing is not None:
#         # Apply preprocessing with map; handle both batched and non-batched forms
#         def _preprocess_batch(batch):
#             texts = batch["OriginalTweet"]
#             if isinstance(texts, list):
#                 return {"OriginalTweet": [preprocessing(t) for t in texts]}
#             else:
#                 return {"OriginalTweet": preprocessing(texts)}

#         ds = ds.map(_preprocess_batch, batched=True)

#     return ds

