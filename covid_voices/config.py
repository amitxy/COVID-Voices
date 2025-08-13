import torch
from dataclasses import dataclass

@dataclass
class Config:
    PROJECT_NAME: str = "corona-NLP-ensemble"
    MODEL_NAME: str = "huawei-noah/TinyBERT_General_4L_312D"

    # data
    SEED: int = 42
    VAL_SIZE: float = 0.2
    MAX_LENGTH: int = 280

    # Training
    BATCH_SIZE: int = 128
    NUM_EPOCHS: int = 2
    LEARNING_RATE: float = 2e-5

    # Device
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_GPUS: int = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # DIRS
    OUTPUT_BASE_DIR: str = "./checkpoints/"
    OUTPUT_DIR: str = "./test_output"