"""
Tokenizer utilities for COVID-19 tweet data.
"""
from transformers import AutoTokenizer


def make_tokenizer(model_name: str, max_length: int = 512):
    """
    Factory function to create a tokenizer function.
    
    Args:
        model_name (str): Name of the pretrained model
        max_length (int): Maximum sequence length
        
    Returns:
        tuple: (tokenize_function, tokenizer_object)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            padding="max_length", #padding=False
            truncation=True,
            max_length=max_length,
        )
    
    return tokenize, tokenizer
