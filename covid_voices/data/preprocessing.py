"""
Data preprocessing utilities for COVID-19 tweet data.
"""

def preprocess_tweet(text: str) -> str:
    """
    Clean and normalize tweet text.
    
    Args:
        text (str): Raw tweet text
        
    Returns:
        str: Preprocessed tweet text
    """
    text = text.lower()
    text = text.replace('#', 'hashtag_')
    text = text.replace('@', 'mention_')
    return text
