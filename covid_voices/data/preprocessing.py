"""
Data preprocessing utilities for COVID-19 tweet data.
"""
import re

url_re = re.compile(r'(?i)\b(?:https?://|ftp://|www\.)[^\s<>\)\]\}"]+')
def remove_urls(text):
    cleaned = url_re.sub('', text)
    # Tidy extra spaces/newlines left behind
    cleaned = re.sub(r'[ \t]+', ' ', cleaned).strip()
    return cleaned


def preprocess_tweet(text):
        """Clean and normalize tweet text"""
        text = remove_urls(text)
        text = text.replace('#', 'hashtag_')
        text = text.replace('@', 'mention_')
        return text