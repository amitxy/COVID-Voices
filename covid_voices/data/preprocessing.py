"""
Data preprocessing utilities for COVID-19 tweet data.
"""
import re

url_re = re.compile(r'(?i)\b(?:https?://|ftp://|www\.)[^\s<>\)\]\}"]+')
def remove_urls(text):
    cleaned = url_re.sub('', text)
    cleaned = re.sub(r'[ \t]+', ' ', cleaned).strip()
    return cleaned

def remove_mentions(text):
    cleaned = re.sub(r'@[A-Za-z0-9]+', '', text)
    return cleaned

def remove_hashtags(text):
    cleaned = re.sub(r'#', '', text)
    return cleaned

def remove_html_tags(text):
    cleaned = re.sub(r'<[^>]*>', '', text)
    return cleaned

def remove_special_characters(text):
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return cleaned



def preprocess_text(text):
        """Clean and normalize tweet text"""
        text = remove_urls(text)
        text = text.replace('#', 'hashtag_')
        text = text.replace('@', 'mention_')
        return text


def preprocess_model_defualt(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def remove_empty_lines(text):
    text = re.sub(r'\n+', '\n', text)
    return text

def preprocess_tweet(text):
    text = remove_empty_lines(text)
    text = remove_urls(text)
    # text = remove_mentions(text)
    # text = remove_hashtags(text)
    # text = remove_html_tags(text)
    # text = remove_special_characters(text)
    text = preprocess_text(text)

    return text