# COVID-Voices: Twitter Sentiment Analysis

## Project Overview

COVID-Voices analyzes Twitter data related to COVID-19 to understand public sentiment during the pandemic. The project uses machine learning and NLP techniques to classify tweets into sentiment categories and identify trends over time.

## Dataset

This project uses the "Corona NLP" dataset containing tweets related to COVID-19 with sentiment annotations:

- **Source**: Corona NLP dataset
- **Files**: 
  - `data/raw/Corona_NLP_train.csv`
  - `data/raw/Corona_NLP_test.csv`
- **Sentiment Classes**: Extremely Negative, Negative, Neutral, Positive, Extremely Positive
- **Encoding**: Latin-1

## Project Structure

```
COVID-Voices/
├── data/
│   └── raw/               # Original dataset files
├── covid_voices/          # Main package
│   └── data/
│       └── datasets/      # Custom dataset implementations
├── EDA.ipynb              # Exploratory data analysis notebook
├── research.ipynb         # Model training and experimentation (separate file for Amit&Carmel )
└── requirements.txt       # Project dependencies
```

## Features

- **Sentiment Analysis**: Classification of tweets into 5 sentiment categories
- **Exploratory Data Analysis**: Visualizations of sentiment distribution and text characteristics
- **Model Ensemble**: Training multiple transformer-based models for improved accuracy
- **Custom Dataset Implementation**: Specialized COVID tweet dataset class with preprocessing

## Models

(TO CHANGE)
The project implements and compares multiple transformer-based models:
- DistilBERT (base-uncased)
- TinyBERT (4-layer)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd COVID-Voices
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Exploratory Data Analysis

Run the EDA notebook to explore the dataset:
```bash
jupyter notebook EDA.ipynb
```

### Training Models

The research notebook contains the model training pipeline:
```bash
jupyter notebook research.ipynb
```

### Using the Custom Dataset

```python
from covid_voices.data.datasets.corona_dataset import CoronaTweetDataset

# Define preprocessing (optional)
def preprocess_tweet(text):
    text = text.lower()
    text = text.replace('#', 'hashtag_')
    text = text.replace('@', 'mention_')
    return text
    
# Load datasets with factory method
datasets = CoronaTweetDataset.load_datasets(preprocessing=preprocess_tweet)

# Access train and test datasets
train_dataset = datasets["train"]
test_dataset = datasets["test"]
```

## Results

The project achieves sentiment classification with the following metrics:
- Accuracy: [Your model accuracy]
- F1 Score: [Your model F1 score]

## Future Work

- Implement additional preprocessing techniques
- Explore other transformer architectures
- Add temporal analysis to track sentiment changes over time
- Deploy a demo web application

## License

[Your chosen license]

## Contributors

[Your name or team information]
