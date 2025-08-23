# COVID-Voices: COVID-19 Tweet Sentiment Analysis

A comprehensive NLP/machine learning project for analyzing sentiment in COVID-19 related tweets using various approaches including Large Language Models (LLMs), fine-tuned transformer models, and model compression techniques.

## 📋 Project Overview

**EDA.ipynb**: Comprehensive data exploration and preprocessing pipeline for COVID-19 tweets. Analyzes sentiment distribution, text characteristics, and implements three preprocessing strategies from basic URL removal to advanced LLM-based content extraction.

**LLM.ipynb**: Evaluates large language models (Llama-3.1-8B and DeepSeek-R1) for zero-shot and few-shot sentiment classification. Tests different prompting strategies and self-consistency techniques to improve prediction reliability.

**Main_FT.ipynb**: Fine-tunes transformer models (DeBERTa-v3-base and Twitter-RoBERTa) with hyperparameter optimization using Optuna. Implements both manual training and Hugging Face Trainer approaches with experiment tracking via Weights & Biases.

**Test_Compression.ipynb**: Comprehensive model compression framework comparing quantization (8-bit), pruning (L1 unstructured), and knowledge distillation techniques. Balances model performance with deployment efficiency for resource-constrained environments. **This is where we RUN the tests** on our fine-tuned models and where we evaluate compression performance.

## 🗂️ Project Structure

```
COVID-Voices/
├── Data/                          # Dataset files
├── EDA.ipynb                      # Exploratory data analysis and preprocessing
├── LLM.ipynb                      # LLM-based sentiment classification
├── Main_FT.ipynb                  # Fine-tuning transformer models
├── Test_Compression.ipynb         # Model compression and optimization
└── README.md                      # This file
```

### 🧪 Quick Start Guide

For immediate results, jump to our [model compression experiments results](#4-test_compressionipynb---model-compression--optimization-and-testing) to see performance comparisons across different optimization techniques and **evaluations of our fine-tuned models on the test set**.


## 📊 Dataset

The project uses the **Corona NLP Tweet Sentiment Analysis** dataset containing:
- **41,157 training tweets** and test set
- **5 sentiment classes**: Extremely Negative, Negative, Neutral, Positive, Extremely Positive
- **Features**: User information, location, tweet text, timestamp, and sentiment labels
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification/data)


## 🔍 Notebook Descriptions

### Prerequisites - please make sure you have all of the required packges

```bash
# Install required packages
pip install torch transformers pandas numpy scikit-learn
pip install optuna wandb kagglehub bitsandbytes accelerate
pip install matplotlib seaborn wordcloud nltk emoji pycountry
```

### 1. EDA.ipynb - Exploratory Data Analysis & Preprocessing

**Purpose**: Comprehensive analysis of the COVID-19 tweet dataset with multiple preprocessing strategies.

**Key Features**:
- **Data Quality Analysis**: Schema validation, null counts, duplicate detection
- **Sentiment Distribution**: Class balance analysis with visualizations
- **Text Analysis**: Tweet length distributions, URL/mention/hashtag patterns
- **Geographic Analysis**: Sentiment patterns by location/country
- **Advanced Preprocessing**: Three different preprocessing strategies

**Preprocessing Strategies**:
1. **Basic**: URL removal, hashtag/mention conversion
2. **Enhanced**: Emphasizes statistically significant n-grams per sentiment
3. **Advanced**: URL content extraction and summarization using LLMs

**Usage Example**:
```python
import pandas as pd
# Load preprocess version of the dataset
preprocess_number = 1 # can be 1,2,3
train_df = pd.read_csv(f"Data/Corona_NLP_train_preprocess{preprocess_number}.csv", encoding="latin-1")
test_df = pd.read_csv(f"Data/Corona_NLP_test_preprocess{preprocess_number}.csv", encoding="latin-1")

# for the original (raw) dataset you can just do
original_train_df = pd.read_csv(f"Data/Corona_NLP_train.csv", encoding="latin-1")
original_test_df = pd.read_csv(f"Data/Corona_NLP_test.csv", encoding="latin-1")
```

**Key Insights**:
- Slight class imbalance (most tweets are Negative/Positive)
- ~48% of tweets contain URLs
- Tweet lengths vary significantly (max 64 words)

### 2. LLM.ipynb - Large Language Model Sentiment Classification

**Purpose**: Evaluate state-of-the-art LLMs for zero-shot and few-shot sentiment classification.

**Models Tested**:
- **Llama-3.1-8B-Instruct**: Meta's instruction-tuned model
- **DeepSeek-R1-Distill-Qwen-7B**: Distilled reasoning model

**Approaches**:
- **Zero-shot**: Direct classification without examples
- **Few-shot**: Classification with labeled examples
- **Self-consistency**: Multiple sampling for robust predictions

**Usage Example**:
```python
# REQUIRED CELLS TO RUN FIRST:
# Cell 1: Install packages and imports
# Cell 2: Configuration setup
# Cell 3: System prompts
# Cell 4: Load model + Tokenizer + Pipeline
# Cell 5: Helpers

# Zero-shot classification with Llama
SYSTEM_PROMPT = "You are a careful annotator for COVID-19 tweet sentiment."
USER_TEMPLATE = """Classify the sentiment of the tweet below into exactly one of:
["Extremely Negative","Negative","Neutral","Positive","Extremely Positive"].

Tweet: {tweet}"""

# Prepare tweets for classification
tweets = ["COVID-19 has been really tough on everyone", "I love how people are helping each other"]

# Generate predictions
prompts = build_prompts(tweets)
outputs = pipe(prompts, max_new_tokens=512, do_sample=False)
```

**Performance Results**:
- **Llama-3.1 Zero-shot**: 30.79% accuracy, 27.65% F1-weighted
- **Llama-3.1 Few-shot**: 31.73% accuracy, 30.19% F1-weighted
- **DeepSeek-R1 Zero-shot**: 32.91% accuracy, 28.30% F1-weighted
- **DeepSeek-R1 Zero-shot**: 32.13% accuracy 27.63% F1-weighted

### 3. Main_FT.ipynb - Fine-tuning Transformer Models

**Purpose**: Fine-tune pre-trained transformer models for COVID-19 sentiment classification with hyperparameter optimization.

**Available Checkpoints**: Our fine-tuned models are available at [CarmelKron/Model_Checkpoints](https://huggingface.co/CarmelKron/Model_Checkpoints) on Hugging Face Hub.

**Models Supported**:
- **DeBERTa-v3-base**: Microsoft's advanced transformer
- **Twitter-RoBERTa**: Domain-specific sentiment model

**Training Approaches**:
1. **Manual Training Loop**: Full control over training process
2. **Hugging Face Trainer**: Standard fine-tuning pipeline

**Key Features**:
- **Hyperparameter Optimization**: Optuna-based search
- **Early Stopping**: Prevents overfitting
- **Model Persistence**: Automatic saving to Hugging Face Hub
- **Experiment Tracking**: Weights & Biases integration



**Usage Example**:
```python
# Running with different preprocesses
# On cell 3 + 9 change PREPROCESS_VERSION variable depanding on the desired preprocess
PREPROCESS_VERSION = ""  # Should be empty if you want to use the original dataset,
# otherwise replace PREPROCESS_VERSION with the following:
VERSION = 1 # can be 1, 2, 3
PREPROCESS_VERSION = f"preprocess{VERSION}"

# Running with different models
# On cell 8 change to one of the following:
model_name = "microsoft/deberta-v3-base"
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
```

### 4. Test_Compression.ipynb - Model Compression & Optimization AND TESTING

**Purpose**: Evaluate various model compression techniques to balance performance and efficiency. **This is where we RUN the tests on our fine-tuned models** and where we compare different compression approaches on our fine-tuned models.

**Compression Techniques**:
- **8-bit Quantization**: Dynamic (CPU) and BitsAndBytes (GPU)
- **Pruning**: L1 unstructured pruning of linear layers
- **Knowledge Distillation**: Training smaller student models


**Usage: Configuration Options**:
To use this notebook, simply change the following configuration variables in cell 2:
```python
# Load preprocess version of the dataset
preprocess_number = 1 # can be 1,2,3
train_df = pd.read_csv(f"Data/Corona_NLP_train_preprocess{preprocess_number}.csv", encoding="latin-1")
test_df = pd.read_csv(f"Data/Corona_NLP_test_preprocess{preprocess_number}.csv", encoding="latin-1")

# for the original (raw) dataset you can just do
#original_train_df = pd.read_csv(f"Data/Corona_NLP_train.csv", encoding="latin-1")
#original_test_df = pd.read_csv(f"Data/Corona_NLP_test.csv", encoding="latin-1")

train_df = train_df.rename(columns={"OriginalTweet":"text", "Sentiment":"label"})
test_df  = test_df.rename(columns={"OriginalTweet":"text", "Sentiment":"label"})

label_map = {'Extremely Negative':0,'Negative':1,'Neutral':2,
             'Positive':3,'Extremely Positive':4}

train_df["label"] = train_df.label.map(label_map).astype(int)
test_df["label"]  = test_df.label.map(label_map).astype(int)


repo_id = "CarmelKron/Model_Checkpoints"
subdir  = "hf_ft_twitter_roberta_orig"  # Change to run with different models: "full_ft_deberta_v3_base", "hf_ft_deberta_v3_base_orig", "full_ft_twitter_roberta"

repo_dir = snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    allow_patterns=[f"{subdir}/*"],
)

folder_path = str(Path(repo_dir) / subdir)
print("Folder path:", folder_path)


cfg = CompressConfig(
    model_id="cardiffnlp/twitter-roberta-base-sentiment-latest",  # Change to "microsoft/deberta-v3-base" for DeBERTa models
    weights_path=folder_path,
    num_labels=None,
    max_len=254,  # Change to 64 for "full_ft_deberta_v3_base" , 128 for "hf_ft_deberta_v3_base_orig"/ "full_ft_twitter_roberta", 254 hf_twitter_roberta_orig
    batch_size=32,
    prune_amount=0.4,
    do_quantized=True, # Change to False if you don't want quantization
    do_pruned=True, # Change to False if you don't want pruning
    do_kd=True, Change to False if you don't want knowledge distillation.
    quantization_backend="bnb",
    force_cpu_for_all=False
)

# NOTE: BY CHANGING ALL do_{} PARAMETERS IN THE ABOVE CONFIG, YOU CAN RUN OUR FINE-TUNED MODELS ON THE TEST SET WITHOUT WAITING FOR THE COMPRESSIONS TO FINISH RUNNING AS WELL.


# To run with other models, change these variables:
# For DeBERTa models: 
#   - model_id = "microsoft/deberta-v3-base"
#   - subdir = "full_ft_deberta_v3_base" or "hf_ft_deberta_v3_base_orig"

# For Twitter-RoBERTa models:
#   - model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"  
#   - subdir = "full_ft_twitter_roberta" or "hf_ft_twitter_roberta_orig"


cmp = CompressionComparator(cfg)
results = cmp.run(test_df=test_df, train_df=train_df)
print(results.to_string(index=False))                                  # Temperature for soft targets
```

**Compression Benefits**:
- **Model Size**: Up to 75% reduction
- **Inference Speed**: Faster processing with minimal accuracy loss
- **Memory Usage**: Reduced GPU/CPU memory requirements
- **Deployment**: Easier deployment on resource-constrained devices

## 🚀 Getting Started

### Running the Notebooks

1. **Start with EDA**: Understand your data and choose preprocessing strategy
2. **Experiment with LLMs**: Test zero-shot and few-shot capabilities
3. **Fine-tune Models**: Optimize transformer models for your task
4. **Compress Models**: Balance performance and efficiency

## 📈 Performance Comparison

| Approach | Model | Accuracy | F1-Weighted | Notes |
|----------|-------|----------|-------------|-------|
| **LLM Zero-shot** | DeepSeek-R1 | 32.91% | 28.3% | No training required |
| **LLM Few-shot** | DeepSeek-R1 | 32.13% | 27.63% | With examples |
| **Fine-tuned** | DeBERTa-v3-base | ~89% | ~89% | Requires training |

**Note**: Due to poor performance of LLMs in zero-shot and few-shot scenarios, we focused our test set evaluation on the fine-tuned transformer models rather than running comprehensive LLM tests on the final test data.


## 🔧 Configuration

### Model Parameters
- **Sequence Length**: 64-256 tokens (dataset-dependent)
- **Batch Size**: 16-64 (GPU memory dependent)
- **Learning Rate**: 1e-6 to 1e-4
- **Training Epochs**: 20 with early stopping


## 📝 Key Findings

1. **LLM Performance**: Current LLMs show poor performance without fine-tuning
2. **Fine-tuning Benefits**: Significant performance improvement over zero-shot approaches
3. **Compression Trade-offs**: 8-bit quantization provides good balance of size and performance

