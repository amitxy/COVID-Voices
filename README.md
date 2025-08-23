# COVID-Voices: COVID-19 Tweet Sentiment Analysis

A comprehensive machine learning project for analyzing sentiment in COVID-19 related tweets using various approaches including Large Language Models (LLMs), fine-tuned transformer models, and model compression techniques.

## 📋 Project Overview

**EDA.ipynb**: Comprehensive data exploration and preprocessing pipeline for COVID-19 tweets. Analyzes sentiment distribution, text characteristics, and implements three preprocessing strategies from basic URL removal to advanced LLM-based content extraction.

**LLM.ipynb**: Evaluates large language models (Llama-3.1-8B and DeepSeek-R1) for zero-shot and few-shot sentiment classification. Tests different prompting strategies and self-consistency techniques to improve prediction reliability.

**Main_FT.ipynb**: Fine-tunes transformer models (DeBERTa-v3-base and Twitter-RoBERTa) with hyperparameter optimization using Optuna. Implements both manual training and Hugging Face Trainer approaches with experiment tracking via Weights & Biases.

**Test_Compression.ipynb**: Comprehensive model compression framework comparing quantization (8-bit), pruning (L1 unstructured), and knowledge distillation techniques. Balances model performance with deployment efficiency for resource-constrained environments.

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

### 3. Main_FT.ipynb - Fine-tuning Transformer Models

**Purpose**: Fine-tune pre-trained transformer models for COVID-19 sentiment classification with hyperparameter optimization.

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


REMOVE:
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

### 4. Test_Compression.ipynb - Model Compression & Optimization

**Purpose**: Evaluate various model compression techniques to balance performance and efficiency.

**Compression Techniques**:
- **8-bit Quantization**: Dynamic (CPU) and BitsAndBytes (GPU)
- **Pruning**: L1 unstructured pruning of linear layers
- **Knowledge Distillation**: Training smaller student models


**Usage: Configuration Options**:
```python
# Change model architecture
model_id = "microsoft/deberta-v3-base"           # or "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Enable/disable compression techniques
do_quantized = True/False                        # 8-bit quantization (dynamic CPU or bnb GPU)
do_pruned = True/False                           # L1 unstructured pruning
do_kd = True/False                               # Knowledge distillation training

# Quantization backend choice
quantization_backend = "dynamic"                  # CPU-based (slower but universal)
quantization_backend = "bnb"                      # GPU-based (faster, requires bitsandbytes)

# Pruning intensity
prune_amount = 0.4                               # Remove 40% of weights (0.1 to 0.8)

# Knowledge distillation settings
kd_student_id = "distilbert-base-uncased"        # Smaller student model
kd_epochs = 2                                    # Training epochs for student
kd_alpha = 0.7                                   # Balance between KD and CE loss
kd_T = 2.0                                       # Temperature for soft targets
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
| **LLM Zero-shot** | Llama-3.1-8B | 30.79% | 27.65% | No training required |
| **LLM Few-shot** | Llama-3.1-8B | 31.73% | 30.19% | With examples |
| **Fine-tuned** | DeBERTa-v3-base | ~85%+ | ~85%+ | Requires training |
| **Compressed** | 8-bit Quantized | ~84%+ | ~84%+ | 75% size reduction |

## 🔧 Configuration

### Model Parameters
- **Sequence Length**: 64-254 tokens (model-dependent)
- **Batch Size**: 16-64 (GPU memory dependent)
- **Learning Rate**: 1e-6 to 1e-4
- **Training Epochs**: 20 with early stopping


## 📝 Key Findings

1. **LLM Performance**: Current LLMs show moderate performance without fine-tuning
2. **Fine-tuning Benefits**: Significant performance improvement over zero-shot approaches
3. **Compression Trade-offs**: 8-bit quantization provides good balance of size and performance
4. **Preprocessing Impact**: Advanced preprocessing strategies can improve model performance

