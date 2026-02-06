# Patient Sentiment Analysis

A 3-class sentiment classifier for patient drug reviews, comparing traditional ML and deep learning approaches.

## Project Overview

This project classifies patient reviews of medications into three sentiment categories:
- **Negative** (ratings 1-3): 22% of dataset
- **Neutral** (ratings 4-7): 18% of dataset
- **Positive** (ratings 8-10): 60% of dataset

## Dataset

- **Source:** UCI Drug Review Dataset
- **Size:** 215,000+ patient reviews
- **Training:** 161,297 reviews
- **Testing:** 53,766 reviews

## Results Summary

| Model | Test Accuracy | Notes |
|-------|---------------|-------|
| Logistic Regression (16k sample) | 64.86% | Baseline with class balancing |
| LSTM (16k sample) | 69.56% | With LR scheduler |
| **LSTM (86k balanced)** | **71.27%** | **Best model** |

The LSTM trained on the balanced dataset achieved the best performance, with a 6.4 percentage point improvement over the logistic regression baseline.

## Approach

### Phase 1: Traditional ML
- Text preprocessing with negation word preservation
- Word2Vec embeddings (Google News, 300 dimensions)
- Logistic Regression with `class_weight='balanced'`
- Hyperparameter tuning via GridSearchCV

### Phase 2: Deep Learning
- LSTM network with sequential embeddings (preserves word order)
- `ReduceLROnPlateau` learning rate scheduler for stable training
- Early stopping to prevent overfitting
- Balanced training via undersampling (28,824 samples per class)

## Project Structure
```
patient_sentiment_analysis/
├── data/
│   ├── drugsComTrain_raw.csv
│   ├── drugsComTest_raw.csv
│   └── processed/          # Pre-computed embeddings
├── notebooks/
│   └── 01_patient_sentiment_training.ipynb
├── models/
│   ├── best_lstm_sentiment.pth      # Sample training
│   └── best_lstm_balanced.pth       # Balanced training (best)
├── requirements.txt
└── README.md
```

## Installation
```bash
# Clone repository
git clone <your-repo-url>
cd patient_sentiment_analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the training notebook:
```bash
cd notebooks
jupyter notebook 01_patient_sentiment_training.ipynb
```

*API deployment instructions coming in the next phase.*

## Key Learnings

1. **Class imbalance matters** - Balanced training improved model fairness across sentiment categories
2. **Negation words are critical** - Preserving "not", "never", etc. significantly improves sentiment detection
3. **Learning rate scheduling helps** - `ReduceLROnPlateau` prevented training instability
4. **More data improves performance** - Scaling from 16k to 86k samples yielded measurable gains

## Tech Stack

- **ML/DL:** scikit-learn, PyTorch, gensim
- **Data:** pandas, numpy
- **Visualisation:** matplotlib, seaborn
- **Deployment:** *(planned)* FastAPI, Docker, Cloud Platform

## Next Steps

- [x] Implement LSTM with sequence embeddings
- [x] Compare LSTM vs Logistic Regression
- [x] Train on balanced full dataset
- [ ] Build REST API with FastAPI
- [ ] Containerise with Docker
- [ ] Deploy to cloud

## Author

Malvin Siew
