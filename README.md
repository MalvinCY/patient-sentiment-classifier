# Patient Sentiment Analysis

A 3-class sentiment classifier for patient drug reviews, comparing traditional ML and deep learning approaches.

## Project Overview

This project classifies patient reviews of medications into three sentiment categories:
- **Negative** (ratings 1-3): 22% of dataset
- **Neutral** (ratings 4-7): 18% of dataset  
- **Positive** (ratings 8-10): 60% of dataset

## Dataset

- **Source:** Drug Review Dataset (Kaggle)
- **Size:** 215,000+ patient reviews
- **Training:** 161,297 reviews
- **Testing:** 53,766 reviews

## Current Progress

### Phase 1: Traditional ML (Completed)
- Text preprocessing with negation preservation
- Word2Vec embeddings (Google News, 300 dimensions)
- Logistic Regression with class balancing
- Hyperparameter tuning via GridSearchCV
- **Result:** 59% accuracy (confirmed model ceiling)

### Phase 2: Deep Learning (In Progress)
- LSTM with sequence padding
- Captures word order and negation patterns
- *Coming soon...*

## Project Structure
```
patient-sentiment-classifier/
├── data/
│   ├── drugsComTrain_raw.csv
│   └── drugsComTest_raw.csv
├── notebooks/
│   └── 01_patient_sentiment_training.ipynb
├── models/
│   └── (trained models saved here)
├── requirements.txt
└── README.md
```

## Installation
```bash
# Clone repository
git clone <your-repo-url>
cd patient-sentiment-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

*Coming soon: API deployment instructions*

## Results Summary

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|-------|----------|-------------------|----------------|------------|
| Logistic Regression (baseline) | 58.8% | 0.54 | 0.56 | 0.54 |
| Logistic Regression (tuned) | 58.9% | 0.54 | 0.56 | 0.54 |
| LSTM | TBD | TBD | TBD | TBD |

## Tech Stack

- **ML/DL:** scikit-learn, PyTorch, gensim
- **Data:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Deployment:** *(planned)* FastAPI, Docker, Cloud Platform

## Next Steps

- [ ] Implement LSTM with padding
- [ ] Compare LSTM vs Logistic Regression
- [ ] Build REST API
- [ ] Containerize with Docker
- [ ] Deploy to cloud

## Author

Malvin Siew
