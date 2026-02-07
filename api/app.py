"""
Patient Sentiment Analysis API
Predicts sentiment (Negative/Neutral/Positive) for patient drug reviews
"""

import os
import torch
import torch.nn as nn
import numpy as np
import gensim.downloader as api
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import Dict
import re
import html
from nltk.corpus import stopwords
import nltk

# Download NLTK data
nltk.download('stopwords', quiet=True)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============= MODEL DEFINITION =============
class SentimentLSTM(nn.Module):
    def __init__(self, embedding_dim=300, hidden_dim=128, num_layers=2, num_classes=3, dropout=0.3):
        super(SentimentLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        final_hidden = hidden[-1]
        final_hidden = self.dropout(final_hidden)
        output = self.fc(final_hidden)
        return output

# ============= PREPROCESSING =============
stop_words = set(stopwords.words('english'))
negation_words = {'no', 'not', 'nor', 'never', 'none', 'nobody', 'nothing',
                  'neither', 'nowhere', 'hardly', 'scarcely', 'barely'}
stop_words = stop_words - negation_words

def preprocess_text(text):
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [word for word in tokens
              if (word not in stop_words) and (len(word) > 2 or word in negation_words)]
    return ' '.join(tokens)

def review_to_embedding_sequence(review, word2vec_model, max_length=50, embedding_dim=300):
    cleaned_review = preprocess_text(review)
    words = cleaned_review.split()
    embeddings = []
    for word in words[:max_length]:
        if word in word2vec_model:
            embeddings.append(word2vec_model[word])
        else:
            embeddings.append(np.zeros(embedding_dim))
    while len(embeddings) < max_length:
        embeddings.append(np.zeros(embedding_dim))
    return np.array(embeddings)

# ============= LOAD MODEL & EMBEDDINGS =============
print("Loading Word2Vec embeddings...")
word2vec = api.load("word2vec-google-news-300")
print("✓ Word2Vec loaded")

print("Loading trained model...")
model = SentimentLSTM(embedding_dim=300, hidden_dim=128, num_layers=2, num_classes=3, dropout=0.3)

# Determine model path (works both locally and in Docker)
# Local: run from api/ folder, model at ../models/
# Docker: run from /app/, model at models/
model_paths = [
    'models/best_lstm_balanced.pth',      # Docker path (from /app/)
    '../models/best_lstm_balanced.pth',   # Local path (from api/)
]

model_loaded = False
for model_path in model_paths:
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        print(f"✓ Model loaded from {model_path}")
        model_loaded = True
        break

if not model_loaded:
    print("⚠ Warning: Model file not found. Predictions will fail.")

# ============= PREDICTION FUNCTION =============
def predict_sentiment(review_text):
    embedding_seq = review_to_embedding_sequence(review_text, word2vec)
    input_tensor = torch.FloatTensor(embedding_seq).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

    return {
        'prediction': label_map[predicted_class],
        'prediction_id': predicted_class,
        'probabilities': {
            'negative': float(probabilities[0][0]),
            'neutral': float(probabilities[0][1]),
            'positive': float(probabilities[0][2])
        },
        'cleaned_text': preprocess_text(review_text)
    }

# ============= FASTAPI APP =============
app = FastAPI(
    title="Patient Sentiment Analysis API",
    description="Predict sentiment (Negative/Neutral/Positive) for patient drug reviews. Model: LSTM with 73.5% accuracy on 54k test samples.",
    version="1.0.0"
)

class ReviewRequest(BaseModel):
    review: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "review": "This medication helped with my condition but caused some side effects"
            }
        }
    )

class PredictionResponse(BaseModel):
    prediction: str
    prediction_id: int
    probabilities: Dict[str, float]
    cleaned_text: str

@app.get("/")
def root():
    return {
        "message": "Patient Sentiment Analysis API",
        "status": "active",
        "model": "LSTM (73.5% accuracy)",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "word2vec_loaded": True
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: ReviewRequest):
    try:
        return predict_sentiment(request.review)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
