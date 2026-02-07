# Patient Sentiment Analysis API

3-class sentiment classification system for patient drug reviews, deployed as a REST API.

![API Demo](docs/screenshots/01_swagger_ui_overview.png)

---

## 🎯 Project Overview

This project classifies patient reviews of medications into three sentiment categories:
- **Negative** (ratings 1-3): Critical reviews, severe side effects
- **Neutral** (ratings 4-7): Mixed experiences, moderate effectiveness
- **Positive** (ratings 8-10): Highly effective, minimal side effects

**Key Achievement:** 73.5% accuracy on 54k test samples with balanced performance across all classes.

---

## 📊 Results

### Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 73.5% |
| **Training Samples** | 86,472 (balanced) |
| **Test Samples** | 53,766 (realistic distribution) |
| **Parameters** | 352,643 |

### Class-Level Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Negative | 0.72 | 0.76 | 0.74 | 11,838 |
| Neutral | 0.42 | 0.61 | 0.49 | 9,579 |
| Positive | 0.90 | 0.76 | 0.83 | 32,349 |

**Key Insight:** Neutral class remains challenging due to inherent ambiguity in ratings 4-7, but balanced training improved neutral recall from 28% → 61%.

---

## 🏗️ Technical Architecture

### Model
- **Type:** 2-layer LSTM with attention to word order
- **Embeddings:** Word2Vec (Google News, 300-dim)
- **Sequence Length:** 50 words (optimized from 100 based on data analysis)
- **Hidden Dimension:** 128
- **Dropout:** 0.3 for regularization

### Key Decisions
1. **Balanced Training:** Stratified sampling (28,824 samples per class) significantly improved minority class performance
2. **Padding Optimization:** Reduced from 100→50 words improved accuracy by 12% (eliminated excessive zero-padding)
3. **Negation Preservation:** Kept words like "not", "never", "no" during preprocessing (critical for sentiment)

### Technology Stack
- **ML/DL:** PyTorch, scikit-learn, gensim
- **API:** FastAPI, Uvicorn
- **Deployment:** Docker
- **Data Processing:** pandas, numpy, NLTK

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.10+
Docker (optional, for containerized deployment)
2GB+ RAM (required for Word2Vec embeddings)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/MalvinCY/patient-sentiment-classifier.git
cd patient-sentiment-classifier
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the API**
```bash
cd api
python app.py
```

The API will start on `http://localhost:8000`

---

## 📡 API Usage

### Interactive Documentation
Visit `http://localhost:8000/docs` for Swagger UI with live testing:

![Swagger UI](docs/screenshots/02_predict_endpoint_expanded.png)

### Prediction Endpoint

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"review": "This medication caused terrible side effects"}'
```

**Response:**
```json
{
  "prediction": "Negative",
  "prediction_id": 0,
  "probabilities": {
    "negative": 0.87,
    "neutral": 0.12,
    "positive": 0.01
  },
  "cleaned_text": "medication caused terrible side effects"
}
```

![Negative Prediction](docs/screenshots/03_prediction_negative_example.png)

### Python Example
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"review": "Amazing drug! Works perfectly with no side effects"}
)

result = response.json()
print(f"Sentiment: {result['prediction']}")
print(f"Confidence: {result['probabilities']['positive']:.2%}")
```

![Positive Prediction](docs/screenshots/04_prediction_positive_example.png)

---

## 🐳 Docker Deployment

### Build and Run
```bash
# Build image
docker build -t patient-sentiment-api .

# Run container
docker run -p 8000:8000 patient-sentiment-api
```

### Cloud Deployment Notes
⚠️ **Memory Requirements:** This application requires 2GB+ RAM due to Word2Vec embeddings (1.6GB).

**Compatible Platforms:**
- ✅ AWS EC2 (t2.small or larger)
- ✅ Google Cloud Run (2GB+ memory)
- ✅ Render/Railway Paid Tiers
- ❌ Free tier services (typically 512MB limit)

**Estimated Cost:** $7-15/month for appropriate hosting

---

## 📈 Model Development Journey

### Experiments Conducted

| Approach | Training Data | Accuracy | Key Learning |
|----------|---------------|----------|--------------|
| Logistic Regression | 16k (imbalanced) | 58.9% | Baseline with averaged embeddings |
| LSTM (100-word pad) | 16k (imbalanced) | 58.5% | Excessive padding hurts performance |
| LSTM (50-word pad) | 16k (imbalanced) | 70.2% | Padding length critical (+11.7 points) |
| **LSTM (50-word pad)** | **86k (balanced)** | **73.5%** | **Class balance matters (+3.3 points)** |

**Total Improvement:** +14.6 percentage points over baseline

### Key Insights
1. **Hyperparameter tuning matters:** Padding length had 12% impact on accuracy
2. **Data quality > Model complexity:** Balanced training improved performance more than architectural changes
3. **Neutral is inherently difficult:** Ratings 4-7 contain mixed sentiment, limiting achievable accuracy

---

## 📁 Project Structure
```
patient-sentiment-classifier/
├── api/
│   └── app.py                  # FastAPI application
├── data/
│   └── processed/              # Preprocessed embeddings (batched)
├── docs/
│   └── screenshots/            # API documentation images
├── models/
│   └── best_lstm_balanced.pth  # Trained model (Git LFS)
├── notebooks/
│   ├── 01_data_exploration_and_training.ipynb
│   └── 02_model_deployment_api.ipynb
├── Dockerfile                  # Container configuration
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

---

## 🔬 Methodology

### Data
- **Source:** UCI Drug Review Dataset
- **Size:** 215,063 reviews (161k train, 54k test)
- **Preprocessing:**
  - HTML entity decoding
  - Lowercase conversion
  - Stop word removal (preserving negations: "not", "never", "no")
  - Special character removal

### Training Strategy
- **Class Balancing:** Stratified sampling to 28,824 samples per class
- **Sequence Padding:** 50 words (based on median=40, covers 63% fully)
- **Optimization:** Adam (lr=0.001) with early stopping (patience=3)
- **Regularization:** Dropout (0.3) to prevent overfitting

---

## 🎓 Future Improvements

While the current model achieves competitive performance (73.5%), potential enhancements include:

- **Attention Mechanisms:** Could improve to ~75% by learning which words matter most
- **Transformer Models (BERT/BioBERT):** Pre-trained language models could reach ~76-77%
  - BioBERT specifically designed for medical text
  - Requires GPU for fine-tuning (4-6 hours on V100)
- **Ensemble Methods:** Combining LSTM + Logistic Regression could add 1-2 points

**Trade-off Considered:** For production deployment, the current LSTM offers optimal balance of performance (73.5%), speed (50ms inference), and resource efficiency (CPU-compatible).

---

## 📝 License

MIT License - see LICENSE file for details

---

## 👤 Author

**Malvin Siew**
- GitHub: [@MalvinCY](https://github.com/MalvinCY)
- Project Link: [https://github.com/MalvinCY/patient-sentiment-classifier](https://github.com/MalvinCY/patient-sentiment-classifier)

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Drug Review Dataset
- Google News Word2Vec pre-trained embeddings
- FastAPI framework for modern API development