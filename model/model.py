import os
import joblib
import numpy as np
from scipy.sparse import hstack
import sys

# Ensure we can find features.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from features import extract_features

# Paths to your new 10/10 brain files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "phish_model.pkl")
VECT_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

def predict_url(url):
    try:
        # 1. Load the hybrid "Brain" and "Translator"
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
            print("ERROR: Model or Vectorizer missing! Run train_model.py first.")
            return 50.0

        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECT_PATH)

        # 2. Structural Analysis (Manual Features)
        manual_feats = np.array([extract_features(url)])

        # 3. Keyword Analysis (TF-IDF Features)
        tfidf_feats = vectorizer.transform([url])

        # 4. Join them together (The Hybrid Step)
        final_input = hstack([tfidf_feats, manual_feats])

        # 5. Get the real probability
        # [0][1] is the probability of it being "Label 1" (Phishing)
        probability = model.predict_proba(final_input)[0][1]
        
        # Multiply by 100 to get the percentage for your Pie Chart
        return round(float(probability) * 100, 2)

    except Exception as e:
        print(f"PREDICTION ERROR: {e}")
        return 0.0 # Safety default