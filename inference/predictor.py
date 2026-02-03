import os
import json
import joblib
import numpy as np
import streamlit as st

# ------ Paths ------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.pkl")
CONFIG_FILE = os.path.join(ARTIFACT_DIR, "config.json")
LABEL_MAP_FILE = os.path.join(ARTIFACT_DIR, "label_map.json")

# ------ Load artifacts (cached) ------

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
        threshold = config["threshold"]

    with open(LABEL_MAP_FILE, "r") as f:
        label_map = json.load(f)

    return model, vectorizer, threshold, label_map

# ------ Prediction Function ------

def predict_emotions(texts):
    model, vectorizer, threshold, label_map = load_artifacts()

    X = vectorizer.transform(texts)
    y_proba = model.predict_proba(X)
    y_pred = (y_proba > threshold).astype(int)

    results = []
    for row in y_pred:
        indices = np.where(row == 1)[0]
        labels = [label_map[str(i)] for i in indices]
        results.append(labels)

    return results
