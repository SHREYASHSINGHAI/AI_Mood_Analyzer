import os
import joblib
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
CONFIG_FILE = os.path.join(ARTIFACT_DIR, "config.json")
LABEL_MAP_FILE = os.path.join(ARTIFACT_DIR, "label_map.json")


# ------ Loading Artifacts ------

model = joblib.load(os.path.join(ARTIFACT_DIR, "model.pkl"))
print("Model loaded successfully.")

vectorizer = joblib.load(os.path.join(ARTIFACT_DIR,"tfidf_vectorizer.pkl"))
print("Vectorizer loaded successfully.")

with open(CONFIG_FILE,"r") as f:
    config = json.load(f)
    threshold = config["threshold"]
print("Configurations loaded successfully.")


with open (LABEL_MAP_FILE,"r") as f:
    label_map = json.load(f)
print("Label map loaded successfully.")

# ------ Prediction Function ------

def predict_emotions(text):
    X = vectorizer.transform(text)
    y_proba = model.predict_proba(X)
    y_pred = (y_proba>threshold).astype(int)

    results = []
    for row in y_pred:
        indices = np.where(row == 1)[0]
        labels = [label_map[str(i)] for i in indices]
        results.append(labels)

    return results
if __name__ == "__main__":
    samples = [
        "I am so happy and excited about my new job!",
        "I feel sad and lonely today.",
        "This is a terrifying experience.",
        "I am angry about the delay in my project.",
        "I love spending time with my family."
    ]
    predictions = predict_emotions(samples)
    for text, pred in zip(samples, predictions):
        print(f"Text: {text}")
        print(f"Predicted Emotions: {pred}")
        print("-"*50)