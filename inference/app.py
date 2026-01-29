from fastapi import FastAPI
from inference.schemas import InferenceRequest, InferenceResponse
from inference.predictor import predict_emotions
import json
from pathlib import Path

app = FastAPI(
    title="AI Mood & Productivity Analyzer",
    version="1.0.0"
)


@app.get("/health")
def health_check():
    return {"status": "AI Mood Analysier is running."}


@app.post("/predict", response_model = InferenceResponse)
def predict(request: InferenceRequest):
    predictions = predict_emotions(request.texts)
    return {"predictions": predictions}


@app.get("/")
def health():
    return {"status": "AI Mood Analyzer API is running"}


@app.get("/version")
def version():
    return {
        "service": "AI Mood Analyzer",
        "api_version": "v1.0",
        "model_version": "1.0.0"
    }

@app.get("/model-info")
def model_info():
    base_dir = Path(__file__).resolve().parent.parent
    artifacts_dir = base_dir / "artifacts"

    with open(artifacts_dir / "label_map.json") as f:
        label_map = json.load(f)

    return {
        "model_type": "LogisticRegression (OneVsRest)",
        "num_labels": len(label_map),
        "labels": list(label_map.keys()),
        "vectorizer": "TF-IDF",
        "framework": "scikit-learn",
        "framework_version": "1.7.0"
    }

