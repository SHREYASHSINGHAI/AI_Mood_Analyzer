def run_evaluation():
    # ------ Imports 
    import os
    import json
    import joblib
    import numpy as np
    from sklearn.metrics import f1_score, hamming_loss


    # --- Directory path
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    artifacts_dir = os.path.join(BASE_DIR, "artifacts")

    # Load data
    X_val = joblib.load(os.path.join(artifacts_dir,"X_val_vec.pkl"))
    y_val = np.load(os.path.join("data", "splitting", "y_val.npy"))


    # Load model
    model = joblib.load("artifacts/model.pkl")


    # Inference with threshold

    threshold = 0.20
    y_proba = model.predict_proba(X_val)
    y_pred = (y_proba>threshold).astype(int)


    # Metrics
    metrics = {
        "threshold" : threshold,
        "micro F1:" : f1_score(y_val, y_pred, average="micro"),
        "macro F1:" : f1_score(y_val, y_pred, average="macro"),
        "hamming Loss:" : hamming_loss(y_val, y_pred)
    }
    print("Evaluation results:", metrics, sep="\n")

    # --- Saving metrics
    with open(os.path.join(artifacts_dir, "metrics.json"),"w") as f:
        json.dump(metrics, f, indent=4)