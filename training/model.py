# ------ Imports
import os
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

import mlflow
import mlflow.sklearn

# Importing application modules
from training.preprocess import run_preprocessing
from training.features import run_feature_engineering
from training.evaluate import run_evaluation


def train_model():

    print("Defining directories...")
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
    SPLITS_DIR = os.path.join(BASE_DIR, "data", "splitting")

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    print("Directories defined successfully!")

    # ---------------- Load data ----------------
    print("Loading training data...")
    X_train = joblib.load(os.path.join(ARTIFACTS_DIR, "X_train_vec.pkl"))
    y_train = np.load(os.path.join(SPLITS_DIR, "y_train.npy"))

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    # ---------------- MLflow setup ----------------
    mlflow.set_experiment("ai-mood-analyzer")

    with mlflow.start_run(run_name="logreg_ovr_v1"):

        # ---------------- Model definition ----------------
        print("Defining model...")
        model = OneVsRestClassifier(
            LogisticRegression(
                max_iter=1000,
                random_state=42
            ),
            n_jobs=-1
        )

        # Log model parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("strategy", "OneVsRest")
        mlflow.log_param("max_iter", 1000)

        # ---------------- Train ----------------
        print("Training model...")
        model.fit(X_train, y_train)
        print("Model trained successfully!")

        # ---------------- Save model ----------------
        model_path = os.path.join(ARTIFACTS_DIR, "model.pkl")
        joblib.dump(model, model_path)
        print("Model saved to artifacts/model.pkl")

        # Log model to MLflow
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Log artifact explicitly
        mlflow.log_artifact(model_path)


def main():
    print("Starting training pipeline...")

    run_preprocessing()
    run_feature_engineering()
    train_model()
    run_evaluation()

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
