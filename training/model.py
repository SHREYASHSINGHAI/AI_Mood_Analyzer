# ------ Imports 

import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Importing application modules
from training.preprocess import run_preprocessing
from training.features import run_feature_engineering
from training.evaluate import run_evaluation

def train_model():

    # Path resolution
    print("Defining directories...")
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    SPLITS_DIR    = os.path.join(BASE_DIR, "data", "splitting")
    print("Directories defined successfully!")


    # Load features
    print("Loading training features and lables...")
    X_train = joblib.load(os.path.join(ARTIFACTS_DIR, "X_train_vec.pkl"))
    print("Loaded training features!")
    y_train = np.load(os.path.join(SPLITS_DIR, "y_train.npy"))
    print("Loaded training lables!")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)


    # Model definition
    print("Defining model...")
    model = OneVsRestClassifier(
        LogisticRegression(
            max_iter=1000,
            random_state=42),n_jobs=-1
    )
    print("model defined!")


    # Train
    print("Fitting model...")
    model.fit(X_train, y_train)
    print("Model fitted!")

    # Save model
    print("Saving model...")
    joblib.dump(model, os.path.join(ARTIFACTS_DIR, "model.pkl"))

    print("Model successfully saved to artifacts/model.pkl.")



def main():
    run_preprocessing()
    run_feature_engineering()
    train_model()
    run_evaluation()



if __name__ == "__main__":
    main()