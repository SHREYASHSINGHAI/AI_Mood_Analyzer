# 🧠 AI Mood Analyzer

An end-to-end AI Mood Analyzer that detects emotions from text using a trained machine learning model.
The project is designed with production-grade ML architecture, separating training, inference, and user interface layers.


🛠️ Tech Stack

- Python 3.11
- Scikit-learn
- FastAPI
- Pydantic
- Streamlit
- MLflow
- Docker


🚀 Project Highlights

🔍 Text-based emotion detection
🧠 ML model trained using Scikit-learn
⚡ FastAPI-based inference service
🎨 Streamlit frontend for user interaction
📦 Docker-ready architecture


🧠 Model Details

Algorithm: Logistic Regression (One-vs-Rest)
Vectorization: TF-IDF
Frameworks:
Scikit-learn
MLflow (experiment tracking ready)

The model is trained offline and saved as artifacts, which are later loaded by the inference API.

🔌 Inference API (FastAPI)

Endpoint: POST /predict
Request Body:
{
  "texts": ["I feel excited about my new job!"]
}
Response:
{
  "emotion": "happy"
}


Features

Input validation using Pydantic
Batch prediction support
Clean error handling
Health-check ready architecture


🎨 Frontend (Streamlit)

Simple and intuitive UI
Sends user input to FastAPI
Displays detected emotion
Handles API failures gracefully
The Streamlit app does NOT load the model directly — it communicates only via the API


🧱 System Architecture

User (Streamlit UI)
        ↓
FastAPI Inference Service
        ↓
Saved ML Artifacts (model, vectorizer)
        ↑
Offline Training Pipeline


------------------------------------------------------------------------------------------
⚙️ How to Run the Project Locally

1. Create & activate virtual environment:
     python -m venv venv
     venv\Scripts\activate   # Windows

2. Install backend dependencies:
     pip install -r requirements.txt
   
3. Run Inference API:
     uvicorn inference.app:app --host 0.0.0.0 --port 8000 --reload

4. Check API on the browser:
      http://localhost:8000/docs

5. Run Streamlit App:
     cd streamlit_app
     pip install -r requirements.txt
     streamlit run app.py

6. Open in browser:
     http://localhost:8501

---------------------------------------------------------------------------------------

🐳 Docker Support

Dockerfile.train → Training pipeline
Dockerfile.infer → Inference service

The project is container-ready and can be extended with Docker Compose.


📌 Configuration Management

config.json stores model-related metadata
Keeps parameters version-controlled
Helps ensure reproducibility and clarity


🎯 Why This Project Matters

This project demonstrates:

✅ End-to-end ML system design
✅ Real-world API + UI integration
✅ Clean code organization
✅ Debugging & production thinking
✅ MLOps fundamentals (without overengineering)

Author: Shreyash Singhai
