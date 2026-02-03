import streamlit as st
import requests

# FastAPI endpoint
API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="AI Mood Analyzer",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI Mood Analyzer")
st.write("Enter a sentence and detect the underlying emotion.")

# User input
user_text = st.text_area(
    "Your text:",
    placeholder="I feel excited about my new job!"
)

if st.button("Analyze Mood"):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        payload = {"texts": [user_text]}

        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()

            result = response.json()
            predictions = result.get("predictions", [])

            if predictions and len(predictions[0]) > 0:
                emotions = predictions[0]
                st.success("Prediction successful!")
                st.write("### Detected Emotion(s):")
                st.write(", ".join(emotions))
            else:
                st.warning("No emotion detected.")

        except requests.exceptions.ConnectionError:
            st.error("Inference API is not running.")
        except Exception as e:
            st.error(f"Error: {e}")
