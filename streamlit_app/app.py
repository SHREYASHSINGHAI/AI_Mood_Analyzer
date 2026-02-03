import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from inference.predictor import predict_emotions

st.set_page_config(
    page_title="AI Mood Analyzer",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI Mood Analyzer")
st.write("Enter a sentence and detect the underlying emotion.")

user_text = st.text_area(
    "Your text:",
    placeholder="I feel excited about my new job!"
)

if st.button("Analyze Mood"):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        try:
            predictions = predict_emotions([user_text])
            st.success("Prediction successful!")
            st.write("### Detected Emotion(s):")
            st.write(", ".join(predictions[0]))

            st.write(predictions[0])

        except Exception as e:
            st.error(f"Inference failed: {e}")
