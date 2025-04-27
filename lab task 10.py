import streamlit as st
import random
import re
import numpy as np
from nltk.chat.util import reflections
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Chat pairs from your existing code (add as many as you want)
pairs = [
    [r"(?i).*hello.*|.*hi.*|.*hey.*",
     ["Hello! How can I assist you with your health-related queries?",
      "Hi there! Feel free to ask any health-related questions.",
      "Hey! I'm here to help you with medical information. How can I assist?"]],
    [r"(?i).*bye.*|.*goodbye.*",
     ["Goodbye! Take care and stay healthy.",
      "See you later! Stay safe and feel free to ask anytime."]],
    [r"(?i).*flu symptoms.*|.*cold symptoms.*",
     ["Flu symptoms include fever, chills, cough, sore throat, body aches, and fatigue."]],
    [r"(?i).*headache treatment.*|.*how to cure headache.*",
     ["Try resting, drinking water, and using over-the-counter pain relief."]]
    
]

# Precompute question embeddings
questions = [pair[0] for pair in pairs]
question_embeddings = model.encode(questions)

# Match user's input with best answer
def find_best_match(user_query):
    user_embedding = model.encode([user_query])
    similarities = cosine_similarity(user_embedding, question_embeddings)
    best_match_idx = np.argmax(similarities)
    best_match_score = similarities[0][best_match_idx]
    threshold = 0.75

    if best_match_score > threshold:
        return random.choice(pairs[best_match_idx][1])
    else:
        return "Sorry, I couldn't find an exact answer. Please provide more details."

# Streamlit App
st.set_page_config(page_title="HealthBot", page_icon="💬")

st.title("🩺 Health Chatbot")
st.markdown("Ask me anything about common health symptoms and tips!")

user_input = st.text_input("You:", placeholder="Type your question here...")

if user_input:
    response = find_best_match(user_input)
    st.markdown(f"**HealthBot:** {response}")
