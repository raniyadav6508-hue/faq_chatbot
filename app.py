import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load FAQ data
data = pd.read_csv('faqs.csv')
questions = data['question'].tolist()
answers = data['answer'].tolist()

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

processed_questions = [preprocess(q) for q in questions]

# TF-IDF setup
vectorizer = TfidfVectorizer(stop_words='english')
question_vectors = vectorizer.fit_transform(processed_questions)

# Response function
def get_response(user_question):
    processed_input = preprocess(user_question)
    input_vector = vectorizer.transform([processed_input])
    similarities = cosine_similarity(input_vector, question_vectors)
    best_match_index = similarities.argmax()
    best_score = similarities[0][best_match_index]

    if best_score < 0.3:
        return "Sorry, I couldn't find a good answer for that. Please rephrase your question."
    return answers[best_match_index]

# ---------- Streamlit UI ----------
st.set_page_config(page_title="FAQ Chatbot", page_icon="🤖")
st.title("🤖 College FAQ Chatbot")
st.write("Ask me anything about college admissions, fees, hostel, etc.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
user_input = st.chat_input("Type your question here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    response = get_response(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)