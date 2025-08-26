# CodeAlpha_QuirkBot


# QuirkBot: Friendly FAQ Chatbot
# Author: Ravi Choudhary
# Internship: CodeAlpha AI Internship (2025)

import json
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# Ensure required resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Preprocessing helpers
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [stemmer.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(tokens)

# Load FAQs
def load_faqs():
    with open("faqs.json", "r") as f:
        return json.load(f)

faqs = load_faqs()
questions = [preprocess(item["question"]) for item in faqs]
answers = [item["answer"] for item in faqs]
# Vectorize
vectorizer = TfidfVectorizer().fit(questions)
q_vectors = vectorizer.transform(questions)

# Streamlit UI
st.title("🤖 QuirkBot - FAQ Assistant")
st.write("Ask me a question and I'll try to answer based on stored FAQs.")

user_input = st.text_input("Ask a question:")

if user_input:
    processed_q = preprocess(user_input)
    user_vector = vectorizer.transform([processed_q])
    similarity = cosine_similarity(user_vector, q_vectors)
    idx = similarity.argmax()
    score = similarity[0][idx]

    if score > 0.3:
        st.success(f"💡 {answers[idx]}")
    else:
        st.warning("❓ Sorry, I don’t know the answer to that yet.")
        
