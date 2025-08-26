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
