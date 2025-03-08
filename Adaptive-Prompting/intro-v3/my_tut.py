import streamlit as st
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize models (cache them for efficiency)
@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer('all-mpnet-base-v2')
    sentiment_analyzer = SentimentIntensityAnalyzer()
    return embedding_model, sentiment_analyzer

embedding_model, sentiment_analyzer = load_models()

def calculate_similarity(student_input, expected_input, embedding_model):
    """Calculates the semantic similarity between two sentences."""
    student_embedding = embedding_model.encode(student_input)
    expected_embedding = embedding_model.encode(expected_input)
    return cosine_similarity([student_embedding], [expected_embedding])[0][0]

def analyze_sentiment(text, sentiment_analyzer):
    """Analyzes the sentiment of a text."""
    scores = sentiment_analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return "Positive"
    elif scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Example usage (inside your main Streamlit loop)
expected_input = "Subtract 3 from both sides of the equation"
student_input = prompt  # The student's input from st.chat_input

student_input_similarity = calculate_similarity(student_input, expected_input, embedding_model)
sentiment = analyze_sentiment(student_input, sentiment_analyzer)
keywords = ["subtract", "3", "both sides"]
keywords_present = all(keyword in student_input.lower() for keyword in keywords)