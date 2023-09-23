import spacy
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the English spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample dataset of filed affidavits and their corresponding counter affidavits
data = {
    "filed_affidavit": [
        "The plaintiff claims that the contract was breached.",
        "The defendant asserts that the contract was fulfilled.",
        "I was injured due to the negligence of the defendant.",
        "He is a murderer"
        # Add more filed affidavits
    ],
    "counter_affidavit": [
        "The defendant denies breaching the contract and provides evidence.",
        "The plaintiff disputes the fulfillment of the contract and presents counter evidence.",
        "I deny the allegations of negligence and provide evidence of due diligence.",
        "I deny the charges on me and provide evidence"
        # Add more counter affidavits
    ],
}

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Tokenize and preprocess the text
def preprocess(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_punct]
    return " ".join(tokens)

df["filed_affidavit"] = df["filed_affidavit"].apply(preprocess)
df["counter_affidavit"] = df["counter_affidavit"].apply(preprocess)

# Create TF-IDF vectors for the text
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df["filed_affidavit"])

# Streamlit app
st.title("Counter Affidavit Suggestion App")

# User input for filed affidavit
user_input = st.text_area("Enter your filed affidavit:")

# Preprocess the user input
user_input = preprocess(user_input)

if st.button("Generate Counter Affidavit"):
    # Calculate cosine similarity between user input and filed affidavits
    user_input_tfidf = tfidf_vectorizer.transform([user_input])
    cosine_similarities = cosine_similarity(user_input_tfidf, tfidf_matrix)

    # Get the index of the most similar filed affidavit
    most_similar_index = cosine_similarities.argmax()
    suggested_counter_affidavit = df.iloc[most_similar_index]["counter_affidavit"]

    # Display the suggested counter affidavit
    st.subheader("Suggested Counter Affidavit:")
    st.write(suggested_counter_affidavit)
