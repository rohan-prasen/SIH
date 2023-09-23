import streamlit as st
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline

# Load RoBERTa tokenizer and model for summarization
tokenizer = RobertaTokenizer.from_pretrained("facebook/roberta-large")
model = RobertaForSequenceClassification.from_pretrained("facebook/roberta-large")

# Streamlit app title and description
st.title("Affidavit Summarizer")
st.write("This app summarizes affidavits using RoBERTa.")

# Input text area for the affidavit
affidavit_text = st.text_area("Paste your affidavit text here:")

# Summarization function
def summarize_affidavit(text):
    if text:
        summary_pipeline = pipeline("summarization", model=model, tokenizer=tokenizer)
        summary = summary_pipeline(text, max_length=150, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    return ""

# Summarize button
if st.button("Summarize"):
    summary = summarize_affidavit(affidavit_text)
    st.subheader("Summarized Affidavit:")
    st.write(summary)

# About section
st.write("\n\n")
st.subheader("About")
st.write(
    "This Streamlit app uses RoBERTa to summarize affidavits. It's a simple demonstration and can be further improved."
)
