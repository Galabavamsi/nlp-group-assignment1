"""Small Streamlit front end for the completed Question 1 work."""

from __future__ import annotations

import streamlit as st

from src.q1_word_segmentation import build_system, sample_inputs


st.set_page_config(page_title="NLP Group Assignment", page_icon="🧩", layout="wide")
st.title("NLP Group Assignment")
st.caption("Question 1: trigram word segmentation + POS tagging")

language = st.selectbox("Language", ["english", "spanish"])
default_text = sample_inputs(language)[0]
text = st.text_input("Enter a sentence with spaces removed", value=default_text)
data_dir = st.text_input("Corpus directory", value="data")


@st.cache_resource(show_spinner="Training the Question 1 models...")
def get_system(selected_language: str, selected_data_dir: str):
    return build_system(selected_language, selected_data_dir)


try:
    system = get_system(language, data_dir)
    decoded = system.decode(text)
    st.subheader("Decoded output")
    st.code(str(decoded), language="text")
    left, right = st.columns(2)
    with left:
        st.metric("Training sentences", len(system.train_sentences))
        st.metric("Vocabulary", len(system.segmenter.vocabulary))
    with right:
        st.metric("Test sentences", len(system.test_sentences))
        st.metric("Max word length", system.segmenter.max_word_length)
    with st.expander("Evaluation summary"):
        st.json(system.evaluate())
except Exception as exc:
    st.error("The model could not be loaded. Run the corpus setup command from the README first.")
    st.exception(exc)
