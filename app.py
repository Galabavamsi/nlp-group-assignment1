"""Streamlit front end for the NLP Group Assignment (Questions 1, 2, 3, and 4)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import nltk
import pandas as pd
import streamlit as st

from scripts.q4_streamlit_app import render_q4_app
from src.q1_word_segmentation import build_system, sample_inputs


st.set_page_config(page_title="NLP Group Assignment 1", page_icon="🧩", layout="wide")


@st.cache_resource
def ensure_nltk_corpora(data_dir: str = "data") -> list[str]:
    """Make the bundled NLTK data visible and fetch anything genuinely missing.

    Three deployment-specific details are handled here:

    1. ``nltk.data.find`` only reports an *unpacked* resource directory. Corpora
       present purely as a ``.zip`` (e.g. ``reuters.zip``) raise ``LookupError``
       even though ``nltk.corpus.reuters`` loads fine from the zip, so the zip is
       checked as well before anything is considered missing.
    2. NLTK refuses to download into a world/group-writable directory, which is
       exactly what a hosted mount such as Streamlit Cloud's ``/mount/src/...``
       looks like (NLTK warns: "will not authorize the non-private download
       directory"). The repository copy is therefore used *read-only*, and any
       genuinely missing resource is downloaded into a private per-user cache
       instead, which NLTK does authorise.
    3. A failed download must never break the app: the repository already ships
       the corpora, so failures are reported back to the caller as a warning.

    Returns the list of resource names that could not be made available.
    """
    data_path = Path(data_dir)
    # Read-only location: the corpora committed under data/ (or the offline zip).
    nltk_data_dir = data_path / "nltk"
    nltk_data_dir.mkdir(parents=True, exist_ok=True)
    # Writable, private fallback for anything that has to be fetched at runtime.
    private_dir = Path.home() / ".cache" / "nlp_assignment_nltk"

    for candidate in (str(nltk_data_dir), str(private_dir)):
        if candidate not in nltk.data.path:
            nltk.data.path.insert(0, candidate)

    def _available(resource_type: str, name: str) -> bool:
        try:
            nltk.data.find(f"{resource_type}/{name}")
            return True
        except LookupError:
            pass
        # nltk.data.find does not see zip-only resources; check for the archive.
        return (nltk_data_dir / resource_type / f"{name}.zip").exists()

    def _fetch(resource_type: str, name: str) -> bool:
        """Download into a private directory; never raise."""
        for target in (nltk_data_dir, private_dir):
            try:
                if nltk.download(name, download_dir=str(target), quiet=True):
                    return True
            except Exception:
                continue
        return False

    missing: list[str] = []
    for corpus in ["brown", "treebank", "gutenberg", "reuters"]:
        if not _available("corpora", corpus) and not _fetch("corpora", corpus):
            missing.append(corpus)
    for tok in ["punkt", "punkt_tab"]:
        if not _available("tokenizers", tok) and not _fetch("tokenizers", tok):
            missing.append(tok)
    return missing


_missing_corpora = ensure_nltk_corpora()
if _missing_corpora:
    st.sidebar.warning(
        "NLTK data missing and could not be downloaded: "
        + ", ".join(_missing_corpora)
        + ". Run `python scripts/download_data.py` locally."
    )

st.sidebar.title("🧩 NLP Assignment Modules")
app_mode = st.sidebar.radio(
    "Select Assignment Question",
    [
        "Question 4: Integrated Background Editor",
        "Question 1: Word Segmentation & POS Tagging",
        "Question 2: Dependency Parser",
        "Question 3: Spelling Corrector",
    ],
)

if app_mode == "Question 4: Integrated Background Editor":
    render_q4_app(standalone=False)

elif app_mode == "Question 1: Word Segmentation & POS Tagging":
    st.title("NLP Group Assignment — Question 1")
    st.caption("Trigram Word Segmentation & POS Tagging (English & Spanish)")

    language = st.selectbox("Language", ["english", "spanish"])
    default_text = sample_inputs(language)[0]
    text = st.text_input("Enter a sentence with spaces removed", value=default_text)
    data_dir = st.text_input("Corpus directory", value="data")

    @st.cache_resource(show_spinner="Training the Question 1 models...")
    def get_q1_system(selected_language: str, selected_data_dir: str):
        return build_system(selected_language, selected_data_dir)

    try:
        system = get_q1_system(language, data_dir)
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
            results_path = Path("reports/q1_results.json")
            if results_path.exists():
                with results_path.open("r", encoding="utf-8") as f:
                    cached_q1 = json.load(f)
                lang_key = "english" if language == "english" else "spanish"
                if lang_key in cached_q1:
                    st.json(cached_q1[lang_key])
                else:
                    st.json(cached_q1)
            else:
                st.json(system.evaluate())
    except Exception as exc:
        st.error("The model could not be loaded. Run the corpus setup command from the README first.")
        st.exception(exc)

elif app_mode == "Question 2: Dependency Parser":
    st.title("NLP Group Assignment — Question 2")
    st.caption("Arc-Standard Transition-Based Dependency Parser (Universal Dependencies English-EWT)")

    from src.q2_dependency_parser import (
        dependency_parse,
        make_example_sentence,
    )

    preset_examples = {
        "The cat sat on the mat.": (
            ["The", "cat", "sat", "on", "the", "mat", "."],
            ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN", "PUNCT"],
        ),
        "She eats a green salad.": (
            ["She", "eats", "a", "green", "salad", "."],
            ["PRON", "VERB", "DET", "ADJ", "NOUN", "PUNCT"],
        ),
        "I saw the man with a telescope.": (
            ["I", "saw", "the", "man", "with", "a", "telescope", "."],
            ["PRON", "VERB", "DET", "NOUN", "ADP", "DET", "NOUN", "PUNCT"],
        ),
    }

    selected_example = st.selectbox("Select Example Sentence", list(preset_examples.keys()))
    words, pos_tags = preset_examples[selected_example]

    @st.cache_resource(show_spinner="Loading Question 2 Dependency Parser...")
    def get_q2_classifier():
        cache_path = Path("data/cache/q2_classifier.pkl")
        if cache_path.exists():
            with cache_path.open("rb") as f:
                return pickle.load(f)
        from src.q2_dependency_parser import build_training_data, parse_conllu, train_classifier
        train_path = Path("data/en_ewt-ud-train.conllu")
        if not train_path.exists():
            return None
        train_sentences = parse_conllu(train_path)
        X, y, _ = build_training_data(train_sentences[:2000])
        clf = train_classifier(X, y)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with cache_path.open("wb") as f:
                pickle.dump(clf, f)
        except Exception:
            pass
        return clf

    classifier = get_q2_classifier()

    if classifier is None:
        st.warning("Training data (data/en_ewt-ud-train.conllu) not found. Run scripts/download_data.py to fetch UD English-EWT.")
    else:
        sentence = make_example_sentence(words, pos_tags)
        try:
            arcs = dependency_parse(sentence, classifier)
            st.subheader("Predicted Dependency Arcs")
            token_map = sentence.token_map()
            table_data = []
            for head_id, dep_id, label in arcs:
                head_word = "ROOT (0)" if head_id == 0 else f"{token_map[head_id].word} ({head_id})"
                dep_word = f"{token_map[dep_id].word} ({dep_id})"
                dep_pos = token_map[dep_id].pos
                table_data.append({
                    "Dependent ID": dep_id,
                    "Dependent Word": dep_word,
                    "POS": dep_pos,
                    "Relation (deprel)": label,
                    "Head": head_word,
                })
            df = pd.DataFrame(table_data).sort_values("Dependent ID")
            st.dataframe(df, use_container_width=True)

            with st.expander("Structured Transition Output"):
                for head_id, dep_id, label in arcs:
                    head_str = "ROOT" if head_id == 0 else f"'{token_map[head_id].word}'"
                    st.write(f"• **{head_str}** ──`[{label}]`──> **'{token_map[dep_id].word}'**")
        except Exception as err:
            st.error(f"Parser error: {err}")

elif app_mode == "Question 3: Spelling Corrector":
    st.title("NLP Group Assignment — Question 3")
    st.caption("Brown Corpus Spelling Corrector: Method A (Edit-1) vs. Method B (Symmetric Delete)")

    from src.q3_spelling_corrector import build_corrector_from_brown

    @st.cache_resource(show_spinner="Loading Brown Corpus Spelling Corrector...")
    def get_q3_corrector(data_dir: str = "data"):
        return build_corrector_from_brown(seed=7, data_dir=data_dir)[0]

    try:
        corrector = get_q3_corrector()
        st.sidebar.subheader("Spelling Settings")
        method = st.sidebar.selectbox(
            "Candidate Generation Method",
            ["symdelete", "edit"],
            format_func=lambda m: "Method B: Symmetric Delete (Fast)" if m == "symdelete" else "Method A: Edit Distance 1",
        )
        margin = st.sidebar.slider("Real-word Confidence Margin", min_value=0.5, max_value=5.0, value=2.0, step=0.1)

        example_choice = st.selectbox(
            "Choose a Test Sentence",
            [
                "I hav a good feeling about this.",
                "This is a test sentnce.",
                "I would like to sea the world.",
                "Please meat me at the station.",
                "Custom",
            ],
        )
        user_input = st.text_input("Sentence to correct", value="" if example_choice == "Custom" else example_choice)

        if st.button("Correct Sentence", type="primary") and user_input:
            corrector.real_word_margin = margin
            result = corrector.correct_sentence(user_input, method=method)
            st.subheader("Correction Result")
            st.markdown(f"**Output:** {result.highlighted_text}")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Latency", f"{result.latency_ms:.2f} ms")
            with col2:
                st.metric("Corrections Applied", len(result.changes))

            if result.changes:
                st.write("**Modifications Detail:**")
                for change in result.changes:
                    st.info(f"Token `{change.original}` → `{change.corrected}` (Kind: {change.kind}, Log-prob: {change.score:.2f})")
            else:
                st.success("No spelling errors detected!")
    except Exception as exc:
        st.error(f"Failed to load spelling corrector: {exc}")
