# NLP Group Assignment 1

This repository is the shared implementation workspace for the NLP group assignment. Question 1 is implemented first; the remaining questions are documented as teammate handoff goals so the work can be extended without restructuring the project.

## Current status

### Question 1 - implemented

The current code uses **English + Spanish** (Spanish is the morphologically richer comparison language):

- Trigram word language model with add-k smoothing.
- Dynamic-programming/Viterbi word segmentation for space-free input.
- Trigram HMM-style POS tagging with emission and transition probabilities.
- Spanish morphology-aware tags such as `NOUN-Masc-Sg` and `ADJ-Fem-Sg` using UD `Gender` and `Number` features.
- Greedy longest-match segmentation baseline.
- Most-frequent-tag POS baseline.
- Exact segmentation/joint accuracy, POS accuracy, confusion matrix, and a segmentation-caused versus genuine POS-error breakdown.
- Streamlit demo for trying both languages interactively.

The main implementation is [src/q1_word_segmentation.py](src/q1_word_segmentation.py). The PDF brief is retained as [Group Assignment 1.pdf](Group%20Assignment%201.pdf) for reference.

## Setup

From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/download_data.py
```

The setup downloads the NLTK Brown corpus and clones the official [UD Spanish-GSD](https://github.com/UniversalDependencies/UD_Spanish-GSD) repository into `data/`. Corpus data is ignored by Git.

## Run Question 1

```powershell
python -m src.q1_word_segmentation --language english --limit 1000
python -m src.q1_word_segmentation --language spanish --limit 1000
python -m src.q1_word_segmentation --language english --text thequickbrownfoxjumpsoverthelazydog
python -m src.q1_word_segmentation --language spanish --text mispadrespuedenviajar
streamlit run app.py
```

The Spanish decoder returns morphology-aware tags internally. The report can show the base UPOS tag and the enriched tag side by side when discussing agreement.

## Suggested teammate workflow

Each member should work in a separate branch and add tests/data-download instructions for their question. Keep reusable models in `src/` and keep large corpora out of Git.

### Question 2 goal - transition-based dependency parser

Add `src/q2_dependency_parser.py` and a small evaluation entry point:

1. Download `UD_English-EWT` and parse train/dev CoNLL-U files.
2. Implement CoNLL-U sentence storage with words, POS tags, heads, and dependency labels.
3. Generate oracle examples for the arc-standard `SHIFT`, `LEFT-ARC(label)`, and `RIGHT-ARC(label)` transitions.
4. Extract the four required POS features (top/second stack and first/second buffer item).
5. Train a scikit-learn transition classifier, parse the dev split, and report LAS.
6. Add examples for “The cat sat on the mat.”, “She eats a green salad.”, and “I saw the man with a telescope.”

### Question 3 goal - spelling corrector and benchmark

Add `src/q3_spelling_corrector.py` and a terminal entry point:

1. Build Brown vocabulary, unigram frequencies, and a bigram model.
2. Implement standard edit-distance-1 candidate generation.
3. Implement symmetric-delete preprocessing and lookup.
4. Correct non-word errors by unigram frequency and real-word errors with contextual bigram scoring.
5. Generate separate non-word and real-word test sets, report accuracy, and benchmark exactly 1,000 words through both candidate methods.
6. Add the continuous `exit`-driven CLI with changed-word highlighting and latency.

### Question 4 goal - integrated Streamlit editor

Extend `app.py` only after Q1 and Q3 APIs are stable. Add `src/q4_editor.py`:

1. Reuse Q1’s English decoder and Q3’s vocabulary/candidate functions; do not retrain them inside Q4.
2. Simulate live typing and also process user-entered text incrementally.
3. Inject merged tokens with a documented probability `p`, emit `[SEGMENT-ALERT]`, `[SPELL-ALERT]`, and interval-based `[GRAMMAR-ALERT]` messages.
4. Train a Penn Treebank PCFG, implement Viterbi/CKY parsing, and document Q1-to-Treebank tag reconciliation.
5. Reuse one shared smoothed Brown bigram/trigram LM for grammar alerts and final sentence scores.
6. Produce the required per-sentence comparison table and exactly-1,000-word speed benchmark.
7. Add the comparative report, two full random sample runs, and a live-deployment screenshot/transcript.

## Repository layout

```text
app.py                         Streamlit entry point (Q1 now; Q4 extension later)
src/q1_word_segmentation.py   Q1 models, decoder, baselines, evaluation
scripts/download_data.py      Corpus setup
tests/                         Shared tests to be added per question
data/                          Local corpora, ignored by Git
```

## Reproducibility notes

English uses the Brown corpus with a seeded 80/20 split. Spanish uses the UD Spanish-GSD train/test split. The trigram decoder uses `max_word_length=20` and add-k `k=0.1`; these settings should be reported and tuned only with the training/dev data available for the language.
