# NLP Group Assignment 1

Concise project report and run guide for all four assignment questions.

## Group Members

- Galaba Vamsi (12340770)
- Bhagyashree Solanki (12240400)
- Anamika Rajesh (12340250)
- Ritika Singh Katoch (12341810)

## Project Overview

This repository implements four NLP systems:

| Question | Main implementation | Summary |
|---|---|---|
| Q1 | `src/q1_word_segmentation.py` | Word segmentation and POS tagging for English and Spanish, with baselines and error analysis. |
| Q2 | `src/q2_dependency_parser.py` | Arc-standard transition-based dependency parser trained on UD English-EWT. |
| Q3 | `src/q3_spelling_corrector.py` | Brown-corpus spelling corrector for non-word and real-word errors. |
| Q4 | `src/q4_live_editor.py` | Integrated Streamlit background editor combining Q1, Q3, PCFG parsing, and n-gram grammar checks. |

The assignment PDF is kept as `Group Assignment 1.pdf`.

## Setup

```powershell
cd C:\Users\anami\Desktop\nlp-group-assignment1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/download_data.py
```

For Q2, download UD English-EWT:

```powershell
git clone --depth 1 https://github.com/UniversalDependencies/UD_English-EWT.git data/UD_English-EWT
Copy-Item data/UD_English-EWT\en_ewt-ud-train.conllu data\en_ewt-ud-train.conllu
Copy-Item data/UD_English-EWT\en_ewt-ud-dev.conllu data\en_ewt-ud-dev.conllu
```

For Q4 sentence splitting:

```powershell
python -c "import nltk; nltk.download('punkt', download_dir='data/nltk'); nltk.download('punkt_tab', download_dir='data/nltk')"
```

## Question 1: Word Segmentation and POS Tagging

Q1 trains an add-k smoothed trigram word language model for segmentation and a trigram HMM-style POS tagger with emission and transition probabilities. English uses Brown. Spanish uses UD Spanish-GSD with morphology-aware tags such as `NOUN-Masc-Pl` and `DET-Fem-Sg`.

### Q1 Results

| Metric | English | Spanish |
|---|---:|---:|
| Evaluated sentences | 200 | 427 |
| Joint exact matches | 51/200 (25.50%) | 32/427 (7.49%) |
| Model segmentation exact | 141/200 (70.50%) | 103/427 (24.12%) |
| Greedy segmentation exact | 45/200 (22.50%) | 25/427 (5.85%) |
| Model POS accuracy | 91.40% | 90.75% |
| Most-frequent-tag baseline | 86.84% | 85.61% |

| Error source | English | Spanish |
|---|---:|---:|
| Gold tag positions | 3,616 | 10,520 |
| Comparable positions | 2,883 | 4,151 |
| Segmentation-caused errors | 733 | 6,369 |
| Genuine POS errors | 248 | 384 |

### Q1 Sample Outputs

```text
thequickbrownfoxjumpsoverthelazydog
-> [('the', 'AT'), ('quick', 'JJ'), ('brown', 'JJ'), ('fox', 'NN'),
    ('jumps', 'NNS'), ('over', 'IN'), ('the', 'AT'), ('lazy', 'JJ'),
    ('dog', 'NN')]

mispadrespuedenviajar
-> [('mis', 'DET-Pl'), ('padres', 'NOUN-Masc-Pl'),
    ('pueden', 'AUX-Pl'), ('viajar', 'VERB')]

lacasarojaesgrande
-> [('la', 'DET-Fem-Sg'), ('casa', 'NOUN-Fem-Sg'),
    ('roja', 'ADJ-Fem-Sg'), ('es', 'AUX-Sg'), ('grande', 'ADJ-Sg')]
```

### Q1 Run Command

```powershell
python scripts/generate_q1_report.py --english-limit 200
```

### Q1 Figures

The generated Q1 plots are stored in `assets/`:

- `assets/q1_segmentation_vs_baseline.png`
- `assets/q1_pos_vs_baseline.png`
- `assets/q1_error_source_breakdown.png`
- `assets/q1_confusion_english.png`
- `assets/q1_confusion_spanish.png`

## Question 2: Transition-Based Dependency Parser

Q2 implements an arc-standard dependency parser with `SHIFT`, `LEFT-ARC(label)`, and `RIGHT-ARC(label)`. It reads CoNLL-U files, generates oracle training transitions, extracts four POS features, trains a scikit-learn logistic regression classifier, and evaluates LAS on UD English-EWT dev data.

### Q2 Results

| Item | Value |
|---|---:|
| Training sentences loaded | 12,544 |
| Training examples generated | 392,242 |
| Skipped training sentences | 287 |
| Development sentences loaded | 2,001 |
| Development tokens evaluated | 25,148 |
| Correct head+label predictions | 14,331 |
| Skipped development sentences | 0 |
| LAS | 56.99% |

### Q2 Example Output

```text
Example: I saw the man with a telescope.

Predicted dependency arcs:
2 (saw         ) ->  1 (I           ) [nsubj]
4 (man         ) ->  3 (the         ) [det]
7 (telescope   ) ->  6 (a           ) [det]
7 (telescope   ) ->  5 (with        ) [case]
4 (man         ) ->  7 (telescope   ) [nmod]
2 (saw         ) ->  4 (man         ) [obj]
2 (saw         ) ->  8 (.           ) [punct]
0 (ROOT        ) ->  2 (saw         ) [root]
```

### Q2 Run Command

```powershell
python -m src.q2_dependency_parser
```

### Q2 Screenshots

- `assets/q2_training_summary.png`
- `assets/q2_las_result.png`
- `assets/q2_dependency_arcs.png`

## Question 3: Spelling Corrector

Q3 builds a Brown-corpus spelling corrector. Method A generates all edit-distance-1 candidates. Method B uses symmetric-delete preprocessing. Non-word errors are corrected by unigram frequency; real-word errors are corrected with local bigram context and a score margin.

### Q3 Results

| Metric | Value |
|---|---:|
| Vocabulary size | 40,387 |
| Training tokens | 888,308 |
| Non-word examples | 500 |
| Real-word examples | 500 |
| Non-word accuracy | 92.4% |
| Real-word accuracy | 70.0% |

### Q3 Speed Demon

| Method | Time for 1,000 words | Words changed |
|---|---:|---:|
| Method A: edit-distance-1 | 310.08 ms | 999 |
| Method B: symmetric delete | 203.80 ms | 1000 |

Method B is faster because it uses precomputed one-deletion keys and dictionary lookup instead of constructing all deletion, replacement, transposition, and insertion strings for every query word.

### Q3 Example Corrections

| Input | Corrected output | Type |
|---|---|---|
| I hav a good feeling about this. | I had a good feeling about this. | non-word |
| This is a test sentnce. | This is a test sentence. | non-word |
| I would like to sea the world. | I would like to see the world. | real-word |
| Please meat me at the station. | Please meet me at the station. | real-word |

### Q3 Run Commands

```powershell
python -m src.q3_spelling_corrector --evaluate --benchmark --max-examples 500
python -m src.q3_spelling_corrector --sentence "I hav a good feeling about this." --sentence "This is a test sentnce." --sentence "I would like to sea the world." --sentence "Please meat me at the station."
python -m src.q3_spelling_corrector --interactive
```

### Q3 Screenshots

- `assets/q3_accuracy.png`
- `assets/q3_benchmark.png`
- `assets/q3_examples.png`

## Question 4: Integrated Background Editor

Q4 is the Streamlit-hosted part of the assignment. It combines Q1 segmentation/POS tagging, Q3 spelling correction, a Penn Treebank PCFG parser, and shared Brown bigram/trigram language models in one live editor.

### Q4 Architecture

| Component | Role |
|---|---|
| `src/q4_live_editor.py` | Token stream processing, merge injection, alerts, latency tracking. |
| `src/q4_pcfg_parser.py` | Penn Treebank PCFG induction and Viterbi CKY parsing. |
| `src/q4_shared_lm.py` | Add-k bigram/trigram LMs and dev perplexity tuning. |
| `src/q4_passage_analysis.py` | Final sentence-level PCFG/n-gram comparison table. |
| `scripts/q4_streamlit_app.py` | Streamlit UI for simulated and genuine live typing. |

### Q4 Main Settings

| Setting | Value |
|---|---:|
| Merge probability `p` | 0.08 |
| Grammar trigger interval `N` | 5 words |
| Bigram smoothing `k` | 0.05 |
| Trigram smoothing `k` | 0.01 |
| Grammar alert threshold | Trigram PPL > 450 or Bigram PPL > 600 |

### Q4 Decision Rule

1. Prefer PCFG when the sentence parses and normalized log probability is at least -6.5.
2. Otherwise use trigram LM if trigram perplexity is below 300.
3. Otherwise use bigram LM if bigram perplexity is below 450.
4. Otherwise mark the sentence ungrammatical.

### Q4 Speed Demon

| Metric | Value |
|---|---:|
| Batch size | 1,000 words |
| Full live pipeline total | 30,457.70 ms |
| Full live pipeline average | 30.4577 ms/word |
| Isolated grammar total | 241.29 ms |
| Isolated grammar average | 0.2413 ms/word |
| Added latency | 30,216.41 ms |
| Segment alerts triggered | 119 |
| Spell alerts triggered | 881 |

### Q4 Run Commands

```powershell
python -m scripts.q4_speed_benchmark
streamlit run app.py
```

In Streamlit, choose `Question 4: Integrated Background Editor`. Use:

- `Genuine Live User Typing` for live alerts and final analysis.
- `Simulated Fast-Typing Demo` for random passage runs with missing-space merges.

### Q4 Screenshots

- `assets/q4_speed_demon.png`
- `assets/q4_sample_run_a.png`
- `assets/q4_sample_run_b.png`

## Screenshot Inventory

All report screenshots/figures should live in `assets/`.

| File | Purpose |
|---|---|
| `assets/logo.png` | Front-page/project logo. |
| `assets/q1_segmentation_vs_baseline.png` | Q1 segmentation baseline comparison. |
| `assets/q1_pos_vs_baseline.png` | Q1 POS baseline comparison. |
| `assets/q1_error_source_breakdown.png` | Q1 segmentation vs POS error split. |
| `assets/q1_confusion_english.png` | Q1 English confusion matrix. |
| `assets/q1_confusion_spanish.png` | Q1 Spanish confusion matrix. |
| `assets/q2_training_summary.png` | Q2 training and classifier summary. |
| `assets/q2_las_result.png` | Q2 LAS evaluation. |
| `assets/q2_dependency_arcs.png` | Q2 required sentence arcs. |
| `assets/q3_accuracy.png` | Q3 accuracy output. |
| `assets/q3_benchmark.png` | Q3 speed benchmark. |
| `assets/q3_examples.png` | Q3 corrected examples. |
| `assets/q4_speed_demon.png` | Q4 speed benchmark. |
| `assets/q4_sample_run_a.png` | Q4 simulated run A. |
| `assets/q4_sample_run_b.png` | Q4 simulated run B. |

## Tests

```powershell
python -m pytest tests/ -v
```

Verified result in the project environment: 18 tests passed.

## Repository Layout


https://nlp-group-assignment1-y92dzndu2g3nu6uavyfsbw.streamlit.app

```text
app.py                         Streamlit entry point for Q1 and Q4
src/q1_word_segmentation.py    Q1 segmentation, POS tagging, baselines, evaluation
src/q2_dependency_parser.py    Q2 dependency parser
src/q3_spelling_corrector.py   Q3 spelling corrector, evaluation, benchmark, CLI
src/q4_live_editor.py          Q4 live editor pipeline
src/q4_pcfg_parser.py          Q4 PCFG parser and tagset reconciliation
src/q4_shared_lm.py            Q4 shared bigram/trigram language models
src/q4_passage_analysis.py     Q4 final passage analysis
scripts/download_data.py       Corpus setup
scripts/generate_q1_report.py  Q1 report/figure generator
scripts/q4_speed_benchmark.py  Q4 benchmark
scripts/q4_streamlit_app.py    Q4 Streamlit interface
tests/                         Unit tests
assets/                        Screenshots and figures
reports/                       Q1 machine-readable report artifacts
```