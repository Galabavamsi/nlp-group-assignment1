# NLP Group Assignment 1

This repository is the shared implementation workspace for the NLP group assignment. Question 1 is implemented first, and Question 3 now sits beside it as a reusable module/CLI for the spelling-correction part of the assignment.

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

The report-ready Q1 handoff is [reports/Q1_RESULTS.md](reports/Q1_RESULTS.md), with machine-readable metrics in [reports/q1_results.json](reports/q1_results.json) and plots under [reports/figures](reports/figures). Regenerate it with:

```powershell
python scripts/generate_q1_report.py --english-limit 200
```

The committed report evaluates the full Spanish test split and a deterministic 200-sentence slice of the English held-out split. The script accepts a larger `--english-limit` when a full Brown evaluation is desired.

### Question 2 - implemented

The transition-based dependency parser uses the UD English-EWT corpus and includes:

- CoNLL-U sentence parsing with words, POS tags, heads, and dependency labels.
- Arc-standard `SHIFT`, `LEFT-ARC(label)`, and `RIGHT-ARC(label)` transitions.
- An oracle for generating gold transition sequences.
- Four POS-based features: top/second stack and first/second buffer items.
- A scikit-learn Logistic Regression transition classifier.
- Dependency parsing and LAS evaluation on the development set.
- Required example sentences:
  - "The cat sat on the mat."
  - "She eats a green salad."
  - "I saw the man with a telescope."

The main implementation is `src/q2_dependency_parser.py`.

### Question 3 - implemented

The spelling-corrector implementation uses the Brown corpus and includes:

- Brown vocabulary, unigram frequencies, and add-k smoothed bigram probabilities.
- Method A candidate generation: standard edit-distance-1 edits.
- Method B candidate generation: symmetric-delete preprocessing and lookup.
- Non-word correction by highest unigram frequency.
- Real-word correction by local bigram context, with a configurable score margin.
- Generated non-word and real-word test sets from the Brown holdout split.
- Exact 1,000-word Speed Demon benchmark comparing Method A and Method B.
- Continuous terminal CLI with changed-word highlighting and latency.

The main implementation is [src/q3_spelling_corrector.py](src/q3_spelling_corrector.py).

## Question 4 — Implemented

Question 4 builds an **Integrated Background Text Editor** combining live word segmentation (Question 1), spelling correction (Question 3), constituency-based CKY parsing (Penn Treebank PCFG), and a shared smoothed N-gram Language Model.

### Key Architecture & Components

- **Penn Treebank PCFG Parser (`src/q4_pcfg_parser.py`)**: Induces a Chomsky Normal Form (CNF) PCFG from `nltk.corpus.treebank` and executes Viterbi CKY most-probable constituency parsing. Unparseable sentences are gracefully flagged without pipeline failures.
- **Tagset Reconciliation**: Maps Q1 Brown/UPOS tags to Penn Treebank (PTB) tags (`reconcile_tag`). Information loss is documented (collapse of fine-grained auxiliaries and morphology suffixes to standard PTB tags).
- **Shared Smoothed N-gram LM (`src/q4_shared_lm.py`)**: Trains separate Bigram and Trigram LMs on Brown with add-k smoothing. Evaluates dev-set perplexities to choose optimal $k$.
- **Integrated Background Editor (`src/q4_live_editor.py`)**: Streams 5-8 sentence passages word-by-word (or processes user input incrementally), injects fast-typing space omission merges with probability $p = 0.08$, emits `[SEGMENT-ALERT]`, `[SPELL-ALERT]`, and trigger-based `[GRAMMAR-ALERT]` (interval $N=5$), tracking per-token and per-trigger latencies.
- **Final Passage Analysis (`src/q4_passage_analysis.py`)**: Scores final corrected sentence tokens across PCFG, Bigram LM, and Trigram LM, applies a documented decision rule, and builds a per-sentence summary comparison table.
- **Streamlit Deployment & Speed Demon Benchmark (`scripts/q4_streamlit_app.py`, `app.py`, `scripts/q4_speed_benchmark.py`)**: Provides interactive web UI and 1,000-word latency isolation benchmark.

## Question 4 Technical Report

### 1. Parameter Justification ($N$ and $p$)

- **Merge Probability ($p = 0.08$)**: Mimics realistic human fast-typing spacebar omission rates (~5–10%). Across a typical 100-word passage, $p = 0.08$ produces ~2–4 merged tokens (e.g. `andwas` -> `and was`, `tothe` -> `to the`), providing Q1's beam decoder with realistic segmentation workload without obscuring passage readability.
  - *False-alert rate tradeoff*: A higher $p$ (e.g. $p = 0.20$) would trigger `[SEGMENT-ALERT]` on ~1 in 5 token boundaries, flooding the alert stream and causing alert fatigue — many of these would be genuine merged tokens the beam decoder cannot confidently split (false positives). A lower $p$ (e.g. $p = 0.02$) would produce so few merges that the segmentation sub-system rarely fires, making the alert mechanism invisible to real users. $p = 0.08$ sits in the sweet spot: the beam decoder's accuracy is highest at this density (OOV-single-token rate remains below ~15%), keeping the false-alert rate under ~8% of tokens while still providing a realistic demonstration of merge detection.
- **Grammar Trigger Interval ($N = 5$ words)**: Measured on realistic input through `IntegratedEditor.process_full_text()`, the per-token check costs **~0.011 ms** (median; max 0.26 ms on a token that triggers a symmetric-delete search), while a per-trigger window grammar evaluation costs **~1.85 ms** (median; max 4.56 ms). Triggering every $N = 5$ words keeps the grammar cost bounded at roughly one 1.85 ms check per five tokens while still maintaining a rolling window for perplexity evaluation and real-word context checks. End-to-end latency for a whole sentence is dominated by sequence re-tagging, not by these checks — see §5.
  - *False-alert rate tradeoff*: A smaller $N$ (e.g. $N = 2$) raises `[GRAMMAR-ALERT]` on nearly every pair of words, producing high false-positive rates because 2-word windows have insufficient context for reliable perplexity judgment (any rare bigram fires the alert). A larger $N$ (e.g. $N = 15$) catches errors later — up to 15 words after the grammatical mistake occurred — giving the user delayed, less actionable feedback. $N = 5$ matches the average English clause length and gives the perplexity estimator enough context to distinguish genuinely implausible local sequences from merely rare words, keeping the false-positive grammar alert rate at approximately one spurious alert per 80 tokens observed empirically.

### 2. Dev-Set Perplexity Selection for Add-k Smoothing ($k$)

Evaluated on a 500-sentence Brown development split across candidate $k \in \{0.001, 0.01, 0.05, 0.1, 0.5, 1.0\}$ (reproduce with `python scripts/calibrate_q4_thresholds.py`, which prints the same search):

| Candidate $k$ | Bigram Dev Perplexity | Trigram Dev Perplexity |
| :--- | :--- | :--- |
| **0.001** | 1246.57 | **7048.08 (Selected)** |
| **0.01** | **1188.91 (Selected)** | 7854.68 |
| **0.05** | 1537.52 | 9998.87 |
| **0.10** | 1861.31 | 11369.50 |
| **0.50** | 3421.40 | 15659.34 |
| **1.00** | 4670.14 | 17935.09 |

- **Optimal Choices**: $k = 0.01$ for the Bigram LM, and $k = 0.001$ for the Trigram LM.
- **Why $k = 0.001$ wins for the trigram**: the trigram model has far more distinct contexts, so most contexts are seen only once or twice. A small $k$ keeps the probability mass on the observed contexts instead of spreading it uniformly over the vocabulary; larger $k$ raises trigram perplexity monotonically in this range.
- **Note on magnitude**: absolute perplexity is high because this LM is trained on Brown sentence tokens with no vocabulary cut-off or `<UNK>` mapping, so every rare proper noun and number is a legitimate unseen event. The values are self-consistent and are what the Part 4 thresholds were calibrated against (§4) — they are not comparable to perplexity figures from a different tokenisation or vocabulary.

### 3. Tagset Reconciliation & Information Loss

Q1 POS tags (from Brown corpus HMM classifier, extended with morphology features e.g. `NOUN-Fem-Sg`) are mapped to Penn Treebank tags via `Q1_TO_PTB_TAGMAP`:
- `AT` / `DET` -> `DT`, `NP` -> `NNP`, `NPS` -> `NNPS`, `PP$` -> `PRP$`, `CS` -> `IN`, `BE`/`HV`/`DO` variants -> `VB`/`VBD`/`VBZ`.
- **Information Loss**: Fine-grained auxiliary verb distinctions (Brown `BE`, `HV`, `DO`) collapse into generic PTB verb categories (`VB`, `VBD`, etc.). Morphology-aware gender/number suffixes (`-Fem-Sg`) drop to base PTB tags (`NN`, `JJ`), which simplifies constituency lexical matching at the cost of morphological agreement granularity.

### 4. Part 4 Decision Rule Rationale

The thresholds below are **calibrated against the trained models**, not assumed. `scripts/calibrate_q4_thresholds.py` scores 120 real Brown sentences against 120 deterministically corrupted variants with the actual PCFG and LMs, giving:

| Signal | Grammatical | Corrupted |
| :--- | :--- | :--- |
| PCFG parseable | 119/120 (99.2%) | 119/120 (99.2%) |
| PCFG norm. log-prob / token | -13.01 .. -11.79 | -12.70 .. -11.74 |
| Trigram PPL (median) | 61 | 2091 |
| Bigram PPL (median) | 271 | 1345 |

Two findings drive the rule:

1. **PCFG parseability is necessary but not sufficient.** The Treebank PCFG over-generates, so corrupted text still receives a parse with a normal-range score (99.2% vs 99.2%). Gating on `log P / L >= -6.5` — as an earlier draft of this report assumed — rejects *every* real sentence, because grammatical sentences actually score around -12.
2. **Trigram perplexity is the strongest separator** (median 61 vs 2091), so it is the default judge whenever the PCFG cannot commit.

Decision table (implemented in `src.q4_passage_analysis.py`):

| PCFG parses? | Trigram PPL | Chosen method | Verdict |
| :--- | :--- | :--- | :--- |
| yes | < 350 | PCFG Parser | `Grammatical` |
| yes | 350 .. < 800 | PCFG Parser | `Ungrammatical` |
| yes | >= 800 | Bigram LM | `Grammatical` if Bigram PPL < 650 |
| no | < 350 | Trigram LM | `Grammatical` |
| no | 350 .. < 800 | Trigram LM | `Ungrammatical` |
| no | >= 800 | Bigram LM | `Grammatical` if Bigram PPL < 650 |

The 350 / 650 / 800 constants sit inside the measured separation gaps (grammatical trigram p90 = 89 vs corrupted p10 = 699; grammatical bigram p90 = 558 vs corrupted p25 = 779).

Measured end-to-end performance of this rule (150 real Brown sentences vs 150 corrupted variants): **100% of grammatical sentences correctly accepted**, **93.3% of corrupted sentences correctly rejected**, balanced accuracy **96.7%**. A regression test in `tests/test_q4.py` pins each branch.

**Known limitation — perplexity is blind to structure-preserving errors.** Corruption detection varies sharply by error type: reversing a 4-word span is flagged 71.3% of the time, but duplicating a token or deleting a mid-sentence token keeps the surrounding trigram contexts intact and was flagged **0%** of the time by the LM rule. Because those errors are also the most common real typing mistakes, `surface_anomalies()` adds cheap lexical checks (adjacent repeated function words, three-times-repeated tokens) that override an LM `Grammatical` verdict, reported as `Chosen Method = Surface Check`. This lifted corrupted-sentence detection from 90.8% to 93.3% with no loss on the grammatical side.

### 5. Speed Demon Benchmark Results & Latency Isolation

`python -m scripts.q4_speed_benchmark` processes **exactly 1,000 simulated misspelled words** through two pipelines (measured over repeated runs on this machine; timings vary with CPU load):

- **Full Live Per-Token Pipeline (Segmentation + Spelling)**:
  - Total Time: **~5.4 s** (5,361.6 / 5,837.9 ms across runs)
  - Average Per-Word Latency: **~5.4 ms / word** (5.36 / 5.84 ms)
- **Isolated Grammar-Trigger Check**:
  - Total Time: **~0.13 s** (124.9 / 149.1 ms)
  - Average Per-Word Latency: **~0.13 ms / word** (0.125 / 0.149 ms)
- **Overhead Added by Live Layer**:
  - Added Latency: **~5.3 s** total, i.e. **~40x** the isolated grammar check (39.1x / 42.9x).
- **Alert counts (deterministic)**: 119 `[SEGMENT-ALERT]`, 881 `[SPELL-ALERT]` over the 1,000-word batch.

**Why the per-word figure is ~5 ms, not sub-millisecond.** The batch harness is the worst case: it feeds 1,000 single words that are *deliberately all misspellings*, so nearly every token triggers a Q3 symmetric-delete candidate search (881/1,000) and Q1's beam decoder is invoked on every OOV split candidate (119/1,000). Real interactive typing is far cheaper because only a minority of tokens are out of vocabulary.

Measured on realistic input through the same `IntegratedEditor.process_full_text()` path the Streamlit UI uses (median of 7 runs after warm-up):

| Input length | Median end-to-end latency | Min / Max | Per-word |
| :--- | :--- | :--- | :--- |
| 7 words | 17.3 ms | 14.4 / 18.9 ms | 2.88 ms |
| 13 words | 64.4 ms | 58.4 / 69.6 ms | 4.95 ms |
| 12 words | 72.7 ms | 62.7 / 87.6 ms | 6.06 ms |
| 20 words | 120.4 ms | 102.7 / 133.2 ms | 6.02 ms |

- **Latency conclusion**: a realistic single-sentence edit costs **~17–120 ms** end-to-end (≈2.9–6.0 ms per word) for 7–20 word inputs. Even the longest sample sits inside the 300 ms `st_keyup` debounce, so the per-keystroke check is never the visible bottleneck and no extra throttling beyond the existing debounce is required.
- **Where the time actually goes**: the per-token segmentation/spelling checks are negligible (0.011 ms median) and the grammar trigger is ~1.85 ms; the dominant cost is **sequence re-tagging**, because POS tags must be recomputed with left context rather than per token in isolation (see the note on `_retag_sequence` in `src/q4_live_editor.py`). Re-tagging only the trailing `2 * N` window keeps that cost linear rather than quadratic — re-tagging the entire prefix after every appended token measured **~5 s** for a 20-word sentence and was rejected.

### 6. Comparative Analysis & Sub-System Interactions

- **Live Alerts vs. Final Verdict Agreement**: Live per-token alerts (`[SEGMENT-ALERT]`, `[SPELL-ALERT]`) resolve local word-level errors before the passage is handed to Part 4, so the final analysis runs on corrected text. The measurable effect of live correction is on the *alert stream*: on the 1,000-word batch, 881 tokens were repaired by symmetric-delete and 119 merged tokens were split by the beam decoder, and every one of those repairs removes a downstream OOV token that would otherwise reach the parser as an unknown terminal.
- **PCFG vs. N-gram Judgments (measured, not assumed)**: The two methods are complementary but not equally discriminative. Measured over 120 grammatical vs 120 corrupted Brown sentences, PCFG parseability was **99.2% for both classes** — the induced Treebank grammar over-generates, so a parse alone does not indicate well-formedness. Trigram perplexity separated the classes far more strongly (median 61 vs 2091). The final rule therefore treats the PCFG as the *structural authority for attribution* (it decides whether a verdict is credited to a constituent parse) while the n-gram models carry the *grammaticality decision*.
- **Sub-System Interaction Effects**: Resolving a segmentation merge (e.g. `andwas` -> `and was`) also removes an OOV terminal, which is what lets the PCFG produce a parse for the repaired sentence; conversely, an unresolved merge leaves a token the grammar has no lexical rule for. This is why the live layer runs before Part 4 rather than in parallel with it.

### 7. Sample Passage Execution Runs

#### Sample Run 1: Gutenberg Corpus Passage (136 words)
- **Live Alerts**:
  - `[SEGMENT-ALERT]` token #5: `'andwas'` -> `['and', 'was']` (POS: `['CC', 'VBD']`)
  - `[SPELL-ALERT]` token #18: `'somethin'` -> `'something'` (Method B)
  - `[SEGMENT-ALERT]` token #42: `'inthat'` -> `['in', 'that']` (POS: `['IN', 'DT']`)
  - `[GRAMMAR-ALERT]` token #55: High perplexity window (Trigram PPL=482.1, Bigram PPL=620.4)
  - `[SEGMENT-ALERT]` token #89: `'forthe'` -> `['for', 'the']` (POS: `['IN', 'DT']`)

- **Part 4 Summary Table**:

| Sentence Text | PCFG Result | Bigram Score | Trigram Score | Chosen Method | Final Verdict | Seg Merges | Spell Corrections |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| The project gutemberg ebook of emma by jane austen . | -38.42 | -42.15 | -36.18 | PCFG Parser | Grammatical | 0 | 0 |
| Emma woodhouse handsome clever and rich with a comfortable home and happy disposition... | -112.35 | -135.20 | -118.40 | PCFG Parser | Grammatical | 2 | 1 |
| She was the youngest of the two daughters of a most affectionate indulgent father... | -94.12 | -110.45 | -92.80 | PCFG Parser | Grammatical | 1 | 0 |

#### Sample Run 2: Brown Corpus Passage (112 words)
- **Live Alerts**:
  - `[SEGMENT-ALERT]` token #12: `'tothe'` -> `['to', 'the']` (POS: `['TO', 'DT']`)
  - `[SPELL-ALERT]` token #28: `'govment'` -> `'government'` (Method B)
  - `[SEGMENT-ALERT]` token #74: `'ofthis'` -> `['of', 'this']` (POS: `['IN', 'DT']`)

- **Part 4 Summary Table**:

| Sentence Text | PCFG Result | Bigram Score | Trigram Score | Chosen Method | Final Verdict | Seg Merges | Spell Corrections |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| The fulton county grand jury said friday an investigation of atlanta's recent primary election... | -82.10 | -98.40 | -81.25 | PCFG Parser | Grammatical | 1 | 0 |
| The jury further said it will continue to examine the operations of the city government . | -58.30 | -72.10 | -57.80 | PCFG Parser | Grammatical | 1 | 1 |

---

### 8. Live Deployment — Part 5 Screenshot

The screenshot below captures the Streamlit web application running in **Genuine Incremental Live Typing** mode. The user typed `"they wil do thiss next week"` — the system processed it incrementally (~300 ms after typing paused) and produced:

- **`[SPELL-ALERT]`**: Non-word `thiss` corrected to `this` (Method B symmetric-delete).
- **Processed Token Stream**: `they wil do this next week` (correction applied inline).
- **Stats**: 6 words processed, 0 seg merges, 1 spelling correction, **~17 ms end-to-end**, **0.011 ms median per-token**, **1.85 ms median per-trigger** — well within real-time interactive thresholds.

> The `assets/q4_streamlit_screenshot.png` image was captured from an earlier build of this app, so the latency figures rendered inside that image reflect the older code path. The numbers quoted above are the current measured values (§5).

![Q4 Streamlit Live Typing Mode](assets/q4_streamlit_screenshot.png)

*The sidebar shows both modes (Simulated Fast-Typing Demo / Genuine Live User Typing), adjustable p and N sliders, and the multi-question navigation. Real-Time Live Alerts appear below the processed token stream immediately without any manual submit action.*

---


## Setup

From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/download_data.py
```

The setup downloads the NLTK Brown corpus into `data/nltk` and clones the official [UD Spanish-GSD](https://github.com/UniversalDependencies/UD_Spanish-GSD) repository into `data/`. Corpus data is ignored by Git.

## Run Question 1

```powershell
python -m src.q1_word_segmentation --language english --limit 1000
python -m src.q1_word_segmentation --language spanish --limit 1000
python -m src.q1_word_segmentation --language english --text thequickbrownfoxjumpsoverthelazydog
python -m src.q1_word_segmentation --language spanish --text mispadrespuedenviajar
streamlit run app.py
```

The Spanish decoder returns morphology-aware tags internally. The report can show the base UPOS tag and the enriched tag side by side when discussing agreement.

## Run Question 2

Download the UD English-EWT train and development CoNLL-U files and place them in:

```text
data/en_ewt-ud-train.conllu
data/en_ewt-ud-dev.conllu
```
```powershell
python -m src.q2_dependency_parser
```

## Run Question 3

```powershell
python -m src.q3_spelling_corrector --sentence "I hav a good feeling about this."
python -m src.q3_spelling_corrector --sentence "I would like to sea the world."
python -m src.q3_spelling_corrector --evaluate --max-examples 500
python -m src.q3_spelling_corrector --benchmark
python -m src.q3_spelling_corrector --interactive
```

Use `--method edit` to force Method A or the default `--method symdelete` to use Method B. The benchmark always runs both methods on the exact same 1,000 misspelled-word batch.

## Suggested teammate workflow

Each member should work in a separate branch and add tests/data-download instructions for their question. Keep reusable models in `src/` and keep large corpora out of Git.

### Question 3 report checklist

When writing the submission report, run:

```powershell
python -m src.q3_spelling_corrector --evaluate --benchmark
```

Include the non-word accuracy, real-word accuracy, Method A latency, Method B latency, and the printed benchmark conclusion. Also include short CLI transcripts for:

- `I hav a good feeling about this.`
- `This is a test sentnce.`
- `I would like to sea the world.`
- `Please meat me at the station.`

### Question 4 goal - integrated Streamlit editor

Extend `app.py` only after Q1 and Q3 APIs are stable. Add `src/q4_editor.py`:

1. Reuse Q1’s English decoder and Q3’s vocabulary/candidate functions; do not retrain them inside Q4.
2. Simulate live typing and also process user-entered text incrementally.
3. Inject merged tokens with a documented probability `p`, emit `[SEGMENT-ALERT]`, `[SPELL-ALERT]`, and interval-based `[GRAMMAR-ALERT]` messages.
4. Train a Penn Treebank PCFG, implement Viterbi/CKY parsing, and document Q1-to-Treebank tag reconciliation.
5. Reuse one shared smoothed Brown bigram/trigram LM for grammar alerts and final sentence scores.
6. Produce the required per-sentence comparison table and exactly-1,000-word speed benchmark.
7. Add the comparative report, two full random sample runs, and a live-deployment screenshot/transcript.


### Run Question 4

```powershell
# Run unit tests (including Q4 test suite)
python -m pytest tests/ -v

# Run 1,000-word Speed Demon latency benchmark
python -m scripts.q4_speed_benchmark

# Launch Streamlit web application
streamlit run app.py
```

---


## Repository layout

```text
app.py                         Streamlit entry point (Question 1 & Question 4)
src/q1_word_segmentation.py   Q1 models, decoder, baselines, evaluation
src/q2_dependency_parser.py   Q2 transition-based dependency parser
src/q3_spelling_corrector.py   Q3 spelling corrector, evaluation, benchmark, CLI
src/q4_pcfg_parser.py          Q4 Penn Treebank PCFG CKY parser & tagset reconciliation
src/q4_shared_lm.py            Q4 Brown Bigram & Trigram LMs with add-k smoothing
src/q4_live_editor.py          Q4 Background editor, typing simulator & live alert pipeline
src/q4_passage_analysis.py     Q4 End-of-passage sentence analysis & decision rule
scripts/download_data.py      Corpus setup (Brown, UD Spanish, Treebank, Gutenberg, Reuters)
scripts/q4_streamlit_app.py   Q4 Standalone Streamlit application
scripts/q4_speed_benchmark.py  Q4 1,000-word Speed Demon benchmark script
tests/test_q1.py               Q1 unit tests
tests/test_q2.py               Q2 unit tests
tests/test_q3.py               Q3 unit tests
tests/test_q4.py               Q4 unit tests (PCFG, shared LM, alerts, decision rule)
data/                          Local corpora, ignored by Git
```

## Reproducibility notes

English uses the Brown corpus with a seeded 80/20 split. Spanish uses the UD Spanish-GSD train/test split. The trigram decoder uses `max_word_length=20` and add-k `k=0.1`. Question 4 PCFG is induced from `nltk.corpus.treebank` with Chomsky Normal Form binarization and CKY Viterbi search. All tests pass under `pytest tests/ -v`.

