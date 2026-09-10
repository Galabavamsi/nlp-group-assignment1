# CONTEXT.md — Handover Notes for the Next Maintainer

**Project:** NLP Group Assignment 1 (Streamlit app + 4 question modules)
**Repo root:** `D:\nlp\grp-assignment1`
**Remote:** https://github.com/Galabavamsi/nlp-group-assignment1
**Entry point:** `app.py` (Streamlit)
**Status:** working — 20/20 tests pass, app boots and renders with 0 exceptions, all four modules run from the CLI.

This document is a handover. Read it before changing anything: several numbers in the
old report were wrong, and the reasons are recorded below so they are not
accidentally "fixed" back.

---

## 1. TL;DR — what to do first

```powershell
cd D:\nlp\grp-assignment1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/download_data.py      # brown/treebank/gutenberg/reuters + UD corpora
python -m pytest tests/ -q           # expect: 20 passed
streamlit run app.py                 # then open http://localhost:8501
```

If corpora are already present under `data/` (they are on this machine), you can skip
`download_data.py` and go straight to the tests and `streamlit run app.py`.

---

## 2. What was broken and what was fixed

Everything below was found by actually running the app and the modules, not by reading
the code. Each item records the *evidence*, because the previous report contained
plausible-looking numbers that turned out to be invented.

### 2.1 Critical: Q4 Part 4 decision rule labelled correct sentences "Ungrammatical"

- **Symptom:** every clean English sentence was returned as `Ungrammatical`.
- **Cause:** the thresholds in `src/q4_passage_analysis.py` were fitted to a fabricated
  perplexity table. It required `log P / L >= -6.5`, but real grammatical Brown
  sentences score about **-12** per token, so the PCFG branch could never fire. The LM
  cut-offs (200/300/350/450) were also far below real values.
- **Fix:** thresholds re-measured empirically with the new
  `scripts/calibrate_q4_thresholds.py` (120 real Brown sentences vs 120 deterministically
  corrupted variants), then the rule rewritten as an explicit decision table.
- **Measured result:** 100% of grammatical sentences accepted, 93.3% of corrupted
  sentences rejected (balanced accuracy 96.7%). Was effectively 0% before.
- **Important caveat:** the PCFG's parseability does **not** discriminate — it returned
  a parse for 99.2% of *both* grammatical and corrupted sentences. Only the n-gram
  perplexities separate the classes. Do not "simplify" the rule back to a PCFG-only gate.

### 2.2 Critical: crash on any sentence longer than 25 tokens

- `src/q4_passage_analysis.py` constructed `ParseResult(is_parseable=..., log_prob=...,
  tree_str=...)`, but `ParseResult` has six required fields. Any sentence over 25 tokens
  raised `TypeError` and killed the whole Part 4 table. Now passes all fields.

### 2.3 Critical: sentence splitting merged the whole passage into one "sentence"

- **Cause:** `nltk.sent_tokenize(text)` was called bare. In NLTK 3.9 that does not load
  the trained Punkt language model, so it returns the entire passage as a single
  sentence. On top of that, the code re-aligned `sent_text.split()` word counts against
  the token list, which cannot work because `detokenize()` glues punctuation onto the
  preceding token. A 4-sentence passage produced **1** analysis instead of 4.
- **Fix:** split directly on the token stream at `.`/`!`/`?`. Tokens and POS tags stay
  exactly aligned and there is no re-alignment step left to get wrong.

### 2.4 Critical: POS tags were predicted one token at a time

- **Cause:** `q1_system.tagger.tag([w])` was called per token. Q1's model is a *trigram*
  HMM, so a single-word call always sees empty context and collapses to nearly the same
  tag for every word.
- **Consequences:** the POS tags handed to the PCFG (which uses them as lexical
  terminals) were garbage, and the grammar alert's LM score was computed over
  mis-tagged input.
- **Fix:** `IntegratedEditor._retag_sequence()` re-tags a bounded trailing window
  (`2 * trigger_interval` tokens) so tags have real left context.
- **Do not** re-tag the entire prefix after every token: that is O(n²) and measured
  **~5 seconds** for a 20-word sentence. The bounded window is deliberate.

### 2.5 Major: tagger was too slow for interactive use (~30× speedup)

- `POSTagger.tag()` iterated over all **317** Brown tags per word →
  **~100–500 ms per short sentence**, unusable live.
- **Fix:** candidate-tag pruning in `src/q1_word_segmentation.py`. For each word the
  search considers only tags observed with that word in training, plus the 30 most
  frequent tags as a fallback for unseen words (`candidate_tags()`).
- **Result:** 21-token tag: 499 ms → **16.5 ms**. Accuracy was *not* sacrificed —
  English POS stayed at 0.9140 and Spanish POS *improved* 90.75% → 90.77% (383 vs 384
  genuine tag errors), because pruning removes wrong-tag attractors as well as cost.

### 2.6 Major: live grammar alerts fired on ~1 window in 7 (alert fatigue)

- Threshold was `trigram_ppl > 450`. Measured over 547 sliding windows of in-domain
  Brown prose: 300 → 24.7%, **450 → 13.9%**, 800 → 1.1%, 1500 → 0%.
- **Fix:** live threshold raised to 800 (trigram) / 1500 (bigram). Note this is
  intentionally *stricter* than the end-of-passage threshold of 350 in
  `q4_passage_analysis.py` — live alerts are tuned for precision while typing, final
  analysis is tuned for decisiveness. They are different jobs; keep them different.

### 2.7 Moderate: stale model cache served outdated behaviour

- `data/cache/q4_editor_models.pkl` pickled trained objects from `src/`. After any code
  change the old pickle kept being loaded, so the app silently ran the old logic.
- **Fix:** `_model_fingerprint()` in `scripts/q4_streamlit_app.py` hashes the relevant
  `src/*.py` files plus corpus presence, stores it beside the models, and invalidates the
  cache on mismatch. The cache file was deleted once, so the next start retrains.

### 2.8 Moderate: `reuters` re-downloaded on every cold start

- `app.py` used `nltk.data.find("corpora/reuters")`, which only reports an *unpacked*
  directory. `reuters` ships here as `reuters.zip` only (valid — `nltk.corpus.reuters`
  loads 10,788 files from it), so the check failed and the app re-downloaded it every
  start.
- **Fix:** the availability check also accepts the `.zip`; failures are collected and
  surfaced as a sidebar warning instead of failing silently offline.

### 2.9 Documentation was materially wrong (README + results)

Corrected in `README.md` and `reports/`:

| Claim in the old README | Reality |
| :--- | :--- |
| Dev perplexity table (248–510 range) | Real: bigram 1189–4670, trigram 7048–17935 |
| Optimal `k`: bigram 0.05, trigram 0.01 | Real: bigram **0.01**, trigram **0.001** |
| Part 4 rule: `log P/L >= -6.5`, PPL < 300 | Real: thresholds 350/650/800 (§2.1) |
| Speed Demon: 117.19 ms total, 0.1172 ms/word | Real: **~5.4 s**, ~5.4 ms/word |
| "per-token check runs in sub-millisecond time" | The benchmark's own number contradicts it; conclusion text rewritten |
| "PCFG parseability rises 45% → 88%" | Never measured; PCFG parses 99.2% of both good and bad input |
| Q2 Example 1 arcs are gold | They are the parser's own output, including a wrong `acl` arc — now labelled as such |
| Q1 Spanish POS 90.75% | Now 90.77% after the tagger fix (regenerated) |

**Q1, Q2 and Q3 headline numbers were re-run and reproduce exactly:**
Q1 English 141/200 segmentation & 0.9140 POS; Q2 LAS **56.99%** on 2,001 dev sentences;
Q3 90.50% non-word / 65.00% real-word at `--max-examples 200` (92.69% / 70.55% on the
full split). Use these; they are verified.

---

## 3. Repository map

```text
app.py                            Streamlit entry point; sidebar switches between Q1–Q4
requirements.txt                  nltk, streamlit, streamlit-keyup, scikit-learn, pandas, matplotlib, pytest
.streamlit/config.toml            headless server + dark theme (committed)
src/q1_word_segmentation.py       Trigram LM + Viterbi segmentation; trigram HMM POS; baselines; eval
src/q2_dependency_parser.py       Arc-standard transition parser + sklearn classifier + LAS eval
src/q3_spelling_corrector.py      Edit-1 (Method A) and symmetric-delete (Method B) corrector; CLI
src/q4_pcfg_parser.py             Penn Treebank PCFG, CNF binarisation, Viterbi CKY, tag reconciliation
src/q4_shared_lm.py               Brown bigram/trigram LMs, add-k tuning, scoring
src/q4_live_editor.py             Live editor: merge injection, SEGMENT/SPELL/GRAMMAR alerts, latency
src/q4_passage_analysis.py        Part 4: sentence splitting, decision rule, comparison table
scripts/download_data.py          Fetches NLTK corpora + clones UD English-EWT / Spanish-GSD
scripts/q4_streamlit_app.py       Q4 UI (rendered inside app.py); model cache + fingerprint
scripts/q4_speed_benchmark.py     1,000-word Speed Demon benchmark
scripts/q4_streamlit_app.py       (see above)
scripts/generate_q1_report.py     Regenerates reports/Q1_RESULTS.md + figures + q1_results.json
scripts/calibrate_q4_thresholds.py NEW — re-measures the Part 4 thresholds and prints justification
tests/test_q1.py … test_q4.py     20 tests, all passing
data/                             Corpora + cache. Git-ignored content, but see §5.
reports/                          Q1/Q2/Q3 results docs, q1_results.json, figures
assets/                           Screenshot used in README §8
```

---

## 4. How to run each piece

```powershell
# All tests
python -m pytest tests/ -q

# Q1 — segmentation + POS (English / Spanish)
python -m src.q1_word_segmentation --language english --text thequickbrownfoxjumpsoverthelazydog --limit 20
python -m src.q1_word_segmentation --language spanish --text lacasarojaesgrande --limit 10

# Q2 — dependency parser (prints LAS + the 3 required example trees)
python -m src.q2_dependency_parser

# Q3 — spelling corrector
python -m src.q3_spelling_corrector --sentence "I hav a good feeling about this."
python -m src.q3_spelling_corrector --evaluate --max-examples 200
python -m src.q3_spelling_corrector --evaluate --benchmark
python -m src.q3_spelling_corrector --interactive

# Q4 — benchmark, threshold calibration, web app
python -m scripts.q4_speed_benchmark
python scripts/calibrate_q4_thresholds.py
streamlit run app.py
```

Expected benchmark output: ~5.4 s full pipeline, ~0.13 s isolated grammar check,
119 `[SEGMENT-ALERT]`, 881 `[SPELL-ALERT]`.

---

## 5. Deployment notes (read before pushing)

### 5.1 Streamlit Community Cloud — current state

The app **is deployed and serving**:
https://nlp-group-assignment1-y92dzndu2g3nu6uavyfsbw.streamlit.app

Cloud clones the repo (including corpora via Git LFS, ~236 MiB) and installs
`requirements.txt` with `uv` on **Python 3.14.7**. The first render trains the Q4
models (~1–3 min).

### 5.2 Log lines that are NOT errors

Two warnings appear in the Cloud log; both are benign, and both are now addressed.

```text
OSError: [Errno 28] inotify watch limit reached
Failed to schedule watch observer for path /mount/src/nlp-group-assignment1
```

Streamlit's file watcher tries to recursively watch ~1,300 corpus files and hits the
container's inotify limit. It happens inside the SDK before any user code runs, so no
`@st.cache_resource` wrapper can suppress it. Fixed at the source with
`fileWatcherType = "none"` in `.streamlit/config.toml`; that file is committed, so a
redeploy picks it up. The watcher only provides local auto-reload, so disabling it is
safe.

```text
UserWarning: NLTK will not authorize the non-private download directory
'/mount/src/nlp-group-assignment1/data/nltk'
```

NLTK refuses to *download into* a world/group-writable directory, which is what a
hosted mount like `/mount/src/...` looks like. The repository copy is now treated as
**read-only**, and `ensure_nltk_corpora()` downloads anything genuinely missing into a
**private per-user cache** (`~/.cache/nlp_assignment_nltk`) that NLTK does authorise.
Because the corpora ship with the repo, nothing normally needs downloading.

### 5.3 Other deployment constraints

- `.streamlit/config.toml` **no longer disables CORS/XSRF protection**. Those were
  switched off previously; they are now at Streamlit's secure defaults. Only re-disable
  them for a local reverse-proxy setup — never for a public app.
- `data/**` is served through Git LFS; Cloud pulls LFS objects automatically.
- **`data/cache/*.pkl` is no longer tracked.** It was, at ~195 MB — the bulk of the
  clone, and it inflates memory when unpickled. It is regenerated on demand, and the
  fingerprint in §2.7 makes a stale copy harmless. Cold start therefore trains the
  models; that is the intended trade-off.
- If a deploy ever fails to build, the lean fallback is to run
  `python scripts/download_data.py` once in the environment.

---

## 6. Known limitations (honest, not bugs)

1. **Perplexity is blind to structure-preserving errors.** Duplicating a token or
   deleting a mid-sentence token leaves the surrounding trigram contexts intact, so the
   LM rule flagged those **0%** of the time (reversing a 4-word span: 71.3%).
   `surface_anomalies()` in `q4_passage_analysis.py` adds lexical checks (adjacent
   repeated function words, three-times-repeated tokens) which lifted corrupted-sentence
   detection from 90.8% → 93.3% with no loss on grammatical input. It still only catches
   ~6.7% of arbitrary duplicate-token corruptions, so this is a real remaining gap.
2. **Part 4 is sensitive to corpus fit — including two of the app's own preset examples.**
   In-domain Brown prose is accepted 100% of the time (99/100 single sentences, 24/25
   sentences in 4-sentence passages), but short sentences built from rare vocabulary can
   still exceed the trigram threshold because the LM was trained on Brown and has few
   contexts for them. Concretely, with the current model:

   | Sentence | Trigram PPL | Verdict |
   | :--- | ---: | :--- |
   | "I saw the man with a telescope ." | 304 | Grammatical |
   | "The cat sat on the mat ." | 13523 | Ungrammatical |
   | "She eats a green salad ." | 23595 | Ungrammatical |

   These are grammatical sentences, so a demo that clicks through the Part 4 analysis on
   those presets will show a false negative. This is a lexical-coverage limit of a
   Brown-trained trigram, **not** a logic error — do not "fix" it by lowering the
   threshold, which would re-break corrupted-sentence detection (§2.1). If it matters for
   the demo, either pick presets with more common vocabulary or train the LM on a larger
   corpus.
3. **Q2 LAS is 56.99%** with POS-only features. The parser attaches `sat` as `acl` under
   `cat` in *"The cat sat on the mat."* — the classifier cannot separate the two because
   the four features are identical at that decision point. Adding lexical or
   distance features is the documented next step.
4. **Q1 Spanish segmentation is weak (24.12% exact).** Spanish is morphologically rich
   and the vocabulary-only segmenter splits unknown words badly (the report shows
   `despejado` → `des`+`pe`+`j`+`ado`). This is a genuine OOV limitation and is already
   documented in `reports/Q1_RESULTS.md`; discuss it rather than hide it.
5. **Live grammar alerts are still noisy on out-of-domain text.** With the 800 threshold
   they fire around 1.1% of windows on Brown-like prose, but far more often on casual
   user-style text. A relative/adaptive threshold is the obvious improvement.
6. **`SESSION`/first-run cost.** The Q4 editor's first use in a process pays model
   training; `@st.cache_resource` and the disk cache exist for exactly that reason.

---

## 7. Invariants — please keep these true

- `python -m pytest tests/ -q` → **20 passed**. The Q4 tests pin every branch of the
  decision rule; if you change thresholds, update the tests deliberately.
- Q1 headline metrics: English 141/200 segmentation, 0.9140 POS; Spanish 103/427,
  ~0.9077 POS. Regenerate with
  `python scripts/generate_q1_report.py --english-limit 200`.
- Q2 LAS 56.99%; Q3 90.50% / 65.00% at `--max-examples 200`.
- Any threshold you touch in `q4_passage_analysis.py` or `q4_live_editor.py` should be
  re-justified with `python scripts/calibrate_q4_thresholds.py` and the numbers pasted
  into the docstring/README. That is how the current values were chosen.

---

## 8. Immediate next steps (in priority order)

1. **Push and deploy.** Run the test + app smoke checks in §1, then commit. Nothing is
   currently known to be broken.
2. **Verify the deploy target** (§5): confirm how corpora arrive (LFS pull vs
   `download_data.py`) and confirm the cold-start training completes.
3. **Decide on `data/cache/*.pkl` in git.** It is LFS-tracked and ~200 MB. Consider
   removing it from tracking so every deploy builds it fresh.
4. Optional improvements, highest value first: lexical/distance features for Q2; adaptive
   grammar-alert threshold; Spanish OOV handling in Q1; extend `surface_anomalies()` to
   cover adjacent swaps (currently 23.3% detected).

---

## 9. Files changed in this pass

```
src/q1_word_segmentation.py     candidate-tag pruning + word_tags/fallback_tags fields
src/q4_live_editor.py           bounded sequence re-tagging; live thresholds 800/1500
src/q4_passage_analysis.py      calibrated decision table; ParseResult fix; token-aligned
                                sentence splitting; surface_anomalies(); Surface Issues column
scripts/q4_streamlit_app.py     model-cache fingerprint invalidation
scripts/q4_speed_benchmark.py   conclusion text now matches its own measurements
scripts/calibrate_q4_thresholds.py  NEW calibration + justification script
app.py                          NLTK availability check accepts .zip; missing-data warning
tests/test_q4.py                decision-rule branches + surface_anomalies tests
README.md                       corrected §1, §2, §4, §5, §6, §8
reports/Q1_RESULTS.md           regenerated (Spanish 90.75% → 90.77%)
reports/Q3_RESULTS.md           exact reproducing command documented
reports/Q2_RESULTS.md           broken code fence fixed; wrong `acl` arc documented
```
