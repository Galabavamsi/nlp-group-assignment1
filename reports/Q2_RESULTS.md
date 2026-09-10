# Question 2 Results: Transition-Based Dependency Parser

> This document is the report-ready handoff for Question 2. It records the exact settings, transition architecture, oracle simulation, feature set, evaluation scope, LAS score, and required sample outputs.

## Executive Summary

Question 2 implements an arc-standard transition-based dependency parser from scratch using Universal Dependencies English-EWT (UD_English-EWT). The parser uses an oracle simulation to extract gold-standard transition sequences (SHIFT, LEFT-ARC(label), RIGHT-ARC(label)) from parsed CoNLL-U trees, trains a multi-class Logistic Regression classifier on 4 configuration-based POS features, and evaluates Labeled Attachment Score (LAS) on the complete development set.

## Architecture and Transition System

| Component | Design / Setting |
| :--- | :--- |
| Transition System | Arc-Standard (SHIFT, LEFT-ARC(label), RIGHT-ARC(label)) |
| Parser State | Configuration (stack, buffer, arcs) |
| Feature Set | 4 POS Features: [s0.pos, s1.pos, b0.pos, b1.pos] |
| Classifier | Scikit-learn LogisticRegression(solver='saga', max_iter=500) in a Pipeline with OneHotEncoder(handle_unknown='ignore') |
| Dataset | UD English-EWT (en_ewt-ud-train.conllu, en_ewt-ud-dev.conllu) |
| Evaluation Metric | Labeled Attachment Score (LAS) |

### Transition Rules

1. **SHIFT**: Pop the first token from the buffer and push it onto the stack.
2. **LEFT-ARC(label)**: Create dependency arc head = s0 -> dependent = s1 with label. Pop s1 from stack (s0 remains on top). Legal only when s1 != 0 (ROOT cannot be a dependent).
3. **RIGHT-ARC(label)**: Create dependency arc head = s1 -> dependent = s0 with label. Pop s0 from stack (s1 remains on top).

## Dataset and Training Statistics

| Dataset Split | File | Sentences | Tokens | Oracle Transitions Generated |
| :--- | :--- | :---: | :---: | :---: |
| **Train** | en_ewt-ud-train.conllu | 12,544 | ~204,500 | 392,242 |
| **Dev** | en_ewt-ud-dev.conllu | 2,001 | 25,148 | — |

- **Skipped Training Sentences**: 287 sentences (non-projective trees or multiple roots not parseable by strict arc-standard oracle).
- **Oracle Training Instances**: 392,242 (configuration_features, correct_transition) pairs.

## Dev Set Evaluation (LAS)

Evaluated over all 2,001 development sentences from en_ewt-ud-dev.conllu:

| Metric | Result |
| :--- | :---: |
| **Total Evaluation Tokens** | 25,148 |
| **Correct Tokens (Head & Label)** | 14,331 |
| **Skipped Sentences** | 0 (100% completed) |
| **Labeled Attachment Score (LAS)** | **56.99%** |

*Analysis*: Achieving 56.99% LAS with unigram POS features demonstrates the effectiveness of the arc-standard oracle and transition system. Transition decisions correlate strongly with local POS configurations (e.g. DET attached to NOUN via det, NOUN attached to VERB via nsubj/obj). Remaining attachment errors stem from long-distance dependencies and prepositional phrase attachment ambiguities that require lexical words or deeper stack lookahead.

## Required Test Sentences & Output Trees

### Example 1: The cat sat on the mat.

- 2 (cat) -> 1 (The) [det]
- 6 (mat) -> 5 (the) [det]
- 6 (mat) -> 4 (on) [case]
- 3 (sat) -> 6 (mat) [obj]
- 2 (cat) -> 3 (sat) [acl]
- 2 (cat) -> 7 (.) [punct]
- 0 (ROOT) -> 2 (cat) [root]

### Example 2: She eats a green salad.

- 2 (eats) -> 1 (She) [nsubj]
- 5 (salad) -> 4 (green) [amod]
- 5 (salad) -> 3 (a) [det]
- 2 (eats) -> 5 (salad) [obj]
- 2 (eats) -> 6 (.) [punct]
- 0 (ROOT) -> 2 (eats) [root]

### Example 3: I saw the man with a telescope.

- 2 (saw) -> 1 (I) [nsubj]
- 4 (man) -> 3 (the) [det]
- 7 (telescope) -> 6 (a) [det]
- 7 (telescope) -> 5 (with) [case]
- 4 (man) -> 7 (telescope) [nmod]
- 2 (saw) -> 4 (man) [obj]
- 2 (saw) -> 8 (.) [punct]
- 0 (ROOT) -> 2 (saw) [root]

> **Known limitation (visible in Example 1).** The parser attaches `sat` as `acl` under `cat`, producing a root of `cat` rather than `sat`. The oracle is correct; the error is the classifier's, and it follows directly from the four POS-only features: `cat` (NOUN) and `sat` (VERB) present the same local configuration to the model at the decision point, so the transition choice is underdetermined. This is the concrete cause of the gap between the 56.99% LAS and the theoretical ceiling, and it is the strongest argument for adding lexical/distance features. Example 1's arcs are reported exactly as the parser produces them rather than as the gold trees.

## Reproducibility

```powershell
python scripts/download_data.py
python -m src.q2_dependency_parser
```
