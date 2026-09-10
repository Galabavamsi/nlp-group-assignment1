"""Calibrate the Question 4 Part 4 decision thresholds against the trained models.

The thresholds used by :meth:`src.q4_passage_analysis.PassageAnalyzer.apply_decision_rule`
are measured rather than assumed. This script scores:

* 120 real Brown sentences (grammatical), and
* 120 deterministically corrupted variants of those same sentences (ungrammatical)

with the trained PCFG parser and the tuned Bigram/Trigram LMs, then prints the
quantiles that justify the constants embedded in the decision rule.

Run from the repository root:

```powershell
python scripts/calibrate_q4_thresholds.py
```
"""

from __future__ import annotations

import argparse
import random
import statistics
from pathlib import Path

from src.q1_word_segmentation import load_brown_sentences
from src.q4_passage_analysis import PassageAnalyzer
from src.q4_pcfg_parser import PCFGParser
from src.q4_shared_lm import SharedLanguageModel


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _corrupt(words: list[str], rng: random.Random) -> list[str]:
    """Deterministic corruption: reverse a short span, then duplicate a token."""
    corrupted = list(words)
    if len(corrupted) >= 6:
        start = rng.randint(1, len(corrupted) - 3)
        end = min(len(corrupted) - 1, start + 3)
        corrupted[start:end] = reversed(corrupted[start:end])
    corrupted.insert(rng.randint(1, len(corrupted) - 1), rng.choice(corrupted))
    return corrupted


def _reverse_span(words: list[str], rng: random.Random) -> list[str]:
    out = list(words)
    if len(out) >= 8:
        start = rng.randint(1, len(out) - 5)
        out[start : start + 4] = reversed(out[start : start + 4])
    return out


def _swap_adjacent(words: list[str], rng: random.Random) -> list[str]:
    out = list(words)
    if len(out) >= 8:
        i = rng.randint(1, len(out) - 3)
        out[i], out[i + 1] = out[i + 1], out[i]
    return out


def _collect(analyzer: PassageAnalyzer, rows: list[list[str]]) -> dict[str, list[float]]:
    pcfg_norm: list[float] = []
    bigram: list[float] = []
    trigram: list[float] = []
    parseable = 0
    scored = 0
    for words in rows:
        if len(words) > 20:
            continue
        report = analyzer.analyze_passage(words, ["NN"] * len(words))
        if not report.sentence_analyses:
            continue
        analysis = report.sentence_analyses[0]
        scored += 1
        if analysis.is_pcfg_parseable:
            parseable += 1
            pcfg_norm.append(analysis.pcfg_logprob / max(1, len(analysis.tokens)))
        if analysis.trigram_perplexity != float("inf"):
            bigram.append(analysis.bigram_perplexity)
            trigram.append(analysis.trigram_perplexity)
    return {
        "pcfg_norm": pcfg_norm,
        "bigram": bigram,
        "trigram": trigram,
        "scored": [float(scored)],
        "parseable": [float(parseable)],
    }


def _describe(name: str, stats: dict[str, list[float]]) -> None:
    scored = int(stats["scored"][0]) if stats["scored"] else 0
    parseable = int(stats["parseable"][0]) if stats["parseable"] else 0
    print(f"\n=== {name} ===")
    print(f"scored={scored}  parseable={parseable}/{scored}")
    for key, label in (("pcfg_norm", "PCFG norm logprob/token"), ("bigram", "Bigram PPL"), ("trigram", "Trigram PPL")):
        values = stats[key]
        if not values:
            print(f"{label:24}: (none)")
            continue
        print(
            f"{label:24}: min={min(values):9.2f} p05={_percentile(values, 0.05):9.2f} "
            f"p10={_percentile(values, 0.10):9.2f} p25={_percentile(values, 0.25):9.2f} "
            f"median={statistics.median(values):9.2f} p75={_percentile(values, 0.75):9.2f} "
            f"p90={_percentile(values, 0.90):9.2f} p95={_percentile(values, 0.95):9.2f} max={max(values):9.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate Question 4 decision thresholds")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--sentences", type=int, default=120)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print("Training PCFG parser and shared Bigram/Trigram LMs...", flush=True)
    pcfg_parser = PCFGParser.train_from_treebank(data_dir=data_dir)
    shared_lm = SharedLanguageModel.train_and_tune(data_dir=data_dir)
    analyzer = PassageAnalyzer(pcfg_parser=pcfg_parser, shared_lm=shared_lm)

    rng = random.Random(args.seed)
    sentences = [s.words for s in load_brown_sentences(data_dir) if 4 <= len(s.words) <= 14]
    rng.shuffle(sentences)
    grammatical = sentences[: args.sentences]
    ungrammatical = [_corrupt(words, rng) for words in grammatical]

    print(f"Best k selected: bigram={shared_lm.best_k_bigram}, trigram={shared_lm.best_k_trigram}")

    gram_stats = _collect(analyzer, grammatical)
    ungram_stats = _collect(analyzer, ungrammatical)
    _describe("GRAMMATICAL (real Brown sentences)", gram_stats)
    _describe("UNGRAMMATICAL (corrupted variants)", ungram_stats)

    print("\n=== Threshold implications ===")
    print(f"PCFG per-token floor    : {min(gram_stats['pcfg_norm']):.2f} (grammatical minimum) -> code uses -13.5")
    print(
        f"Trigram Grammatical if  : < {_percentile(gram_stats['trigram'], 0.90):.0f} "
        f"(gram p90); Ungrammatical from {_percentile(ungram_stats['trigram'], 0.10):.0f} (corrupt p10)"
        " -> code uses 450"
    )
    print(
        f"Trigram fallback cutoff : {_percentile(gram_stats['trigram'], 0.95):.0f} (gram p95) -> code uses 800"
    )
    print(
        f"Bigram Grammatical if   : < {_percentile(gram_stats['bigram'], 0.90):.0f} "
        f"(gram p90); Ungrammatical from {_percentile(ungram_stats['bigram'], 0.10):.0f} (corrupt p10)"
        " -> code uses 650"
    )

    # Why the surface checks exist: perplexity is blind to errors that leave the
    # surrounding trigram contexts intact.
    print("\n=== Detection rate by corruption type (LM rule only) ===")
    base = [list(s.words) for s in grammatical]
    for name, fn in (
        ("duplicate_token", lambda w, r: w[: len(w) // 2] + [r.choice(w)] + w[len(w) // 2 :]),
        ("drop_middle_token", lambda w, r: w[: len(w) // 2] + w[len(w) // 2 + 1 :]),
        ("reverse_span4", lambda w, r: _reverse_span(w, r)),
        ("swap_adjacent", lambda w, r: _swap_adjacent(w, r)),
    ):
        detected = 0
        total = 0
        for words in base:
            variant = fn(list(words), rng)
            if len(variant) > 20:
                continue
            scores = shared_lm.score_sentence(variant)
            tri, bi = scores["trigram_perplexity"], scores["bigram_perplexity"]
            total += 1
            if not (tri < 350.0 or (tri >= 800.0 and bi < 650.0)):
                detected += 1
        print(f"  {name:18}: {detected}/{total} = {100 * detected / max(1, total):.1f}% flagged")
    print("  -> repeated/dropped tokens barely change trigram context, which is why")
    print("     src.q4_passage_analysis.surface_anomalies() adds lexical checks.")


if __name__ == "__main__":
    main()
