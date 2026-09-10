"""Generate report-ready Question 1 metrics, figures, and Markdown.

The default report evaluates all Spanish test sentences and a deterministic
200-sentence slice of the Brown English held-out split. The English slice is
intentional: Brown's original fine-grained tagset makes full decoding costly;
the command accepts a larger limit when more compute is available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# When launched as ``python scripts/generate_q1_report.py``, Python puts the
# scripts directory on sys.path rather than the repository root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.q1_word_segmentation import Q1System, build_system


REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"


def evaluate_language(language: str, limit: int | None) -> tuple[Q1System, dict]:
    system = build_system(language, ROOT / "data")
    total_test = len(system.test_sentences)
    if limit is not None:
        system.test_sentences = system.test_sentences[:limit]
    metrics = system.evaluate()
    metrics.update(
        {
            "language": language,
            "training_sentences": len(system.train_sentences),
            "development_sentences": len(system.dev_sentences),
            "available_test_sentences": total_test,
            "evaluated_test_sentences": len(system.test_sentences),
            "evaluation_scope": "full test split" if limit is None else f"first {limit} sentences of the seeded held-out split",
            "vocabulary_size": len(system.segmenter.vocabulary),
            "max_word_length": system.segmenter.max_word_length,
            "segmentation_beam_size": system.segmenter.beam_size,
            "tagging_beam_size": system.tagger.beam_size,
            "smoothing_k": system.lm.k,
        }
    )
    return system, metrics


def draw_bar_plot(results: dict[str, dict], keys: list[tuple[str, str]], title: str, ylabel: str, path: Path) -> None:
    languages = list(results)
    x = np.arange(len(languages))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for offset, (metric_key, label) in enumerate(keys):
        values = [results[language][metric_key] for language in languages]
        bars = ax.bar(x + (offset - (len(keys) - 1) / 2) * width, values, width, label=label)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xticks(x, [language.title() for language in languages])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def draw_error_plot(results: dict[str, dict], path: Path) -> None:
    languages = list(results)
    genuine = [results[language]["genuine_tag_errors"] for language in languages]
    segmentation = [results[language]["segmentation_caused_tag_errors"] for language in languages]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(languages, genuine, label="Genuine POS errors", color="#4575b4")
    ax.bar(languages, segmentation, bottom=genuine, label="Segmentation-caused errors", color="#d73027")
    ax.set_title("Question 1 tagging error-source breakdown")
    ax.set_ylabel("Error count")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def draw_confusion(metrics: dict, path: Path) -> None:
    matrix = metrics["confusion_matrix"]
    labels = metrics["confusion_labels"]
    # Keep the figure readable by selecting the tags with the most support.
    support = {label: sum(matrix.get(label, {}).values()) for label in labels}
    labels = sorted(labels, key=lambda label: support[label], reverse=True)[:15]
    values = np.array([[matrix.get(actual, {}).get(predicted, 0) for predicted in labels] for actual in labels])
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(values, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Count")
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted tag")
    ax.set_ylabel("Actual tag")
    ax.set_title(f"{metrics['language'].title()} POS confusion matrix (top-supported tags)")
    for row in range(len(labels)):
        for column in range(len(labels)):
            if values[row, column]:
                ax.text(column, row, str(values[row, column]), ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def sample_markdown(system: Q1System, language: str) -> str:
    samples = {
        "english": ["thequickbrownfoxjumpsoverthelazydog"],
        "spanish": ["mispadrespuedenviajar", "elcielodespejadoesazul", "lacasarojaesgrande"],
    }[language]
    lines = []
    for text in samples:
        lines.append(f"- `{text}` -> `{system.decode(text)}`")
    return "\n".join(lines)


def build_report(results: dict[str, dict], systems: dict[str, Q1System], english_limit: int) -> str:
    lines = [
        "# Question 1 Results: Word Segmentation and POS Tagging",
        "",
        "> This document is the report-ready handoff for Question 1. It records the exact settings, evaluation scope, metrics, sample outputs, figures, and limitations used by the current implementation.",
        "",
        "## Executive summary",
        "",
        "Question 1 trains a trigram word language model for dynamic-programming segmentation, then tags the resulting words with a trigram HMM-style POS decoder using emission and transition probabilities. English uses the NLTK Brown corpus; Spanish uses UD Spanish-GSD. Spanish tags add gender and number where UD provides those features.",
        "",
        "The Spanish metrics below use the complete UD Spanish-GSD test split. English metrics use a deterministic first-200 slice of the seeded Brown 80/20 held-out split because Brown's original fine-grained Penn tagset makes exhaustive decoding much slower. Run `python scripts/generate_q1_report.py --english-limit 11165` for a full English report when more runtime is available.",
        "",
        "## Method and reproducibility context",
        "",
        "| Setting | Value |",
        "|---|---|",
        "| Segmentation model | Add-k trigram LM + dynamic programming with a practical beam |",
        "| POS model | Emission probabilities + trigram tag transitions + Viterbi beam |",
        "| Smoothing | add-k, k = 0.1 |",
        "| Maximum word length | 20 characters |",
        "| Segmentation beam | 32 histories per reachable character position |",
        "| Tagging beam | 16 tag histories |",
        "| English split | Seeded 80/20 split of Brown tagged sentences, seed 7 |",
        "| Spanish split | UD Spanish-GSD train/dev/test files |",
        "| English tags | Brown's original Penn-style tags |",
        "| Spanish tags | UD UPOS plus `Gender`/`Number` suffixes, e.g. `NOUN-Masc-Pl` |",
        "",
        "## Dataset and evaluation counts",
        "",
        "| Language | Train sentences | Dev sentences | Available test sentences | Evaluated sentences | Vocabulary |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for language, metrics in results.items():
        lines.append(f"| {language.title()} | {metrics['training_sentences']:,} | {metrics['development_sentences']:,} | {metrics['available_test_sentences']:,} | {metrics['evaluated_test_sentences']:,} | {metrics['vocabulary_size']:,} |")

    lines += [
        "",
        "## Main metrics",
        "",
        "Segmentation exact-match is the proportion of sentences whose complete predicted word sequence matches the gold sequence. POS accuracy is conditional on words whose predicted segmentation aligns with the gold word at that position; this makes the later error-source split explicit.",
        "",
        "| Language | Model segmentation exact | Greedy segmentation exact | Model POS accuracy | Most-frequent-tag baseline | Joint exact |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for language, metrics in results.items():
        lines.append(
            f"| {language.title()} | {metrics['model_segmentation_exact']}/{metrics['evaluated_test_sentences']} ({fmt_pct(metrics['model_segmentation_exact'] / max(1, metrics['evaluated_test_sentences']))}) | {metrics['greedy_segmentation_exact']}/{metrics['evaluated_test_sentences']} ({fmt_pct(metrics['greedy_segmentation_exact'] / max(1, metrics['evaluated_test_sentences']))}) | {fmt_pct(metrics['model_pos_accuracy'])} | {fmt_pct(metrics['baseline_pos_accuracy'])} | {metrics['joint_exact']}/{metrics['evaluated_test_sentences']} ({fmt_pct(metrics['joint_exact'] / max(1, metrics['evaluated_test_sentences']))}) |"
        )

    lines += [
        "",
        "### Error-source breakdown",
        "",
        "A tag error is counted as segmentation-caused when the predicted word at that gold position does not align with the gold word. Otherwise, a wrong tag is counted as a genuine POS error.",
        "",
        "| Language | Gold tag positions | Comparable positions | Segmentation-caused errors | Genuine POS errors | Segmentation-caused rate | Genuine-error rate among comparable positions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for language, metrics in results.items():
        lines.append(
            f"| {language.title()} | {metrics['gold_tag_count']:,} | {metrics['comparable_tag_count']:,} | {metrics['segmentation_caused_tag_errors']:,} | {metrics['genuine_tag_errors']:,} | {fmt_pct(metrics['segmentation_caused_tag_error_rate'])} | {fmt_pct(metrics['genuine_tag_error_rate'])} |"
        )

    lines += [
        "",
        "## Sample outputs",
        "",
        "### English",
        "",
        sample_markdown(systems["english"], "english"),
        "",
        "### Spanish",
        "",
        sample_markdown(systems["spanish"], "spanish"),
        "",
        "The Spanish sample containing `despejado` demonstrates an important limitation: that word is not in the training vocabulary, so a vocabulary-only segmenter can split it incorrectly. This should be discussed as an out-of-vocabulary failure rather than hidden from the report.",
        "",
        "## Figures",
        "",
        "![Segmentation exact match versus greedy baseline](figures/q1_segmentation_vs_baseline.png)",
        "",
        "![POS accuracy versus most-frequent-tag baseline](figures/q1_pos_vs_baseline.png)",
        "",
        "![Tagging error-source breakdown](figures/q1_error_source_breakdown.png)",
        "",
        "![English confusion matrix](figures/q1_confusion_english.png)",
        "",
        "![Spanish confusion matrix](figures/q1_confusion_spanish.png)",
        "",
        "## Interpretation for the comparative report",
        "",
        "- The trigram decoder should be compared against greedy longest-match segmentation, because greedy matching ignores context and can choose locally long but globally implausible words.",
        "- The POS model should be compared against the most-frequent-tag baseline. The conditional POS accuracy and the error-source table must be read together: segmentation errors reduce the number of positions where a tag comparison is meaningful.",
        "- Spanish's enriched tags expose agreement information that plain coarse POS tags hide. For example, the output can distinguish `NOUN-Masc-Pl` from `NOUN-Fem-Sg` and can expose whether adjective/noun agreement is being learned.",
        "- English and Spanish are not directly comparable by raw accuracy alone because they use different tag inventories and different evaluation scopes in this report. Report the data split and tag definition beside every number.",
        "",
        "## Teammate handoff",
        "",
        "- Reuse `build_system(\"english\")` and the returned `lm`, `segmenter`, and `tagger` objects in Question 4; do not retrain them inside the live editor.",
        "- Reuse `POSTagger.morphology_tag` and preserve the Spanish agreement suffix convention in the final report.",
        "- Use the JSON file for tables or further plotting: `reports/q1_results.json`.",
        "- Q2 can follow the same data-download pattern with UD English-EWT; Q3 can reuse Brown loading and the shared vocabulary conventions.",
        "",
        "## Reproduce",
        "",
        "```powershell",
        f"python scripts/generate_q1_report.py --english-limit {english_limit}",
        "python -m pytest -q",
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--english-limit", type=int, default=200)
    parser.add_argument("--spanish-limit", type=int, default=None)
    args = parser.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    systems: dict[str, Q1System] = {}
    results: dict[str, dict] = {}
    for language, limit in (("english", args.english_limit), ("spanish", args.spanish_limit)):
        print(f"Evaluating {language}...")
        system, metrics = evaluate_language(language, limit)
        systems[language] = system
        results[language] = metrics
        print(f"  sentences={metrics['evaluated_test_sentences']}, segmentation={metrics['model_segmentation_exact']}, POS={metrics['model_pos_accuracy']:.4f}")

    draw_bar_plot(
        results,
        [("model_segmentation_exact", "Trigram + DP"), ("greedy_segmentation_exact", "Greedy longest-match")],
        "Segmentation exact match versus baseline",
        "Exact-match sentence count",
        FIGURE_DIR / "q1_segmentation_vs_baseline.png",
    )
    draw_bar_plot(
        results,
        [("model_pos_accuracy", "Trigram POS model"), ("baseline_pos_accuracy", "Most-frequent-tag")],
        "POS accuracy versus baseline",
        "Accuracy",
        FIGURE_DIR / "q1_pos_vs_baseline.png",
    )
    draw_error_plot(results, FIGURE_DIR / "q1_error_source_breakdown.png")
    for language, metrics in results.items():
        draw_confusion(metrics, FIGURE_DIR / f"q1_confusion_{language}.png")

    (REPORT_DIR / "q1_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (REPORT_DIR / "Q1_RESULTS.md").write_text(build_report(results, systems, args.english_limit), encoding="utf-8")
    print(f"Wrote report to {REPORT_DIR / 'Q1_RESULTS.md'}")


if __name__ == "__main__":
    main()
