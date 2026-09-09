"""Question 4: Final Passage Analysis & Method Comparison.

This module splits the final (corrected) token stream into sentences, scores each
sentence using the PCFG parser, Bigram LM, and Trigram LM, applies a documented
decision rule to choose the optimal analysis method and assign a final
grammaticality verdict, and formats the summary comparison table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import nltk
from src.q1_word_segmentation import is_word
from src.q3_spelling_corrector import detokenize
from src.q4_pcfg_parser import PCFGParser, reconcile_tag
from src.q4_shared_lm import SharedLanguageModel


@dataclass
class SentenceAnalysis:
    sentence_index: int
    text: str
    tokens: list[str]
    pos_tags: list[str]
    pcfg_result: str  # Log-prob string or "unparseable"
    pcfg_logprob: float
    is_pcfg_parseable: bool
    bigram_logprob: float
    bigram_perplexity: float
    trigram_logprob: float
    trigram_perplexity: float
    chosen_method: str  # "PCFG Parser", "Trigram LM", or "Bigram LM"
    verdict: str  # "Grammatical" or "Ungrammatical"
    seg_merges_resolved: int
    spell_corrections_applied: int


@dataclass
class PassageAnalysisReport:
    sentence_analyses: list[SentenceAnalysis] = field(default_factory=list)

    def to_dict_list(self) -> list[dict[str, object]]:
        """Convert sentence analyses to table dictionary representation."""
        table = []
        for sa in self.sentence_analyses:
            table.append(
                {
                    "Sentence Text": sa.text,
                    "PCFG Result": sa.pcfg_result,
                    "Bigram Score": f"{sa.bigram_logprob:.2f}",
                    "Trigram Score": f"{sa.trigram_logprob:.2f}",
                    "Chosen Method": sa.chosen_method,
                    "Final Verdict": sa.verdict,
                    "Seg Merges Resolved": sa.seg_merges_resolved,
                    "Spell Corrections Applied": sa.spell_corrections_applied,
                }
            )
        return table


class PassageAnalyzer:
    """Analyzer executing Part 4 end-of-passage evaluation."""

    def __init__(self, pcfg_parser: PCFGParser, shared_lm: SharedLanguageModel) -> None:
        self.pcfg_parser = pcfg_parser
        self.shared_lm = shared_lm

    def split_into_sentences(self, tokens: Sequence[str], pos_tags: Sequence[str]) -> list[tuple[list[str], list[str]]]:
        """Split token stream and aligned POS tags into individual sentences."""
        if not tokens:
            return []

        text = detokenize(tokens)
        raw_sents = nltk.sent_tokenize(text)

        output: list[tuple[list[str], list[str]]] = []
        curr_idx = 0
        for sent_text in raw_sents:
            sent_words = sent_text.strip().split()
            sent_tokens: list[str] = []
            sent_tags: list[str] = []
            for _ in sent_words:
                if curr_idx < len(tokens):
                    sent_tokens.append(tokens[curr_idx])
                    sent_tags.append(pos_tags[curr_idx] if curr_idx < len(pos_tags) else "NN")
                    curr_idx += 1
            if sent_tokens:
                output.append((sent_tokens, sent_tags))

        # Handle remaining unassigned tokens if any
        if curr_idx < len(tokens):
            rem_tokens = list(tokens[curr_idx:])
            rem_tags = list(pos_tags[curr_idx:]) if curr_idx < len(pos_tags) else ["NN"] * len(rem_tokens)
            output.append((rem_tokens, rem_tags))

        return output

    def apply_decision_rule(
        self,
        is_pcfg_parseable: bool,
        pcfg_logprob: float,
        num_tokens: int,
        trigram_ppl: float,
        bigram_ppl: float,
    ) -> tuple[str, str]:
        """Apply documented decision rule to pick chosen method and grammaticality verdict.

        Decision Logic:
        1. PCFG Parser: Selected if sentence parses AND normalized log-prob per word >= -6.5.
           Verdict -> Grammatical.
        2. Trigram LM: Selected if PCFG fails or is outlier AND Trigram PPL < 300.0.
           Verdict -> Grammatical (if PPL < 200.0) else Ungrammatical.
        3. Bigram LM: Selected if Trigram PPL >= 300.0 AND Bigram PPL < 450.0.
           Verdict -> Grammatical (if PPL < 350.0) else Ungrammatical.
        """
        norm_pcfg_score = pcfg_logprob / max(1, num_tokens)

        if is_pcfg_parseable and norm_pcfg_score >= -6.5:
            return "PCFG Parser", "Grammatical"
        elif trigram_ppl < 300.0:
            verdict = "Grammatical" if trigram_ppl < 200.0 else "Ungrammatical"
            return "Trigram LM", verdict
        elif bigram_ppl < 450.0:
            verdict = "Grammatical" if bigram_ppl < 350.0 else "Ungrammatical"
            return "Bigram LM", verdict
        else:
            return "Bigram LM", "Ungrammatical"

    def analyze_passage(
        self,
        tokens: Sequence[str],
        pos_tags: Sequence[str],
        total_seg_merges: int = 0,
        total_spell_corrections: int = 0,
    ) -> PassageAnalysisReport:
        """Analyze passage sentences and compile structural summary table."""
        sentences = self.split_into_sentences(tokens, pos_tags)
        report = PassageAnalysisReport()

        total_words = max(1, len(tokens))

        for idx, (sent_tokens, sent_tags) in enumerate(sentences, start=1):
            sent_text = detokenize(sent_tokens)

            # PCFG parse
            pcfg_res = self.pcfg_parser.parse(sent_tokens, sent_tags)
            pcfg_str = f"{pcfg_res.log_prob:.2f}" if pcfg_res.is_parseable else "unparseable"

            # Shared LM scores
            lm_scores = self.shared_lm.score_sentence(sent_tokens)

            # Proportional attribution of alerts per sentence
            sent_word_count = len(sent_tokens)
            prop = sent_word_count / total_words
            seg_count = max(0, round(total_seg_merges * prop))
            spell_count = max(0, round(total_spell_corrections * prop))

            chosen_method, verdict = self.apply_decision_rule(
                is_pcfg_parseable=pcfg_res.is_parseable,
                pcfg_logprob=pcfg_res.log_prob,
                num_tokens=len(sent_tokens),
                trigram_ppl=lm_scores["trigram_perplexity"],
                bigram_ppl=lm_scores["bigram_perplexity"],
            )

            report.sentence_analyses.append(
                SentenceAnalysis(
                    sentence_index=idx,
                    text=sent_text,
                    tokens=sent_tokens,
                    pos_tags=sent_tags,
                    pcfg_result=pcfg_str,
                    pcfg_logprob=pcfg_res.log_prob,
                    is_pcfg_parseable=pcfg_res.is_parseable,
                    bigram_logprob=lm_scores["bigram_logprob"],
                    bigram_perplexity=lm_scores["bigram_perplexity"],
                    trigram_logprob=lm_scores["trigram_logprob"],
                    trigram_perplexity=lm_scores["trigram_perplexity"],
                    chosen_method=chosen_method,
                    verdict=verdict,
                    seg_merges_resolved=seg_count,
                    spell_corrections_applied=spell_count,
                )
            )

        return report
