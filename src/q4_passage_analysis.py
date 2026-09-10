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
from src.q4_pcfg_parser import PCFGParser, ParseResult, reconcile_tag
from src.q4_shared_lm import SharedLanguageModel

# Function words whose immediate repetition is almost never grammatical. An
# adjacent repeated function word ("the the", "and and", "to to") is the
# highest-precision surface cue for a duplicated-token typing error.
_FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "by",
    "for", "with", "from", "as", "is", "was", "were", "are", "be", "been",
    "that", "this", "these", "those", "it", "he", "she", "they", "we", "i",
    "his", "her", "their", "its", "my", "your", "our", "not", "no", "so",
}


def surface_anomalies(tokens: Sequence[str]) -> list[str]:
    """Cheap, high-precision surface checks that n-gram models cannot see.

    A repeated token does not damage the surrounding trigram contexts, so the LM
    perplexity detectors are blind to it (measured at 0% detection on duplicated
    tokens). These structural checks catch the common typing errors that keep the
    word sequence locally fluent.
    """
    problems: list[str] = []
    lowered = [t.lower() for t in tokens]
    for i in range(len(lowered) - 1):
        if lowered[i] in _FUNCTION_WORDS and lowered[i] == lowered[i + 1]:
            problems.append(f"repeated function word '{tokens[i]} {tokens[i + 1]}'")
    for i in range(len(lowered) - 2):
        if lowered[i] == lowered[i + 1] == lowered[i + 2]:
            problems.append(f"token '{tokens[i]}' repeated three times")
    return problems



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
    surface_issues: list[str] = field(default_factory=list)


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
                    "Surface Issues": "; ".join(sa.surface_issues) if sa.surface_issues else "-",
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

        # Split on the token stream, not on a detokenized string.
        #
        # The previous implementation called ``nltk.sent_tokenize`` and then tried
        # to re-align the returned word counts with the token list. That is both
        # fragile (detokenize collapses punctuation onto the preceding token, so
        # whitespace word counts do not match token counts) and silently wrong:
        # a bare ``nltk.sent_tokenize`` does not load the trained Punkt language
        # model, so it returns the entire passage as a single sentence. A
        # 4-sentence passage was therefore analysed as one sentence.
        #
        # Splitting directly on sentence-final punctuation keeps the tokens and
        # their POS tags exactly aligned, with no re-alignment step to get wrong.
        sentence_end = {".", "!", "?"}
        output: list[tuple[list[str], list[str]]] = []
        curr_tokens: list[str] = []
        curr_tags: list[str] = []
        for index, token in enumerate(tokens):
            curr_tokens.append(token)
            curr_tags.append(pos_tags[index] if index < len(pos_tags) else "NN")
            if token in sentence_end:
                output.append((curr_tokens, curr_tags))
                curr_tokens, curr_tags = [], []

        if curr_tokens:
            output.append((curr_tokens, curr_tags))

        return output

    def apply_decision_rule(
        self,
        is_pcfg_parseable: bool,
        pcfg_logprob: float,
        num_tokens: int,
        trigram_ppl: float,
        bigram_ppl: float,
    ) -> tuple[str, str]:
        """Apply the calibrated decision rule to pick a method and grammaticality verdict.

        Thresholds are *measured*, not assumed. ``scripts/calibrate_q4_thresholds.py``
        scores 120 real Brown sentences against 120 deterministically corrupted
        variants of the same sentences with the trained PCFG and LMs, giving:

        | Signal                          | Grammatical     | Corrupted       |
        | :------------------------------ | :-------------- | :-------------- |
        | PCFG parseable                  | 100%            | 100%            |
        | PCFG norm. log-prob / token     | -13.01 .. -11.79 | -12.70 .. -11.74 |
        | Trigram PPL (median)            | 61              | 2091            |
        | Bigram PPL (median)             | 271             | 1345            |

        Two consequences drive the rule below:

        1. PCFG parseability is **necessary but not sufficient**. The Treebank PCFG
           over-generates, so corrupted sentences still receive a parse with a
           normal-range score; parseability alone would label every input
           grammatical. It is therefore used as the structural authority, but a
           parse is not accepted as *Grammatical* unless an n-gram model also
           confirms the sentence is lexically plausible.
        2. Trigram perplexity is the strongest single separator (61 vs 2091), so it
           is the default judge whenever the PCFG cannot commit.

        Decision table:

        =================== ================= ==================== =============
        PCFG parses?        Trigram PPL       Chosen method        Verdict
        =================== ================= ==================== =============
        yes                 < 350             PCFG Parser          Grammatical
        yes                 350 .. < 800      PCFG Parser          Ungrammatical
        yes                 >= 800            Bigram LM            Grammatical if
                                                                Bigram PPL < 650
        no                  < 350             Trigram LM           Grammatical
        no                  350 .. < 800      Trigram LM           Ungrammatical
        no                  >= 800            Bigram LM            Grammatical if
                                                                Bigram PPL < 650
        =================== ================= ==================== =============

        The 350 / 650 / 800 constants sit inside the measured separation gaps
        (grammatical trigram p90 = 89 vs corrupted p10 = 699; grammatical bigram
        p90 = 558 vs corrupted p25 = 779), which is why they are used here instead
        of the round numbers previously assumed.
        """
        TRIGRAM_GRAMMATICAL_PPL = 350.0
        TRIGRAM_FALLBACK_PPL = 800.0
        BIGRAM_GRAMMATICAL_PPL = 650.0

        if trigram_ppl < TRIGRAM_GRAMMATICAL_PPL:
            return ("PCFG Parser" if is_pcfg_parseable else "Trigram LM"), "Grammatical"
        if trigram_ppl < TRIGRAM_FALLBACK_PPL:
            return ("PCFG Parser" if is_pcfg_parseable else "Trigram LM"), "Ungrammatical"
        verdict = "Grammatical" if bigram_ppl < BIGRAM_GRAMMATICAL_PPL else "Ungrammatical"
        return "Bigram LM", verdict

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

            # PCFG parse (cap at 25 words to prevent cubic O(n^3) chart explosion on long sentences)
            if len(sent_tokens) <= 25:
                pcfg_res = self.pcfg_parser.parse(sent_tokens, sent_tags)
                pcfg_str = f"{pcfg_res.log_prob:.2f}" if pcfg_res.is_parseable else "unparseable"
            else:
                # Every ParseResult field is required; constructing it with only
                # the keyword subset used to raise TypeError on any sentence over
                # 25 tokens, crashing the whole Part 4 table.
                ptb_tags = [reconcile_tag(t) for t in sent_tags]
                pcfg_res = ParseResult(
                    sentence=list(sent_tokens),
                    pos_tags=list(sent_tags),
                    ptb_tags=ptb_tags,
                    log_prob=float("-inf"),
                    is_parseable=False,
                    tree_str=None,
                )
                pcfg_str = "unparseable (length > 25)"

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

            # Surface checks override an LM "Grammatical" verdict: repeated
            # function words keep the trigram contexts fluent, so perplexity
            # cannot see them (measured 0% detection without this check).
            issues = surface_anomalies(sent_tokens)
            if issues and verdict == "Grammatical":
                verdict = "Ungrammatical"
                chosen_method = "Surface Check"

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
                    surface_issues=issues,
                )
            )

        return report
