"""Question 4: Shared smoothed Bigram and Trigram Language Models.

This module trains separate add-k smoothed Bigram and Trigram language models on
the Brown corpus. It evaluates candidate smoothing parameter values (k) on a
development set perplexity benchmark and scores arbitrary token sequences for
grammar alerting and final sentence-level analysis.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from src.q1_word_segmentation import END, START, is_word, normalize_word


def _log_prob_addk(count: int, context_total: int, vocab_size: int, k: float) -> float:
    return math.log((count + k) / (context_total + k * vocab_size))


@dataclass
class AddKNgramLM:
    """Add-k smoothed N-gram language model (N=2 for bigram, N=3 for trigram)."""

    n: int = 2
    k: float = 0.1
    counts: collections.Counter = field(default_factory=collections.Counter)
    context_totals: collections.Counter = field(default_factory=collections.Counter)
    vocabulary: set[str] = field(default_factory=set)
    total_unigrams: int = 0

    def fit(self, sentences: Iterable[Sequence[str]]) -> "AddKNgramLM":
        """Train the N-gram model on cleaned, tokenized sentences."""
        for sentence in sentences:
            words = [normalize_word(w) for w in sentence if is_word(w)]
            if not words:
                continue
            self.vocabulary.update(words)

            if self.n == 2:
                padded = [START, *words, END]
                bigrams = list(zip(padded, padded[1:]))
                for left, right in bigrams:
                    self.counts[(left, right)] += 1
                    self.context_totals[left] += 1
                self.total_unigrams += len(words)
            elif self.n == 3:
                padded = [START, START, *words, END]
                trigrams = list(zip(padded, padded[1:], padded[2:]))
                for left, middle, right in trigrams:
                    self.counts[(left, middle, right)] += 1
                    self.context_totals[(left, middle)] += 1
                self.total_unigrams += len(words)
        return self

    @property
    def vocab_size(self) -> int:
        return max(1, len(self.vocabulary) + 1)

    def log_prob(self, word: str, context: Sequence[str]) -> float:
        """Compute log probability log P(word | context)."""
        word = normalize_word(word)
        if self.n == 2:
            left = normalize_word(context[0]) if context else START
            count = self.counts[(left, word)]
            total = self.context_totals[left]
            return _log_prob_addk(count, total, self.vocab_size, self.k)
        elif self.n == 3:
            left1 = normalize_word(context[0]) if len(context) > 0 else START
            left2 = normalize_word(context[1]) if len(context) > 1 else START
            count = self.counts[(left1, left2, word)]
            total = self.context_totals[(left1, left2)]
            return _log_prob_addk(count, total, self.vocab_size, self.k)
        return 0.0

    def sentence_logprob(self, sentence: Sequence[str]) -> float:
        """Compute total log probability of a sentence under this model."""
        words = [normalize_word(w) for w in sentence if is_word(w)]
        if not words:
            return 0.0

        if self.n == 2:
            padded = [START, *words, END]
            return sum(self.log_prob(right, [left]) for left, right in zip(padded, padded[1:]))
        elif self.n == 3:
            padded = [START, START, *words, END]
            return sum(self.log_prob(right, [left1, left2]) for left1, left2, right in zip(padded, padded[1:], padded[2:]))
        return 0.0

    def sentence_perplexity(self, sentence: Sequence[str]) -> float:
        """Compute perplexity of a sentence under this model."""
        words = [normalize_word(w) for w in sentence if is_word(w)]
        if not words:
            return float("inf")
        logprob = self.sentence_logprob(sentence)
        num_tokens = len(words) + 1  # include END token
        return math.exp(-logprob / num_tokens)


@dataclass
class SharedLanguageModel:
    """Container hosting both Question 4 Bigram and Trigram models with tuned add-k smoothing."""

    bigram: AddKNgramLM = field(default_factory=lambda: AddKNgramLM(n=2, k=0.1))
    trigram: AddKNgramLM = field(default_factory=lambda: AddKNgramLM(n=3, k=0.1))
    best_k_bigram: float = 0.1
    best_k_trigram: float = 0.1
    dev_perplexities: dict[str, dict[float, float]] = field(default_factory=dict)

    @classmethod
    def train_and_tune(cls, data_dir: str | Path = "data", candidate_ks: Sequence[float] = (0.001, 0.01, 0.05, 0.1, 0.5, 1.0)) -> "SharedLanguageModel":
        """Train Bigram & Trigram LMs on Brown and tune k using dev-set perplexity."""
        from src.q1_word_segmentation import load_brown_sentences

        all_sentences = [s.words for s in load_brown_sentences(data_dir)]
        split = int(len(all_sentences) * 0.8)
        train_sents, dev_sents = all_sentences[:split], all_sentences[split:]

        dev_results: dict[str, dict[float, float]] = {"bigram": {}, "trigram": {}}

        # Tune Bigram
        best_k_bi = candidate_ks[0]
        best_ppl_bi = float("inf")
        for k in candidate_ks:
            model = AddKNgramLM(n=2, k=k).fit(train_sents)
            total_logprob = sum(model.sentence_logprob(s) for s in dev_sents[:500])
            total_words = sum(len([w for w in s if is_word(w)]) + 1 for s in dev_sents[:500])
            ppl = math.exp(-total_logprob / max(1, total_words))
            dev_results["bigram"][k] = ppl
            if ppl < best_ppl_bi:
                best_ppl_bi = ppl
                best_k_bi = k

        # Tune Trigram
        best_k_tri = candidate_ks[0]
        best_ppl_tri = float("inf")
        for k in candidate_ks:
            model = AddKNgramLM(n=3, k=k).fit(train_sents)
            total_logprob = sum(model.sentence_logprob(s) for s in dev_sents[:500])
            total_words = sum(len([w for w in s if is_word(w)]) + 1 for s in dev_sents[:500])
            ppl = math.exp(-total_logprob / max(1, total_words))
            dev_results["trigram"][k] = ppl
            if ppl < best_ppl_tri:
                best_ppl_tri = ppl
                best_k_tri = k

        # Fit final models on full dataset using best k
        final_bigram = AddKNgramLM(n=2, k=best_k_bi).fit(all_sentences)
        final_trigram = AddKNgramLM(n=3, k=best_k_tri).fit(all_sentences)

        return cls(
            bigram=final_bigram,
            trigram=final_trigram,
            best_k_bigram=best_k_bi,
            best_k_trigram=best_k_tri,
            dev_perplexities=dev_results,
        )

    def score_sentence(self, sentence: Sequence[str]) -> dict[str, float]:
        """Compute sentence log probabilities and perplexities for both models."""
        return {
            "bigram_logprob": self.bigram.sentence_logprob(sentence),
            "bigram_perplexity": self.bigram.sentence_perplexity(sentence),
            "trigram_logprob": self.trigram.sentence_logprob(sentence),
            "trigram_perplexity": self.trigram.sentence_perplexity(sentence),
        }
