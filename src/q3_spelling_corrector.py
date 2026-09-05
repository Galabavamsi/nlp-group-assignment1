"""Question 3: spelling correction with candidate-method benchmarks.

This module builds a Brown-corpus vocabulary, unigram frequencies, and a
smoothed bigram language model. It implements both required candidate
generation methods:

* Method A: standard edit-distance-1 generation.
* Method B: symmetric-delete lookup over a preprocessed delete index.

The command-line entry point supports evaluation, the exact 1,000-word
"Speed Demon" benchmark, one-off sentence correction, and a continuous CLI.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
import string
import sys
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence


START = "<s>"
END = "</s>"
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\w\s]", re.UNICODE)
ALPHA_RE = re.compile(r"^[a-z]+(?:'[a-z]+)?$")


def normalize_word(word: str) -> str:
    return word.strip().lower()


def is_word(word: str) -> bool:
    return bool(ALPHA_RE.fullmatch(normalize_word(word)))


def tokenize_text(text: str) -> list[str]:
    return WORD_RE.findall(text)


def detokenize(tokens: Sequence[str]) -> str:
    text = ""
    no_space_before = {".", ",", "!", "?", ";", ":", ")", "]", "}", "'", '"'}
    no_space_after = {"(", "[", "{", '"'}
    for token in tokens:
        if not text:
            text = token
        elif token in no_space_before or text[-1] in no_space_after:
            text += token
        else:
            text += " " + token
    return text


def match_case(source: str, correction: str) -> str:
    if len(source) > 1 and source.isupper():
        return correction.upper()
    if source[:1].isupper():
        return correction.capitalize()
    return correction


def edits1(word: str, alphabet: str = string.ascii_lowercase) -> set[str]:
    """Return all strings one edit away from ``word``."""

    word = normalize_word(word)
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = {left + right[1:] for left, right in splits if right}
    transposes = {left + right[1] + right[0] + right[2:] for left, right in splits if len(right) > 1}
    replaces = {left + char + right[1:] for left, right in splits if right for char in alphabet}
    inserts = {left + char + right for left, right in splits for char in alphabet}
    return (deletes | transposes | replaces | inserts) - {word}


def deletion_variants(word: str) -> set[str]:
    word = normalize_word(word)
    return {word[:i] + word[i + 1 :] for i in range(len(word))}


def within_edit_distance_one(source: str, target: str) -> bool:
    """Fast check for whether two words are at Levenshtein/edit distance <= 1.

    Transposition is also accepted because Method A includes it as a single
    spelling edit.
    """

    source = normalize_word(source)
    target = normalize_word(target)
    if source == target:
        return True
    if abs(len(source) - len(target)) > 1:
        return False
    if len(source) == len(target):
        mismatches = [i for i, (a, b) in enumerate(zip(source, target)) if a != b]
        if len(mismatches) <= 1:
            return True
        if len(mismatches) == 2:
            first, second = mismatches
            return second == first + 1 and source[first] == target[second] and source[second] == target[first]
        return False

    if len(source) > len(target):
        source, target = target, source
    i = j = edits = 0
    while i < len(source) and j < len(target):
        if source[i] == target[j]:
            i += 1
            j += 1
        else:
            edits += 1
            if edits > 1:
                return False
            j += 1
    return True


@dataclass
class Correction:
    original: str
    corrected: str
    index: int
    kind: str
    score: float


@dataclass
class SentenceCorrection:
    original_text: str
    corrected_text: str
    changes: list[Correction]
    latency_ms: float

    @property
    def highlighted_text(self) -> str:
        changed = {change.index: change.corrected for change in self.changes}
        tokens = tokenize_text(self.corrected_text)
        highlighted = [f"**{token}**" if index in changed else token for index, token in enumerate(tokens)]
        return detokenize(highlighted)


@dataclass
class SpellingCorrector:
    k: float = 0.1
    real_word_margin: float = 0.75
    alphabet: str = string.ascii_lowercase
    vocabulary: set[str] = field(default_factory=set)
    unigrams: collections.Counter[str] = field(default_factory=collections.Counter)
    bigrams: collections.Counter[tuple[str, str]] = field(default_factory=collections.Counter)
    bigram_context_totals: collections.Counter[str] = field(default_factory=collections.Counter)
    delete_index: dict[str, set[str]] = field(default_factory=lambda: collections.defaultdict(set))

    def fit(self, sentences: Iterable[Sequence[str]]) -> "SpellingCorrector":
        for sentence in sentences:
            words = [normalize_word(word) for word in sentence if is_word(word)]
            if not words:
                continue
            self.vocabulary.update(words)
            self.unigrams.update(words)
            padded = [START, *words, END]
            sentence_bigrams = list(zip(padded, padded[1:]))
            self.bigrams.update(sentence_bigrams)
            self.bigram_context_totals.update(left for left, _ in sentence_bigrams)
        self.build_delete_index()
        return self

    @property
    def vocab_size(self) -> int:
        return max(1, len(self.vocabulary) + 1)

    @property
    def total_words(self) -> int:
        return max(1, sum(self.unigrams.values()))

    def build_delete_index(self) -> None:
        index: dict[str, set[str]] = collections.defaultdict(set)
        for word in self.vocabulary:
            index[word].add(word)
            for variant in deletion_variants(word):
                index[variant].add(word)
        self.delete_index = index

    def unigram_logprob(self, word: str) -> float:
        word = normalize_word(word)
        return math.log((self.unigrams[word] + self.k) / (self.total_words + self.k * self.vocab_size))

    def bigram_logprob(self, previous: str, word: str) -> float:
        previous = normalize_word(previous)
        word = normalize_word(word)
        total = self.bigram_context_totals[previous]
        return math.log((self.bigrams[(previous, word)] + self.k) / (total + self.k * self.vocab_size))

    def phrase_score(self, previous: str | None, word: str, next_word: str | None = None) -> float:
        score = self.unigram_logprob(word) * 0.10
        if previous:
            score += self.bigram_logprob(previous, word)
        if next_word:
            score += self.bigram_logprob(word, next_word)
        return score

    def candidates_edit1(self, word: str) -> set[str]:
        return {candidate for candidate in edits1(word, self.alphabet) if candidate in self.vocabulary}

    def candidates_symmetric_delete(self, word: str) -> set[str]:
        word = normalize_word(word)
        keys = {word, *deletion_variants(word)}
        candidates: set[str] = set()
        for key in keys:
            candidates.update(self.delete_index.get(key, set()))
        if word in self.vocabulary:
            candidates.add(word)
        return {candidate for candidate in candidates if candidate != word and within_edit_distance_one(word, candidate)}

    def candidates(self, word: str, method: str = "symdelete", include_self: bool = False) -> set[str]:
        method = method.lower()
        if method in {"edit", "edits", "method-a", "a"}:
            candidates = self.candidates_edit1(word)
        elif method in {"symdelete", "symmetric-delete", "method-b", "b"}:
            candidates = self.candidates_symmetric_delete(word)
        else:
            raise ValueError("method must be 'edit' or 'symdelete'")
        if include_self and normalize_word(word) in self.vocabulary:
            candidates.add(normalize_word(word))
        return candidates

    def correct_non_word(self, word: str, method: str = "symdelete") -> tuple[str, float]:
        word = normalize_word(word)
        candidates = self.candidates(word, method=method)
        if not candidates:
            return word, self.unigram_logprob(word)
        best = max(candidates, key=lambda candidate: (self.unigrams[candidate], candidate))
        return best, self.unigram_logprob(best)

    def correct_real_word(
        self,
        word: str,
        previous: str | None = None,
        next_word: str | None = None,
        method: str = "symdelete",
    ) -> tuple[str, float]:
        word = normalize_word(word)
        original_score = self.phrase_score(previous, word, next_word)
        candidates = self.candidates(word, method=method)
        if not candidates:
            return word, original_score
        best = max(candidates, key=lambda candidate: (self.phrase_score(previous, candidate, next_word), self.unigrams[candidate]))
        best_score = self.phrase_score(previous, best, next_word)
        if best_score - original_score >= self.real_word_margin:
            return best, best_score
        return word, original_score

    def correct_tokens(self, tokens: Sequence[str], method: str = "symdelete") -> tuple[list[str], list[Correction]]:
        output = list(tokens)
        word_positions = [index for index, token in enumerate(tokens) if is_word(token)]
        normalized_words = {index: normalize_word(tokens[index]) for index in word_positions}
        changes: list[Correction] = []
        for position, index in enumerate(word_positions):
            original_token = tokens[index]
            word = normalized_words[index]
            previous = normalized_words.get(word_positions[position - 1]) if position > 0 else START
            next_word = normalized_words.get(word_positions[position + 1]) if position + 1 < len(word_positions) else END
            if word not in self.vocabulary:
                corrected, score = self.correct_non_word(word, method=method)
                kind = "non-word"
            elif len(word) <= 2:
                corrected, score = word, self.phrase_score(previous, word, next_word)
                kind = "real-word"
            else:
                corrected, score = self.correct_real_word(word, previous, next_word, method=method)
                kind = "real-word"
            if corrected != word:
                display = match_case(original_token, corrected)
                output[index] = display
                normalized_words[index] = corrected
                changes.append(Correction(original_token, display, index, kind, score))
        return output, changes

    def correct_sentence(self, text: str, method: str = "symdelete") -> SentenceCorrection:
        start = time.perf_counter()
        tokens = tokenize_text(text)
        corrected_tokens, changes = self.correct_tokens(tokens, method=method)
        latency_ms = (time.perf_counter() - start) * 1000
        return SentenceCorrection(text, detokenize(corrected_tokens), changes, latency_ms)


def load_brown_sentences(data_dir: str = "data") -> list[list[str]]:
    try:
        import nltk
        from nltk.corpus import brown
    except ImportError as exc:
        raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc

    nltk_data_dir = f"{data_dir}/nltk"
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_dir)
    try:
        raw_sentences = brown.sents()
    except LookupError as exc:
        raise RuntimeError("Download the Brown corpus with: python scripts/download_data.py --english-only") from exc

    sentences: list[list[str]] = []
    for sentence in raw_sentences:
        words = [normalize_word(word) for word in sentence if is_word(word)]
        if len(words) >= 3:
            sentences.append(words)
    return sentences


def train_test_split(sentences: Sequence[Sequence[str]], seed: int = 7, test_fraction: float = 0.10) -> tuple[list[list[str]], list[list[str]]]:
    shuffled = [list(sentence) for sentence in sentences]
    random.Random(seed).shuffle(shuffled)
    split = max(1, int(len(shuffled) * (1 - test_fraction)))
    return shuffled[:split], shuffled[split:]


def build_corrector_from_brown(seed: int = 7, data_dir: str = "data") -> tuple[SpellingCorrector, list[list[str]]]:
    sentences = load_brown_sentences(data_dir)
    train_sentences, test_sentences = train_test_split(sentences, seed=seed, test_fraction=0.10)
    return SpellingCorrector().fit(train_sentences), test_sentences


def corrupt_non_word(word: str, rng: random.Random, vocabulary: set[str], alphabet: str = string.ascii_lowercase) -> str:
    options = sorted(candidate for candidate in edits1(word, alphabet) if candidate and candidate not in vocabulary and is_word(candidate))
    if not options:
        return word
    sameish = [candidate for candidate in options if abs(len(candidate) - len(word)) <= 1]
    return rng.choice(sameish or options)


def real_word_replacement(word: str, rng: random.Random, corrector: SpellingCorrector) -> str | None:
    candidates = sorted(candidate for candidate in corrector.candidates_edit1(word) if candidate != word)
    if not candidates:
        return None
    return rng.choice(candidates)


@dataclass
class ErrorExample:
    sentence: list[str]
    error_sentence: list[str]
    index: int
    gold: str
    observed: str


def generate_test_sets(
    corrector: SpellingCorrector,
    test_sentences: Sequence[Sequence[str]],
    seed: int = 11,
    max_examples: int | None = None,
) -> tuple[list[ErrorExample], list[ErrorExample]]:
    rng = random.Random(seed)
    non_word_examples: list[ErrorExample] = []
    real_word_examples: list[ErrorExample] = []
    for sentence in test_sentences:
        eligible = [i for i, word in enumerate(sentence) if len(word) >= 3 and word in corrector.vocabulary]
        if not eligible:
            continue
        index = rng.choice(eligible)
        gold = sentence[index]
        non_word = corrupt_non_word(gold, rng, corrector.vocabulary, corrector.alphabet)
        if non_word != gold:
            error_sentence = list(sentence)
            error_sentence[index] = non_word
            non_word_examples.append(ErrorExample(list(sentence), error_sentence, index, gold, non_word))
        real_word = real_word_replacement(gold, rng, corrector)
        if real_word and real_word != gold:
            error_sentence = list(sentence)
            error_sentence[index] = real_word
            real_word_examples.append(ErrorExample(list(sentence), error_sentence, index, gold, real_word))
        if max_examples and len(non_word_examples) >= max_examples and len(real_word_examples) >= max_examples:
            break
    return non_word_examples[:max_examples], real_word_examples[:max_examples]


def evaluate_examples(corrector: SpellingCorrector, examples: Sequence[ErrorExample], method: str) -> float:
    correct = 0
    for example in examples:
        observed = example.error_sentence[example.index]
        previous = example.error_sentence[example.index - 1] if example.index > 0 else START
        next_word = example.error_sentence[example.index + 1] if example.index + 1 < len(example.error_sentence) else END
        if observed not in corrector.vocabulary:
            corrected, _ = corrector.correct_non_word(observed, method=method)
        elif len(observed) <= 2:
            corrected = observed
        else:
            corrected, _ = corrector.correct_real_word(observed, previous, next_word, method=method)
        if corrected == example.gold:
            correct += 1
    return correct / max(1, len(examples))


def evaluate_corrector(
    corrector: SpellingCorrector,
    test_sentences: Sequence[Sequence[str]],
    method: str = "symdelete",
    seed: int = 11,
    max_examples: int | None = None,
) -> dict:
    non_word_examples, real_word_examples = generate_test_sets(corrector, test_sentences, seed=seed, max_examples=max_examples)
    return {
        "method": method,
        "non_word_examples": len(non_word_examples),
        "real_word_examples": len(real_word_examples),
        "non_word_accuracy": evaluate_examples(corrector, non_word_examples, method),
        "real_word_accuracy": evaluate_examples(corrector, real_word_examples, method),
    }


def build_benchmark_batch(
    corrector: SpellingCorrector,
    sentences: Sequence[Sequence[str]],
    size: int = 1000,
    seed: int = 19,
) -> list[str]:
    rng = random.Random(seed)
    words = [word for sentence in sentences for word in sentence if len(word) >= 4 and word in corrector.vocabulary]
    if not words:
        raise ValueError("Need at least one vocabulary word to build a benchmark batch")
    batch: list[str] = []
    attempts = 0
    while len(batch) < size and attempts < size * 50:
        attempts += 1
        word = rng.choice(words)
        corrupted = corrupt_non_word(word, rng, corrector.vocabulary, corrector.alphabet)
        if corrupted != word:
            batch.append(corrupted)
    if len(batch) < size:
        raise ValueError(f"Could only build {len(batch)} misspellings, not the required {size}")
    return batch


def benchmark_candidate_methods(corrector: SpellingCorrector, batch: Sequence[str]) -> dict:
    if len(batch) != 1000:
        raise ValueError("Speed Demon benchmark requires exactly 1,000 misspelled words")

    timings: dict[str, float] = {}
    corrections: dict[str, int] = {}
    for method in ("edit", "symdelete"):
        start = time.perf_counter()
        changed = 0
        for word in batch:
            corrected, _ = corrector.correct_non_word(word, method=method)
            if corrected != normalize_word(word):
                changed += 1
        timings[method] = (time.perf_counter() - start) * 1000
        corrections[method] = changed
    faster = "symdelete" if timings["symdelete"] < timings["edit"] else "edit"
    return {
        "words": len(batch),
        "method_a_edit_ms": timings["edit"],
        "method_b_symdelete_ms": timings["symdelete"],
        "method_a_changed": corrections["edit"],
        "method_b_changed": corrections["symdelete"],
        "faster_method": faster,
        "conclusion": (
            "Method B uses precomputed one-deletion keys, so lookup is mostly dictionary access "
            "over a small set of query deletes instead of constructing every deletion, replacement, "
            "transposition, and insertion string for each word."
        ),
    }


def run_interactive(corrector: SpellingCorrector, method: str = "symdelete") -> None:
    print("Continuous spelling corrector. Type 'exit' to stop.")
    while True:
        sentence = input("Sentence> ")
        if sentence.strip().lower() == "exit":
            break
        result = corrector.correct_sentence(sentence, method=method)
        print(f"Corrected: {result.highlighted_text}")
        print(f"Latency: {result.latency_ms:.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Question 3 spelling corrector")
    parser.add_argument("--method", choices=["edit", "symdelete"], default="symdelete")
    parser.add_argument("--sentence", action="append", help="Correct a sentence; may be repeated")
    parser.add_argument("--evaluate", action="store_true", help="Generate Brown test sets and report accuracy")
    parser.add_argument("--benchmark", action="store_true", help="Run the exact 1,000-word Speed Demon benchmark")
    parser.add_argument("--interactive", action="store_true", help="Start the continuous terminal CLI")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional cap for faster evaluation runs")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    try:
        corrector, test_sentences = build_corrector_from_brown(seed=args.seed, data_dir=args.data_dir)
    except RuntimeError as exc:
        print(f"Setup error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    output: dict[str, object] = {
        "vocabulary_size": len(corrector.vocabulary),
        "training_tokens": sum(corrector.unigrams.values()),
    }
    if args.evaluate:
        output["evaluation"] = evaluate_corrector(corrector, test_sentences, method=args.method, max_examples=args.max_examples)
    if args.benchmark:
        batch = build_benchmark_batch(corrector, test_sentences, size=1000, seed=args.seed + 12)
        output["speed_demon"] = benchmark_candidate_methods(corrector, batch)
    if args.sentence:
        sentence_results = []
        for sentence in args.sentence:
            result = corrector.correct_sentence(sentence, method=args.method)
            sentence_results.append(
                {
                    "input": sentence,
                    "corrected": result.corrected_text,
                    "highlighted": result.highlighted_text,
                    "latency_ms": result.latency_ms,
                    "changes": [change.__dict__ for change in result.changes],
                }
            )
        output["sentences"] = sentence_results
    if args.evaluate or args.benchmark or args.sentence:
        print(json.dumps(output, indent=2))
    if args.interactive or not (args.evaluate or args.benchmark or args.sentence):
        run_interactive(corrector, method=args.method)


if __name__ == "__main__":
    main()
