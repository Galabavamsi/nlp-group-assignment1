"""Question 1: joint word segmentation and POS tagging.

The implementation deliberately keeps the models small and inspectable:

* segmentation: trigram word language model + dynamic programming;
* tagging: trigram HMM-style Viterbi decoder with word/tag emissions;
* Spanish morphology: UD UPOS tags are extended with Gender and Number;
* baselines: greedy longest-match segmentation and most-frequent-tag tagging.

Corpora are downloaded separately by ``scripts/download_data.py`` and are not
committed to the repository.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


START = "<s>"
END = "</s>"
UNK = "<UNK>"
TOKEN_RE = re.compile(r"^[\wáéíóúüñÁÉÍÓÚÜÑ]+$", re.UNICODE)


def normalize_word(word: str) -> str:
    return word.strip().lower()


def is_word(word: str) -> bool:
    return bool(TOKEN_RE.fullmatch(word)) and any(ch.isalpha() for ch in word)


def _log_prob(count: int, total: int, vocabulary_size: int, k: float) -> float:
    return math.log((count + k) / (total + k * vocabulary_size))


@dataclass
class TaggedSentence:
    words: list[str]
    tags: list[str]


@dataclass
class NgramLM:
    """Add-k trigram word language model."""

    k: float = 0.1
    unigrams: collections.Counter = field(default_factory=collections.Counter)
    bigrams: collections.Counter = field(default_factory=collections.Counter)
    bigram_context_totals: collections.Counter = field(default_factory=collections.Counter)
    trigrams: collections.Counter = field(default_factory=collections.Counter)
    trigram_context_totals: collections.Counter = field(default_factory=collections.Counter)
    vocabulary: set[str] = field(default_factory=set)

    def fit(self, sentences: Iterable[Sequence[str]]) -> "NgramLM":
        for sentence in sentences:
            words = [normalize_word(w) for w in sentence if is_word(w)]
            if not words:
                continue
            padded = [START, START, *words, END]
            self.vocabulary.update(words)
            self.unigrams.update(padded[2:])
            bigrams = list(zip(padded, padded[1:]))
            self.bigrams.update(bigrams)
            self.bigram_context_totals.update(a for a, _ in bigrams)
            trigrams = list(zip(padded, padded[1:], padded[2:]))
            self.trigrams.update(trigrams)
            self.trigram_context_totals.update((a, b) for a, b, _ in trigrams)
        return self

    @property
    def vocab_size(self) -> int:
        return max(1, len(self.vocabulary) + 1)

    def score(self, previous2: str, previous1: str, word: str) -> float:
        word = normalize_word(word)
        tri = math.exp(_log_prob(self.trigrams[(previous2, previous1, word)], self.trigram_context_totals[(previous2, previous1)], self.vocab_size, self.k))
        bi = math.exp(_log_prob(self.bigrams[(previous1, word)], self.bigram_context_totals[previous1], self.vocab_size, self.k))
        uni = math.exp(_log_prob(self.unigrams[word], sum(self.unigrams.values()), self.vocab_size, self.k))
        # Interpolation gives unseen sentence contexts a useful backoff signal;
        # this is especially important for the artificial no-space test strings.
        return math.log(0.60 * tri + 0.30 * bi + 0.10 * uni)

    def sentence_logprob(self, words: Sequence[str]) -> float:
        normalized = [normalize_word(w) for w in words]
        padded = [START, START, *normalized, END]
        return sum(self.score(a, b, c) for a, b, c in zip(padded, padded[1:], padded[2:]))


@dataclass
class SegmentationModel:
    lm: NgramLM
    max_word_length: int = 20
    beam_size: int = 32

    def __post_init__(self) -> None:
        self.vocabulary = self.lm.vocabulary

    @staticmethod
    def usable_word(word: str) -> bool:
        # Single-character Brown/UD artefacts create pathological splits in
        # space-free text. Preserve English's valid ``a`` and ``i``.
        return len(word) >= 2 or word in {"a", "i"}

    def candidates_at(self, text: str, start: int) -> list[str]:
        end = min(len(text), start + self.max_word_length)
        candidates = [text[start:i] for i in range(start + 1, end + 1) if text[start:i] in self.vocabulary and self.usable_word(text[start:i])]
        # Unknown single characters keep the decoder total for fully OOV input.
        if not candidates:
            candidates = [text[start : start + 1]]
        return candidates

    def segment(self, text: str) -> list[str]:
        text = normalize_word(text).replace(" ", "")
        if not text:
            return []

        # position -> (last-two, last-one) -> (score, words). Keeping a bucket
        # per position avoids repeatedly scanning states from earlier positions.
        states: dict[int, dict[tuple[str, str], tuple[float, list[str]]]] = {0: {(START, START): (0.0, [])}}
        candidate_cache = {position: self.candidates_at(text, position) for position in range(len(text))}
        score_cache: dict[tuple[str, str, str], float] = {}
        for position in range(len(text)):
            for (previous2, previous1), (score, words) in states.get(position, {}).items():
                for candidate in candidate_cache[position]:
                    next_pos = position + len(candidate)
                    score_key = (previous2, previous1, candidate)
                    segment_score = score_cache.get(score_key)
                    if segment_score is None:
                        segment_score = self.lm.score(previous2, previous1, candidate)
                        score_cache[score_key] = segment_score
                    new_score = score + segment_score
                    next_states = states.setdefault(next_pos, {})
                    next_key = (previous1, candidate)
                    incumbent = next_states.get(next_key)
                    if incumbent is None or new_score > incumbent[0]:
                        next_states[next_key] = (new_score, [*words, candidate])
            # The exact state space can contain many rare word histories on
            # Brown/UD. Keep the best histories at each reachable position;
            # this is the same practical beam approximation used by decoders.
            for next_pos in range(position + 1, min(len(text), position + self.max_word_length) + 1):
                bucket = states.get(next_pos)
                if bucket is not None and len(bucket) > self.beam_size:
                    states[next_pos] = dict(sorted(bucket.items(), key=lambda item: item[1][0], reverse=True)[: self.beam_size])

        finals = list(states.get(len(text), {}).values())
        return max(finals, key=lambda item: item[0])[1] if finals else [text]


@dataclass
class POSTagger:
    """Trigram HMM POS tagger with optional morphology-aware tags."""

    morphology_aware: bool = False
    k: float = 0.1
    beam_size: int = 64
    emissions: collections.Counter = field(default_factory=collections.Counter)
    tag_unigrams: collections.Counter = field(default_factory=collections.Counter)
    tag_bigrams: collections.Counter = field(default_factory=collections.Counter)
    tag_trigrams: collections.Counter = field(default_factory=collections.Counter)
    tag_context_totals: collections.Counter = field(default_factory=collections.Counter)
    word_counts: collections.Counter = field(default_factory=collections.Counter)
    word_tag_totals: collections.Counter = field(default_factory=collections.Counter)
    most_frequent_tag: dict[str, str] = field(default_factory=dict)
    tags: set[str] = field(default_factory=set)

    @staticmethod
    def morphology_tag(upos: str, feats: str | None) -> str:
        values: dict[str, str] = {}
        if feats and feats != "_":
            for item in feats.split("|"):
                if "=" in item:
                    key, value = item.split("=", 1)
                    values[key] = value
        suffix = []
        if values.get("Gender") in {"Masc", "Fem", "Neut"}:
            suffix.append(values["Gender"])
        if values.get("Number") in {"Sing", "Plur"}:
            suffix.append({"Sing": "Sg", "Plur": "Pl"}[values["Number"]])
        return upos + ("-" + "-".join(suffix) if suffix else "")

    def fit(self, sentences: Iterable[TaggedSentence]) -> "POSTagger":
        for sentence in sentences:
            words = [normalize_word(w) for w in sentence.words]
            tags = list(sentence.tags)
            self.word_counts.update(words)
            self.tags.update(tags)
            self.emissions.update(zip(words, tags))
            self.word_tag_totals.update(words)
            padded = [START, START, *tags, END]
            self.tag_unigrams.update(padded[2:])
            self.tag_bigrams.update(zip(padded, padded[1:]))
            tag_trigrams = list(zip(padded, padded[1:], padded[2:]))
            self.tag_trigrams.update(tag_trigrams)
            self.tag_context_totals.update((a, b) for a, b, _ in tag_trigrams)
        best: dict[str, tuple[int, str]] = {}
        for (word, tag), count in self.emissions.items():
            if word not in best or count > best[word][0]:
                best[word] = (count, tag)
        self.most_frequent_tag = {word: tag for word, (_, tag) in best.items()}
        return self

    def emission_logprob(self, word: str, tag: str) -> float:
        word = normalize_word(word)
        count = self.emissions[(word, tag)]
        total = self.word_tag_totals[word]
        if not total:
            count = 0
            total = self.word_tag_totals[UNK]
        return _log_prob(count, total, max(1, len(self.tags)), self.k)

    def transition_logprob(self, previous2: str, previous1: str, tag: str) -> float:
        count = self.tag_trigrams[(previous2, previous1, tag)]
        total = self.tag_context_totals[(previous2, previous1)]
        return _log_prob(count, total, max(1, len(self.tags) + 1), self.k)

    def tag(self, words: Sequence[str]) -> list[str]:
        words = [normalize_word(w) for w in words]
        if not words or not self.tags:
            return []
        states: dict[tuple[str, str], tuple[float, list[str]]] = {(START, START): (0.0, [])}
        for word in words:
            next_states: dict[tuple[str, str], tuple[float, list[str]]] = {}
            for (previous2, previous1), (score, path) in states.items():
                for tag in sorted(self.tags):
                    candidate_score = score + self.transition_logprob(previous2, previous1, tag) + self.emission_logprob(word, tag)
                    key = (previous1, tag)
                    incumbent = next_states.get(key)
                    if incumbent is None or candidate_score > incumbent[0]:
                        next_states[key] = (candidate_score, [*path, tag])
            # Brown's original Penn tagset contains many rare tags. A small
            # beam keeps trigram Viterbi practical without changing the model.
            states = dict(sorted(next_states.items(), key=lambda item: item[1][0], reverse=True)[: self.beam_size])
        return max(states.values(), key=lambda item: item[0])[1]


@dataclass
class Q1System:
    language: str
    lm: NgramLM
    segmenter: SegmentationModel
    tagger: POSTagger
    train_sentences: list[TaggedSentence]
    test_sentences: list[TaggedSentence]
    dev_sentences: list[TaggedSentence] = field(default_factory=list)

    def decode(self, text: str) -> list[tuple[str, str]]:
        words = self.segmenter.segment(text)
        return list(zip(words, self.tagger.tag(words)))

    def greedy_segment(self, text: str) -> list[str]:
        text = normalize_word(text).replace(" ", "")
        output: list[str] = []
        position = 0
        vocabulary = self.segmenter.vocabulary
        while position < len(text):
            options = [text[position:i] for i in range(position + 1, min(len(text), position + self.segmenter.max_word_length) + 1) if text[position:i] in vocabulary]
            word = max(options, key=len) if options else text[position]
            output.append(word)
            position += len(word)
        return output

    def most_frequent_tags(self, words: Sequence[str]) -> list[str]:
        return [self.tagger.most_frequent_tag.get(normalize_word(word), "UNK") for word in words]

    def evaluate(self) -> dict:
        metrics = {
            "sentences": len(self.test_sentences),
            "joint_exact": 0,
            "model_segmentation_exact": 0,
            "greedy_segmentation_exact": 0,
            "model_pos_accuracy": 0.0,
            "baseline_pos_accuracy": 0.0,
            "morphology_aware_pos_accuracy": 0.0,
            "segmentation_caused_tag_errors": 0,
            "genuine_tag_errors": 0,
        }
        confusion: collections.Counter[tuple[str, str]] = collections.Counter()
        comparable_tags = 0
        morphology_correct = 0
        all_gold_tags = sum(len(s.tags) for s in self.test_sentences)
        for gold in self.test_sentences:
            raw = "".join(gold.words)
            predicted_words = self.segmenter.segment(raw)
            predicted_tags = self.tagger.tag(predicted_words)
            greedy_words = self.greedy_segment(raw)
            baseline_tags = self.most_frequent_tags(gold.words)
            if predicted_words == gold.words:
                metrics["model_segmentation_exact"] += 1
            if greedy_words == gold.words:
                metrics["greedy_segmentation_exact"] += 1
            if list(zip(predicted_words, predicted_tags)) == list(zip(gold.words, gold.tags)):
                metrics["joint_exact"] += 1
            for index, gold_tag in enumerate(gold.tags):
                if index < len(predicted_words) and predicted_words[index] == gold.words[index] and index < len(predicted_tags):
                    predicted_tag = predicted_tags[index]
                    comparable_tags += 1
                    confusion[(gold_tag, predicted_tag)] += 1
                    if predicted_tag == gold_tag:
                        metrics["model_pos_accuracy"] += 1
                        morphology_correct += 1
                    else:
                        metrics["genuine_tag_errors"] += 1
                else:
                    metrics["segmentation_caused_tag_errors"] += 1
            metrics["baseline_pos_accuracy"] += sum(tag == gold_tag for tag, gold_tag in zip(baseline_tags, gold.tags))
        metrics["model_pos_accuracy"] /= max(1, comparable_tags)
        metrics["baseline_pos_accuracy"] /= max(1, all_gold_tags)
        metrics["morphology_aware_pos_accuracy"] = morphology_correct / max(1, comparable_tags)
        labels = sorted(set(gold for gold, _ in confusion) | set(pred for _, pred in confusion))
        metrics["confusion_labels"] = labels
        metrics["confusion_matrix"] = {gold: {pred: confusion[(gold, pred)] for pred in labels} for gold in labels}
        return metrics


def load_brown_sentences(data_dir: str | Path = "data") -> list[TaggedSentence]:
    import nltk
    from nltk.corpus import brown

    nltk_data_dir = Path(data_dir) / "nltk"
    if nltk_data_dir.exists():
        nltk.data.path.insert(0, str(nltk_data_dir))
    sentences = []
    for sentence in brown.tagged_sents(tagset=None):
        filtered = [(normalize_word(word), tag) for word, tag in sentence if is_word(word)]
        if len(filtered) >= 2:
            sentences.append(TaggedSentence([w for w, _ in filtered], [t for _, t in filtered]))
    return sentences


def parse_conllu(path: str | Path, morphology_aware: bool = True) -> list[TaggedSentence]:
    sentences: list[TaggedSentence] = []
    words: list[str] = []
    tags: list[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                if words:
                    sentences.append(TaggedSentence(words, tags))
                    words, tags = [], []
                continue
            if line.startswith("#"):
                continue
            columns = line.split("\t")
            if len(columns) < 6 or "-" in columns[0] or "." in columns[0]:
                continue
            word, upos, feats = columns[1], columns[3], columns[5]
            if is_word(word):
                words.append(normalize_word(word))
                tags.append(POSTagger.morphology_tag(upos, feats) if morphology_aware else upos)
        if words:
            sentences.append(TaggedSentence(words, tags))
    return [s for s in sentences if len(s.words) >= 2]


def build_system(language: str, data_dir: str | Path = "data", seed: int = 7) -> Q1System:
    language = language.lower()
    if language == "english":
        all_sentences = load_brown_sentences(data_dir)
        random.Random(seed).shuffle(all_sentences)
        split = int(len(all_sentences) * 0.8)
        train_sentences, test_sentences = all_sentences[:split], all_sentences[split:]
        dev_sentences = []
        morphology_aware = False
    elif language == "spanish":
        root = Path(data_dir) / "UD_Spanish-GSD"
        train_sentences = parse_conllu(root / "es_gsd-ud-train.conllu")
        dev_sentences = parse_conllu(root / "es_gsd-ud-dev.conllu")
        test_sentences = parse_conllu(root / "es_gsd-ud-test.conllu")
        morphology_aware = True
    else:
        raise ValueError("language must be 'english' or 'spanish'")
    lm = NgramLM(k=0.1).fit(sentence.words for sentence in train_sentences)
    segmenter = SegmentationModel(lm, max_word_length=20)
    tagger = POSTagger(morphology_aware=morphology_aware, k=0.1).fit(train_sentences)
    return Q1System(language, lm, segmenter, tagger, train_sentences, test_sentences, dev_sentences)


def sample_inputs(language: str) -> list[str]:
    if language == "english":
        return ["thequickbrownfoxjumpsoverthelazydog"]
    return ["mispadrespuedenviajar", "elcielodespejadoesazul", "lacasarojaesgrande"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate Question 1 segmentation and POS tagging")
    parser.add_argument("--language", choices=["english", "spanish"], default="english")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--text", action="append", help="space-free input; may be repeated")
    parser.add_argument("--limit", type=int, default=None, help="limit test sentences for a quick run")
    args = parser.parse_args()
    system = build_system(args.language, args.data_dir)
    if args.limit:
        system.test_sentences = system.test_sentences[: args.limit]
    print(json.dumps({"language": args.language, "evaluation": system.evaluate()}, indent=2, ensure_ascii=False))
    for text in args.text or sample_inputs(args.language):
        print(f"{text} -> {system.decode(text)}")


if __name__ == "__main__":
    main()
