"""Question 4: Integrated Background Editor — Live Typing Simulator & Alerts Pipeline.

This module implements the real-time background editor pipeline combining:
* Space-omission merge generator (probability p);
* [SEGMENT-ALERT] via Q1's beam decoder;
* [SPELL-ALERT] via Q3's symmetric-delete candidate generator (Method B);
* [GRAMMAR-ALERT] via Q4's shared LM perplexity and Q3's real-word bigram check (trigger interval N);
* Latency tracking for per-token and per-trigger processing.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Sequence

import nltk
from src.q1_word_segmentation import Q1System, build_system, is_word, normalize_word
from src.q3_spelling_corrector import SpellingCorrector, build_corrector_from_brown
from src.q4_shared_lm import SharedLanguageModel


@dataclass
class Alert:
    alert_type: str  # "SEGMENT-ALERT", "SPELL-ALERT", "GRAMMAR-ALERT"
    token_index: int
    original: str
    replacement: str | list[str]
    details: str
    pos_tags: list[str] = field(default_factory=list)


@dataclass
class EditorState:
    tokens: list[str] = field(default_factory=list)
    pos_tags: list[str] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    processed_count: int = 0
    segment_merges_resolved: int = 0
    spelling_corrections_applied: int = 0
    per_token_latencies_ms: list[float] = field(default_factory=list)
    per_trigger_latencies_ms: list[float] = field(default_factory=list)

    @property
    def avg_per_token_latency_ms(self) -> float:
        return sum(self.per_token_latencies_ms) / max(1, len(self.per_token_latencies_ms))

    @property
    def avg_per_trigger_latency_ms(self) -> float:
        return sum(self.per_trigger_latencies_ms) / max(1, len(self.per_trigger_latencies_ms))


def has_valid_word_split(token: str, vocab: set[str]) -> bool:
    """Fast check whether token can be partitioned into >= 2 valid words from vocab."""
    if len(token) < 4:
        return False
    # Check 2-word split
    for i in range(2, len(token) - 1):
        w1, w2 = token[:i], token[i:]
        if (w1 in vocab or w1 in {"a", "i"}) and (w2 in vocab or w2 in {"a", "i"}):
            return True
        # Check 3-word split
        if (w1 in vocab or w1 in {"a", "i"}) and len(w2) >= 4:
            for j in range(2, len(w2) - 1):
                w2a, w2b = w2[:j], w2[j:]
                if (w2a in vocab or w2a in {"a", "i"}) and (w2b in vocab or w2b in {"a", "i"}):
                    return True
    return False


class IntegratedEditor:
    """Integrated Live Text Editor hosting Q1, Q3, and Q4 models."""

    def __init__(
        self,
        p_merge: float = 0.08,
        trigger_interval: int = 5,
        data_dir: str | Path = "data",
        q1_system: Q1System | None = None,
        q3_corrector: SpellingCorrector | None = None,
        shared_lm: SharedLanguageModel | None = None,
    ) -> None:
        self.p_merge = p_merge
        self.trigger_interval = trigger_interval
        self.data_dir = Path(data_dir)

        # Load sub-system models
        self.q1_system = q1_system or build_system("english", data_dir=self.data_dir)
        self.q3_corrector = q3_corrector or build_corrector_from_brown(seed=7, data_dir=str(self.data_dir))[0]
        self.shared_lm = shared_lm or SharedLanguageModel.train_and_tune(data_dir=self.data_dir)

    def sample_passage(self, corpus_name: str = "gutenberg", seed: int | None = None) -> list[str]:
        """Sample a 5-8 sentence passage from gutenberg, brown, or reuters."""
        nltk_data_dir = self.data_dir / "nltk"
        if nltk_data_dir.exists() and str(nltk_data_dir) not in nltk.data.path:
            nltk.data.path.insert(0, str(nltk_data_dir))

        rng = random.Random(seed)
        if corpus_name == "gutenberg":
            try:
                from nltk.corpus import gutenberg
                fileid = rng.choice(gutenberg.fileids())
                sents = gutenberg.sents(fileid)
            except Exception:
                from nltk.corpus import brown
                sents = brown.sents()
        elif corpus_name == "reuters":
            try:
                from nltk.corpus import reuters
                fileid = rng.choice(reuters.fileids())
                sents = reuters.sents(fileid)
            except Exception:
                from nltk.corpus import brown
                sents = brown.sents()
        else:
            from nltk.corpus import brown
            sents = brown.sents()

        num_sents = rng.randint(5, 8)
        if len(sents) <= num_sents:
            chosen = sents
        else:
            start_idx = rng.randint(0, len(sents) - num_sents)
            chosen = sents[start_idx : start_idx + num_sents]

        words: list[str] = []
        for sent in chosen:
            words.extend(sent)
        return words

    def inject_merges(self, words: Sequence[str], rng: random.Random | None = None) -> list[str]:
        """Inject space omission merges with probability p_merge between consecutive words."""
        if rng is None:
            rng = random.Random()
        if not words:
            return []

        merged: list[str] = []
        i = 0
        while i < len(words):
            current = words[i]
            if i + 1 < len(words) and is_word(current) and is_word(words[i + 1]) and rng.random() < self.p_merge:
                # Merge current and next token
                merged.append(current.lower() + words[i + 1].lower())
                i += 2
            else:
                merged.append(current)
                i += 1
        return merged

    def process_token(self, token: str, state: EditorState) -> list[str]:
        """Process a single incoming token through per-token checks (Segmentation -> Spelling).

        Returns list of resulting word tokens (1 word if unchanged/spelling-corrected, >=2 if split).
        """
        start_time = time.perf_counter()
        normalized = normalize_word(token)
        token_index = len(state.tokens)

        # Step 1: [SEGMENT-ALERT]
        # Check if token is out of vocabulary and splits into >= 2 valid words
        if is_word(token) and normalized not in self.q3_corrector.vocabulary:
            if has_valid_word_split(normalized, self.q3_corrector.vocabulary):
                split_words = self.q1_system.segmenter.segment(normalized)
                if len(split_words) >= 2 and all(w in self.q3_corrector.vocabulary for w in split_words):
                    tags = self.q1_system.tagger.tag(split_words)
                    state.alerts.append(
                        Alert(
                            alert_type="SEGMENT-ALERT",
                            token_index=token_index,
                            original=token,
                            replacement=split_words,
                            details=f"Split merged token '{token}' into {split_words} with POS tags {tags}",
                            pos_tags=tags,
                        )
                    )
                    state.segment_merges_resolved += 1
                    state.per_token_latencies_ms.append((time.perf_counter() - start_time) * 1000)
                    return split_words

        # Step 2: [SPELL-ALERT]
        # Check for remaining non-word errors using Q3 Method B (Symmetric Delete)
        if is_word(token) and normalized not in self.q3_corrector.vocabulary:
            corrected, _ = self.q3_corrector.correct_non_word(normalized, method="symdelete")
            if corrected != normalized:
                state.alerts.append(
                    Alert(
                        alert_type="SPELL-ALERT",
                        token_index=token_index,
                        original=token,
                        replacement=corrected,
                        details=f"Corrected non-word '{token}' -> '{corrected}' (Method B Symmetric Delete)",
                        pos_tags=self.q1_system.tagger.tag([corrected]),
                    )
                )
                state.spelling_corrections_applied += 1
                state.per_token_latencies_ms.append((time.perf_counter() - start_time) * 1000)
                return [corrected]

        # Token unchanged
        state.per_token_latencies_ms.append((time.perf_counter() - start_time) * 1000)
        return [token]

    def _retag_sequence(self, state: EditorState) -> None:
        """Refresh the trailing POS tags within a bounded right-context window.

        Tags must not be predicted per token in isolation: Q1's model is a trigram
        HMM, so ``tagger.tag([w])`` always sees the empty context and collapses to
        the same high-frequency tag for almost every word. Downstream consumers
        need real tags, because the tags are the PCFG's lexical terminals and are
        also reported verbatim in the Part 4 table.

        Re-tagging the *entire* prefix after every appended token would be O(n^2)
        and was measured at ~250 ms/word for a 20-word sentence -- far outside the
        interactive budget. Instead this re-tags only the trailing window that the
        live grammar check actually inspects. The window is ``2 * trigger_interval``
        tokens, which also supplies the previous two tags the trigram model needs,
        so tags inside the inspected span are computed with real context while the
        per-token cost stays constant. The frozen tags before the window were
        themselves computed with adequate context when they were the window.
        """
        tokens = state.tokens
        if not tokens:
            state.pos_tags = []
            return

        window = max(4, 2 * self.trigger_interval)
        start = max(0, len(tokens) - window)
        suffix_tags = self.q1_system.tagger.tag(tokens[start:])
        if not suffix_tags or len(suffix_tags) != len(tokens) - start:
            suffix_tags = ["NN"] * (len(tokens) - start)

        state.pos_tags = state.pos_tags[:start] + list(suffix_tags)

    def trigger_grammar_check(self, state: EditorState) -> None:
        """Run interval-based grammar and real-word error check on accumulated window.

        The perplexity cut-offs here are tuned for *precision during typing*, which
        is deliberately stricter than the end-of-passage thresholds in
        :mod:`src.q4_passage_analysis`. Measured over 547 sliding windows of
        in-domain Brown prose:

        ================== ==========================
        Trigram threshold  Windows raising an alert
        ================== ==========================
        300                24.7%
        450                13.9%
        800                 1.1%
        1500                0.0%
        ================== ==========================

        A live alert that fires on one window in seven is alert fatigue, so the
        threshold sits at 800 (roughly one spurious alert per 90 windows) while
        still firing on genuinely anomalous windows, which measure above 3000.
        """
        start_time = time.perf_counter()
        if len(state.tokens) < 3:
            return

        window_size = min(len(state.tokens), self.trigger_interval * 2)
        window_tokens = state.tokens[-window_size:]
        scores = self.shared_lm.score_sentence(window_tokens)

        # Threshold for implausible window perplexity (live: precision-first).
        if scores["trigram_perplexity"] > 800.0 or scores["bigram_perplexity"] > 1500.0:
            state.alerts.append(
                Alert(
                    alert_type="GRAMMAR-ALERT",
                    token_index=len(state.tokens) - 1,
                    original=" ".join(window_tokens),
                    replacement=" ".join(window_tokens),
                    details=f"High perplexity window: Trigram PPL={scores['trigram_perplexity']:.1f}, Bigram PPL={scores['bigram_perplexity']:.1f}",
                )
            )

        # Real-word error check on last word in window using Q3 bigram model
        if len(state.tokens) >= 3:
            idx = len(state.tokens) - 2
            prev_w = normalize_word(state.tokens[idx - 1]) if idx > 0 else "<s>"
            curr_w = normalize_word(state.tokens[idx])
            next_w = normalize_word(state.tokens[idx + 1]) if idx + 1 < len(state.tokens) else "</s>"

            if is_word(curr_w) and len(curr_w) >= 3:
                corrected_rw, _ = self.q3_corrector.correct_real_word(curr_w, prev_w, next_w, method="symdelete")
                if corrected_rw != curr_w:
                    state.alerts.append(
                        Alert(
                            alert_type="GRAMMAR-ALERT",
                            token_index=idx,
                            original=state.tokens[idx],
                            replacement=corrected_rw,
                            details=f"Real-word error in context: '{prev_w} {curr_w} {next_w}' -> suggested '{corrected_rw}'",
                            pos_tags=self.q1_system.tagger.tag([corrected_rw]),
                        )
                    )
                    state.tokens[idx] = corrected_rw
                    state.spelling_corrections_applied += 1

        state.per_trigger_latencies_ms.append((time.perf_counter() - start_time) * 1000)

    def process_full_text(self, text: str) -> EditorState:
        """Process a complete string of user text from scratch, emitting all live alerts."""
        state = EditorState()
        raw_tokens = text.strip().split()
        if not raw_tokens:
            return state

        for token in raw_tokens:
            resulting_words = self.process_token(token, state)
            for w in resulting_words:
                state.tokens.append(w)

            state.processed_count += 1
            self._retag_sequence(state)
            if len(state.tokens) % self.trigger_interval == 0:
                self.trigger_grammar_check(state)

        if len(state.tokens) % self.trigger_interval != 0:
            self.trigger_grammar_check(state)

        return state

    def process_text_incremental(self, text: str, state: EditorState) -> EditorState:
        """Re-derive editor state for the current full text value.

        NOTE: this deliberately does *not* keep a "last processed index" and
        diff against it — it always rebuilds a fresh EditorState from the
        complete current `text` via process_full_text(). That is intentional:
        with a per-keystroke UI widget (see scripts/q4_streamlit_app.py) the
        text can shrink or be reset entirely (e.g. switching the sample-preset
        dropdown back to "-- Type Custom Text --"), and a stateful diff
        tracker would need explicit reset-detection to avoid processing
        garbage or throwing on a negative diff. Recomputing from scratch each
        time sidesteps that failure mode entirely — the trade-off is
        reprocessing the whole string per rerun, which is cheap at the
        ~24ms/token, ~9ms/trigger latencies measured for this pipeline. The
        `state` argument is accepted for API compatibility but intentionally
        ignored; callers should use the returned EditorState.
        """
        return self.process_full_text(text)

    def simulate_typing_stream(
        self,
        words: Sequence[str],
        sleep_delay_sec: float = 0.05,
        seed: int | None = None,
    ) -> Generator[tuple[EditorState, Alert | None], None, None]:
        """Stream passage word-by-word with simulated fast-typing merges and live alerts."""
        rng = random.Random(seed)
        merged_words = self.inject_merges(words, rng=rng)
        state = EditorState()

        for token in merged_words:
            if sleep_delay_sec > 0:
                time.sleep(sleep_delay_sec)

            alerts_before = len(state.alerts)
            resulting_words = self.process_token(token, state)
            for w in resulting_words:
                state.tokens.append(w)

            state.processed_count += 1
            self._retag_sequence(state)
            if len(state.tokens) % self.trigger_interval == 0:
                self.trigger_grammar_check(state)

            new_alert = state.alerts[-1] if len(state.alerts) > alerts_before else None
            yield state, new_alert