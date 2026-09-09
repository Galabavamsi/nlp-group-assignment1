"""Question 4: PCFG constituency parser with CKY search and tagset reconciliation.

This module induces a probabilistic context-free grammar (PCFG) from the Penn
Treebank sample (via NLTK), implements a robust Viterbi CKY most-probable parse
algorithm, and provides tagset reconciliation from Question 1's Brown POS tags
to Penn Treebank tags.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import nltk
from nltk.grammar import PCFG, Nonterminal, ProbabilisticProduction


# Mapping table from Question 1 Brown/UPOS tags to Penn Treebank (PTB) tags
Q1_TO_PTB_TAGMAP: dict[str, str] = {
    # Determiners / Articles
    "AT": "DT",
    "DET": "DT",
    "DT": "DT",
    # Nouns & Proper Nouns
    "NN": "NN",
    "NNS": "NNS",
    "NP": "NNP",
    "NPS": "NNPS",
    "NOUN": "NN",
    "PROPN": "NNP",
    # Verbs
    "VB": "VB",
    "VBD": "VBD",
    "VBG": "VBG",
    "VBN": "VBN",
    "VBP": "VBP",
    "VBZ": "VBZ",
    "VERB": "VB",
    "AUX": "VB",
    # Auxiliary verbs (Brown specific)
    "BE": "VB",
    "BED": "VBD",
    "BEDZ": "VBD",
    "BEG": "VBG",
    "BEM": "VBP",
    "BEN": "VBN",
    "BER": "VBP",
    "BEZ": "VBZ",
    "HV": "VB",
    "HVD": "VBD",
    "HVG": "VBG",
    "HVN": "VBN",
    "HVZ": "VBZ",
    "DO": "VB",
    "DOD": "VBD",
    "DOG": "VBG",
    "DON": "VBN",
    "DOZ": "VBZ",
    "MD": "MD",
    # Adjectives & Adverbs
    "JJ": "JJ",
    "JJR": "JJR",
    "JJS": "JJS",
    "ADJ": "JJ",
    "RB": "RB",
    "RBR": "RBR",
    "RBS": "RBS",
    "RBT": "RBS",
    "ADV": "RB",
    "QL": "RB",
    # Pronouns
    "PP$": "PRP$",
    "PPO": "PRP",
    "PPS": "PRP",
    "PPSS": "PRP",
    "PN": "PRP",
    "PRP": "PRP",
    "PRP$": "PRP$",
    "PRON": "PRP",
    # Prepositions & Conjunctions
    "IN": "IN",
    "CS": "IN",
    "CC": "CC",
    "ADP": "IN",
    "CCONJ": "CC",
    "SCONJ": "IN",
    # Numerals & Specials
    "CD": "CD",
    "NUM": "CD",
    "OD": "JJ",
    "RP": "RP",
    "PRT": "RP",
    "TO": "TO",
    "UH": "UH",
    "INTJ": "UH",
    "EX": "EX",
    "FW": "FW",
    "WDT": "WDT",
    "WP": "WP",
    "WP$": "WP$",
    "WRB": "WRB",
    # Punctuation
    ".": ".",
    ",": ",",
    ":": ":",
    ";": ":",
    "(": "(",
    ")": ")",
    "--": ":",
    "PUNCT": ".",
}


def reconcile_tag(q1_tag: str) -> str:
    """Map a Question 1 POS tag (including morphology extensions) to Penn Treebank tag."""
    base_tag = q1_tag.split("-")[0].strip()
    return Q1_TO_PTB_TAGMAP.get(base_tag, Q1_TO_PTB_TAGMAP.get(q1_tag, "NN"))


@dataclass
class ParseResult:
    sentence: list[str]
    pos_tags: list[str]
    ptb_tags: list[str]
    log_prob: float
    is_parseable: bool
    tree_str: str | None = None


@dataclass
class PCFGParser:
    """Constituency parser trained on Penn Treebank with CKY Viterbi search."""

    pcfg: PCFG | None = field(default=None)
    binary_rules: dict[str, list[tuple[str, str, float]]] = field(default_factory=lambda: collections.defaultdict(list))
    unary_rules: dict[str, list[tuple[str, float]]] = field(default_factory=lambda: collections.defaultdict(list))
    terminal_rules: dict[str, list[tuple[str, float]]] = field(default_factory=lambda: collections.defaultdict(list))
    start_symbol: str = "S"

    @classmethod
    def train_from_treebank(cls, data_dir: str | Path = "data") -> "PCFGParser":
        """Induce a CNF PCFG from the NLTK Penn Treebank sample."""
        nltk_data_dir = Path(data_dir) / "nltk"
        if nltk_data_dir.exists() and str(nltk_data_dir) not in nltk.data.path:
            nltk.data.path.insert(0, str(nltk_data_dir))

        try:
            from nltk.corpus import treebank
            trees = treebank.parsed_sents()
        except Exception as exc:
            raise RuntimeError("Treebank corpus not found. Run scripts/download_data.py first.") from exc

        productions: list[ProbabilisticProduction] = []
        for tree in trees:
            # Binarize tree to Chomsky Normal Form (CNF)
            t = tree.copy(deep=True)
            t.collapse_unary(collapsePOS=False, collapseRoot=False)
            t.chomsky_normal_form(horzMarkov=2)
            for prod in t.productions():
                productions.append(prod)

        # Induce PCFG rule probabilities
        pcfg = nltk.induce_pcfg(Nonterminal("S"), productions)
        parser = cls(pcfg=pcfg, start_symbol="S")
        parser._index_rules()
        return parser

    def _index_rules(self) -> None:
        """Index rules by RHS for O(1) lookups during CKY decoding."""
        if not self.pcfg:
            return

        for prod in self.pcfg.productions():
            lhs = str(prod.lhs())
            prob = prod.prob()
            if prob <= 0:
                continue
            log_p = math.log(prob)
            rhs = prod.rhs()

            if len(rhs) == 2 and isinstance(rhs[0], Nonterminal) and isinstance(rhs[1], Nonterminal):
                b, c = str(rhs[0]), str(rhs[1])
                self.binary_rules[lhs].append((b, c, log_p))
            elif len(rhs) == 1 and isinstance(rhs[0], Nonterminal):
                b = str(rhs[0])
                self.unary_rules[lhs].append((b, log_p))
            elif len(rhs) == 1:
                term = str(rhs[0])
                self.terminal_rules[term].append((lhs, log_p))

    def parse(self, words: Sequence[str], pos_tags: Sequence[str] | None = None) -> ParseResult:
        """Parse a sentence (or POS tag sequence) using Viterbi CKY search.

        If pos_tags are provided, they are reconciled to Penn Treebank tags and used
        as the lexical terminals for high-coverage constituency parsing.
        """
        words_list = list(words)
        if not words_list:
            return ParseResult([], [], [], float("-inf"), False, None)

        if pos_tags:
            ptb_tags = [reconcile_tag(t) for t in pos_tags]
            terminals = ptb_tags
        else:
            ptb_tags = [reconcile_tag(w) for w in words_list]
            terminals = [w.lower() for w in words_list]

        n = len(terminals)
        # chart[i][j][nonterminal] = (max_log_prob, backpointer)
        chart: list[list[dict[str, tuple[float, object]]]] = [
            [{} for _ in range(n + 1)] for _ in range(n + 1)
        ]

        # Initialization (span length = 1)
        for i, term in enumerate(terminals):
            cell = chart[i][i + 1]
            matching_lhs = self.terminal_rules.get(term, [])
            if not matching_lhs:
                # Fallback matching for OOV terminal symbols: try matching base POS tag
                fallback_tag = reconcile_tag(term)
                matching_lhs = self.terminal_rules.get(fallback_tag, [("NN", math.log(1e-4))])

            for lhs, log_p in matching_lhs:
                if lhs not in cell or log_p > cell[lhs][0]:
                    cell[lhs] = (log_p, term)

            # Unary closure for length 1
            self._apply_unary_closure(cell)

        # CKY Span iteration
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length
                cell = chart[i][j]
                for k in range(i + 1, j):
                    left_cell = chart[i][k]
                    right_cell = chart[k][j]
                    if not left_cell or not right_cell:
                        continue

                    for lhs, rules in self.binary_rules.items():
                        for b, c, log_p in rules:
                            if b in left_cell and c in right_cell:
                                score = log_p + left_cell[b][0] + right_cell[c][0]
                                if lhs not in cell or score > cell[lhs][0]:
                                    cell[lhs] = (score, (k, b, c))

                self._apply_unary_closure(cell)

        # Search for valid parse spanning 0..n
        root_cell = chart[0][n]
        best_lhs = None
        best_score = float("-inf")

        # Prefer start symbol S, else fallback to top nonterminal
        if self.start_symbol in root_cell:
            best_lhs = self.start_symbol
            best_score = root_cell[self.start_symbol][0]
        elif root_cell:
            best_lhs, (best_score, _) = max(root_cell.items(), key=lambda item: item[1][0])

        if best_lhs is None:
            return ParseResult(words_list, list(pos_tags or []), ptb_tags, float("-inf"), False, None)

        tree_str = self._build_tree_str(chart, 0, n, best_lhs, words_list, ptb_tags)
        return ParseResult(words_list, list(pos_tags or []), ptb_tags, best_score, True, tree_str)

    def _apply_unary_closure(self, cell: dict[str, tuple[float, object]]) -> None:
        """Apply unary productions (e.g. A -> B) to fill cell transitively."""
        updated = True
        while updated:
            updated = False
            for lhs, rules in self.unary_rules.items():
                for b, log_p in rules:
                    if b in cell:
                        score = log_p + cell[b][0]
                        if lhs not in cell or score > cell[lhs][0]:
                            cell[lhs] = (score, (b,))
                            updated = True

    def _build_tree_str(
        self,
        chart: list[list[dict[str, tuple[float, object]]]],
        i: int,
        j: int,
        lhs: str,
        words: list[str],
        ptb_tags: list[str],
    ) -> str:
        """Reconstruct bracketed tree string representation from CKY backpointers."""
        if i + 1 == j:
            word = words[i] if i < len(words) else ptb_tags[i]
            return f"({lhs} {word})"

        backpointer = chart[i][j][lhs][1]
        if isinstance(backpointer, tuple) and len(backpointer) == 3:
            k, b, c = backpointer
            left_tree = self._build_tree_str(chart, i, k, b, words, ptb_tags)
            right_tree = self._build_tree_str(chart, k, j, c, words, ptb_tags)
            return f"({lhs} {left_tree} {right_tree})"
        elif isinstance(backpointer, tuple) and len(backpointer) == 1:
            (b,) = backpointer
            child_tree = self._build_tree_str(chart, i, j, b, words, ptb_tags)
            return f"({lhs} {child_tree})"
        return f"({lhs} {words[i]})"
