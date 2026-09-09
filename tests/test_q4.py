"""Unit tests for Question 4 implementation."""

import math

from src.q4_live_editor import Alert, EditorState, IntegratedEditor
from src.q4_passage_analysis import PassageAnalyzer, PassageAnalysisReport
from src.q4_pcfg_parser import PCFGParser, reconcile_tag
from src.q4_shared_lm import AddKNgramLM, SharedLanguageModel


def test_tagset_reconciliation_maps_q1_tags_to_ptb():
    assert reconcile_tag("AT") == "DT"
    assert reconcile_tag("NP") == "NNP"
    assert reconcile_tag("NOUN-Fem-Sg") == "NN"
    assert reconcile_tag("ADJ-Masc-Pl") == "JJ"
    assert reconcile_tag("PUNCT") == "."
    assert reconcile_tag("UNKNOWN_TAG") == "NN"


def test_pcfg_parser_handles_valid_and_unparseable_sentences():
    # Construct a tiny dummy PCFG parser for unit testing
    parser = PCFGParser(start_symbol="S")
    # Rule: S -> NP VP (p=1.0), NP -> DT NN (p=1.0), VP -> VB NP (p=1.0)
    # Terminal: DT -> "dt", NN -> "nn", VB -> "vb"
    parser.binary_rules["S"].append(("NP", "VP", math.log(1.0)))
    parser.binary_rules["NP"].append(("DT", "NN", math.log(1.0)))
    parser.binary_rules["VP"].append(("VB", "NP", math.log(1.0)))
    parser.terminal_rules["DT"].append(("DT", math.log(1.0)))
    parser.terminal_rules["NN"].append(("NN", math.log(1.0)))
    parser.terminal_rules["VB"].append(("VB", math.log(1.0)))

    # Test valid parse: "the dog bit the cat" with POS tags ["DT", "NN", "VB", "DT", "NN"]
    res_valid = parser.parse(["the", "dog", "bit", "the", "cat"], ["DT", "NN", "VB", "DT", "NN"])
    assert res_valid.is_parseable is True
    assert res_valid.log_prob == 0.0
    assert res_valid.tree_str is not None
    assert "S" in res_valid.tree_str

    # Test unparseable sentence: invalid POS tag sequence ["VB", "VB"]
    res_unparseable = parser.parse(["bit", "bit"], ["VB", "VB"])
    assert res_unparseable.is_parseable is False
    assert res_unparseable.log_prob == float("-inf")
    assert res_unparseable.tree_str is None

    # Test empty input
    res_empty = parser.parse([], [])
    assert res_empty.is_parseable is False


def test_shared_ngram_lm_score_and_perplexity():
    sents = [["the", "quick", "brown", "fox"], ["the", "lazy", "dog"]]
    bigram_lm = AddKNgramLM(n=2, k=0.1).fit(sents)
    trigram_lm = AddKNgramLM(n=3, k=0.1).fit(sents)

    # Logprobs should be negative finite numbers
    bi_logprob = bigram_lm.sentence_logprob(["the", "quick", "fox"])
    tri_logprob = trigram_lm.sentence_logprob(["the", "quick", "fox"])
    assert bi_logprob < 0.0
    assert tri_logprob < 0.0

    # Perplexities should be positive numbers
    bi_ppl = bigram_lm.sentence_perplexity(["the", "quick", "fox"])
    tri_ppl = trigram_lm.sentence_perplexity(["the", "quick", "fox"])
    assert bi_ppl > 1.0
    assert tri_ppl > 1.0


def test_passage_analyzer_decision_rule():
    parser = PCFGParser(start_symbol="S")
    lm = SharedLanguageModel(
        bigram=AddKNgramLM(n=2, k=0.1),
        trigram=AddKNgramLM(n=3, k=0.1),
    )
    analyzer = PassageAnalyzer(pcfg_parser=parser, shared_lm=lm)

    # Condition 1: PCFG parseable with good score -> PCFG Parser, Grammatical
    method, verdict = analyzer.apply_decision_rule(
        is_pcfg_parseable=True,
        pcfg_logprob=-10.0,
        num_tokens=5,  # norm score = -2.0 >= -6.5
        trigram_ppl=100.0,
        bigram_ppl=150.0,
    )
    assert method == "PCFG Parser"
    assert verdict == "Grammatical"

    # Condition 2: PCFG unparseable, Trigram PPL low -> Trigram LM
    method, verdict = analyzer.apply_decision_rule(
        is_pcfg_parseable=False,
        pcfg_logprob=float("-inf"),
        num_tokens=5,
        trigram_ppl=150.0,
        bigram_ppl=250.0,
    )
    assert method == "Trigram LM"
    assert verdict == "Grammatical"

    # Condition 3: High perplexity everywhere -> Ungrammatical
    method, verdict = analyzer.apply_decision_rule(
        is_pcfg_parseable=False,
        pcfg_logprob=float("-inf"),
        num_tokens=5,
        trigram_ppl=600.0,
        bigram_ppl=700.0,
    )
    assert verdict == "Ungrammatical"


def test_merge_generator_drops_spaces_with_probability():
    editor = IntegratedEditor(p_merge=1.0)  # Always merge consecutive words
    words = ["the", "quick", "brown", "fox"]
    merged = editor.inject_merges(words)
    # With p=1.0, ("the", "quick") -> "thequick", ("brown", "fox") -> "brownfox"
    assert len(merged) == 2
    assert merged[0] == "thequick"
    assert merged[1] == "brownfox"
