import tempfile
from pathlib import Path

from src.q2_dependency_parser import (
    ParserState,
    Sentence,
    Token,
    extract_features,
    generate_oracle_examples,
    left_arc,
    parse_conllu,
    right_arc,
    shift,
)

def test_shift():
    state = ParserState(stack=[0], buffer=[1, 2], arcs=[])
    shift(state)
    assert state.stack == [0, 1]
    assert state.buffer == [2]
    assert state.arcs == []

def test_left_arc():
    state = ParserState(stack=[0, 1, 2], buffer=[], arcs=[])
    arc = left_arc(state, "det")
    assert arc == (2, 1, "det")
    assert state.stack == [0, 2]
    assert state.arcs == [(2, 1, "det")]

def test_right_arc():
    state = ParserState(stack=[0, 1, 2], buffer=[], arcs=[])
    arc = right_arc(state, "nsubj")
    assert arc == (1, 2, "nsubj")
    assert state.stack == [0, 1]
    assert state.arcs == [(1, 2, "nsubj")]

def test_extract_features():
    sentence = Sentence(
        [
            Token(1, "The", "DET", 2, "det"),
            Token(2, "cat", "NOUN", 3, "nsubj"),
            Token(3, "sat", "VERB", 0, "root"),
        ]
    )
    state = ParserState(
        stack=[0, 1],
        buffer=[2, 3],
        arcs=[],
    )
    features = extract_features(state, sentence)
    assert features == ["DET", "ROOT", "NOUN", "VERB"]

def test_oracle():
    sentence = Sentence(
        [
            Token(1, "The", "DET", 2, "det"),
            Token(2, "cat", "NOUN", 3, "nsubj"),
            Token(3, "sat", "VERB", 0, "root"),
        ]
    )
    examples = generate_oracle_examples(sentence)
    transitions = [example.transition for example in examples]
    assert transitions == [
        "SHIFT",
        "SHIFT",
        "LEFT-ARC(det)",
        "SHIFT",
        "LEFT-ARC(nsubj)",
        "RIGHT-ARC(root)",
    ]

def test_parse_conllu():
    content = """# sent_id = 1
1\tThe\t_\tDET\t_\t_\t2\tdet\t_\t_
2\tcat\t_\tNOUN\t_\t_\t3\tnsubj\t_\t_
3\tsat\t_\tVERB\t_\t_\t0\troot\t_\t_
4-5\tcan't\t_\t_\t_\t_\t_\t_\t_\t_
4\t.\t_\tPUNCT\t_\t_\t3\tpunct\t_\t_

"""
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        suffix=".conllu",
        delete=False,
    ) as file:
        file.write(content)
        path = file.name
    try:
        sentences = parse_conllu(path)
        assert len(sentences) == 1
        assert [
            (token.id, token.word, token.pos, token.head, token.deprel)
            for token in sentences[0].tokens
        ] == [
            (1, "The", "DET", 2, "det"),
            (2, "cat", "NOUN", 3, "nsubj"),
            (3, "sat", "VERB", 0, "root"),
            (4, ".", "PUNCT", 3, "punct"),
        ]
    finally:
        Path(path).unlink()
