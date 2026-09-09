# -*- coding: utf-8 -*-
"""Question 2: transition-based dependency parser."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class Token:
    id: int
    word: str
    pos: str
    head: int
    deprel: str


@dataclass
class Sentence:
    tokens: list[Token]

    def token_map(self) -> dict[int, Token]:
        return {token.id: token for token in self.tokens}


@dataclass
class ParserState:
    stack: list[int]
    buffer: list[int]
    arcs: list[tuple[int, int, str]]


@dataclass
class OracleExample:
    stack: tuple[int, ...]
    buffer: tuple[int, ...]
    transition: str


def parse_conllu(path: str | Path) -> list[Sentence]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Could not find CoNLL-U file: {path}")

    sentences: list[Sentence] = []
    current_tokens: list[Token] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                if current_tokens:
                    sentences.append(Sentence(current_tokens))
                    current_tokens = []
                continue

            if line.startswith("#"):
                continue

            columns = line.split("\t")

            if len(columns) != 10:
                continue

            token_id = columns[0]

            if "-" in token_id or "." in token_id:
                continue

            token = Token(
                id=int(columns[0]),
                word=columns[1],
                pos=columns[3],
                head=int(columns[6]),
                deprel=columns[7],
            )

            current_tokens.append(token)

    if current_tokens:
        sentences.append(Sentence(current_tokens))

    return sentences


def shift(state: ParserState) -> None:
    if not state.buffer:
        raise ValueError("Cannot SHIFT: buffer is empty.")

    token_id = state.buffer.pop(0)
    state.stack.append(token_id)


def left_arc(
    state: ParserState,
    label: str,
) -> tuple[int, int, str]:
    if len(state.stack) < 2:
        raise ValueError("Cannot LEFT-ARC: stack has fewer than 2 items.")

    second = state.stack[-2]
    top = state.stack[-1]

    if second == 0:
        raise ValueError("Cannot LEFT-ARC with ROOT as the dependent.")

    arc = (top, second, label)

    state.stack.pop(-2)

    state.arcs.append(arc)

    return arc


def right_arc(
    state: ParserState,
    label: str,
) -> tuple[int, int, str]:
    if len(state.stack) < 2:
        raise ValueError("Cannot RIGHT-ARC: stack has fewer than 2 items.")

    second = state.stack[-2]
    top = state.stack[-1]

    if top == 0:
        raise ValueError("Cannot RIGHT-ARC with ROOT as the dependent.")

    arc = (second, top, label)
    state.stack.pop()
    state.arcs.append(arc)

    return arc


def apply_transition(
    state: ParserState,
    transition: str,
) -> tuple[int, int, str] | None:

    if transition == "SHIFT":
        shift(state)
        return None

    if transition.startswith("LEFT-ARC("):
        label = transition[len("LEFT-ARC("):-1]
        return left_arc(state, label)

    if transition.startswith("RIGHT-ARC("):
        label = transition[len("RIGHT-ARC("):-1]
        return right_arc(state, label)

    raise ValueError(f"Unknown transition: {transition}")


def get_gold_children(sentence: Sentence) -> dict[int, set[int]]:

    children: dict[int, set[int]] = {
        token.id: set() for token in sentence.tokens
    }

    # ROOT has ID 0.
    children[0] = set()

    for token in sentence.tokens:
        children[token.head].add(token.id)

    return children


def all_dependents_attached(
    dependent_id: int,
    gold_children: dict[int, set[int]],
    attached_dependents: set[int],
) -> bool:

    return gold_children[dependent_id].issubset(attached_dependents)


def oracle_transition(
    state: ParserState,
    sentence: Sentence,
    gold_children: dict[int, set[int]],
    attached_dependents: set[int],
) -> str | None:

    if not state.buffer and state.stack == [0]:
        return None

    token_map = sentence.token_map()


    if len(state.stack) >= 2:
        second = state.stack[-2]
        top = state.stack[-1]

        if (
            second != 0
            and token_map[second].head == top
            and all_dependents_attached(
                second,
                gold_children,
                attached_dependents,
            )
        ):
            label = token_map[second].deprel
            return f"LEFT-ARC({label})"


    if len(state.stack) >= 2:
        second = state.stack[-2]
        top = state.stack[-1]

        if (
            top != 0
            and token_map[top].head == second
            and all_dependents_attached(
                top,
                gold_children,
                attached_dependents,
            )
        ):
            label = token_map[top].deprel
            return f"RIGHT-ARC({label})"

    if state.buffer:
        return "SHIFT"
    return None


def generate_oracle_examples(
    sentence: Sentence,
) -> list[OracleExample]:

    gold_children = get_gold_children(sentence)
    state = ParserState(
        stack=[0],
        buffer=[token.id for token in sentence.tokens],
        arcs=[],
    )

    examples: list[OracleExample] = []

    attached_dependents: set[int] = set()
    max_steps = 2 * len(sentence.tokens) + 10

    for _ in range(max_steps):
        transition = oracle_transition(
            state,
            sentence,
            gold_children,
            attached_dependents,
        )

        if transition is None:
            if not state.buffer and state.stack == [0]:
                return examples

            raise ValueError(
                "Oracle became stuck. "
                "The sentence may be non-projective or incompatible "
                "with the basic arc-standard transition system."
            )

        examples.append(
            OracleExample(
                stack=tuple(state.stack),
                buffer=tuple(state.buffer),
                transition=transition,
            )
        )

        arc = apply_transition(state, transition)

        if arc is not None:
            _, dependent, _ = arc
            attached_dependents.add(dependent)

    raise ValueError("Oracle exceeded the maximum number of parsing steps.")


def get_pos(
    token_id: int | None,
    token_map: dict[int, Token],
) -> str:

    if token_id is None:
        return "NONE"

    if token_id == 0:
        return "ROOT"

    return token_map[token_id].pos


def extract_features(
    state: ParserState,
    sentence: Sentence,
) -> list[str]:

    token_map = sentence.token_map()

    stack_top = (
        state.stack[-1]
        if len(state.stack) >= 1
        else None
    )

    stack_second = (
        state.stack[-2]
        if len(state.stack) >= 2
        else None
    )

    buffer_first = (
        state.buffer[0]
        if len(state.buffer) >= 1
        else None
    )

    buffer_second = (
        state.buffer[1]
        if len(state.buffer) >= 2
        else None
    )

    return [
        get_pos(stack_top, token_map),
        get_pos(stack_second, token_map),
        get_pos(buffer_first, token_map),
        get_pos(buffer_second, token_map),
    ]


def build_training_data(
    sentences: list[Sentence],
) -> tuple[list[list[str]], list[str], int]:

    X: list[list[str]] = []
    y: list[str] = []

    skipped = 0

    for sentence_number, sentence in enumerate(sentences, start=1):

        try:
            examples = generate_oracle_examples(sentence)

        except ValueError:
            skipped += 1
            continue

        state = ParserState(
            stack=[0],
            buffer=[token.id for token in sentence.tokens],
            arcs=[],
        )

        for example in examples:

            features = extract_features(state, sentence)

            X.append(features)
            y.append(example.transition)

            apply_transition(state, example.transition)

        if sentence_number % 1000 == 0:
            print(
                f"Processed {sentence_number:,} / "
                f"{len(sentences):,} sentences..."
            )

    return X, y, skipped



def train_classifier(
    X: list[list[str]],
    y: list[str],
) -> Pipeline:

    model = Pipeline(
        [
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=500,
                    solver="saga",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("\nTraining classifier...")
    model.fit(X, y)

    print("Classifier training complete.")
    return model


def dependency_parse(
    sentence: Sentence,
    classifier: Pipeline,
) -> list[tuple[int, int, str]]:

    state = ParserState(
        stack=[0],
        buffer=[token.id for token in sentence.tokens],
        arcs=[],
    )

    max_steps = 2 * len(sentence.tokens) + 10

    for _ in range(max_steps):

        if not state.buffer and state.stack == [0]:
            break

        features = extract_features(state, sentence)
        predicted_transition = classifier.predict([features])[0]

        try:
            apply_transition(state, predicted_transition)

        except ValueError:
            raise ValueError(
                f"Invalid predicted transition "
                f"'{predicted_transition}' "
                f"for stack={state.stack}, "
                f"buffer={state.buffer}"
            )

    else:
        raise ValueError(
            "Parser exceeded the maximum number of steps."
        )

    return state.arcs


def print_predicted_arcs(
    sentence: Sentence,
    arcs: list[tuple[int, int, str]],
) -> None:

    token_map = sentence.token_map()

    print("\nPredicted dependency arcs:")
    print("-" * 70)

    for head, dependent, label in arcs:

        head_word = "ROOT" if head == 0 else token_map[head].word
        dependent_word = token_map[dependent].word

        print(
            f"{head:>2} ({head_word:<12}) "
            f"-> "
            f"{dependent:>2} ({dependent_word:<12}) "
            f"[{label}]"
        )

def calculate_las(
    sentence: Sentence,
    predicted_arcs: list[tuple[int, int, str]],
) -> float:

    predicted_map = {
        dependent: (head, label)
        for head, dependent, label in predicted_arcs
    }

    correct = 0
    total = len(sentence.tokens)

    for token in sentence.tokens:

        predicted = predicted_map.get(token.id)

        if predicted is None:
            continue

        predicted_head, predicted_label = predicted

        if (
            predicted_head == token.head
            and predicted_label == token.deprel
        ):
            correct += 1

    if total == 0:
        return 0.0

    return 100.0 * correct / total


def evaluate(
    sentences: list[Sentence],
    classifier: Pipeline,
) -> float:

    total_tokens = 0
    correct_tokens = 0
    skipped_sentences = 0

    for sentence_number, sentence in enumerate(
        sentences,
        start=1,
    ):

        try:
            predicted_arcs = dependency_parse(
                sentence,
                classifier,
            )

        except ValueError:
            skipped_sentences += 1
            continue

        predicted_map = {
            dependent: (head, label)
            for head, dependent, label in predicted_arcs
        }

        for token in sentence.tokens:

            total_tokens += 1

            predicted = predicted_map.get(token.id)

            if predicted is None:
                continue

            predicted_head, predicted_label = predicted

            if (
                predicted_head == token.head
                and predicted_label == token.deprel
            ):
                correct_tokens += 1

        if sentence_number % 500 == 0:
            print(
                f"Evaluated {sentence_number:,} / "
                f"{len(sentences):,} sentences..."
            )

    if total_tokens == 0:
        return 0.0

    las = 100.0 * correct_tokens / total_tokens

    print("\n" + "=" * 70)
    print("DEV SET EVALUATION")
    print("=" * 70)

    print(f"Total tokens       : {total_tokens:,}")
    print(f"Correct tokens     : {correct_tokens:,}")
    print(f"Skipped sentences  : {skipped_sentences:,}")
    print(f"LAS                : {las:.2f}%")

    return las


def make_example_sentence(
    words: list[str],
    pos_tags: list[str],
) -> Sentence:

    if len(words) != len(pos_tags):
        raise ValueError(
            "Number of words and POS tags must match."
        )

    tokens = [
        Token(
            id=i,
            word=word,
            pos=pos,
            head=0,
            deprel="",
        )
        for i, (word, pos) in enumerate(
            zip(words, pos_tags),
            start=1,
        )
    ]

    return Sentence(tokens)


def run_required_examples(
    classifier: Pipeline,
) -> None:

    examples = [
        (
            "The cat sat on the mat.",
            ["The", "cat", "sat", "on", "the", "mat", "."],
            ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN", "PUNCT"],
        ),
        (
            "She eats a green salad.",
            ["She", "eats", "a", "green", "salad", "."],
            ["PRON", "VERB", "DET", "ADJ", "NOUN", "PUNCT"],
        ),
        (
            "I saw the man with a telescope.",
            ["I", "saw", "the", "man", "with", "a", "telescope", "."],
            ["PRON", "VERB", "DET", "NOUN", "ADP", "DET", "NOUN", "PUNCT"],
        ),
    ]

    for text, words, pos_tags in examples:

        sentence = make_example_sentence(
            words,
            pos_tags,
        )

        print("\n" + "=" * 70)
        print(f"Example: {text}")
        print("=" * 70)
        print("\nTokens and POS tags:")

        for token in sentence.tokens:
            print(
                f"  {token.id:>2}  "
                f"{token.word:<12} "
                f"{token.pos}"
            )

        try:
            arcs = dependency_parse(
                sentence,
                classifier,
            )

            print_predicted_arcs(
                sentence,
                arcs,
            )

        except ValueError as error:
            print(f"\nParser error: {error}")


def main() -> None:
    train_path = Path("data/en_ewt-ud-train.conllu")
    dev_path = Path("data/en_ewt-ud-dev.conllu")
    print(f"Reading training data: {train_path}")
    train_sentences = parse_conllu(train_path)

    print(
        f"Loaded {len(train_sentences):,} training sentences."
    )

    print("\n" + "=" * 70)
    print("GENERATING TRAINING DATA")
    print("=" * 70)

    X, y, skipped = build_training_data(
        train_sentences
    )

    print("\nTraining data generation complete.")
    print(f"Training examples : {len(X):,}")
    print(f"Skipped sentences : {skipped:,}")

    classifier = train_classifier(X, y)

    print(f"\nReading development data: {dev_path}")
    dev_sentences = parse_conllu(dev_path)

    print(
        f"Loaded {len(dev_sentences):,} development sentences."
    )

    evaluate(
        dev_sentences,
        classifier,
    )

    run_required_examples(classifier)


if __name__ == "__main__":
    main()

