from src.q3_spelling_corrector import SpellingCorrector, edits1


def build_tiny_corrector():
    return SpellingCorrector(real_word_margin=0.5).fit(
        [
            ["i", "would", "like", "to", "see", "the", "world"],
            ["i", "would", "like", "to", "see", "the", "world"],
            ["the", "sea", "is", "blue"],
            ["a", "good", "sentence"],
            ["a", "good", "sentence"],
        ]
    )


def test_edit_distance_one_generation_includes_required_edit_types():
    generated = edits1("abc")
    assert "ab" in generated
    assert "acb" in generated
    assert "adc" in generated
    assert "abbc" in generated


def test_symmetric_delete_finds_deletion_error_candidate():
    corrector = SpellingCorrector().fit([["hello"], ["hello"], ["help"]])
    assert "hello" in corrector.candidates_symmetric_delete("hell")


def test_non_word_correction_uses_unigram_frequency():
    corrector = build_tiny_corrector()
    corrected, _ = corrector.correct_non_word("sentnce", method="edit")
    assert corrected == "sentence"


def test_real_word_correction_uses_bigram_context():
    corrector = build_tiny_corrector()
    result = corrector.correct_sentence("I would like to sea the world.", method="edit")
    assert result.corrected_text == "I would like to see the world."
    assert result.changes[0].kind == "real-word"
