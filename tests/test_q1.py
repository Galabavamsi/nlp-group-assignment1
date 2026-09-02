from src.q1_word_segmentation import NgramLM, POSTagger, SegmentationModel, TaggedSentence


def test_morphology_tag_uses_gender_and_number():
    assert POSTagger.morphology_tag("NOUN", "Gender=Fem|Number=Sing") == "NOUN-Fem-Sg"


def test_segmentation_dynamic_programming_prefers_known_words():
    lm = NgramLM(k=0.1).fit([["the", "quick", "fox"], ["the", "quick", "dog"]])
    model = SegmentationModel(lm, max_word_length=10)
    assert model.segment("thequickfox") == ["the", "quick", "fox"]


def test_tagger_returns_one_tag_per_word():
    tagger = POSTagger().fit([
        TaggedSentence(["the", "dog"], ["DT", "NN"]),
        TaggedSentence(["a", "dog"], ["DT", "NN"]),
    ])
    words = ["the", "dog"]
    assert len(tagger.tag(words)) == len(words)
