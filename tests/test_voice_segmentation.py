from guidebot.voice.segmentation import TextSegmenter


def test_segmenter_emits_speakable_phrases_from_arbitrary_tokens() -> None:
    segmenter = TextSegmenter(min_chars=4, max_chars=20)

    first = segmenter.feed("你好，今天")
    second = segmenter.feed("过得怎么样？我很好")
    final = segmenter.flush()

    assert first == ()
    assert second == ("你好，今天过得怎么样？",)
    assert final == "我很好"
