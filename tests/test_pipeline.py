from app.pipeline.chunker import chunk_sentences, format_timestamp
from app.pipeline.cleaner import Sentence, clean_caption_text, sentences_from_segments
from app.pipeline.youtube import RawSegment, extract_video_id


def test_extract_video_id_watch_and_short() -> None:
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_clean_caption_text_strips_fillers() -> None:
    text = "Um, inheritance  is useful, you know, in OOP."
    cleaned = clean_caption_text(text)
    assert "um" not in cleaned.lower()
    assert "you know" not in cleaned.lower()
    assert "inheritance is useful" in cleaned.lower()


def test_sentences_preserve_timestamps() -> None:
    segments = [
        RawSegment("Classes are blueprints.", 3.0, 2.0),
        RawSegment("Objects are instances of classes.", 6.0, 3.0),
    ]
    sentences, full = sentences_from_segments(segments)
    assert "Classes are blueprints." in full
    assert sentences[0].start == 3.0
    assert sentences[-1].end == 9.0


def test_chunking_keeps_sentence_boundaries_and_times() -> None:
    sentences = [
        Sentence("This is the introduction to object oriented programming in Python.", 0.0, 8.0),
        Sentence("We will first talk about classes as reusable blueprints for objects.", 8.2, 16.0),
        Sentence("Then we will look at inheritance as a way to reuse existing behavior.", 28.0, 36.0),
        Sentence("Finally we cover polymorphism and how it relates to inheritance.", 36.5, 44.0),
    ]
    chunks = chunk_sentences(sentences)
    assert chunks
    assert all(chunk.word_count > 0 for chunk in chunks)
    assert chunks[0].start_label == format_timestamp(chunks[0].start_timestamp)
    joined = " ".join(c.text for c in chunks)
    assert "blueprints" in joined
    assert "polymorphism" in joined
