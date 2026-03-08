import pytest
from app.ingestion import parse_text, chunk_text, parse_document


class TestParseText:
    def test_parse_plain_text(self):
        assert parse_text(b"Hello World") == "Hello World"

    def test_parse_utf8(self):
        text = "Hello World".encode("utf-8")
        assert parse_text(text) == "Hello World"

    def test_parse_with_bom(self):
        text = b"\xef\xbb\xbfHello"
        result = parse_text(text)
        assert "Hello" in result


class TestChunkText:
    def test_short_text_single_chunk(self):
        text = "This is a short text with enough length to pass the filter."
        chunks = chunk_text(text, chunk_size=500)
        assert len(chunks) == 1

    def test_long_text_multiple_chunks(self):
        text = "\n\n".join([f"Paragraph {i}. " * 10 for i in range(20)])
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1

    def test_empty_text(self):
        assert chunk_text("") == []

    def test_whitespace_only(self):
        assert chunk_text("   \n\n   ") == []

    def test_respects_paragraph_boundaries(self):
        text = "First paragraph with enough content here.\n\nSecond paragraph with more content here."
        chunks = chunk_text(text, chunk_size=1000)
        assert len(chunks) == 1

    def test_filters_short_chunks(self):
        text = "Ok\n\nThis is a valid chunk with enough content to pass the minimum length filter."
        chunks = chunk_text(text, chunk_size=500)
        for chunk in chunks:
            assert len(chunk) >= 20

    def test_overlap_is_applied(self):
        # Create text that forces multiple chunks
        text = "\n\n".join([f"Section {i} content here." * 5 for i in range(10)])
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1


class TestParseDocument:
    def test_parse_txt(self):
        result = parse_document(b"Hello from a text file", "test.txt")
        assert result == "Hello from a text file"

    def test_parse_md(self):
        result = parse_document(b"# Title\n\nContent here", "test.md")
        assert "Title" in result
        assert "Content" in result

    def test_parse_markdown_extension(self):
        result = parse_document(b"# Title", "test.markdown")
        assert "Title" in result

    def test_unsupported_format(self):
        with pytest.raises(ValueError, match="Unsupported"):
            parse_document(b"data", "test.xyz")

    def test_unsupported_format_docx(self):
        with pytest.raises(ValueError, match="Unsupported"):
            parse_document(b"data", "test.docx")
