import re
import io
from typing import List
from PyPDF2 import PdfReader


def parse_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()


def parse_text(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="replace").strip()


def parse_document(file_bytes: bytes, filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext == "pdf":
        return parse_pdf(file_bytes)
    elif ext in ("txt", "md", "markdown"):
        return parse_text(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: .{ext}. Supported: pdf, txt, md")


def chunk_text(
    text: str, chunk_size: int = 500, chunk_overlap: int = 50
) -> List[str]:
    """Split text into semantic chunks based on paragraphs and sentences."""
    text = re.sub(r"\r\n", "\n", text)

    # Split by double newlines (paragraphs) or markdown headings
    sections = re.split(r"\n\s*\n|\n(?=#)", text)

    chunks = []
    current_chunk = ""

    for section in sections:
        section = section.strip()
        if not section:
            continue

        if len(current_chunk) + len(section) + 2 <= chunk_size:
            current_chunk = (
                f"{current_chunk}\n\n{section}" if current_chunk else section
            )
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                overlap_text = (
                    current_chunk[-chunk_overlap:]
                    if len(current_chunk) > chunk_overlap
                    else ""
                )
                current_chunk = (
                    f"{overlap_text} {section}" if overlap_text else section
                )
            else:
                # Section alone exceeds chunk_size — split by sentences
                sentences = re.split(r"(?<=[.!?])\s+", section)
                for sent in sentences:
                    if len(current_chunk) + len(sent) + 1 <= chunk_size:
                        current_chunk = (
                            f"{current_chunk} {sent}" if current_chunk else sent
                        )
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sent

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Filter out very short chunks (less than 20 chars)
    return [c for c in chunks if len(c) >= 20]
