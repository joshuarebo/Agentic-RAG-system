import io
from typing import List
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
    """Split text into semantic chunks using LangChain's RecursiveCharacterTextSplitter."""
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)

    # Filter out very short chunks (less than 20 chars)
    return [c for c in chunks if len(c.strip()) >= 20]
