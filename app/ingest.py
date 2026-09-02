"""
ingest.py
Handles: loading a PDF/text file, extracting text, and splitting it into
overlapping chunks that are small enough for the embedding model + LLM
to work with effectively.
"""

from pypdf import PdfReader
from typing import List, Dict


def load_pdf_text(file_path: str) -> str:
    """Extract raw text from a PDF file, page by page."""
    reader = PdfReader(file_path)
    full_text = ""
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        full_text += f"\n[PAGE {i + 1}]\n{text}"
    return full_text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """
    Split text into overlapping chunks.

    chunk_size: approx number of words per chunk
    overlap: number of words repeated between consecutive chunks
             (this helps preserve context across chunk boundaries)
    """
    words = text.split()
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words)

        chunks.append({
            "id": f"chunk_{chunk_id}",
            "text": chunk_str
        })

        chunk_id += 1
        start += chunk_size - overlap  # slide window forward with overlap

    return chunks


def process_document(file_path: str) -> List[Dict]:
    """Full pipeline: load PDF -> extract text -> chunk it."""
    raw_text = load_pdf_text(file_path)
    chunks = chunk_text(raw_text)
    print(f"Processed '{file_path}': {len(chunks)} chunks created.")
    return chunks


if __name__ == "__main__":
    # quick manual test
    import sys
    if len(sys.argv) > 1:
        result = process_document(sys.argv[1])
        print(result[0])  # preview first chunk
