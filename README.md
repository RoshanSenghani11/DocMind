# DocMind — Chat with your PDFs (RAG-based Q&A System)

DocMind lets you upload any PDF and ask questions about it in natural language.
It uses Retrieval-Augmented Generation (RAG) to ground every answer in the
actual document content — reducing hallucination and showing exactly which
part of the document was used to answer.

## How it works

1. **Ingestion** — PDF is parsed and split into overlapping text chunks.
2. **Embeddings** — Each chunk is converted into a vector using a local
   sentence-transformer model (`all-MiniLM-L6-v2`).
3. **Vector Search** — ChromaDB stores the vectors and retrieves the most
   relevant chunks for any given question.
4. **Generation** — The retrieved chunks + question are sent to an LLM
   (Llama 3.1 via Groq's free API) to generate a grounded answer.
5. **Source citations** — The app shows which chunks were used to answer.

## Tech Stack

- Python
- Streamlit (frontend)
- ChromaDB (vector database)
- Sentence-Transformers (embeddings)
- Groq API + Llama 3.1 (LLM inference)
- pypdf (PDF parsing)

## Setup

```bash
git clone https://github.com/RoshanSenghani11/DocMind
cd docmind
pip install -r requirements.txt
```

Get a free API key from [console.groq.com](https://console.groq.com) and set it:

```bash
export GROQ_API_KEY="your_key_here"
```

## Run

```bash
cd app
streamlit run main.py
```

## Project Structure

```
docmind/
├── app/
│   ├── main.py          # Streamlit UI
│   ├── ingest.py         # PDF loading + chunking
│   ├── vectorstore.py    # Embeddings + ChromaDB
│   └── generate.py       # LLM prompt + answer generation
├── requirements.txt
└── README.md
```

## Future Improvements

- Support multiple file formats (docx, txt, web pages)
- Multi-document chat (query across several files at once)
- Conversation memory (follow-up questions)
- Dockerize + deploy to HuggingFace Spaces

## Why this project

Built to demonstrate practical understanding of Retrieval-Augmented
Generation — an architecture used in real-world products like ChatPDF,
Notion AI, and enterprise document assistants.

## Live Demo Link of DocMind

RAG-based Q&A system that lets you chat with your PDFs using semantic search and LLM-powered answers with source citations.
```
https://docmind-2611.streamlit.app/
```