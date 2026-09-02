"""
vectorstore.py
Handles: converting text chunks into embeddings (vectors) and storing them
in ChromaDB (a free, local vector database) for fast similarity search.
"""

import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# Free, local embedding model — no API key or cost needed.
# Runs on CPU, ~80MB, good quality for this use case.
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


class VectorStore:
    def __init__(self, persist_path: str = "./vectorstore", collection_name: str = "docmind"):
        self.model = SentenceTransformer(EMBED_MODEL_NAME)
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_chunks(self, chunks: List[Dict]):
        """Embed each chunk and store it in ChromaDB."""
        texts = [c["text"] for c in chunks]
        ids = [c["id"] for c in chunks]

        embeddings = self.model.encode(texts).tolist()

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts
        )
        print(f"Added {len(chunks)} chunks to vector store.")

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Find the top_k most relevant chunks for a given query."""
        query_embedding = self.model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

        matches = []
        for doc, doc_id in zip(results["documents"][0], results["ids"][0]):
            matches.append({"id": doc_id, "text": doc})

        return matches

    def reset(self):
        """Clear the collection (useful when uploading a new document)."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(name=self.collection.name)


if __name__ == "__main__":
    # quick manual test
    vs = VectorStore()
    vs.add_chunks([{"id": "test_1", "text": "The sky is blue during the day."}])
    print(vs.search("what color is the sky?"))
