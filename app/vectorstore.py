import os
from typing import List, Dict, Optional
import chromadb


class VectorStore:
    def __init__(self, persist_dir: str = "./chroma_data"):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )

    def _make_chunk_id(self, doc_id: str, index: int) -> str:
        return f"{doc_id}_chunk_{index}"

    def add_document(self, doc_id: str, filename: str, chunks: List[str]) -> int:
        if not chunks:
            return 0

        ids = [self._make_chunk_id(doc_id, i) for i in range(len(chunks))]
        metadatas = [
            {"doc_id": doc_id, "filename": filename, "chunk_index": i}
            for i in range(len(chunks))
        ]

        self.collection.add(ids=ids, documents=chunks, metadatas=metadatas)
        return len(chunks)

    def search(
        self, query: str, top_k: int = 5, doc_filter: Optional[str] = None
    ) -> List[Dict]:
        if self.collection.count() == 0:
            return []

        where_filter = {"doc_id": doc_filter} if doc_filter else None

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        items = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                distance = results["distances"][0][i]
                # ChromaDB cosine distance: 0=identical, 2=opposite
                similarity = 1 - (distance / 2)
                items.append(
                    {
                        "content": doc,
                        "metadata": results["metadatas"][0][i],
                        "relevance_score": round(similarity, 4),
                    }
                )

        return items

    def delete_document(self, doc_id: str):
        results = self.collection.get(where={"doc_id": doc_id}, include=[])
        if results["ids"]:
            self.collection.delete(ids=results["ids"])

    def list_documents(self) -> List[Dict]:
        all_items = self.collection.get(include=["metadatas"])
        docs: Dict[str, Dict] = {}
        for meta in all_items["metadatas"]:
            doc_id = meta["doc_id"]
            if doc_id not in docs:
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "filename": meta["filename"],
                    "chunk_count": 0,
                }
            docs[doc_id]["chunk_count"] += 1
        return list(docs.values())

    def get_document_count(self) -> int:
        return len(self.list_documents())

    def get_total_chunks(self) -> int:
        return self.collection.count()
