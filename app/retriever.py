from typing import List, Optional
from app.vectorstore import VectorStore
from app.router import ModelRouter
from app.models import ChunkReference

QUERY_EXPANSION_PROMPT = """You are a query expansion assistant for a document retrieval system.
Given a user question, generate an expanded version that:
1. Includes relevant synonyms and related terms
2. Clarifies the intent
3. Maintains the original meaning

Respond with ONLY the expanded query, nothing else."""


class Retriever:
    def __init__(self, vector_store: VectorStore, router: ModelRouter):
        self.vector_store = vector_store
        self.router = router

    async def expand_query(self, query: str) -> str:
        """Use LLM to expand the query for better retrieval."""
        messages = [
            {"role": "system", "content": QUERY_EXPANSION_PROMPT},
            {"role": "user", "content": f"Expand this query: {query}"},
        ]

        result = await self.router.complete(
            messages=messages,
            complexity="simple",
            max_tokens=200,
            temperature=0.3,
        )

        return result["content"].strip()

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        expand: bool = True,
        doc_filter: Optional[str] = None,
    ) -> List[ChunkReference]:
        """Retrieve relevant chunks using query expansion and vector search."""

        # Step 1: Expand query for better retrieval
        search_query = query
        if expand:
            try:
                search_query = await self.expand_query(query)
            except Exception:
                search_query = query  # Fallback to original

        # Step 2: Vector search with expanded query
        results = self.vector_store.search(
            query=search_query,
            top_k=top_k * 2,  # Get extra candidates for diversity
            doc_filter=doc_filter,
        )

        # Step 3: Also search with original query for different perspectives
        if expand and search_query != query:
            original_results = self.vector_store.search(
                query=query,
                top_k=top_k,
                doc_filter=doc_filter,
            )

            seen_contents = {r["content"] for r in results}
            for r in original_results:
                if r["content"] not in seen_contents:
                    results.append(r)
                    seen_contents.add(r["content"])

        # Step 4: Sort by relevance and take top_k
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        results = results[:top_k]

        # Convert to ChunkReference objects
        return [
            ChunkReference(
                document_source=r["metadata"]["filename"],
                chunk_index=r["metadata"]["chunk_index"],
                content=r["content"],
                relevance_score=r["relevance_score"],
            )
            for r in results
        ]
