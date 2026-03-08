import pytest
import tempfile
import shutil
from app.vectorstore import VectorStore


@pytest.fixture
def temp_store():
    temp_dir = tempfile.mkdtemp()
    store = VectorStore(persist_dir=temp_dir)
    yield store
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestVectorStore:
    def test_add_and_search(self, temp_store):
        chunks = [
            "The payment policy requires invoices to be submitted within 30 days.",
            "All invoices must include a purchase order number.",
            "Late payments incur a 2% penalty fee.",
        ]
        temp_store.add_document("doc1", "policy.txt", chunks)

        results = temp_store.search("payment deadline", top_k=2)
        assert len(results) > 0
        assert results[0]["relevance_score"] > 0

    def test_add_empty_chunks(self, temp_store):
        count = temp_store.add_document("doc1", "empty.txt", [])
        assert count == 0

    def test_list_documents(self, temp_store):
        temp_store.add_document("doc1", "file1.txt", ["chunk one", "chunk two"])
        temp_store.add_document("doc2", "file2.txt", ["chunk three"])

        docs = temp_store.list_documents()
        assert len(docs) == 2

        doc_ids = {d["doc_id"] for d in docs}
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids

    def test_delete_document(self, temp_store):
        temp_store.add_document("doc1", "file1.txt", ["chunk1", "chunk2"])
        assert temp_store.get_total_chunks() == 2

        temp_store.delete_document("doc1")
        assert temp_store.get_total_chunks() == 0

    def test_search_with_filter(self, temp_store):
        temp_store.add_document("doc1", "policy.txt", ["Payment terms are net 30."])
        temp_store.add_document("doc2", "invoice.txt", ["Invoice total: $5000."])

        results = temp_store.search("payment", top_k=5, doc_filter="doc1")
        for r in results:
            assert r["metadata"]["doc_id"] == "doc1"

    def test_empty_search(self, temp_store):
        results = temp_store.search("anything", top_k=5)
        assert len(results) == 0

    def test_document_count(self, temp_store):
        assert temp_store.get_document_count() == 0
        temp_store.add_document("doc1", "file1.txt", ["a valid chunk here"])
        assert temp_store.get_document_count() == 1

    def test_total_chunks(self, temp_store):
        assert temp_store.get_total_chunks() == 0
        temp_store.add_document("doc1", "f.txt", ["chunk a", "chunk b", "chunk c"])
        assert temp_store.get_total_chunks() == 3

    def test_relevance_scores_ordered(self, temp_store):
        temp_store.add_document(
            "doc1",
            "policy.txt",
            [
                "The payment policy requires net 30 payment terms.",
                "Office supplies should be ordered on Mondays.",
                "Employee vacation policy allows 20 days per year.",
            ],
        )

        results = temp_store.search("payment terms policy", top_k=3)
        # Results should be ordered by relevance (highest first)
        for i in range(len(results) - 1):
            assert results[i]["relevance_score"] >= results[i + 1]["relevance_score"]
