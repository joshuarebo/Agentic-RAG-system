import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.api import reset_instances

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_state():
    """Reset singletons between test classes to avoid cross-contamination."""
    reset_instances()
    yield
    reset_instances()


class TestHealthAndRoot:
    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "docs" in data

    def test_health(self):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "documents_count" in data
        assert "models_available" in data
        assert len(data["models_available"]) == 2


class TestDocumentUpload:
    def test_upload_unsupported_format(self):
        response = client.post(
            "/api/documents/upload",
            files={"file": ("test.xyz", b"content", "application/octet-stream")},
        )
        assert response.status_code == 400
        assert "Unsupported" in response.json()["detail"]

    def test_upload_empty_file(self):
        response = client.post(
            "/api/documents/upload",
            files={"file": ("test.txt", b"", "text/plain")},
        )
        assert response.status_code == 400

    def test_upload_txt_file(self):
        content = b"This is a test document with enough content for chunking. " * 5
        response = client.post(
            "/api/documents/upload",
            files={"file": ("test.txt", content, "text/plain")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "doc_id" in data
        assert data["filename"] == "test.txt"
        assert data["chunk_count"] > 0

    def test_upload_md_file(self):
        content = b"# Policy\n\nAll invoices must include a PO number.\n\nPayment is due within 30 days."
        response = client.post(
            "/api/documents/upload",
            files={"file": ("policy.md", content, "text/markdown")},
        )
        assert response.status_code == 200
        assert response.json()["filename"] == "policy.md"


class TestDocumentManagement:
    def test_list_documents(self):
        response = client.get("/api/documents")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_delete_document(self):
        # Upload first
        content = b"Temporary document for deletion test with some content."
        upload = client.post(
            "/api/documents/upload",
            files={"file": ("temp.txt", content, "text/plain")},
        )
        doc_id = upload.json()["doc_id"]

        # Delete
        response = client.delete(f"/api/documents/{doc_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"


class TestQueryValidation:
    def test_query_empty_question(self):
        response = client.post("/api/query", json={"question": ""})
        assert response.status_code == 400

    def test_query_whitespace_only(self):
        response = client.post("/api/query", json={"question": "   "})
        assert response.status_code == 400


class TestLogs:
    def test_get_logs(self):
        response = client.get("/api/logs")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_logs_with_limit(self):
        response = client.get("/api/logs?limit=10")
        assert response.status_code == 200
