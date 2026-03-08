"""End-to-end test script for the Policy-Aware AI Decision Agent.

Run the FastAPI server first:
    uvicorn app.main:app --port 8000

Then run this script:
    python test_e2e.py

This tests the full pipeline: document upload, retrieval, and agent decisions.
"""
import httpx
import json
import sys
import time
import os

BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
API = f"{BASE_URL}/api"

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name} -- {detail}")


def main():
    global passed, failed
    client = httpx.Client(timeout=120.0)

    # -------------------------------------------------------
    print("\n=== 1. Health Check ===")
    # -------------------------------------------------------
    resp = client.get(f"{API}/health")
    check("Health endpoint returns 200", resp.status_code == 200)
    data = resp.json()
    check("Status is healthy", data["status"] == "healthy")
    check("Models listed", len(data["models_available"]) == 2, str(data.get("models_available")))

    # -------------------------------------------------------
    print("\n=== 2. Document Upload ===")
    # -------------------------------------------------------
    docs_dir = os.path.join(os.path.dirname(__file__), "example_documents")

    # Upload payment policy
    with open(os.path.join(docs_dir, "payment_policy.md"), "rb") as f:
        resp = client.post(f"{API}/documents/upload", files={"file": ("payment_policy.md", f, "text/markdown")})
    check("Upload payment_policy.md returns 200", resp.status_code == 200)
    policy = resp.json()
    check("Policy has chunks", policy["chunk_count"] > 0, f"chunks={policy.get('chunk_count')}")

    # Upload sample invoice
    with open(os.path.join(docs_dir, "sample_invoice.txt"), "rb") as f:
        resp = client.post(f"{API}/documents/upload", files={"file": ("sample_invoice.txt", f, "text/plain")})
    check("Upload sample_invoice.txt returns 200", resp.status_code == 200)
    invoice = resp.json()
    check("Invoice has chunks", invoice["chunk_count"] > 0, f"chunks={invoice.get('chunk_count')}")

    # Upload supplier offer
    with open(os.path.join(docs_dir, "supplier_offer.txt"), "rb") as f:
        resp = client.post(f"{API}/documents/upload", files={"file": ("supplier_offer.txt", f, "text/plain")})
    check("Upload supplier_offer.txt returns 200", resp.status_code == 200)
    offer = resp.json()
    check("Offer has chunks", offer["chunk_count"] > 0, f"chunks={offer.get('chunk_count')}")

    # List documents
    resp = client.get(f"{API}/documents")
    check("List documents returns 200", resp.status_code == 200)
    docs = resp.json()
    check("All 3 documents indexed", len(docs) >= 3, f"count={len(docs)}")

    # -------------------------------------------------------
    print("\n=== 3. Validation Tests ===")
    # -------------------------------------------------------
    resp = client.post(f"{API}/query", json={"question": ""})
    check("Empty question rejected (400)", resp.status_code == 400)

    resp = client.post(f"{API}/documents/upload", files={"file": ("bad.xyz", b"data", "application/octet-stream")})
    check("Unsupported format rejected (400)", resp.status_code == 400)

    resp = client.post(f"{API}/documents/upload", files={"file": ("empty.txt", b"", "text/plain")})
    check("Empty file rejected (400)", resp.status_code == 400)

    # -------------------------------------------------------
    print("\n=== 4. Agent Decision - Compliance Check ===")
    # -------------------------------------------------------
    resp = client.post(f"{API}/query", json={
        "question": "Does the invoice from TechSupply Corp include a valid Purchase Order number and vendor address as required by the payment policy?"
    })
    check("Query returns 200", resp.status_code == 200)
    data = resp.json()

    # Validate response structure
    check("Has answer field", "answer" in data)
    check("Has decision field", "decision" in data)
    check("Has model_usage field", "model_usage" in data)

    decision = data["decision"]
    check("Decision is PASS/FAIL/NEEDS_INFO", decision["decision"] in ("PASS", "FAIL", "NEEDS_INFO"),
          decision["decision"])
    check("Has reasons", len(decision["reasons"]) > 0)
    check("Has evidence", len(decision["evidence"]) > 0)
    check("Confidence is 0-1", 0 <= decision["confidence"] <= 1, str(decision["confidence"]))
    check("Has reasoning steps", len(decision["reasoning_steps"]) == 5,
          f"steps={len(decision.get('reasoning_steps', []))}")

    # Validate evidence structure
    if decision["evidence"]:
        ev = decision["evidence"][0]
        check("Evidence has document_source", "document_source" in ev)
        check("Evidence has content", "content" in ev and len(ev["content"]) > 0)
        check("Evidence has relevance_score", "relevance_score" in ev)

    # Validate model usage
    usage = data["model_usage"]
    check("Model name logged", len(usage["model"]) > 0)
    check("Input tokens logged", usage["tokens_input"] > 0, str(usage["tokens_input"]))
    check("Output tokens logged", usage["tokens_output"] > 0, str(usage["tokens_output"]))
    check("Latency logged", usage["latency_ms"] > 0, f"{usage['latency_ms']}ms")

    print(f"\n    Agent decided: {decision['decision']} (confidence: {decision['confidence']:.0%})")
    print(f"    Model: {usage['model']} | Tokens: {usage['tokens_input']}in/{usage['tokens_output']}out | Latency: {usage['latency_ms']:.0f}ms")

    # -------------------------------------------------------
    print("\n=== 5. Agent Decision - Threshold Check ===")
    # -------------------------------------------------------
    resp = client.post(f"{API}/query", json={
        "question": "Does the GlobalTech supplier offer exceed the $50,000 approval threshold?"
    })
    check("Threshold query returns 200", resp.status_code == 200)
    data2 = resp.json()
    check("Decision structure valid", data2["decision"]["decision"] in ("PASS", "FAIL", "NEEDS_INFO"))
    check("Evidence cites supplier offer", any(
        "supplier_offer" in e["document_source"] for e in data2["decision"]["evidence"]
    ))
    check("Evidence cites payment policy", any(
        "payment_policy" in e["document_source"] for e in data2["decision"]["evidence"]
    ))

    print(f"\n    Agent decided: {data2['decision']['decision']} (confidence: {data2['decision']['confidence']:.0%})")

    # -------------------------------------------------------
    print("\n=== 6. Multi-Model Routing Logs ===")
    # -------------------------------------------------------
    resp = client.get(f"{API}/logs")
    check("Logs endpoint returns 200", resp.status_code == 200)
    logs = resp.json()
    check("Logs are recorded", len(logs) > 0, f"count={len(logs)}")

    models_used = set(log["model"] for log in logs)
    check("Multiple models used", len(models_used) >= 2, str(models_used))

    total_in = sum(log["tokens_input"] for log in logs)
    total_out = sum(log["tokens_output"] for log in logs)
    print(f"\n    Models used: {models_used}")
    print(f"    Total API calls: {len(logs)}")
    print(f"    Total tokens: {total_in} input + {total_out} output = {total_in + total_out}")

    # -------------------------------------------------------
    print("\n=== 7. Document Deletion ===")
    # -------------------------------------------------------
    resp = client.delete(f"{API}/documents/{offer['doc_id']}")
    check("Delete returns 200", resp.status_code == 200)
    resp = client.get(f"{API}/documents")
    remaining = resp.json()
    check("Document removed", len(remaining) < len(docs), f"remaining={len(remaining)}")

    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------
    total = passed + failed
    print(f"\n{'='*50}")
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    print(f"{'='*50}")

    if failed > 0:
        sys.exit(1)
    else:
        print("\n  All tests passed! The system is working correctly.\n")


if __name__ == "__main__":
    main()
