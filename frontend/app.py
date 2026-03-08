import os
import streamlit as st
import requests
import json

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000/api")

st.set_page_config(
    page_title="Policy-Aware AI Decision Agent",
    layout="wide",
)

st.title("Policy-Aware AI Decision Agent")
st.caption("Upload documents, ask questions, get structured decisions with evidence.")

# --- Sidebar: Document Management ---
with st.sidebar:
    st.header("Document Management")

    uploaded_file = st.file_uploader(
        "Upload Document",
        type=["pdf", "txt", "md"],
        help="Supported: PDF, TXT, Markdown",
    )

    if uploaded_file and st.button("Index Document"):
        with st.spinner("Processing..."):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "application/octet-stream",
                )
            }
            try:
                resp = requests.post(f"{API_BASE}/documents/upload", files=files)
                if resp.status_code == 200:
                    d = resp.json()
                    st.success(
                        f"Indexed: {d['filename']} ({d['chunk_count']} chunks)"
                    )
                else:
                    st.error(resp.json().get("detail", "Upload failed"))
            except requests.ConnectionError:
                st.error("Cannot connect to backend. Is it running?")

    st.divider()
    st.subheader("Indexed Documents")

    try:
        docs_resp = requests.get(f"{API_BASE}/documents", timeout=5)
        if docs_resp.status_code == 200:
            docs = docs_resp.json()
            if docs:
                for doc in docs:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(f"{doc['filename']} ({doc['chunk_count']} chunks)")
                    with col2:
                        if st.button("X", key=f"del_{doc['doc_id']}"):
                            requests.delete(f"{API_BASE}/documents/{doc['doc_id']}")
                            st.rerun()
            else:
                st.info("No documents indexed yet.")
    except requests.ConnectionError:
        st.warning("Backend not available.")
    except Exception:
        st.warning("Could not load documents.")

# --- Main: Query Interface ---
st.header("Ask a Question")

question = st.text_area(
    "Enter your question:",
    placeholder="e.g., Is this invoice compliant with the payment policy?",
    height=100,
)

col1, col2 = st.columns([3, 1])
with col1:
    submit = st.button("Analyze", type="primary", use_container_width=True)
with col2:
    model_pref = st.selectbox(
        "Model",
        [None, "google/gemini-2.0-flash-001", "meta-llama/llama-3.3-70b-instruct"],
        format_func=lambda x: "Auto" if x is None else x.split("/")[-1],
    )

if submit and question:
    with st.spinner("Agent is reasoning..."):
        try:
            payload = {"question": question}
            if model_pref:
                payload["model_preference"] = model_pref

            resp = requests.post(f"{API_BASE}/query", json=payload, timeout=120)

            if resp.status_code == 200:
                data = resp.json()
                decision = data["decision"]

                # Decision banner
                dec = decision["decision"]
                colors = {"PASS": "green", "FAIL": "red", "NEEDS_INFO": "orange"}
                color = colors.get(dec, "gray")
                st.markdown(f"### Decision: :{color}[{dec}]")
                st.metric("Confidence", f"{decision['confidence']:.0%}")

                # Answer
                st.subheader("Answer")
                st.write(data["answer"])

                # Reasons
                st.subheader("Reasons")
                for reason in decision["reasons"]:
                    st.markdown(f"- {reason}")

                # Evidence
                st.subheader("Evidence")
                for ev in decision["evidence"]:
                    with st.expander(
                        f"{ev['document_source']} (chunk {ev['chunk_index']}, "
                        f"score: {ev['relevance_score']:.4f})"
                    ):
                        st.text(ev["content"])

                # Reasoning Steps
                st.subheader("Reasoning Flow")
                for step in decision["reasoning_steps"]:
                    st.markdown(f"**Step {step['step_number']}: {step['action']}**")
                    st.markdown(f"_{step['detail']}_")
                    if step.get("result"):
                        st.markdown(f"> {step['result']}")

                # Model Usage
                usage = data["model_usage"]
                st.subheader("Model Usage")
                cols = st.columns(4)
                cols[0].metric("Model", usage["model"].split("/")[-1])
                cols[1].metric("Input Tokens", usage["tokens_input"])
                cols[2].metric("Output Tokens", usage["tokens_output"])
                cols[3].metric("Latency", f"{usage['latency_ms']:.0f}ms")

                if usage.get("cached_tokens", 0) > 0:
                    st.info(
                        f"Prompt caching saved {usage['cached_tokens']} tokens!"
                    )
            else:
                st.error(resp.json().get("detail", "Query failed"))
        except requests.ConnectionError:
            st.error("Cannot connect to backend.")
        except requests.Timeout:
            st.error("Request timed out.")
