import json
import re
import time
from typing import List, Optional
from app.models import (
    DecisionResult,
    DecisionEnum,
    ChunkReference,
    ReasoningStep,
    ModelUsage,
    QueryResponse,
)
from app.retriever import Retriever
from app.router import ModelRouter
from app.config import get_settings

DECISION_SYSTEM_PROMPT = """You are a Policy-Aware AI Decision Agent. Your role is to analyze documents, policies, and evidence to produce structured decisions.

You MUST respond with valid JSON in this exact format:
{
    "decision": "PASS" | "FAIL" | "NEEDS_INFO",
    "reasons": ["reason1", "reason2"],
    "evidence_indices": [0, 1],
    "confidence": 0.85,
    "answer": "A clear explanation of your decision"
}

Rules:
- PASS: The query condition is satisfied based on evidence
- FAIL: The query condition is NOT satisfied based on evidence
- NEEDS_INFO: Insufficient evidence to make a determination
- confidence must be between 0.0 and 1.0
- evidence_indices must reference the chunk numbers provided in the context
- If confidence is below 0.6, set decision to NEEDS_INFO
- Every claim must be backed by evidence from the provided chunks
- Do NOT hallucinate or invent information not in the evidence
- Be thorough but concise in your reasoning"""


class DecisionAgent:
    def __init__(self, retriever: Retriever, router: ModelRouter):
        self.retriever = retriever
        self.router = router
        self.settings = get_settings()

    async def analyze(
        self,
        question: str,
        model_preference: Optional[str] = None,
        doc_filter: Optional[str] = None,
    ) -> QueryResponse:
        """Run the full Reason -> Evaluate -> Execute -> Explain pipeline."""
        steps: List[ReasoningStep] = []
        total_start = time.time()

        # Step 1: Identify intent
        steps.append(
            ReasoningStep(
                step_number=1,
                action="identify_intent",
                detail="Analyzing user question to determine intent and required documents",
                result=f"Question: {question}",
            )
        )

        # Step 2: Retrieve relevant documents
        steps.append(
            ReasoningStep(
                step_number=2,
                action="retrieve_documents",
                detail="Searching vector database for relevant document chunks",
            )
        )

        chunks = await self.retriever.retrieve(
            query=question,
            top_k=self.settings.top_k_final,
            expand=True,
            doc_filter=doc_filter,
        )

        unique_sources = set(c.document_source for c in chunks)
        steps[-1].result = (
            f"Retrieved {len(chunks)} relevant chunks from {len(unique_sources)} document(s)"
        )

        # Step 3: Evaluate evidence quality
        steps.append(
            ReasoningStep(
                step_number=3,
                action="evaluate_evidence",
                detail="Assessing quality and relevance of retrieved evidence",
            )
        )

        if not chunks:
            steps[-1].result = "No relevant documents found in the knowledge base"
            elapsed = round((time.time() - total_start) * 1000, 2)
            return QueryResponse(
                answer="I could not find any relevant documents to answer your question. Please upload the relevant documents first.",
                decision=DecisionResult(
                    decision=DecisionEnum.NEEDS_INFO,
                    reasons=["No relevant documents found in the knowledge base"],
                    evidence=[],
                    confidence=0.0,
                    reasoning_steps=steps,
                ),
                model_usage=ModelUsage(
                    model="none",
                    tokens_input=0,
                    tokens_output=0,
                    latency_ms=elapsed,
                ),
            )

        avg_relevance = sum(c.relevance_score for c in chunks) / len(chunks)
        steps[-1].result = f"Average relevance score: {avg_relevance:.4f}"

        # Step 4: Compare and analyze
        steps.append(
            ReasoningStep(
                step_number=4,
                action="analyze_compliance",
                detail="Comparing document contents against query requirements",
            )
        )

        context = self._build_context(chunks)

        # Step 5: Generate decision
        steps.append(
            ReasoningStep(
                step_number=5,
                action="generate_decision",
                detail="Producing structured decision with evidence and confidence",
            )
        )

        messages = [
            {"role": "system", "content": DECISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"## Retrieved Evidence\n\n{context}\n\n"
                    f"## Question\n\n{question}\n\n"
                    "Analyze the evidence and produce your decision as JSON."
                ),
            },
        ]

        result = await self.router.complete(
            messages=messages,
            model=model_preference,
            complexity="normal",
            temperature=0.1,
            max_tokens=self.settings.max_tokens,
        )

        model_usage = result["model_usage"]

        decision_data = self._parse_decision(result["content"], chunks)
        decision_data = self._enforce_governance(decision_data, chunks)

        steps[3].result = f"Analysis complete. Found {len(decision_data['reasons'])} reasons."
        steps[4].result = (
            f"Decision: {decision_data['decision']} "
            f"(confidence: {decision_data['confidence']:.0%})"
        )

        decision_result = DecisionResult(
            decision=DecisionEnum(decision_data["decision"]),
            reasons=decision_data["reasons"],
            evidence=decision_data["evidence"],
            confidence=decision_data["confidence"],
            reasoning_steps=steps,
        )

        return QueryResponse(
            answer=decision_data["answer"],
            decision=decision_result,
            model_usage=model_usage,
        )

    def _build_context(self, chunks: List[ChunkReference]) -> str:
        parts = []
        for i, chunk in enumerate(chunks):
            parts.append(
                f"[Chunk {i}] Source: {chunk.document_source} | "
                f"Relevance: {chunk.relevance_score:.4f}\n"
                f"{chunk.content}"
            )
        return "\n\n---\n\n".join(parts)

    def _parse_decision(
        self, response: str, chunks: List[ChunkReference]
    ) -> dict:
        data = None
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

        if data is None:
            return {
                "decision": "NEEDS_INFO",
                "reasons": ["Could not parse AI response into structured format"],
                "evidence": chunks[:3],
                "confidence": 0.3,
                "answer": response,
            }

        # Map evidence indices to actual chunks
        evidence_indices = data.get(
            "evidence_indices", list(range(min(3, len(chunks))))
        )
        evidence = [
            chunks[idx]
            for idx in evidence_indices
            if isinstance(idx, int) and 0 <= idx < len(chunks)
        ]
        if not evidence and chunks:
            evidence = chunks[:3]

        return {
            "decision": data.get("decision", "NEEDS_INFO"),
            "reasons": data.get("reasons", ["No reasons provided"]),
            "evidence": evidence,
            "confidence": float(data.get("confidence", 0.5)),
            "answer": data.get("answer", response),
        }

    def _enforce_governance(
        self, decision_data: dict, chunks: List[ChunkReference]
    ) -> dict:
        confidence = decision_data["confidence"]

        # Low confidence -> NEEDS_INFO
        if confidence < self.settings.confidence_threshold:
            decision_data["decision"] = "NEEDS_INFO"
            threshold_msg = (
                f"Confidence ({confidence:.0%}) is below the required "
                f"threshold ({self.settings.confidence_threshold:.0%})"
            )
            if threshold_msg not in str(decision_data["reasons"]):
                decision_data["reasons"].append(threshold_msg)

        # Must have evidence
        if not decision_data["evidence"]:
            decision_data["decision"] = "NEEDS_INFO"
            decision_data["reasons"].append("No supporting evidence found")
            decision_data["confidence"] = min(decision_data["confidence"], 0.3)

        # Clamp confidence
        decision_data["confidence"] = max(0.0, min(1.0, decision_data["confidence"]))

        # Validate decision value
        if decision_data["decision"] not in ("PASS", "FAIL", "NEEDS_INFO"):
            decision_data["decision"] = "NEEDS_INFO"

        return decision_data
