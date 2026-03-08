import pytest
from unittest.mock import AsyncMock, MagicMock
from app.agent import DecisionAgent
from app.models import ChunkReference, DecisionEnum, ModelUsage


class TestDecisionAgent:
    def setup_method(self):
        self.mock_retriever = MagicMock()
        self.mock_router = MagicMock()
        self.agent = DecisionAgent(self.mock_retriever, self.mock_router)

    def _make_chunk(self, source="policy.txt", index=0, content="Test", score=0.9):
        return ChunkReference(
            document_source=source,
            chunk_index=index,
            content=content,
            relevance_score=score,
        )

    def test_build_context(self):
        chunks = [self._make_chunk(content="Payment must be within 30 days.")]
        context = self.agent._build_context(chunks)
        assert "policy.txt" in context
        assert "Payment must be within 30 days." in context
        assert "0.9" in context

    def test_build_context_multiple_chunks(self):
        chunks = [
            self._make_chunk(content="Chunk one", index=0),
            self._make_chunk(content="Chunk two", index=1),
        ]
        context = self.agent._build_context(chunks)
        assert "[Chunk 0]" in context
        assert "[Chunk 1]" in context
        assert "---" in context

    def test_parse_valid_decision(self):
        chunks = [self._make_chunk()]
        response = (
            '{"decision": "PASS", "reasons": ["Compliant"], '
            '"evidence_indices": [0], "confidence": 0.85, '
            '"answer": "The invoice is compliant."}'
        )

        result = self.agent._parse_decision(response, chunks)
        assert result["decision"] == "PASS"
        assert result["confidence"] == 0.85
        assert len(result["evidence"]) == 1

    def test_parse_json_in_markdown(self):
        chunks = [self._make_chunk()]
        response = 'Here is my analysis:\n```json\n{"decision": "FAIL", "reasons": ["Non-compliant"], "evidence_indices": [0], "confidence": 0.7, "answer": "Failed."}\n```'

        result = self.agent._parse_decision(response, chunks)
        assert result["decision"] == "FAIL"

    def test_parse_invalid_json_fallback(self):
        chunks = [self._make_chunk()]
        result = self.agent._parse_decision("This is not JSON at all", chunks)
        assert result["decision"] == "NEEDS_INFO"
        assert result["confidence"] == 0.3

    def test_parse_missing_evidence_indices(self):
        chunks = [self._make_chunk(), self._make_chunk(index=1)]
        response = '{"decision": "PASS", "reasons": ["OK"], "confidence": 0.8, "answer": "Good."}'

        result = self.agent._parse_decision(response, chunks)
        assert len(result["evidence"]) > 0

    def test_enforce_governance_low_confidence(self):
        data = {
            "decision": "PASS",
            "reasons": ["Some reason"],
            "evidence": [self._make_chunk()],
            "confidence": 0.3,
            "answer": "Test",
        }
        result = self.agent._enforce_governance(data, [])
        assert result["decision"] == "NEEDS_INFO"
        assert any("below" in r.lower() for r in result["reasons"])

    def test_enforce_governance_no_evidence(self):
        data = {
            "decision": "PASS",
            "reasons": ["Some reason"],
            "evidence": [],
            "confidence": 0.9,
            "answer": "Test",
        }
        result = self.agent._enforce_governance(data, [])
        assert result["decision"] == "NEEDS_INFO"
        assert result["confidence"] <= 0.3

    def test_enforce_governance_valid_pass(self):
        evidence = [self._make_chunk()]
        data = {
            "decision": "PASS",
            "reasons": ["Compliant"],
            "evidence": evidence,
            "confidence": 0.85,
            "answer": "Test",
        }
        result = self.agent._enforce_governance(data, evidence)
        assert result["decision"] == "PASS"
        assert result["confidence"] == 0.85

    def test_enforce_governance_invalid_decision_value(self):
        data = {
            "decision": "MAYBE",
            "reasons": ["Unclear"],
            "evidence": [self._make_chunk()],
            "confidence": 0.9,
            "answer": "Test",
        }
        result = self.agent._enforce_governance(data, [])
        assert result["decision"] == "NEEDS_INFO"

    def test_enforce_governance_clamps_confidence(self):
        data = {
            "decision": "PASS",
            "reasons": ["OK"],
            "evidence": [self._make_chunk()],
            "confidence": 1.5,
            "answer": "Test",
        }
        result = self.agent._enforce_governance(data, [])
        assert result["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_analyze_no_chunks(self):
        self.mock_retriever.retrieve = AsyncMock(return_value=[])

        response = await self.agent.analyze("Is this compliant?")

        assert response.decision.decision == DecisionEnum.NEEDS_INFO
        assert response.decision.confidence == 0.0
        assert "relevant documents" in response.answer

    @pytest.mark.asyncio
    async def test_analyze_with_chunks(self):
        chunks = [
            self._make_chunk(content="Payment terms are Net 30.", score=0.92),
        ]
        self.mock_retriever.retrieve = AsyncMock(return_value=chunks)
        self.mock_router.complete = AsyncMock(
            return_value={
                "content": (
                    '{"decision": "PASS", "reasons": ["Net 30 terms match"], '
                    '"evidence_indices": [0], "confidence": 0.88, '
                    '"answer": "The invoice complies with payment terms."}'
                ),
                "model_usage": ModelUsage(
                    model="anthropic/claude-3.5-haiku",
                    tokens_input=500,
                    tokens_output=100,
                    latency_ms=1200.0,
                    cached_tokens=0,
                    timestamp="2024-01-01T00:00:00",
                ),
            }
        )

        response = await self.agent.analyze("Does this invoice comply?")

        assert response.decision.decision == DecisionEnum.PASS
        assert response.decision.confidence == 0.88
        assert len(response.decision.reasoning_steps) == 5
