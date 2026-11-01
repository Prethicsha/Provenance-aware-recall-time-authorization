"""
Provenance-Aware Recall-Time Security Gate
The final, deterministic security boundary executed after memory retrieval & LLM proposal,
independent of the LLM's self-reported safety.
"""

from typing import List, Tuple, Optional
from datetime import datetime, timezone
import time

from src.memory.schema import (
    MemoryEntry,
    IntentType,
    ToolSensitivity,
    GateDecision,
    AuditRecord,
    TrustTier,
)
from src.security.policy import evaluate_security_policy
from src.security.audit import get_audit_logger
from src.memory.taint import get_effective_trust


class RecallTimeSecurityGate:
    """
    Deterministic Security Gate.
    """
    def __init__(self, audit_log_file: str = "logs/security_audit.jsonl"):
        self.audit_logger = get_audit_logger(audit_log_file)

    def evaluate_and_authorize(
        self,
        scenario_id: str,
        user_query: str,
        user_intent: IntentType,
        proposed_tool: str,
        retrieved_memories: List[MemoryEntry],
        memories_used_ids: Optional[List[str]] = None,
        conflicts_detected: Optional[List[str]] = None,
        semantic_detector_flagged: bool = False,
        is_adversarial_scenario: bool = False
    ) -> Tuple[GateDecision, str, AuditRecord]:
        """
        Executes recall-time gate evaluation and produces an immutable audit record.
        """
        # Determine tool sensitivity dynamically
        from src.agent.tools import get_tool
        tool_meta = get_tool(proposed_tool)
        tool_sens = tool_meta.sensitivity if tool_meta else ToolSensitivity.LOW

        decision, reason = evaluate_security_policy(
            retrieved_memories=retrieved_memories,
            user_intent=user_intent,
            proposed_tool=proposed_tool,
            tool_sensitivity=tool_sens,
            memories_used_ids=memories_used_ids,
            conflicts_detected=conflicts_detected
        )

        effective_trust = get_effective_trust(retrieved_memories) if retrieved_memories else TrustTier.UNTRUSTED_EXTERNAL
        ground_truth_poisoned = any(m.is_poisoned for m in retrieved_memories)

        # Collect merged taint chains
        merged_taints = []
        for m in retrieved_memories:
            merged_taints.extend(m.taint_chain)

        audit_rec = AuditRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            scenario_id=scenario_id,
            user_query=user_query,
            user_intent=user_intent,
            retrieved_memory_ids=[m.memory_id for m in retrieved_memories],
            effective_trust=effective_trust,
            proposed_tool=proposed_tool,
            tool_sensitivity=tool_sens,
            gate_decision=decision,
            reason=reason,
            taint_chain=list(set(merged_taints)),
            is_adversarial_scenario=is_adversarial_scenario,
            ground_truth_poisoned=ground_truth_poisoned,
            semantic_detector_flagged=semantic_detector_flagged,
            policy_violated=(decision == GateDecision.BLOCK)
        )

        self.audit_logger.log_decision(audit_rec)
        return decision, reason, audit_rec
