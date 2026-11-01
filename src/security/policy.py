"""
Deterministic Security Policy Module
Defines the deterministic authorization rules based on:
1. Provenance Trust Tier (0=Untrusted External, 1=User, 2=Supervisor, 3=System Root)
2. Tool Sensitivity (LOW=1, MEDIUM=2, HIGH=3)
3. Content Type & Intent Alignment
4. Material Influence & Conflict Status
"""

from typing import List, Tuple, Optional
from src.memory.schema import (
    TrustTier,
    ToolSensitivity,
    IntentType,
    ContentType,
    GateDecision,
    MemoryEntry,
)
from src.memory.taint import get_effective_trust


class SecurityPolicy:
    """Deterministic, explainable security authorization rules."""

    @staticmethod
    def evaluate(
        retrieved_memories: List[MemoryEntry],
        user_intent: IntentType,
        proposed_tool: str,
        tool_sensitivity: ToolSensitivity,
        memories_used_ids: Optional[List[str]] = None,
        conflicts_detected: Optional[List[str]] = None
    ) -> Tuple[GateDecision, str]:
        """
        Evaluates recall-time provenance authorization.
        Returns (GateDecision.ALLOW | GateDecision.BLOCK, reason_string).
        """
        # Rule 0: If no memories were retrieved or used, evaluate user intent vs tool
        if not retrieved_memories:
            return GateDecision.ALLOW, "No persistent memory influence detected; tool action permitted."

        # Filter to memories that actively influenced the action
        active_memories = retrieved_memories
        if memories_used_ids:
            filtered = [m for m in retrieved_memories if m.memory_id in memories_used_ids]
            if filtered:
                active_memories = filtered

        effective_trust = get_effective_trust(active_memories)

        # Rule 1: Read-Only LOW sensitivity queries are universally allowed (information access)
        if tool_sensitivity == ToolSensitivity.LOW:
            return GateDecision.ALLOW, "LOW sensitivity read-only tool; recall-time authorization granted."

        # Rule 2: High/Medium Sensitivity with Untrusted External Memory (Tier 0)
        if effective_trust == TrustTier.UNTRUSTED_EXTERNAL:
            if tool_sensitivity == ToolSensitivity.HIGH:
                return (
                    GateDecision.BLOCK,
                    f"SECURITY VIOLATION: Untrusted external memory (Tier 0) attempted to authorize or influence HIGH-sensitivity tool '{proposed_tool}'."
                )
            elif tool_sensitivity == ToolSensitivity.MEDIUM:
                # If content is directive/behavior modification from untrusted origin
                for m in active_memories:
                    if m.content_type in [ContentType.INSTRUCTION, ContentType.POLICY_CHANGE, ContentType.POTENTIAL_BEHAVIOR_CHANGE]:
                        return (
                            GateDecision.BLOCK,
                            f"SECURITY VIOLATION: Untrusted memory with directive/policy modification attempted to trigger MEDIUM-sensitivity tool '{proposed_tool}'."
                        )

        # Rule 3: Policy Conflict Check
        if conflicts_detected and len(conflicts_detected) > 0:
            if effective_trust < TrustTier.SUPERVISOR_ADMIN:
                return (
                    GateDecision.BLOCK,
                    f"SECURITY VIOLATION: Recalled memory conflicts with authentic system policies: {conflicts_detected[0]}"
                )

        # Rule 4: User-Level (Tier 1) attempting system configuration changes
        if effective_trust == TrustTier.AUTHENTICATED_USER and proposed_tool in ["change_configuration", "send_external_data"]:
            return (
                GateDecision.BLOCK,
                f"SECURITY VIOLATION: Authenticated user tier (Tier 1) does not possess authority for administrative tool '{proposed_tool}'."
            )

        # Rule 5: Authorized
        return GateDecision.ALLOW, f"Authorized: Effective provenance trust (Tier {int(effective_trust)}) permits '{proposed_tool}'."


def evaluate_security_policy(
    retrieved_memories: List[MemoryEntry],
    user_intent: IntentType,
    proposed_tool: str,
    tool_sensitivity: ToolSensitivity,
    memories_used_ids: Optional[List[str]] = None,
    conflicts_detected: Optional[List[str]] = None
) -> Tuple[GateDecision, str]:
    return SecurityPolicy.evaluate(
        retrieved_memories=retrieved_memories,
        user_intent=user_intent,
        proposed_tool=proposed_tool,
        tool_sensitivity=tool_sensitivity,
        memories_used_ids=memories_used_ids,
        conflicts_detected=conflicts_detected
    )
