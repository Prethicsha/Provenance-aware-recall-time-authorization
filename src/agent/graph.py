"""
LangGraph Agent Workflow Pipeline
Orchestrates:
1. User Query & Intent Classification
2. Semantic Persistent Memory Retrieval
3. Memory NLP Analysis (Poisoning, Pragmatics, Policy Conflict)
4. LLM Reasoning & Action Proposal
5. Provenance-Aware Recall-Time Security Gate (ALLOW / BLOCK)
6. Tool Execution (Only if authorized)
7. Final Synthesis & Response
"""

from typing import Dict, Any, Optional, List
from src.memory.schema import (
    MemoryEntry,
    TrustTier,
    ToolSensitivity,
    IntentType,
    GateDecision,
)
from src.agent.state import AgentState
from src.agent.tools import get_tool, execute_tool_action
from src.agent.llm import LLMProvider, get_llm_provider
from src.retrieval.retriever import SemanticRetriever
from src.nlp.intent_classifier import classify_user_intent
from src.nlp.instruction_analyzer import analyze_content_type
from src.nlp.conflict_detector import detect_policy_conflict
from src.nlp.poisoning_detector import PoisoningDetector
from src.security.gate import RecallTimeSecurityGate
from src.memory.taint import get_effective_trust


class AgentGraphRunner:
    """
    Modular Agent State Graph Runner.
    Can run with full LangGraph workflow or direct state machine graph.
    """
    def __init__(
        self,
        retriever: SemanticRetriever,
        security_gate: RecallTimeSecurityGate,
        llm_provider: Optional[LLMProvider] = None,
        poison_detector: Optional[PoisoningDetector] = None
    ):
        self.retriever = retriever
        self.security_gate = security_gate
        self.llm = llm_provider or get_llm_provider("mock")
        self.poison_detector = poison_detector

    def run(
        self,
        user_query: str,
        scenario_id: str = "LIVE-RUN",
        is_adversarial: bool = False,
        defense_mode: str = "BASELINE_C"
    ) -> AgentState:
        # Step 1: Intent Classification
        intent = classify_user_intent(user_query)
        state = AgentState(query=user_query, user_intent=intent)

        # Step 2: Semantic Memory Retrieval
        retrieved = self.retriever.retrieve(user_query, top_k=5)
        state.retrieved_memories = retrieved

        # Step 3: NLP Analysis (Pragmatics, Conflict, Poisoning)
        conflicts = []
        scores = []
        for mem in retrieved:
            mem.content_type = analyze_content_type(mem.content)
            has_conflict, c_reason = detect_policy_conflict(mem, intent)
            if has_conflict and c_reason:
                conflicts.append(f"[{mem.memory_id}] {c_reason}")
            
            if self.poison_detector and self.poison_detector.is_fitted:
                try:
                    prob = float(self.poison_detector.predict_proba([mem.content])[0])
                    mem.poisoning_score = prob
                    scores.append(prob)
                except Exception:
                    scores.append(0.0)
            else:
                scores.append(0.0)

        state.conflicts_detected = conflicts
        state.poisoning_scores = scores
        state.effective_trust = get_effective_trust(retrieved) if retrieved else TrustTier.AUTHENTICATED_USER
        
        all_taints = []
        for m in retrieved:
            all_taints.extend(m.taint_chain)
        state.taint_chain = list(set(all_taints))

        # Step 4: LLM Reasoning & Tool Proposal
        proposal = self.llm.generate_proposal(user_query, retrieved)
        state.proposed_tool = proposal.proposed_tool
        state.proposed_tool_args = proposal.proposed_tool_args
        state.reasoning_summary = proposal.reasoning_summary
        state.memories_used_ids = proposal.memories_used_ids

        tool_meta = get_tool(state.proposed_tool)
        state.tool_sensitivity = tool_meta.sensitivity if tool_meta else ToolSensitivity.LOW

        # Step 5: Security Gate Evaluation based on Defense Mode
        detector_flagged = any(s > 0.5 for s in scores)

        if defense_mode == "BASELINE_A":
            decision = GateDecision.ALLOW
            reason = "Baseline A (Unprotected): No recall-time security monitor active. Tool execution permitted unconditionally."
            audit_rec = None
        elif defense_mode == "BASELINE_B":
            if detector_flagged:
                decision = GateDecision.BLOCK
                reason = "Baseline B (Semantic Classifier Alone): Blocked because statistical text detector flagged candidate memory as suspicious."
            else:
                decision = GateDecision.ALLOW
                reason = "Baseline B (Semantic Classifier Alone): Passed statistical detector threshold."
            audit_rec = None
        else:
            # Baseline C: NEXUS Recall-Time Provenance Authorization Gate
            decision, reason, audit_rec = self.security_gate.evaluate_and_authorize(
                scenario_id=scenario_id,
                user_query=user_query,
                user_intent=intent,
                proposed_tool=state.proposed_tool,
                retrieved_memories=retrieved,
                memories_used_ids=state.memories_used_ids,
                conflicts_detected=conflicts,
                semantic_detector_flagged=detector_flagged,
                is_adversarial_scenario=is_adversarial
            )
        state.gate_decision = decision
        state.gate_reason = reason

        # Step 6: Conditional Execution
        if decision == GateDecision.ALLOW:
            result = execute_tool_action(state.proposed_tool, state.proposed_tool_args)
            state.tool_execution_result = result
            state.final_response = f"Action Authorized ({defense_mode}). Execution Result: {result}\nReasoning: {state.reasoning_summary}"
        else:
            state.tool_execution_result = "[BLOCKED BY SECURITY GATE]"
            state.final_response = (
                f"ACTION BLOCKED ({defense_mode}).\n"
                f"Reason: {reason}\n"
                f"Attempted Tool: {state.proposed_tool} (Sensitivity: {state.tool_sensitivity.name})\n"
                f"Effective Trust: Tier {int(state.effective_trust)} ({state.effective_trust.name})"
            )

        return state


def build_agent_graph(
    retriever: SemanticRetriever,
    security_gate: RecallTimeSecurityGate,
    llm_provider: Optional[LLMProvider] = None,
    poison_detector: Optional[PoisoningDetector] = None
) -> AgentGraphRunner:
    return AgentGraphRunner(
        retriever=retriever,
        security_gate=security_gate,
        llm_provider=llm_provider,
        poison_detector=poison_detector
    )
