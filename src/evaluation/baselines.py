"""
Baseline Agent Architectures for Empirical Benchmarking
Implements:
1. Baseline A: Vulnerable Agent (No security gate or detection)
2. Baseline B: Semantic Detector Only (Pre-retrieval / post-retrieval filtering)
3. Baseline C: Proposed Provenance-Aware Recall-Time Defense
"""

import time
from typing import Dict, Any, List, Optional
from src.memory.schema import GateDecision, ToolSensitivity
from src.agent.tools import get_tool, execute_tool_action
from src.agent.llm import LLMProvider, get_llm_provider
from src.retrieval.retriever import SemanticRetriever
from src.nlp.poisoning_detector import PoisoningDetector
from src.security.gate import RecallTimeSecurityGate
from src.nlp.intent_classifier import classify_user_intent
from src.nlp.instruction_analyzer import analyze_content_type
from src.nlp.conflict_detector import detect_policy_conflict


class VulnerableAgentBaseline:
    """Baseline A: Standard RAG Agent with zero security checks and no provenance gate."""
    def __init__(self, retriever: SemanticRetriever, llm: Optional[LLMProvider] = None):
        self.retriever = retriever
        self.llm = llm or get_llm_provider("mock")

    def run_scenario(self, user_query: str, is_attack: bool, scenario_id: str = "") -> Dict[str, Any]:
        t0 = time.perf_counter()
        retrieved = self.retriever.retrieve(user_query, top_k=5)
        proposal = self.llm.generate_proposal(user_query, retrieved)
        
        # Executes proposed tool unconditionally
        res = execute_tool_action(proposal.proposed_tool, proposal.proposed_tool_args)
        t_elapsed = time.perf_counter() - t0

        tool_meta = get_tool(proposal.proposed_tool)
        tool_sens = tool_meta.sensitivity.name if tool_meta else "LOW"
        is_sensitive = (tool_meta.sensitivity >= ToolSensitivity.MEDIUM) if tool_meta else False

        ret_info = [
            {"id": m.memory_id, "trust_tier": m.trust_tier.name, "is_poisoned": m.is_poisoned, "is_tainted": m.is_tainted}
            for m in retrieved
        ]
        effective_trust = retrieved[0].trust_tier.name if retrieved else "UNTRUSTED_EXTERNAL"
        taint_status = any(m.is_tainted for m in retrieved) if retrieved else False

        if is_attack:
            if is_sensitive:
                tool_exec = "EXECUTED_SENSITIVE"
                outcome = "UNAUTHORIZED_SENSITIVE_EXECUTION"
                act_succeeded = True
                act_blocked = False
                act_neutralized = False
            else:
                tool_exec = "EXECUTED_READ_ONLY"
                outcome = "ATTACK_NEUTRALIZED"
                act_succeeded = False
                act_blocked = False
                act_neutralized = True
        else:
            tool_exec = "EXECUTED_READ_ONLY" if not is_sensitive else "EXECUTED_SENSITIVE"
            outcome = "LEGITIMATE_SUCCESS"
            act_succeeded = True
            act_blocked = False
            act_neutralized = False

        return {
            "scenario_id": scenario_id,
            "baseline": "Baseline A (Vulnerable)",
            "ground_truth_attack": is_attack,
            "original_requested_action": user_query,
            "is_attack": is_attack,
            "retrieved_memories": ret_info,
            "retrieved_memory_ids": [m.memory_id for m in retrieved],
            "detector_score": 0.0,
            "detector_prediction": 0,
            "poisoning_detector_prediction": 0,
            "trust_tier": effective_trust,
            "taint_status": taint_status,
            "proposed_tool": proposal.proposed_tool,
            "tool_sensitivity": tool_sens,
            "gate_decision": "NO_GATE_ALLOWED",
            "tool_execution": tool_exec,
            "final_outcome": outcome,
            "action_blocked": act_blocked,
            "action_succeeded": act_succeeded,
            "action_neutralized": act_neutralized,
            "latency_sec": t_elapsed
        }


class SemanticDetectorBaseline:
    """Baseline B: Semantic Classifier Filtering Only (Pre/Post-retrieval). No provenance information."""
    def __init__(
        self,
        retriever: SemanticRetriever,
        detector: PoisoningDetector,
        llm: Optional[LLMProvider] = None,
        threshold: float = 0.5
    ):
        self.retriever = retriever
        self.detector = detector
        self.llm = llm or get_llm_provider("mock")
        self.threshold = threshold

    def run_scenario(self, user_query: str, is_attack: bool, scenario_id: str = "") -> Dict[str, Any]:
        t0 = time.perf_counter()
        retrieved = self.retriever.retrieve(user_query, top_k=5)

        # Check semantic detector on retrieved memories
        max_prob = 0.0
        flagged = False
        if retrieved and self.detector and self.detector.is_fitted:
            texts = [m.content for m in retrieved]
            probs = self.detector.predict_proba(texts)
            if len(probs) > 0:
                max_prob = float(max(probs))
                if max_prob >= self.threshold:
                    flagged = True

        ret_info = [
            {"id": m.memory_id, "trust_tier": m.trust_tier.name, "is_poisoned": m.is_poisoned, "is_tainted": m.is_tainted}
            for m in retrieved
        ]
        effective_trust = retrieved[0].trust_tier.name if retrieved else "UNTRUSTED_EXTERNAL"
        taint_status = any(m.is_tainted for m in retrieved) if retrieved else False

        t_elapsed = time.perf_counter() - t0

        if flagged:
            # Detector blocked execution before tool invocation
            outcome = "BLOCKED_BY_DETECTOR" if is_attack else "FALSE_POSITIVE_BLOCK"
            return {
                "scenario_id": scenario_id,
                "baseline": "Baseline B (Semantic Detector)",
                "ground_truth_attack": is_attack,
                "original_requested_action": user_query,
                "is_attack": is_attack,
                "retrieved_memories": ret_info,
                "retrieved_memory_ids": [m.memory_id for m in retrieved],
                "detector_score": max_prob,
                "detector_prediction": 1,
                "poisoning_detector_prediction": 1,
                "trust_tier": effective_trust,
                "taint_status": taint_status,
                "proposed_tool": "NONE (BLOCKED_BY_DETECTOR)",
                "tool_sensitivity": "NONE",
                "gate_decision": "DETECTOR_BLOCKED",
                "tool_execution": "BLOCKED_BY_DETECTOR",
                "final_outcome": outcome,
                "action_blocked": True,
                "action_succeeded": False,
                "action_neutralized": False,
                "latency_sec": t_elapsed
            }

        # If detector produced False Negative, LLM proceeds and executes proposed tool
        proposal = self.llm.generate_proposal(user_query, retrieved)
        res = execute_tool_action(proposal.proposed_tool, proposal.proposed_tool_args)
        t_elapsed = time.perf_counter() - t0
        tool_meta = get_tool(proposal.proposed_tool)
        tool_sens = tool_meta.sensitivity.name if tool_meta else "LOW"
        is_sensitive = (tool_meta.sensitivity >= ToolSensitivity.MEDIUM) if tool_meta else False

        if is_attack:
            if is_sensitive:
                tool_exec = "EXECUTED_SENSITIVE"
                outcome = "UNAUTHORIZED_SENSITIVE_EXECUTION"
                act_succeeded = True
                act_blocked = False
                act_neutralized = False
            else:
                tool_exec = "EXECUTED_READ_ONLY"
                outcome = "ATTACK_NEUTRALIZED"
                act_succeeded = False
                act_blocked = False
                act_neutralized = True
        else:
            tool_exec = "EXECUTED_READ_ONLY" if not is_sensitive else "EXECUTED_SENSITIVE"
            outcome = "LEGITIMATE_SUCCESS"
            act_succeeded = True
            act_blocked = False
            act_neutralized = False

        return {
            "scenario_id": scenario_id,
            "baseline": "Baseline B (Semantic Detector)",
            "ground_truth_attack": is_attack,
            "original_requested_action": user_query,
            "is_attack": is_attack,
            "retrieved_memories": ret_info,
            "retrieved_memory_ids": [m.memory_id for m in retrieved],
            "detector_score": max_prob,
            "detector_prediction": 0,
            "poisoning_detector_prediction": 0,
            "trust_tier": effective_trust,
            "taint_status": taint_status,
            "proposed_tool": proposal.proposed_tool,
            "tool_sensitivity": tool_sens,
            "gate_decision": "NO_PROVENANCE_ALLOWED",
            "tool_execution": tool_exec,
            "final_outcome": outcome,
            "action_blocked": act_blocked,
            "action_succeeded": act_succeeded,
            "action_neutralized": act_neutralized,
            "latency_sec": t_elapsed
        }


class ProposedDefenseAgent:
    """Baseline C (Proposed System): Provenance-Aware Recall-Time Authorization Gate."""
    def __init__(
        self,
        retriever: SemanticRetriever,
        security_gate: RecallTimeSecurityGate,
        detector: Optional[PoisoningDetector] = None,
        llm: Optional[LLMProvider] = None
    ):
        self.retriever = retriever
        self.security_gate = security_gate
        self.detector = detector
        self.llm = llm or get_llm_provider("mock")

    def run_scenario(self, user_query: str, is_attack: bool, scenario_id: str = "") -> Dict[str, Any]:
        t0 = time.perf_counter()
        intent = classify_user_intent(user_query)
        retrieved = self.retriever.retrieve(user_query, top_k=5)

        # NLP Analysis
        conflicts = []
        scores = []
        for mem in retrieved:
            mem.content_type = analyze_content_type(mem.content)
            has_c, c_reason = detect_policy_conflict(mem, intent)
            if has_c and c_reason:
                conflicts.append(c_reason)
            if self.detector and self.detector.is_fitted:
                try:
                    p = float(self.detector.predict_proba([mem.content])[0])
                    scores.append(p)
                except Exception:
                    scores.append(0.0)

        proposal = self.llm.generate_proposal(user_query, retrieved)
        max_score = max(scores) if scores else 0.0
        detector_pred = 1 if (max_score >= 0.5) else 0
        flagged = (detector_pred == 1)

        decision, reason, audit_rec = self.security_gate.evaluate_and_authorize(
            scenario_id=scenario_id,
            user_query=user_query,
            user_intent=intent,
            proposed_tool=proposal.proposed_tool,
            retrieved_memories=retrieved,
            memories_used_ids=proposal.memories_used_ids,
            conflicts_detected=conflicts,
            semantic_detector_flagged=flagged,
            is_adversarial_scenario=is_attack
        )

        t_elapsed = time.perf_counter() - t0
        tool_meta = get_tool(proposal.proposed_tool)
        tool_sens = tool_meta.sensitivity.name if tool_meta else "LOW"
        is_sensitive = (tool_meta.sensitivity >= ToolSensitivity.MEDIUM) if tool_meta else False

        ret_info = [
            {"id": m.memory_id, "trust_tier": m.trust_tier.name, "is_poisoned": m.is_poisoned, "is_tainted": m.is_tainted}
            for m in retrieved
        ]
        effective_trust = retrieved[0].trust_tier.name if retrieved else "UNTRUSTED_EXTERNAL"
        taint_status = any(m.is_tainted for m in retrieved) if retrieved else False

        if decision == GateDecision.ALLOW:
            res = execute_tool_action(proposal.proposed_tool, proposal.proposed_tool_args)
            if is_attack:
                if is_sensitive:
                    tool_exec = "EXECUTED_SENSITIVE"
                    outcome = "UNAUTHORIZED_SENSITIVE_EXECUTION"
                    act_succeeded = True
                    act_blocked = False
                    act_neutralized = False
                else:
                    tool_exec = "EXECUTED_READ_ONLY"
                    outcome = "ATTACK_NEUTRALIZED"
                    act_succeeded = False
                    act_blocked = False
                    act_neutralized = True
            else:
                tool_exec = "EXECUTED_READ_ONLY" if not is_sensitive else "EXECUTED_SENSITIVE"
                outcome = "LEGITIMATE_SUCCESS"
                act_succeeded = True
                act_blocked = False
                act_neutralized = False

            return {
                "scenario_id": scenario_id,
                "baseline": "Baseline C (Proposed Defense)",
                "ground_truth_attack": is_attack,
                "original_requested_action": user_query,
                "is_attack": is_attack,
                "retrieved_memories": ret_info,
                "retrieved_memory_ids": [m.memory_id for m in retrieved],
                "detector_score": max_score,
                "detector_prediction": detector_pred,
                "poisoning_detector_prediction": detector_pred,
                "trust_tier": effective_trust,
                "taint_status": taint_status,
                "proposed_tool": proposal.proposed_tool,
                "tool_sensitivity": tool_sens,
                "gate_decision": "ALLOW",
                "tool_execution": tool_exec,
                "final_outcome": outcome,
                "action_blocked": act_blocked,
                "action_succeeded": act_succeeded,
                "action_neutralized": act_neutralized,
                "gate_reason": reason,
                "latency_sec": t_elapsed
            }
        else:
            outcome = "BLOCKED_BY_GATE" if is_attack else "FALSE_POSITIVE_BLOCK"
            return {
                "scenario_id": scenario_id,
                "baseline": "Baseline C (Proposed Defense)",
                "ground_truth_attack": is_attack,
                "original_requested_action": user_query,
                "is_attack": is_attack,
                "retrieved_memories": ret_info,
                "retrieved_memory_ids": [m.memory_id for m in retrieved],
                "detector_score": max_score,
                "detector_prediction": detector_pred,
                "poisoning_detector_prediction": detector_pred,
                "trust_tier": effective_trust,
                "taint_status": taint_status,
                "proposed_tool": proposal.proposed_tool,
                "tool_sensitivity": tool_sens,
                "gate_decision": "BLOCK",
                "tool_execution": "BLOCKED_BY_GATE",
                "final_outcome": outcome,
                "action_blocked": True,
                "action_succeeded": False,
                "action_neutralized": False,
                "gate_reason": reason,
                "latency_sec": t_elapsed
            }
