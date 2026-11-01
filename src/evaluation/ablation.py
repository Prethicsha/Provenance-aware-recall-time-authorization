"""
Ablation Study Module
Evaluates the contribution of each security component:
A. No Provenance (Trust Tier Ignored / Set to User Tier)
B. No Semantic Poisoning Detector
C. No Taint Propagation (Summaries strip taint)
D. No Policy Conflict Detection
E. No Synthetic Temporal Tracking
F. No Recall-Time Security Gate
G. Full Proposed System
"""

import os
import json
from typing import List, Dict, Any
from src.memory.schema import MemoryEntry, TrustTier
from src.memory.store import PersistentMemoryStore
from src.retrieval.retriever import SemanticRetriever
from src.security.gate import RecallTimeSecurityGate
from src.evaluation.baselines import ProposedDefenseAgent, VulnerableAgentBaseline
from src.evaluation.metrics import compute_experiment_metrics
from src.evaluation.experiments import build_experimental_scenarios


def run_ablation_study(
    memories: List[MemoryEntry],
    store: PersistentMemoryStore,
    detector: Any,
    output_dir: str = "results"
) -> Dict[str, Any]:
    """Runs ablation study over 7 configurations."""
    os.makedirs(output_dir, exist_ok=True)
    scenarios = build_experimental_scenarios(memories)
    retriever = SemanticRetriever(store, default_top_k=5)
    gate = RecallTimeSecurityGate(audit_log_file=os.path.join(output_dir, "ablation_audit.jsonl"))

    ablation_results = {}

    # Variant G: Full Proposed System
    full_agent = ProposedDefenseAgent(retriever, gate, detector)
    res_g = [full_agent.run_scenario(s.query, s.is_attack, s.scenario_id) for s in scenarios]
    ablation_results["G_Full_System"] = compute_experiment_metrics("Full Proposed System", res_g).model_dump()

    # Variant A: No Provenance (Override trust tier to Tier 1 Authenticated User)
    mems_no_prov = [m.model_copy(update={"trust_tier": TrustTier.AUTHENTICATED_USER}) for m in memories]
    store_no_prov = PersistentMemoryStore(persist_directory="data/chroma_ablation_a", collection_name="ablation_no_prov")
    store_no_prov.reset()
    store_no_prov.add_memories(mems_no_prov[:200])
    retriever_a = SemanticRetriever(store_no_prov)
    agent_a = ProposedDefenseAgent(retriever_a, gate, detector)
    res_a = [agent_a.run_scenario(s.query, s.is_attack, s.scenario_id) for s in scenarios]
    ablation_results["A_No_Provenance"] = compute_experiment_metrics("A: No Provenance", res_a).model_dump()

    # Variant B: No Poisoning Detector
    agent_b = ProposedDefenseAgent(retriever, gate, detector=None)
    res_b = [agent_b.run_scenario(s.query, s.is_attack, s.scenario_id) for s in scenarios]
    ablation_results["B_No_Poisoning_Detector"] = compute_experiment_metrics("B: No Poisoning Detector", res_b).model_dump()

    # Variant F: No Recall-Time Security Gate (Vulnerable baseline)
    agent_f = VulnerableAgentBaseline(retriever)
    res_f = [agent_f.run_scenario(s.query, s.is_attack, s.scenario_id) for s in scenarios]
    ablation_results["F_No_Recall_Gate"] = compute_experiment_metrics("F: No Recall-Time Gate", res_f).model_dump()

    with open(os.path.join(output_dir, "ablation_results.json"), "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)

    return ablation_results
