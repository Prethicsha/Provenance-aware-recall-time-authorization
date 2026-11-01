"""
Agent Execution CLI Script
Executes interactive query flow through the complete pipeline:
User Query -> Intent -> Retrieval -> NLP Analysis -> LLM Reasoning -> Security Gate -> Execution
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.memory.store import PersistentMemoryStore
from src.retrieval.retriever import SemanticRetriever
from src.security.gate import RecallTimeSecurityGate
from src.agent.llm import get_llm_provider
from src.agent.graph import build_agent_graph
from src.nlp.poisoning_detector import EmbeddingPoisoningDetector


def main():
    query = "A customer wants a refund. Please process the request."
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])

    print("=" * 60)
    print("RUNNING NLP AGENT WITH PROVENANCE-AWARE SECURITY GATE")
    print("=" * 60)
    print(f"User Query: {query}\n")

    store = PersistentMemoryStore(persist_directory="data/chroma_db")
    retriever = SemanticRetriever(store, default_top_k=5)
    gate = RecallTimeSecurityGate(audit_log_file="logs/security_audit.jsonl")
    
    # Load detector if available
    detector = EmbeddingPoisoningDetector()
    try:
        detector.load("data/processed/models/embedding_detector.pkl")
    except Exception:
        detector = None

    llm = get_llm_provider("mock")
    agent = build_agent_graph(retriever, gate, llm_provider=llm, poison_detector=detector)

    state = agent.run(user_query=query, scenario_id="CLI-RUN")

    print(f"1. Classified User Intent: {state.user_intent.value}")
    print(f"2. Retrieved Memories ({len(state.retrieved_memories)}):")
    for m in state.retrieved_memories:
        print(f"   - [{m.memory_id}] (Tier {int(m.trust_tier)}, Type={m.content_type.value}, Sim={m.similarity_score:.3f}): {m.content[:80]}...")
    
    print(f"\n3. Effective Trust: Tier {int(state.effective_trust)} ({state.effective_trust.name})")
    print(f"4. LLM Proposed Tool: {state.proposed_tool} (Sensitivity: {state.tool_sensitivity.name})")
    print(f"5. LLM Reasoning: {state.reasoning_summary}")
    print(f"\n6. Security Gate Decision: {state.gate_decision.value}")
    print(f"   Reason: {state.gate_reason}")
    print(f"\n7. Tool Execution Status: {state.tool_execution_result}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
