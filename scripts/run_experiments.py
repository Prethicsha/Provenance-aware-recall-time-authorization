"""
Benchmark Experiments Orchestration Script
Runs real empirical benchmarks:
- Baseline A (Vulnerable) vs Baseline B (Semantic Detector) vs Baseline C (Proposed Defense)
- False Negative Evasion Experiment
- Synthetic Temporal Dormancy Simulation (0, 7, 30 days)
- Multi-Hop Taint Retention
- Ablation Studies (A through G)
- Retrieval Metrics (Recall@K, Precision@K, MRR)
"""

import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.memory.schema import MemoryEntry
from src.memory.store import PersistentMemoryStore
from src.retrieval.retriever import SemanticRetriever, compute_retrieval_metrics
from src.nlp.poisoning_detector import EmbeddingPoisoningDetector
from src.evaluation.experiments import run_full_benchmark_experiments
from src.evaluation.ablation import run_ablation_study


def main():
    print("=" * 60)
    print("PHASE 22 & 23: RUNNING EXPERIMENTAL PROTOCOL & ABLATION")
    print("=" * 60)

    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # 1. Load memories
    input_file = "data/processed/normalized_memories.jsonl"
    if not os.path.exists(input_file):
        print("Please run scripts/preprocess_data.py first.")
        return

    memories = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                memories.append(MemoryEntry.model_validate_json(line))

    # 2. Connect to ChromaDB
    store = PersistentMemoryStore(persist_directory="data/chroma_db")

    # 3. Load Trained Detector
    detector = EmbeddingPoisoningDetector()
    detector.load("data/processed/models/embedding_detector.pkl")

    # 4. Evaluate Semantic Retrieval Metrics
    print("\n[1/3] Computing Semantic Retrieval Metrics (Recall@5, Precision@5, MRR)...")
    retriever = SemanticRetriever(store, default_top_k=5)
    sample_queries = [m.content[:60] for m in memories[:50]]
    sample_targets = [[m.memory_id] for m in memories[:50]]
    ret_metrics = compute_retrieval_metrics(sample_queries, sample_targets, retriever, k=5)
    print(f"  Recall@5:    {ret_metrics['recall@k']:.4f}")
    print(f"  Precision@5: {ret_metrics['precision@k']:.4f}")
    print(f"  MRR:         {ret_metrics['mrr']:.4f}")

    with open("results/retrieval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(ret_metrics, f, indent=2)

    # 5. Run Full Benchmark Comparison (Baselines A, B, C, Dormancy, Evasion)
    print("\n[2/3] Running Full Baseline Benchmark Experiments...")
    bench_results = run_full_benchmark_experiments(
        memories=memories,
        store=store,
        detector=detector,
        output_dir="results"
    )

    print("\nBaseline Performance Summary:")
    for b in bench_results["baseline_comparison"]:
        print(f"\n--- {b['baseline_name']} ---")
        print(f"  Sensitive Attack Success Rate (ASR): {b['sensitive_attack_success_rate']:.4f} (lower is better)")
        print(f"  Sensitive Attack Block Rate (ABR):   {b['sensitive_attack_block_rate']:.4f} (higher is better)")
        print(f"  Legitimate Task Success Rate (LTS):  {b['legitimate_task_success_rate']:.4f} (higher is better)")
        print(f"  False Positive Rate (FPR):           {b['false_positive_rate']:.4f} (lower is better)")
        print(f"  Mean Gate Latency:                   {b['mean_gate_latency_ms']:.2f} ms")
        print(f"  Median Gate Latency:                 {b['median_gate_latency_ms']:.2f} ms")

    # 6. Run Ablation Studies
    print("\n[3/3] Running Ablation Studies (Ablations A-G)...")
    ablation_results = run_ablation_study(
        memories=memories,
        store=store,
        detector=detector,
        output_dir="results"
    )

    print("\nAblation Results Summary:")
    for k, v in ablation_results.items():
        print(f"  {k:25s} | Sensitive ASR: {v['sensitive_attack_success_rate']:.4f} | Sensitive ABR: {v['sensitive_attack_block_rate']:.4f} | LTS: {v['legitimate_task_success_rate']:.4f} | FPR: {v['false_positive_rate']:.4f}")

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY! Results stored in 'results/'.")
    print("=" * 60)


if __name__ == "__main__":
    main()
