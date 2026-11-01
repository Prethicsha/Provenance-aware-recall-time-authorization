"""
Data Preprocessing Script
Loads raw datasets (MemShield-Bench, BIPIA, Bitext, Enron), cleans text, creates standardized MemoryEntry records,
and trains NLP Poisoning Detectors (TF-IDF and Embedding baselines).
"""

import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loaders import (
    load_memshield_dataset,
    load_bipia_dataset,
    load_bitext_dataset,
    load_enron_sample,
)
from src.nlp.poisoning_detector import train_and_evaluate_detectors


def main():
    print("=" * 60)
    print("PHASE 2 & 3: DATASET LOADING & PREPROCESSING")
    print("=" * 60)
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    # 1. Load MemShield-Bench (Primary Benchmark)
    print("\n[1/4] Loading MemShield-Bench (train / val / test / full)...")
    train_mems = load_memshield_dataset(split="train")
    val_mems = load_memshield_dataset(split="validation")
    test_mems = load_memshield_dataset(split="test")
    full_mems = load_memshield_dataset(split="full")
    print(f"  Loaded MemShield: Train={len(train_mems)}, Val={len(val_mems)}, Test={len(test_mems)}, Full={len(full_mems)}")

    # 2. Load BIPIA + GPT-4o-mini
    print("\n[2/4] Loading BIPIA Benchmark sample...")
    bipia_mems = load_bipia_dataset(max_samples=1000)
    print(f"  Loaded BIPIA entries: {len(bipia_mems)}")

    # 3. Load Bitext Customer Support
    print("\n[3/4] Loading Bitext Customer Support dataset...")
    bitext_mems = load_bitext_dataset(max_samples=1000)
    print(f"  Loaded Bitext entries: {len(bitext_mems)}")

    # 4. Load Enron Email Stream Sample
    print("\n[4/4] Loading Enron Email sample (streaming reader)...")
    enron_mems = load_enron_sample(max_samples=300)
    print(f"  Loaded Enron entries: {len(enron_mems)}")

    # Save normalized combined dataset for persistent store indexing
    all_normalized = full_mems + bipia_mems + bitext_mems + enron_mems
    out_jsonl = os.path.join(processed_dir, "normalized_memories.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for m in all_normalized:
            f.write(m.model_dump_json() + "\n")
    print(f"\nSaved {len(all_normalized)} normalized memory entries to: {out_jsonl}")

    # Train NLP Poisoning Detectors on MemShield train split and evaluate on unseen test split
    print("\n" + "=" * 60)
    print("PHASE 8: TRAINING NLP POISONING DETECTORS (TF-IDF & Embeddings)")
    print("=" * 60)
    
    # Strict group-safety check: exclude any exact text in test that exists in train
    train_text_set = set(m.content.strip().lower() for m in train_mems)
    deduped_test_mems = [m for m in test_mems if m.content.strip().lower() not in train_text_set]
    if len(deduped_test_mems) < len(test_mems):
        print(f"  [Group-Safety Note] Deduplicated {len(test_mems) - len(deduped_test_mems)} cross-split duplicate entries from Test evaluation split.")

    train_texts = [m.content for m in train_mems]
    train_labels = [int(m.is_poisoned) for m in train_mems]
    test_texts = [m.content for m in deduped_test_mems]
    test_labels = [int(m.is_poisoned) for m in deduped_test_mems]

    detector_results = train_and_evaluate_detectors(
        train_texts=train_texts,
        train_labels=train_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        save_dir=os.path.join(processed_dir, "models")
    )

    print("\nDetector Evaluation on Test Split:")
    for model_name, res in detector_results.items():
        m = res["metrics"]
        print(f"\n--- {model_name.upper()} Model ---")
        print(f"  Accuracy:  {m['accuracy']:.4f}")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall:    {m['recall']:.4f}")
        print(f"  F1 Score:  {m['f1']:.4f}")
        print(f"  AUROC:     {m['auroc']:.4f}")
        print(f"  FPR:       {m['fpr']:.4f}")
        print(f"  FNR:       {m['fnr']:.4f}")

    # Save detector metrics summary
    metrics_path = os.path.join(processed_dir, "detector_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({k: v["metrics"] for k, v in detector_results.items()}, f, indent=2)
    print(f"\nSaved detector metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
