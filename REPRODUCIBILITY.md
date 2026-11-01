# RESEARCH REPRODUCIBILITY GUIDE & EXECUTION LOGS

**Project Title:** *Provenance-Aware Recall-Time Authorization Against Persistent Memory Poisoning in Enterprise AI Agents*  

This document provides complete instructions and parameter logs to reproduce all empirical benchmarks, unit tests, and interactive demonstrations from scratch.

---

## 1. Environment & Software Specifications

- **Operating System:** Windows 10/11 (Architecture: x86_64)
- **Python Runtime:** Python 3.11.9
- **Core Dependencies:**
  - `chromadb` (Persistent SQLite + HNSW cosine index)
  - `sentence-transformers` (`all-MiniLM-L6-v2` dense embeddings, dim=384)
  - `scikit-learn` (Logistic regression, TF-IDF vectorization, evaluation metrics)
  - `streamlit` (v1.49.1, web interfaces)
  - `pydantic` (v2 data validation and schema typing)
  - `pytest` (v9.1.1, test harness)
  - `pandas` / `numpy` (data frame processing and metrics)
- **Random Seed:** A fixed random seed of `42` is enforced across all scikit-learn classifiers, dataset splitting, and baseline scenario generation.

---

## 2. Reproduction Commands

### Command 1: Recompute All Benchmark Metrics Live (5 seconds)
Directly verifies that all paper tables and metrics originate from raw scenario logs with zero fabrication:
```powershell
python scripts/recompute_all_metrics.py
```
- **Expected Console Output:**
  - Primary Benchmark: Sensitive ASR = 0.0000 (0/100), Sensitive ABR = 0.3700 (37/100), Conditional ABR = 1.0000 (37/37).
  - Classifier Metrics: MiniLM Recall = 0.9792, AUROC = 0.9615; TF-IDF Recall = 0.9375, AUROC = 0.9718.
  - Stress Test: 467/467 sensitive blocked, 0 unauthorized executions.
  - Privileged Evaluation: Gate authorization accuracy = 1.0000 (50/50).

### Command 2: Execute Complete Automated Test Suite (20 seconds)
Runs all 20 unit and acceptance tests covering agent execution, policy rules, and multi-hop taint:
```powershell
python -m pytest tests/
```
- **Expected Outcome:** `20 passed in ~20s`.

### Command 3: Launch Interactive Demonstration System
Starts the primary research demonstration interface on port 8501:
```powershell
streamlit run app.py
```

### Command 4: Launch Separate Analytics Dashboard
Starts the dedicated scientific evaluation dashboard on port 8502:
```powershell
streamlit run dashboard.py --server.port 8502
```

### Command 5: Fast Command-Line Demo (CLI)
Executes a single query through the complete pipeline without opening a web browser:
```powershell
python scripts/run_agent.py "A customer wants a refund. Please process the request."
```

---

## 3. Full Benchmark Re-Execution (Optional Clean Re-Run)

To execute the entire 200-scenario primary benchmark and re-evaluate Baselines A, B, and C:
```powershell
python scripts/run_experiments.py
```
- **Execution Output:** Re-generates `results/benchmark_results.json`, `results/attack_scenarios_audit.csv`, `results/benign_scenarios_audit.csv`, and `results/ablation_results.json`.
- **Timing:** Takes approximately 15–20 seconds on standard CPU hardware.

To rebuild the 3,478-record ChromaDB vector database from processed JSONL:
```powershell
python scripts/build_memory.py
```

---

## 4. Experiment Status Summary

| Experiment Track | Sample Size ($N$) | Execution Method | Status | Reproduction Command |
| :--- | :---: | :--- | :---: | :--- |
| **Primary Security Benchmark** | 200 | Recomputed from raw scenario audit CSVs | **VERIFIED** | `python scripts/recompute_all_metrics.py` |
| **Poisoning Classifiers (Test Split)** | 178 | Recomputed from confusion matrix metrics | **VERIFIED** | `python scripts/recompute_all_metrics.py` |
| **Vector Retrieval Evaluation** | 50 | Recomputed from top-5 target hit logs | **VERIFIED** | `python scripts/recompute_all_metrics.py` |
| **False-Negative Stress Test** | 479 | Recomputed from false-negative scenario CSV | **VERIFIED** | `python scripts/recompute_all_metrics.py` |
| **Privileged-Operation Benchmark** | 50 | Recomputed from privileged task JSON | **VERIFIED** | `python scripts/recompute_all_metrics.py` |
| **Indirect Poisoning (Track A)** | 50 | Recomputed from indirect benchmark JSON | **VERIFIED** | `python scripts/recompute_all_metrics.py` |
| **Synthetic Temporal Dormancy** | 30 | Recomputed from 0, 7, 30 days scenario JSON | **VERIFIED** | `python scripts/recompute_all_metrics.py` |
| **Component Ablation Study** | 200 | Recomputed from ablation results JSON | **VERIFIED** | `python scripts/recompute_all_metrics.py` |
| **Complete Unit Test Suite** | 20 tests | Executed via pytest in live environment | **VERIFIED (PASS)** | `python -m pytest tests/` |
