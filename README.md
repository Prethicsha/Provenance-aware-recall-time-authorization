# Provenance-Aware Recall-Time Authorization Against Persistent Memory Poisoning in Enterprise AI Agents

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Format-blue.svg)](paper/revised_draft.tex)
[![Python](https://img.shields.io/badge/Python-3.11-brightgreen.svg)](https://www.python.org/)
[![VectorDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![Tests](https://img.shields.io/badge/Tests-20%2F20%20Passing-success.svg)](tests/)

Official empirical research demonstration system and conference codebase for:  
**"Provenance-Aware Recall-Time Authorization Against Persistent Memory Poisoning in Enterprise AI Agents"**

---

## 📌 Executive Summary

Stateful AI agents utilize persistent vector stores (e.g. ChromaDB) to preserve episodic memories and operational context across user sessions. When untrusted external data (such as emails, scraped websites, or customer support transcripts) is ingested, attackers can embed malicious directive payloads that remain dormant until recalled into the context window. 

Traditional defenses attempt to filter incoming text using statistical NLP classifiers. However, our empirical audit demonstrates that statistical classifiers exhibit an inherent **13.08% to 14.62% False Positive Rate**, causing them to catastrophically block legitimate operational queries (yielding 0.0% task success).

This repository implements a **provenance-aware, recall-time authorization reference monitor**:
1. **Four-Tier Provenance Trust Hierarchy ($T_{\mathrm{eff}} \in \{0, 1, 2, 3\}$):** Segregates external untrusted data (Tier 0), user session input (Tier 1), supervisor administration (Tier 2), and system root policies (Tier 3).
2. **Multi-Hop Taint Algebra:** Enforces strict provenance inheritance ($T_{\mathrm{eff}}(m_{\mathrm{derived}}) = \min_{p \in \mathcal{P}} \mathrm{tier}(p)$) across LLM summarization and reflection loops so untrusted taint cannot be laundered.
3. **Deterministic Reference Monitor:** Intercepts proposed actions before tool execution, evaluating tool sensitivity (`LOW`, `MEDIUM`, `HIGH`) against effective trust $T_{\mathrm{eff}}$, completely independent of the LLM's self-reported reasoning.
4. **Decoupled Architecture:** Read-only informational queries proceed unconditionally (Rule 1), while sensitive actions influenced by untrusted memories are deterministically blocked (Rule 2).

---

## 🏆 Key Empirical Results ($N = 200$ Primary Scenarios)

| Metric | Baseline A (Vulnerable) | Baseline B (Detector Only) | Baseline C (Proposed Defense) |
| :--- | :---: | :---: | :---: |
| **Sensitive Attack Success Rate (ASR $\downarrow$)** | 37.0% (37/100) | 0.0% (0/100) | **0.0% (0/100)** |
| **Sensitive Attack Block Rate (ABR $\uparrow$)** | 0.0% (0/100) | 100.0% (100/100) | **37.0% (37/100)** |
| **Conditional ABR on Sensitive Proposals ($\uparrow$)**| 0.0% (0/37) | 100.0% (37/37) | **100.0% (37/37)** |
| **Read-Only Outcome Rate** | 63.0% (63/100) | 0.0% (0/100) | **63.0% (63/100)** |
| **Legitimate Task Success (LTS $\uparrow$)** | 100.0% (100/100) | 0.0% (0/100) | **100.0% (100/100)** |
| **Gate False Positive Rate (FPR $\downarrow$)** | 0.0% (0/100) | 100.0% (100/100) | **0.0% (0/100)** |
| **Mean Pipeline Latency** | 13.69 ms | 87.86 ms | **96.35 ms** *(Gate: ~13 ms)* |

* **Central Evasion Stress Test ($N = 479$):** Across 479 confirmed classifier false negatives ($P(\text{poison}) < 0.50$), our gate blocked 100% (467/467) of sensitive proposals with **0.0% unauthorized executions**.
* **Privileged-Operation Benchmark ($N = 50$):** Zero false-positive blocks on authorized tasks (0% PFPR); 100% negative controls blocked (100% PSER).
* **Synthetic Temporal Dormancy ($T+0, T+7, T+30$ Days):** Maintained **10/10 blocked (1.0000 ABR)** across all horizons, proving metadata does not degrade over time.

---

## 🚀 Quickstart & Demonstration Commands

### 1. Interactive Research Demonstration System (Streamlit)
Launches the primary interactive testing lab with live query execution, curated scenarios, policy playground, and provenance lab:
```powershell
streamlit run app.py
```

### 2. Scientific Analytics & Evaluation Dashboard
Launches the dedicated 7-page research dashboard with deep metrics, distribution charts, and supplementary experiment analysis:
```powershell
streamlit run dashboard.py --server.port 8502
```

### 3. Fast Command-Line Demo (CLI)
Executes a single natural language request through the full agent pipeline in your terminal:
```powershell
python scripts/run_agent.py "A customer wants a refund. Please process the request."
```

### 4. Live Benchmark Metrics Recomputation (5 Seconds)
Recomputes all paper tables and metrics directly from raw scenario records without fabrication:
```powershell
python scripts/recompute_all_metrics.py
```

### 5. Automated Unit Test Suite (20/20 Tests Passing)
Executes all unit tests covering agent pipelines, reference monitor rules, taint algebra, and service APIs:
```powershell
python -m pytest tests/
```

---

## 📂 Repository Architecture

```text
├── app.py                       # *** Interactive Research Demonstration System (Streamlit) ***
├── dashboard.py                 # *** Separate Scientific Analytics Dashboard (Streamlit) ***
├── DEMO_GUIDE.md                # 5-8 minute presentation talk track for evaluators & professors
├── RESULTS_INTERPRETATION.md    # Exhaustive metric formulas, denominators, and scientific guide
├── IMPLEMENTATION_MAP.md        # Manuscript-to-code traceability matrix
├── REPRODUCIBILITY.md           # Reproduction parameters, seeds, and execution logs
├── configs/
│   └── config.yaml              # Global framework and retrieval configuration
├── data/
│   ├── raw/                     # Raw corpora (MemShield-Bench, BIPIA, Bitext, Enron)
│   ├── processed/               # Normalized memories (JSONL) and serialized models
│   └── chroma_db/               # Persistent ChromaDB store (3,478 indexed memories)
├── logs/
│   └── security_audit.jsonl     # Structured immutable JSONL forensic audit ledger
├── paper/
│   ├── revised_draft.tex        # Audited IEEE conference LaTeX manuscript (3,721 words)
│   └── references.bib           # 16 IEEE BibTeX citations
├── results/
│   ├── benchmark_results.json   # Primary benchmark metrics and baseline latency
│   ├── attack_scenarios_audit.csv # Scenario-level trace of 100 attack evaluations
│   ├── benign_scenarios_audit.csv # Scenario-level trace of 100 benign evaluations
│   └── false_negative_experiment.csv # 479 evasion stress test scenario records
├── scripts/
│   ├── run_agent.py             # CLI execution entry point
│   ├── recompute_all_metrics.py # Mathematical verification engine
│   ├── run_experiments.py       # Primary 200-scenario benchmark harness
│   ├── build_memory.py          # ChromaDB index builder
│   └── generate_results.py      # Plot and table generator
├── src/
│   ├── services/
│   │   └── research_service.py  # Central decoupled service layer for UI and experiments
│   ├── agent/                   # AgentGraphRunner, MockProvider, and tool registry
│   ├── memory/                  # Schema, ChromaDB store, provenance, and taint algebra
│   ├── nlp/                     # Intent classifier, conflict analyzer, and poisoning detectors
│   ├── retrieval/               # Semantic vector retriever (all-MiniLM-L6-v2)
│   ├── security/                # SecurityPolicy (Rules 0-5), Gate, and AuditLogger
│   └── evaluation/              # Baselines A/B/C, ablation studies, and metric calculations
└── tests/                       # Complete pytest suite (20 unit tests, all passing)
```

---

## 🔬 Documented Research Limitations

In accordance with scientific integrity guidelines:
1. **Deterministic Mock Action Proposer:** Experiments utilize a deterministic `MockProvider` rather than a stochastic live LLM (e.g. Llama-3). This isolates reference monitor mechanics from generative variance.
2. **Classifier Training-Memory Overlap:** Exactly 71/100 primary attack scenarios utilized memories present in the classifier training split. Therefore, the primary benchmark validates *gate policy enforcement*, not unseen detector generalization.
3. **Prefix-Based Retrieval Queries:** Retrieval queries were constructed from the first 60 characters of source memories with single-target relevance.
4. **Privileged Evaluation Mock Fallbacks:** In the privileged benchmark, 21/40 positive scenarios fell back to read-only retrieval due to keyword matching constraints in the mock proposer.
5. **Audit Ledger Format:** Audit records are committed to local JSONL files without cryptographic tamper-proof hardware attestation.

---

## 📄 License
This research project is licensed under the Apache 2.0 License.
