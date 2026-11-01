import json
import csv
import math
import statistics
import pandas as pd
import numpy as np

def recompute_all():
    print("=" * 70)
    print("INDEPENDENT RECOMPUTATION OF ALL EXPERIMENTAL BENCHMARKS")
    print("=" * 70)

    # 1. PRIMARY BENCHMARK (N=200)
    print("\n--- 1. PRIMARY BENCHMARK AUDIT (results/attack_scenarios_audit.csv & benchmark_results.json) ---")
    df_att = pd.read_csv("results/attack_scenarios_audit.csv")
    print(f"Total attack audit rows: {len(df_att)}")
    
    # Baseline C outcomes in attack audit
    # Check proposed tools and sensitivities
    sens_tools = {"execute_payment", "update_account", "send_external_data", "change_configuration"}
    df_att["is_sensitive_proposed"] = df_att["proposed_tool"].isin(sens_tools)
    n_sens_prop = df_att["is_sensitive_proposed"].sum()
    n_ro_prop = (df_att["proposed_tool"] == "read_knowledge_base").sum()
    
    print(f"Attack scenarios proposing sensitive tools: {n_sens_prop}")
    print(f"Attack scenarios proposing read_knowledge_base: {n_ro_prop}")
    
    # Gate decisions on sensitive proposals under Baseline C
    blocked_sens = ((df_att["is_sensitive_proposed"]) & (df_att["gate_decision"] == "BLOCK")).sum()
    allowed_sens = ((df_att["is_sensitive_proposed"]) & (df_att["gate_decision"] == "ALLOW")).sum()
    executed_sens = ((df_att["is_sensitive_proposed"]) & (df_att["tool_execution"] == "EXECUTED_SENSITIVE")).sum()
    
    print(f"Baseline C: Sensitive proposals blocked: {blocked_sens}/{n_sens_prop}")
    print(f"Baseline C: Sensitive proposals allowed: {allowed_sens}/{n_sens_prop}")
    print(f"Baseline C: Sensitive executions succeeded: {executed_sens}/100")
    
    # Read-only queries under Baseline C
    allowed_ro = ((df_att["proposed_tool"] == "read_knowledge_base") & (df_att["gate_decision"] == "ALLOW")).sum()
    print(f"Baseline C: Read-only queries allowed: {allowed_ro}/{n_ro_prop}")

    with open("results/benchmark_results.json", "r") as f:
        bench_json = json.load(f)

    for arch in bench_json["baseline_comparison"]:
        print(f"\n{arch['baseline_name']}:")
        print(f"  Sensitive ASR: {arch['sensitive_attack_success_rate']:.4f} ({arch['sensitive_attacks_succeeded']}/{arch['sensitive_attack_scenarios']})")
        print(f"  Sensitive ABR: {arch['sensitive_attack_block_rate']:.4f} ({arch['sensitive_attacks_blocked']}/{arch['sensitive_attack_scenarios']})")
        print(f"  Outcome Rate (Read-Only): {arch['attack_neutralization_rate']:.4f} ({arch['sensitive_attacks_neutralized']}/{arch['sensitive_attack_scenarios']})")
        print(f"  Legitimate Task Success: {arch['legitimate_task_success_rate']:.4f} ({arch['legitimate_succeeded']}/{arch['legitimate_scenarios']})")
        print(f"  Gate False Positive Rate: {arch['false_positive_rate']:.4f} ({arch['legitimate_blocked']}/{arch['legitimate_scenarios']})")
        print(f"  Mean Latency: {arch['mean_gate_latency_ms']:.2f} ms | Median: {arch['median_gate_latency_ms']:.2f} ms")

    # 2. SEMANTIC CLASSIFIER TEST SET (N=178)
    print("\n--- 2. SEMANTIC CLASSIFIER AUDIT (data/processed/detector_metrics.json) ---")
    with open("data/processed/detector_metrics.json", "r") as f:
        det_metrics = json.load(f)
    
    for model_name, m in det_metrics.items():
        print(f"\nModel: {model_name}")
        tp, tn, fp, fn = m["true_positives"], m["true_negatives"], m["false_positives"], m["false_negatives"]
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
        fpr = fp / (fp + tn)
        fnr = fn / (tp + fn)
        auroc = m["auroc"]
        print(f"  Confusion: TP={tp}, TN={tn}, FP={fp}, FN={fn} (Total N={tp+tn+fp+fn})")
        print(f"  Precision: {prec:.4f} (Reported: {m['precision']:.4f})")
        print(f"  Recall:    {rec:.4f} (Reported: {m['recall']:.4f})")
        print(f"  F1:        {f1:.4f} (Reported: {m['f1']:.4f})")
        print(f"  AUROC:     {auroc:.4f} (Reported: {m['auroc']:.4f})")
        print(f"  FPR:       {fpr:.4f} (Reported: {m['fpr']:.4f})")
        print(f"  FNR:       {fnr:.4f} (Reported: {m['fnr']:.4f})")

    # 3. RETRIEVAL METRICS (N=50 queries)
    print("\n--- 3. RETRIEVAL SUBSYSTEM AUDIT (results/retrieval_metrics.json) ---")
    with open("results/retrieval_metrics.json", "r") as f:
        ret_metrics = json.load(f)
    print(f"  Recall@5:    {ret_metrics['recall@k']:.4f}")
    print(f"  Precision@5: {ret_metrics['precision@k']:.4f}")
    print(f"  MRR:         {ret_metrics['mrr']:.4f}")

    # 4. FALSE NEGATIVE STRESS TEST (N=479)
    print("\n--- 4. FALSE NEGATIVE STRESS TEST AUDIT (results/false_negative_experiment.csv) ---")
    df_fn = pd.read_csv("results/false_negative_experiment.csv")
    n_fn = len(df_fn)
    print(f"Total false negative test cases: {n_fn}")
    fn_sens = df_fn["tool_sensitivity"].isin(["HIGH", "MEDIUM"]).sum()
    fn_low = (df_fn["tool_sensitivity"] == "LOW").sum()
    fn_sens_blocked = ((df_fn["tool_sensitivity"].isin(["HIGH", "MEDIUM"])) & (df_fn["gate_decision"] == "BLOCK")).sum()
    fn_low_allowed = ((df_fn["tool_sensitivity"] == "LOW") & (df_fn["gate_decision"] == "ALLOW")).sum()
    fn_unauth_exec = (df_fn["final_execution_result"] == "EXECUTED (UNAUTHORIZED)").sum()
    print(f"  Final execution outcomes: {dict(df_fn['final_execution_result'].value_counts())}")
    print(f"  HIGH/MEDIUM sensitivity proposals: {fn_sens} | Blocked: {fn_sens_blocked}/{fn_sens}")
    print(f"  LOW sensitivity proposals:         {fn_low} | Allowed: {fn_low_allowed}/{fn_low}")
    print(f"  Unauthorized executions:           {fn_unauth_exec}/{n_fn}")

    # 5. PRIVILEGED TASK EXPERIMENT (N=50)
    print("\n--- 5. PRIVILEGED TASK EXPERIMENT AUDIT (results/privileged_task_experiment.json) ---")
    with open("results/privileged_task_experiment.json", "r") as f:
        priv_json = json.load(f)
    for arch_k, summary in priv_json["summary_by_architecture"].items():
        print(f"\n{arch_k}:")
        print(f"  Positive cases: Sensitive Executed={summary['sensitive_action_execution_count']}/40, Read-Only Fallback={summary['read_only_fallback_count']}/40, Blocked={summary['authorized_action_blocked_count']}/40")
        print(f"  Negative cases: Blocked={summary['negative_controls_blocked_count']}/10, Leaked={summary['unauthorized_sensitive_leakage_count']}/10")
        print(f"  Gate Authorization Accuracy: {summary['gate_authorization_accuracy']:.4f}")

    # 6. INDIRECT POISONING EXPERIMENT (N=50)
    print("\n--- 6. INDIRECT POISONING BENCHMARK AUDIT (results/indirect_poisoning_experiment.json) ---")
    with open("results/indirect_poisoning_experiment.json", "r") as f:
        ind_json = json.load(f)
    for arch_k, summary in ind_json["summary_by_architecture"].items():
        print(f"\n{arch_k}:")
        print(f"  Hijack Proposal Rate: {summary['hijack_proposal_rate']:.4f} ({summary['hijacked_proposal_count']}/50)")
        print(f"  Attack Block Rate:    {summary['indirect_attack_block_rate']:.4f} ({summary['indirect_attack_block_count']}/50)")
        print(f"  Attack Success Rate:  {summary['indirect_attack_success_rate']:.4f} ({summary['indirect_attack_success_count']}/50)")
        print(f"  Attribution Integrity:{summary['attribution_integrity_rate']:.4f} ({summary['attribution_integrity_count']}/50)")

    # 7. TEMPORAL DORMANCY (N=30)
    print("\n--- 7. TEMPORAL DORMANCY AUDIT (results/benchmark_results.json) ---")
    dorm = bench_json["synthetic_temporal_dormancy"]
    for horizon, res in dorm.items():
        print(f"  {horizon}: Blocked = {res['sensitive_attacks_blocked']}/{res['total_scenarios']} | ABR = {res['sensitive_attack_block_rate']:.4f} | ASR = {res['sensitive_attack_success_rate']:.4f}")

    # 8. COMPONENT ABLATION (N=200)
    print("\n--- 8. COMPONENT ABLATION AUDIT (results/ablation_results.json) ---")
    with open("results/ablation_results.json", "r") as f:
        abl_json = json.load(f)
    for abl_key, res in abl_json.items():
        print(f"\n{abl_key}: {res.get('baseline_name', abl_key)}")
        print(f"  Sensitive ASR: {res['sensitive_attack_success_rate']:.4f} ({res['sensitive_attacks_succeeded']}/{res['sensitive_attack_scenarios']})")
        print(f"  Sensitive ABR: {res['sensitive_attack_block_rate']:.4f} ({res['sensitive_attacks_blocked']}/{res['sensitive_attack_scenarios']})")
        print(f"  Outcome Rate:  {res['attack_neutralization_rate']:.4f} ({res['sensitive_attacks_neutralized']}/{res['sensitive_attack_scenarios']})")
        print(f"  Legit Success: {res['legitimate_task_success_rate']:.4f} ({res['legitimate_succeeded']}/{res['legitimate_scenarios']})")
        print(f"  Gate FPR:      {res['false_positive_rate']:.4f} ({res['legitimate_blocked']}/{res['legitimate_scenarios']})")

if __name__ == "__main__":
    recompute_all()
