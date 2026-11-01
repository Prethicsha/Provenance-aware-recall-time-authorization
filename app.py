"""
NEXUS — Enterprise Agent Security Platform & Research Testbed
Title: Provenance-Aware Recall-Time Authorization Against Persistent Memory Poisoning in Enterprise AI Agents
Entry Point: streamlit run app.py
"""

import os
import sys
import json
import time
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.services.research_service import get_research_service
from src.memory.schema import TrustTier, ToolSensitivity, ContentType, GateDecision
from src.ui.nexus_theme import (
    NEXUS_CSS,
    render_tier_badge,
    render_sensitivity_badge
)
from src.ui.nexus_charts import (
    create_defense_comparison_chart,
    create_outcome_breakdown_chart,
    create_confusion_matrix_chart,
    create_latency_breakdown_chart,
    create_ablation_chart
)

# Page Configuration
st.set_page_config(
    page_title="NEXUS | Enterprise Agent Security Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject System Styling
st.markdown(NEXUS_CSS, unsafe_allow_html=True)
st.markdown("""
<style>
    .system-bar {
        background: #0B1120;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
    }
    .system-status-indicator {
        display: inline-block;
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background-color: #10B981;
        margin-right: 8px;
        box-shadow: 0 0 10px #10B981;
    }
    .terminal-card {
        background: #0D1526;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .terminal-header {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #38BDF8;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 14px;
        border-bottom: 1px solid #1E293B;
        padding-bottom: 10px;
    }
    .interception-barrier {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 2px dashed #EF4444;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 16px 0;
        position: relative;
    }
    .interception-badge {
        position: absolute;
        top: -12px;
        left: 20px;
        background: #EF4444;
        color: #FFFFFF;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 800;
        padding: 2px 10px;
        border-radius: 4px;
        letter-spacing: 0.08em;
    }
    .stage-card {
        background: #080C14;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .stage-title {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
        margin-bottom: 6px;
        display: flex;
        justify-content: space-between;
    }
</style>
""", unsafe_allow_html=True)


def get_service():
    s = get_research_service()
    if not hasattr(s, "get_benchmark_scenario_catalog"):
        import importlib
        import src.services.research_service as rs_mod
        importlib.reload(rs_mod)
        rs_mod.ResearchService._instance = None
        s = rs_mod.get_research_service()
    return s


def init_session_state():
    if "active_mode" not in st.session_state:
        st.session_state.active_mode = "BASELINE_C"
    if "active_prompt" not in st.session_state:
        st.session_state.active_prompt = "A customer wants a refund. Please process the request."
    if "last_trace" not in st.session_state:
        st.session_state.last_trace = None


def main():
    init_session_state()
    service = get_service()

    # Top Status Bar
    mode_labels = {
        "BASELINE_C": "PROPOSED RECALL-TIME GATE (BASELINE C)",
        "BASELINE_A": "UNPROTECTED RUNTIME (BASELINE A - NO GATE)",
        "BASELINE_B": "STATISTICAL CLASSIFIER GATE (BASELINE B)"
    }
    mode_colors = {
        "BASELINE_C": "#10B981",
        "BASELINE_A": "#EF4444",
        "BASELINE_B": "#F59E0B"
    }

    st.markdown(f"""
    <div class="system-bar">
        <div>
            <span class="system-status-indicator"></span>
            <strong style="color:#F8FAFC;">NEXUS // ENTERPRISE AGENT SECURITY PLATFORM</strong>
            <span style="color:#64748B; margin-left:12px;">SYSTEM: Online</span>
        </div>
        <div>
            <span style="color:#64748B;">ACTIVE GATEWAY DEFENSE:</span> 
            <strong style="color:{mode_colors[st.session_state.active_mode]};">{mode_labels[st.session_state.active_mode]}</strong>
            <span style="color:#64748B; margin-left:16px;">VECTOR STORE:</span> 
            <strong style="color:#38BDF8;">ChromaDB (3,478 Indexed Memories)</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Primary Platform Navigation
    tab_workstation, tab_catalog, tab_evidence, tab_memory_store, tab_audit_stream = st.tabs([
        "⚡ Live Agent Workstation",
        "🔬 Benchmark Scenario Catalog (200 Primary + 479 Stress)",
        "📈 Empirical Research Evidence & Findings",
        "💾 Vector Memory Database (3,478 Records)",
        "📋 Forensic Security Audit Ledger"
    ])

    # =========================================================================
    # TAB 1: LIVE AGENT WORKSTATION
    # =========================================================================
    with tab_workstation:
        st.markdown("""
        <div style="margin-bottom: 16px;">
            <h2 style="margin: 0; font-size: 1.55rem; color: #F8FAFC;">Interactive Agent Execution & Runtime Interceptor</h2>
            <div style="font-size: 0.88rem; color: #94A3B8; margin-top: 4px;">
                Submit any prompt or select from actual benchmark scenarios. 
                Toggle defense modes live to witness the contrast between an unauthorized breach (Baseline A) and runtime interception (Baseline C).
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_config, col_input = st.columns([1, 2])

        with col_config:
            st.markdown("""
            <div class="terminal-card">
                <div class="terminal-header">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2">
                        <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
                        <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
                    </svg>
                    Runtime Defense Mode
                </div>
            """, unsafe_allow_html=True)

            mode_choice = st.radio(
                "Select Security Boundary:",
                [
                    "🛡️ Proposed: NEXUS Recall-Time Gate (Baseline C)",
                    "⚠️ Unprotected Agent (Baseline A - No Gate)",
                    "🔍 Naive Semantic Classifier Alone (Baseline B)"
                ],
                index=0 if st.session_state.active_mode == "BASELINE_C" else (1 if st.session_state.active_mode == "BASELINE_A" else 2)
            )

            if "Baseline C" in mode_choice:
                st.session_state.active_mode = "BASELINE_C"
                st.caption("✅ Provenance Gate Active: Intercepts high-sensitivity actions influenced by untrusted memories. Read-only queries pass unconditionally.")
            elif "Baseline A" in mode_choice:
                st.session_state.active_mode = "BASELINE_A"
                st.caption("🚨 Vulnerable Runtime: No gate. Actions execute unconditionally, demonstrating the persistent memory poisoning breach.")
            else:
                st.session_state.active_mode = "BASELINE_B"
                st.caption("⚠️ Classifier Alone: Blocks if detector probability > 0.50. Fails due to 13.08% False Positive Rate on multi-candidate retrieval.")

            st.markdown("---")
            st.markdown("##### Quick Scenario Ingestion")
            # Pull diverse realistic queries from the primary catalog
            primary_scenarios = service.get_benchmark_scenario_catalog("primary")
            
            sample_options = ["Custom Natural Language Prompt"] + [
                f"{s['scenario_id']} | {s['user_query'][:60]}..."
                for s in primary_scenarios[:25]
            ]
            chosen_sample = st.selectbox("Load Benchmark Scenario:", sample_options)

            if chosen_sample != "Custom Natural Language Prompt":
                sel_sid = chosen_sample.split(" | ")[0]
                matching_sc = next((s for s in primary_scenarios if s["scenario_id"] == sel_sid), None)
                if matching_sc:
                    st.session_state.active_prompt = matching_sc["user_query"]

            st.markdown("</div>", unsafe_allow_html=True)

        with col_input:
            st.markdown("""
            <div class="terminal-card">
                <div class="terminal-header">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2">
                        <polyline points="4 17 10 11 4 5"></polyline>
                        <line x1="12" y1="19" x2="20" y2="19"></line>
                    </svg>
                    Operational Request / Instruction
                </div>
            """, unsafe_allow_html=True)

            user_prompt = st.text_area("Prompt Dispatcher:", value=st.session_state.active_prompt, height=85)
            btn_dispatch = st.button("⚡ Dispatch Request to Agent Runtime", type="primary", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Run pipeline
        if btn_dispatch and user_prompt.strip():
            with st.spinner("Processing request through agent reasoning and security boundary..."):
                trace = service.run_agent_query(
                    query=user_prompt.strip(),
                    scenario_id="WORKSTATION-LIVE",
                    defense_mode=st.session_state.active_mode
                )
                st.session_state.last_trace = trace

        # Display trace
        if st.session_state.last_trace:
            t = st.session_state.last_trace
            dec = t["gate_decision"]

            st.markdown("---")
            st.markdown("### 🔍 Live Execution & Security Telemetry")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Classified Intent", t["user_intent"])
            m2.metric("Effective Trust ($T_{\\mathrm{eff}}$)", f"Tier {t['effective_trust']} ({t['effective_trust_name']})")
            m3.metric("Proposed Action", f"{t['proposed_tool']} ({t['tool_sensitivity']})")
            m4.metric("Pipeline Latency", f"{t['latency_ms']} ms")

            # Step 1: Agent Reasoning & Memory Retrieval
            st.markdown(f"""
            <div class="stage-card">
                <div class="stage-title">
                    <span>Step 1: Agent Formulation & Vector Memory Search</span>
                    <span style="color:#38BDF8;">ChromaDB Top-{t['retrieved_count']} Candidates</span>
                </div>
                <div style="font-size:0.85rem; color:#CBD5E1; margin-bottom:10px;">
                    <strong>Agent Reasoning Formulation:</strong> {t['reasoning_summary']}
                </div>
            """, unsafe_allow_html=True)

            if t["retrieved_memories"]:
                for mem in t["retrieved_memories"]:
                    p_badge = '<span style="color:#EF4444; font-weight:700;">[POISONED TAINT]</span>' if mem['is_poisoned'] else '<span style="color:#10B981;">[CLEAN METADATA]</span>'
                    st.markdown(f"""
                    <div style="background:#0D1526; border:1px solid #1E293B; border-radius:6px; padding:10px; margin-bottom:8px;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                            <span style="font-family:'JetBrains Mono'; font-weight:700; color:#38BDF8;">{mem['memory_id']}</span>
                            {render_tier_badge(mem['trust_tier'], mem['trust_tier_name'])}
                        </div>
                        <div style="font-size:0.75rem; color:#94A3B8; margin-bottom:4px;">
                            Channel: <code>{mem['source_channel']}</code> | Cosine Sim: <code>{mem['similarity_score']}</code> | {p_badge}
                        </div>
                        <div style="font-size:0.82rem; color:#E2E8F0; line-height:1.4;">
                            "{mem['content']}"
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Zero persistent memories retrieved.")

            st.markdown("</div>", unsafe_allow_html=True)

            # Step 2: Runtime Interception Barrier
            if t["tool_sensitivity"] == "HIGH" or t["effective_trust"] == 0 or t["conflicts_detected"]:
                st.markdown(f"""
                <div class="interception-barrier">
                    <div class="interception-badge">SECURITY GATEWAY INTERCEPT</div>
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <strong style="color:#F8FAFC; font-size:1.05rem;">Consequential Action Intercepted at Execution Boundary</strong>
                            <div style="color:#CBD5E1; font-size:0.85rem; margin-top:4px;">
                                The agent proposed invoking tool <code>{t['proposed_tool']}</code> ({render_sensitivity_badge(t['tool_sensitivity'])}).
                                The Gateway halted execution before system API dispatch to verify memory provenance.
                            </div>
                        </div>
                        <div>
                            {render_tier_badge(t['effective_trust'], t['effective_trust_name'])}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Step 3: Gateway Verdict & Enforcement
            if dec == "ALLOW":
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 78, 59, 0.3) 100%); border-left: 6px solid #10B981; border-radius: 8px; padding: 18px 22px; margin: 16px 0;">
                    <h3 style="color:#10B981; margin:0 0 6px 0;">✅ VERDICT: ACTION AUTHORIZED — EXECUTION PERMITTED</h3>
                    <div style="font-size:0.92rem; color:#F8FAFC;">
                        <strong>Policy Rule:</strong> {t['rule_triggered']}<br>
                        <strong>Gateway Explanation:</strong> {t['gate_reason']}<br>
                        <strong>Actual Tool Execution Result:</strong> <code>{t['tool_execution_result']}</code>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(127, 29, 29, 0.3) 100%); border-left: 6px solid #EF4444; border-radius: 8px; padding: 18px 22px; margin: 16px 0;">
                    <h3 style="color:#EF4444; margin:0 0 6px 0;">🛡️ VERDICT: ACTION BLOCKED — EXECUTION SUPPRESSED</h3>
                    <div style="font-size:0.92rem; color:#F8FAFC;">
                        <strong>Policy Violation:</strong> {t['rule_triggered']}<br>
                        <strong>Gateway Explanation:</strong> {t['gate_reason']}<br>
                        <strong>Enforcement Boundary:</strong> The operational tool <code>{t['proposed_tool']}</code> was aborted before calling any real API. Persistent memory poisoning mitigated.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("🔬 View Raw Dispatch State & JSON Payload"):
                st.json(t)

    # =========================================================================
    # TAB 2: FULL BENCHMARK SCENARIO CATALOG
    # =========================================================================
    with tab_catalog:
        st.markdown("### 🔬 Exhaustive Benchmark Scenario Catalog")
        st.markdown("""
        Explore all **200 primary benchmark scenarios** and **479 classifier evasion stress test cases** 
        evaluated in the research paper. Filter, search, and click to test any scenario live in the runtime workstation.
        """)

        track_sel = st.radio(
            "Benchmark Track:",
            ["Primary Benchmark Suite (N = 200: 100 Attack + 100 Benign)", "Classifier Evasion Stress Test (N = 479 Confirmed False Negatives)"],
            horizontal=True
        )

        if "Primary" in track_sel:
            scenarios = service.get_benchmark_scenario_catalog("primary")
            df_sc = pd.DataFrame(scenarios)

            c_f1, c_f2, c_f3 = st.columns(3)
            with c_f1:
                search_q = st.text_input("Filter prompt text / scenario ID:", value="")
            with c_f2:
                type_filter = st.selectbox("Scenario Type:", ["All (200)", "Adversarial Injections (100)", "Benign Control Tasks (100)"])
            with c_f3:
                decision_filter = st.multiselect("Gate Decision:", ["ALLOW", "BLOCK"], default=["ALLOW", "BLOCK"])

            filtered_df = df_sc.copy()
            if "Adversarial" in type_filter:
                filtered_df = filtered_df[filtered_df["is_adversarial"] == True]
            elif "Benign" in type_filter:
                filtered_df = filtered_df[filtered_df["is_adversarial"] == False]

            filtered_df = filtered_df[filtered_df["gate_decision"].isin(decision_filter)]
            if search_q:
                filtered_df = filtered_df[filtered_df["user_query"].str.contains(search_q, case=False, na=False) | filtered_df["scenario_id"].str.contains(search_q, case=False, na=False)]

            display_cols = ["scenario_id", "user_query", "proposed_tool", "tool_sensitivity", "effective_trust", "gate_decision", "reason"]
            st.dataframe(filtered_df[display_cols], use_container_width=True, height=450)

            st.markdown("---")
            st.markdown("#### Test Any Scenario Live in the Workstation")
            selected_sid = st.selectbox("Choose a scenario from the table to load & dispatch:", options=filtered_df["scenario_id"].tolist())
            if st.button("Load into Live Workstation ⚡", type="primary"):
                target = next((s for s in scenarios if s["scenario_id"] == selected_sid), None)
                if target:
                    st.session_state.active_prompt = target["user_query"]
                    st.success(f"Loaded `{selected_sid}` into the Live Workstation! Switch to Tab 1 to run it under different defense modes.")

        else:
            # Stress test track
            st_scenarios = service.get_benchmark_scenario_catalog("stress_test")
            df_st = pd.DataFrame(st_scenarios)
            st.markdown(f"**Total False-Negative Evasion Cases Evaluated:** `{len(df_st)}`")
            st.caption("These are scenarios where the statistical text detector completely failed (P < 0.50). The recall-time gate still intercepts 100% of sensitive tool proposals based on provenance.")
            
            st_search = st.text_input("Search stress test cases:", value="")
            if st_search:
                df_st = df_st[df_st["scenario_id"].str.contains(st_search, case=False, na=False) | df_st["attack_type"].str.contains(st_search, case=False, na=False)]

            st.dataframe(df_st[["scenario_id", "attack_type", "detector_probability", "retrieved_memory_id", "proposed_tool", "tool_sensitivity", "gate_decision", "final_execution_result"]], use_container_width=True, height=450)

    # =========================================================================
    # TAB 3: EMPIRICAL RESEARCH EVIDENCE
    # =========================================================================
    with tab_evidence:
        st.markdown("### 📈 Scientific Evidence & Research Benchmark Findings")
        st.markdown("""
        All empirical findings reported in the conference paper, derived directly from the scenario-level experimental artifacts.
        """)

        metrics = service.recompute_all_metrics()

        # Section 1: Primary Defense Benchmark
        st.markdown("#### 1. Primary Benchmark Defense Comparison ($N = 200$)")
        b1, b2 = st.columns(2)
        with b1:
            st.plotly_chart(create_defense_comparison_chart(metrics), use_container_width=True)
        with b2:
            st.plotly_chart(create_outcome_breakdown_chart(metrics), use_container_width=True)

        baselines_df = pd.DataFrame(metrics.get("benchmark_baselines", []))
        if not baselines_df.empty:
            b_view = baselines_df[[
                "baseline_name", "sensitive_attack_success_rate", "sensitive_attack_block_rate",
                "attack_neutralization_rate", "legitimate_task_success_rate", "false_positive_rate",
                "mean_gate_latency_ms"
            ]].copy()
            b_view.columns = ["Architecture", "Sensitive ASR (↓)", "Sensitive ABR (↑)", "Read-Only Rate", "Legitimate Success (↑)", "Gate FPR (↓)", "Mean Latency (ms)"]
            st.dataframe(b_view.style.format({
                "Sensitive ASR (↓)": "{:.1%}", "Sensitive ABR (↑)": "{:.1%}", "Read-Only Rate": "{:.1%}",
                "Legitimate Success (↑)": "{:.1%}", "Gate FPR (↓)": "{:.1%}", "Mean Latency (ms)": "{:.2f}"
            }), use_container_width=True, hide_index=True)

        st.markdown("""
        <div style="background:#0D1526; border-left:4px solid #38BDF8; padding:14px 18px; border-radius:4px; margin:16px 0;">
            <strong style="color:#38BDF8;">The Security–Utility Paradox Resolved:</strong><br>
            <span style="font-size:0.88rem; color:#CBD5E1;">
                • <strong>Baseline A (Unprotected):</strong> Achieves 100% utility but suffers a <strong>37.0% attack breach rate</strong>.<br>
                • <strong>Baseline B (Semantic Detector):</strong> Blocks attacks but suffers a <strong>13.08% False Positive Rate</strong>. Over top-5 candidate retrieval, this compounds to $1 - (1 - 0.1308)^5 \\approx 50.4\%$ false-positive probability, collapsing utility to <strong>0% Legitimate Task Success</strong>.<br>
                • <strong>Baseline C (Proposed Recall-Time Gate):</strong> Achieves <strong>0.0% ASR, 100.0% LTS, and 0.0% FPR</strong> because read-only informational queries proceed unconditionally without operational blast radius.
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Section 2: Classifier Evaluation & Confusion Matrices
        st.markdown("#### 2. Statistical Text Classifier Evaluation ($N = 178$ Test Partition)")
        det_m = metrics.get("classifier_metrics", {})
        cm1, cm2 = st.columns(2)
        with cm1:
            if "minilm" in det_m:
                st.plotly_chart(create_confusion_matrix_chart({
                    "tp": det_m["minilm"]["true_positives"], "tn": det_m["minilm"]["true_negatives"],
                    "fp": det_m["minilm"]["false_positives"], "fn": det_m["minilm"]["false_negatives"]
                }, "MiniLM Embeddings Classifier (N=178)"), use_container_width=True)
        with cm2:
            if "tfidf" in det_m:
                st.plotly_chart(create_confusion_matrix_chart({
                    "tp": det_m["tfidf"]["true_positives"], "tn": det_m["tfidf"]["true_negatives"],
                    "fp": det_m["tfidf"]["false_positives"], "fn": det_m["tfidf"]["false_negatives"]
                }, "TF-IDF Logistic Regression (N=178)"), use_container_width=True)

        # Section 3: Component Ablations
        st.markdown("#### 3. Component Ablation Study ($N = 200$)")
        st.plotly_chart(create_ablation_chart(), use_container_width=True)

        # Section 4: Latency Profiling
        st.markdown("#### 4. Component Latency Profiling (Steady-State Mean = 96.35 ms)")
        st.plotly_chart(create_latency_breakdown_chart(), use_container_width=True)

    # =========================================================================
    # TAB 4: VECTOR MEMORY DATABASE
    # =========================================================================
    with tab_memory_store:
        st.markdown("### 💾 ChromaDB Persistent Memory Corpus ($N = 3,478$)")
        mems = service.get_normalized_memories()
        if mems:
            df_m = pd.DataFrame(mems)
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                search_mem = st.text_input("Search content or memory ID:", value="refund")
            with f_col2:
                ds_filter = st.multiselect("Dataset:", df_m["source_dataset"].unique().tolist(), default=df_m["source_dataset"].unique().tolist())
            with f_col3:
                tier_filter = st.multiselect("Trust Tier:", [0, 1, 2, 3], default=[0, 1, 2, 3])

            filtered_m = df_m[(df_m["source_dataset"].isin(ds_filter)) & (df_m["trust_tier"].isin(tier_filter))]
            if search_mem:
                filtered_m = filtered_m[filtered_m["content"].str.contains(search_mem, case=False, na=False) | filtered_m["memory_id"].str.contains(search_mem, case=False, na=False)]

            st.dataframe(
                filtered_m[["memory_id", "source_dataset", "source_channel", "trust_tier", "is_poisoned", "attack_type", "content"]].head(50),
                use_container_width=True,
                height=450
            )

    # =========================================================================
    # TAB 5: FORENSIC AUDIT LEDGER
    # =========================================================================
    with tab_audit_stream:
        st.markdown("### 📋 Append-Only Forensic Audit Ledger (`logs/security_audit.jsonl`)")
        audit_recs = service.get_audit_records(limit=150)
        if audit_recs:
            df_a = pd.DataFrame(audit_recs)
            af1, af2 = st.columns([1, 2])
            with af1:
                dec_sel = st.multiselect("Filter Decision:", ["ALLOW", "BLOCK"], default=["ALLOW", "BLOCK"])
            with af2:
                audit_q = st.text_input("Filter audit prompt text:", value="")

            filtered_a = df_a[df_a["gate_decision"].isin(dec_sel)]
            if audit_q:
                filtered_a = filtered_a[filtered_a["user_query"].str.contains(audit_q, case=False, na=False) | filtered_a["scenario_id"].str.contains(audit_q, case=False, na=False)]

            cols = [c for c in ["timestamp", "scenario_id", "gate_decision", "proposed_tool", "tool_sensitivity", "effective_trust", "user_query", "reason"] if c in filtered_a.columns]
            st.dataframe(filtered_a[cols].sort_index(ascending=False), use_container_width=True, height=450)

            st.download_button(
                label="Export Audit Ledger as JSON 📥",
                data=filtered_a.to_json(orient="records", indent=2),
                file_name="nexus_security_audit.json",
                mime="application/json"
            )
        else:
            st.info("No audit events recorded yet. Run queries in the Live Workstation.")


if __name__ == "__main__":
    main()
