"""
NEXUS Design System & Cyber-Observatory Theme
Provides CSS styling, SVG brand marks, timeline generators, and UI components
for the NEXUS AI Agent Security Observatory.
"""

# Color Palette Tokens
COLOR_BG_DARK = "#080C14"
COLOR_BG_SURFACE = "#0D1526"
COLOR_BG_CARD = "#111C33"
COLOR_BORDER = "#1E293B"
COLOR_BORDER_FOCUS = "#38BDF8"
COLOR_CYAN = "#06B6D4"
COLOR_VIOLET = "#8B5CF6"
COLOR_EMERALD = "#10B981"
COLOR_AMBER = "#F59E0B"
COLOR_RED = "#EF4444"
COLOR_TEXT_PRIMARY = "#F8FAFC"
COLOR_TEXT_MUTED = "#94A3B8"

NEXUS_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Global Dark Theme Overrides */
    .stApp {
        background-color: #080C14 !important;
        color: #F8FAFC !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Streamlit Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #0B1120 !important;
        border-right: 1px solid #1E293B !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {
        color: #CBD5E1 !important;
    }

    /* Brand Header Box */
    .nexus-brand-header {
        background: linear-gradient(135deg, rgba(13, 21, 38, 0.95) 0%, rgba(17, 28, 51, 0.85) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5), inset 0 1px 0 0 rgba(255, 255, 255, 0.05);
        position: relative;
        overflow: hidden;
    }
    .nexus-brand-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #06B6D4, #8B5CF6, #10B981);
    }
    .nexus-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #FFFFFF 0%, #38BDF8 60%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 6px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .nexus-descriptor {
        font-size: 1.05rem;
        color: #94A3B8;
        margin: 0 0 14px 0;
        font-weight: 500;
    }
    .nexus-tagline {
        font-size: 1.15rem;
        font-weight: 700;
        color: #38BDF8;
        margin-bottom: 8px;
        text-shadow: 0 0 20px rgba(56, 189, 248, 0.3);
    }
    .nexus-abstract {
        font-size: 0.92rem;
        color: #CBD5E1;
        line-height: 1.55;
        max-width: 900px;
        margin-bottom: 12px;
    }

    /* Central Flow Diagram Container */
    .flow-container {
        background: rgba(11, 17, 32, 0.8);
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 20px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        overflow-x: auto;
    }
    .flow-node {
        background: #111C33;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 10px 14px;
        text-align: center;
        min-width: 130px;
        flex: 1;
        transition: all 0.2s ease;
    }
    .flow-node.gate {
        border-color: #06B6D4;
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
        box-shadow: 0 0 15px rgba(6, 182, 212, 0.25);
    }
    .flow-node-title {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin-bottom: 4px;
    }
    .flow-node.gate .flow-node-title {
        color: #38BDF8;
    }
    .flow-node-value {
        font-size: 0.88rem;
        font-weight: 700;
        color: #F8FAFC;
        font-family: 'JetBrains Mono', monospace;
    }
    .flow-arrow {
        color: #64748B;
        font-size: 1.2rem;
        font-weight: 700;
        padding: 0 4px;
    }

    /* Live Agent Trace Timeline (8 Stages) */
    .timeline-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin: 20px 0;
        padding: 16px;
        background: #0B1120;
        border: 1px solid #1E293B;
        border-radius: 10px;
    }
    .timeline-step {
        flex: 1;
        min-width: 125px;
        background: #0F172A;
        border-left: 3px solid #334155;
        border-radius: 6px;
        padding: 10px 12px;
        position: relative;
    }
    .timeline-step.completed {
        border-left-color: #06B6D4;
        background: rgba(6, 182, 212, 0.05);
    }
    .timeline-step.allowed {
        border-left-color: #10B981;
        background: rgba(16, 185, 129, 0.08);
    }
    .timeline-step.blocked {
        border-left-color: #EF4444;
        background: rgba(239, 68, 68, 0.08);
    }
    .timeline-step-num {
        font-size: 0.7rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
    }
    .timeline-step.completed .timeline-step-num { color: #38BDF8; }
    .timeline-step.allowed .timeline-step-num { color: #10B981; }
    .timeline-step.blocked .timeline-step-num { color: #EF4444; }
    .timeline-step-name {
        font-size: 0.82rem;
        font-weight: 600;
        color: #E2E8F0;
        margin-top: 2px;
    }
    .timeline-step-detail {
        font-size: 0.72rem;
        color: #94A3B8;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* High Density Metric Cards */
    .nexus-metric-card {
        background: #0D1526;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 16px 20px;
        position: relative;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .nexus-metric-card:hover {
        border-color: #38BDF8;
    }
    .nexus-metric-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 6px;
    }
    .nexus-metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        color: #F8FAFC;
    }
    .nexus-metric-sub {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 4px;
    }

    /* 3-Column Analysis Cards */
    .analysis-card {
        background: #0D1526;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 18px 20px;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    .analysis-card-title {
        font-size: 0.88rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #38BDF8;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid #1E293B;
        padding-bottom: 8px;
    }
    .analysis-field {
        margin-bottom: 10px;
    }
    .analysis-field-label {
        font-size: 0.75rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
    }
    .analysis-field-value {
        font-size: 0.9rem;
        color: #F1F5F9;
        font-weight: 500;
    }

    /* Decision Result Banners */
    .banner-allow {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(6, 78, 59, 0.25) 100%);
        border: 1px solid #10B981;
        border-left: 6px solid #10B981;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 16px 0;
    }
    .banner-block {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(127, 29, 29, 0.25) 100%);
        border: 1px solid #EF4444;
        border-left: 6px solid #EF4444;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 16px 0;
    }

    /* Trust Tier Badges */
    .tier-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .tier-0 { background: rgba(239, 68, 68, 0.2); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    .tier-1 { background: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .tier-2 { background: rgba(59, 130, 246, 0.2); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.4); }
    .tier-3 { background: rgba(139, 92, 246, 0.2); color: #A78BFA; border: 1px solid rgba(139, 92, 246, 0.4); }

    /* Sensitivity Badges */
    .sens-low { background: rgba(16, 185, 129, 0.15); color: #34D399; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }
    .sens-medium { background: rgba(245, 158, 11, 0.15); color: #FBBF24; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }
    .sens-high { background: rgba(239, 68, 68, 0.2); color: #F87171; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }

    /* Code & Pre styling */
    pre, code {
        font-family: 'JetBrains Mono', monospace !important;
        background-color: #0F172A !important;
        color: #38BDF8 !important;
    }

    /* Clean Streamlit widgets */
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        background-color: #0F172A !important;
        border: 1px solid #334155 !important;
        color: #F8FAFC !important;
        border-radius: 6px !important;
    }
    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stTextArea"] textarea:focus {
        border-color: #06B6D4 !important;
        box-shadow: 0 0 8px rgba(6, 182, 212, 0.3) !important;
    }

    /* Primary Buttons */
    button[kind="primary"] {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
        border: 1px solid #38BDF8 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        box-shadow: 0 4px 14px 0 rgba(2, 132, 199, 0.39) !important;
        transition: all 0.2s ease !important;
    }
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0369A1 0%, #075985 100%) !important;
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.5) !important;
        transform: translateY(-1px);
    }
    button[kind="secondary"] {
        background: #0F172A !important;
        border: 1px solid #334155 !important;
        color: #CBD5E1 !important;
    }
</style>
"""

def render_nexus_header():
    """Renders the top branding header for NEXUS."""
    return f"""
    <div class="nexus-brand-header">
        <div class="nexus-title">
            <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
            </svg>
            NEXUS
            <span style="font-size: 0.95rem; font-weight: 500; color: #64748B; margin-left: 8px;">v2.4-RELEASE</span>
        </div>
        <div class="nexus-descriptor">Persistent AI Memory Security Observatory & Reference Monitor</div>
        <div class="nexus-tagline">“Trust the memory. Verify the action.”</div>
        <div class="nexus-abstract">
            <strong>Recall-Time Provenance Authorization</strong> eliminates the fundamental security–utility paradox in enterprise AI agents. 
            By decoupling epistemological memory storage from operational tool execution authority, persistent memory poisoning attacks are 
            deterministically blocked at runtime without impairing benign read-only operations.
        </div>
        <div style="font-size: 0.78rem; color: #64748B; font-family: 'JetBrains Mono', monospace;">
            RESEARCH BENCHMARK MODE: Deterministic MockProvider isolation enabled • Zero LLM prompt-stochasticity
        </div>
    </div>
    """

def render_central_security_flow():
    """Renders the core security flow architecture diagram."""
    return """
    <div class="flow-container">
        <div class="flow-node">
            <div class="flow-node-title">Stage 1</div>
            <div class="flow-node-value">User Query</div>
            <div style="font-size: 0.7rem; color: #94A3B8; margin-top: 2px;">Intent Analyzed</div>
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-node">
            <div class="flow-node-title">Stage 2</div>
            <div class="flow-node-value">Vector Retrieval</div>
            <div style="font-size: 0.7rem; color: #94A3B8; margin-top: 2px;">ChromaDB Top-K</div>
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-node">
            <div class="flow-node-title">Stage 3</div>
            <div class="flow-node-value">Provenance Trace</div>
            <div style="font-size: 0.7rem; color: #94A3B8; margin-top: 2px;">Trust Tiers 0-3</div>
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-node gate">
            <div class="flow-node-title">🛡️ SECURITY GATE</div>
            <div class="flow-node-value" style="color: #38BDF8;">Recall-Time Policy</div>
            <div style="font-size: 0.7rem; color: #06B6D4; margin-top: 2px;">Reference Monitor</div>
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-node">
            <div class="flow-node-title">Stage 5</div>
            <div class="flow-node-value">Tool Execution</div>
            <div style="font-size: 0.7rem; color: #94A3B8; margin-top: 2px;">Allowed or Blocked</div>
        </div>
    </div>
    """

def render_trace_timeline(stages: list):
    """
    Renders the 8-stage Live Agent Trace timeline.
    stages format: [{"num": 1, "name": "Query Received", "status": "completed", "detail": "Intent: ACTION"}]
    """
    html = '<div class="timeline-container">'
    for s in stages:
        status_cls = s.get("status", "completed")  # completed, allowed, blocked, pending
        html += f"""
        <div class="timeline-step {status_cls}">
            <div class="timeline-step-num">Step {s['num']} • {status_cls.upper()}</div>
            <div class="timeline-step-name">{s['name']}</div>
            <div class="timeline-step-detail" title="{s.get('detail', '')}">{s.get('detail', '')}</div>
        </div>
        """
    html += '</div>'
    return html

def render_tier_badge(tier: int, name: str = "") -> str:
    """Returns styled HTML tier badge."""
    t_name = name if name else f"Tier {tier}"
    return f'<span class="tier-badge tier-{tier}">T{tier} • {t_name}</span>'

def render_sensitivity_badge(sensitivity: str) -> str:
    """Returns styled HTML sensitivity badge."""
    s_lower = sensitivity.lower()
    return f'<span class="sens-{s_lower}">{sensitivity.upper()}</span>'
