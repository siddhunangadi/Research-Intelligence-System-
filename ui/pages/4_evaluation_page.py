"""
Evaluation Console — Research Intelligence System
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import streamlit as st
import altair as alt

from ui.style_helper import apply_custom_style, render_header
from database.connection import DatabaseConnectionManager
from indexing.postgres_store import EvaluationModel, ChunkModel
from evaluation.evaluation_runner import RISEvaluationRunner

st.set_page_config(page_title="Evaluation Console — RIS", page_icon="🧬", layout="wide", initial_sidebar_state="collapsed")
apply_custom_style()
render_header("evaluation")

# ── DB PROBE ──────────────────────────────────────────────────────────────────
db_manager = DatabaseConnectionManager()
db_empty   = True
try:
    with db_manager.get_session() as s:
        db_empty = s.query(ChunkModel).count() == 0
except Exception:
    pass

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:1rem 0 2rem 0;">
    <div class="section-label">Continuous Evaluation</div>
    <h1 class="gradient-text" style="font-size:2.8rem; margin:0 0 0.6rem 0;">
        Quality Gate Console
    </h1>
    <p style="color:#64748b; font-size:1.05rem; margin:0;">
        Verify hallucination constraints, check retrieval precision,
        and block regressions with golden dataset gates.
    </p>
</div>
""", unsafe_allow_html=True)

# ── TOP LAYOUT: Runner | Why This Matters ─────────────────────────────────────
action_col, explain_col = st.columns([1, 1], gap="large")

# ── ACTION CARD ───────────────────────────────────────────────────────────────
with action_col:
    st.markdown("""
    <div class="glass-card">
        <div class="section-label">Diagnostic Runner</div>
        <h3 style="margin:0 0 0.5rem 0; font-size:1.2rem; color:#ffffff;">
            ⚙️ Run Quality Gate
        </h3>
        <p style="color:#475569; font-size:0.87rem; line-height:1.6; margin:0;">
            Calculates Faithfulness &amp; Context Precision over the golden dataset.
            Blocks CI/CD merges if SLAs are breached.
        </p>
    </div>
    """, unsafe_allow_html=True)

    evaluate_live = st.checkbox(
        "Use Live Database Chunks (requires uploaded papers)",
        value=not db_empty,
        disabled=db_empty,
        help="Live mode evaluates against real indexed chunks. Mock mode uses built-in fixtures."
    )

    if db_empty:
        st.info("No documents in index — running with mock fixtures.")

    trigger = st.button("🚀  Trigger Evaluation", use_container_width=True)

    if trigger:
        mock_db = {
            "sparse_db": [
                {"chunk_id": "c1", "paper_id": "p01", "section": "methodology",
                 "page_number": 4,
                 "content": "The architecture was optimized using AdamW with a learning rate of 1e-4.",
                 "embedding": [0.1]*1024},
                {"chunk_id": "c2", "paper_id": "p01", "section": "results",
                 "page_number": 6,
                 "content": "We achieved a validation accuracy of 92.4% on GLUE benchmarks.",
                 "embedding": [0.2]*1024},
            ],
            "dense_db": [
                {"chunk_id": "c1", "paper_id": "p01", "section": "methodology",
                 "page_number": 4,
                 "content": "The architecture was optimized using AdamW with a learning rate of 1e-4.",
                 "embedding": [0.1]*1024},
                {"chunk_id": "c2", "paper_id": "p01", "section": "results",
                 "page_number": 6,
                 "content": "We achieved a validation accuracy of 92.4% on GLUE benchmarks.",
                 "embedding": [0.2]*1024},
            ],
        }

        with st.spinner("Calculating Faithfulness · Context Precision · MRR…"):
            runner = RISEvaluationRunner()
            report = runner.run_evaluations(mock_db=None if evaluate_live else mock_db)

        m      = report["metrics"]
        passed = report["passed_gate"]

        st.markdown("<br>", unsafe_allow_html=True)

        if passed:
            st.markdown("""
            <div class="gate-passed">
                <div style="font-size:2rem; margin-bottom:0.4rem;">✅</div>
                <div style="font-family:'Outfit',sans-serif; font-size:1.5rem;
                            font-weight:900; color:#10b981;">QUALITY GATE PASSED</div>
                <div style="color:#6ee7b7; font-size:0.85rem; margin-top:0.35rem;">
                    All SLA thresholds satisfied — pipeline is production-safe
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="gate-failed">
                <div style="font-size:2rem; margin-bottom:0.4rem;">🚨</div>
                <div style="font-family:'Outfit',sans-serif; font-size:1.5rem;
                            font-weight:900; color:#ef4444;">QUALITY GATE BREACHED</div>
                <div style="color:#fca5a5; font-size:0.85rem; margin-top:0.35rem;">
                    SLA violation — CI/CD build would be blocked
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        faith_col = "#10b981" if m['faithfulness']     >= 0.85 else "#ef4444"
        prec_col  = "#10b981" if m['context_precision'] >= 0.80 else "#ef4444"

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.markdown(f"""
            <div class="metric-card" style="--accent:linear-gradient(90deg,{faith_col},{faith_col}88);">
                <div class="metric-label">Faithfulness</div>
                <div class="metric-value" style="color:{faith_col};">{m['faithfulness']:.0%}</div>
                <div style="font-size:0.72rem; color:#475569; margin-top:0.3rem;">SLA ≥ 85%</div>
            </div>
            """, unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""
            <div class="metric-card" style="--accent:linear-gradient(90deg,{prec_col},{prec_col}88);">
                <div class="metric-label">Context Precision</div>
                <div class="metric-value" style="color:{prec_col};">{m['context_precision']:.0%}</div>
                <div style="font-size:0.72rem; color:#475569; margin-top:0.3rem;">SLA ≥ 80%</div>
            </div>
            """, unsafe_allow_html=True)
        with mc3:
            mrr = m.get('retrieval_mrr', m.get('mrr', 0.0))
            st.markdown(f"""
            <div class="metric-card" style="--accent:linear-gradient(90deg,#f97316,#fb923c);">
                <div class="metric-label">Retrieval MRR</div>
                <div class="metric-value" style="color:#f97316;">{mrr:.2f}</div>
                <div style="font-size:0.72rem; color:#475569; margin-top:0.3rem;">Mean Reciprocal Rank</div>
            </div>
            """, unsafe_allow_html=True)
        with mc4:
            rec = m.get('retrieval_recall_at_5', m.get('recall_at_5', 0.0))
            st.markdown(f"""
            <div class="metric-card" style="--accent:linear-gradient(90deg,#fb923c,#ea580c);">
                <div class="metric-label">Recall@5</div>
                <div class="metric-value" style="color:#fb923c;">{rec:.2f}</div>
                <div style="font-size:0.72rem; color:#475569; margin-top:0.3rem;">Top-5 Recall</div>
            </div>
            """, unsafe_allow_html=True)

# ── WHY THIS MATTERS — native Streamlit (no raw HTML) ────────────────────────
with explain_col:
    st.markdown("""
    <div class="glass-card">
        <div class="section-label">Why This Matters</div>
        <h3 style="margin:0 0 1.1rem 0; font-size:1.2rem; color:#ffffff;">
            🛡️ SLA Guardrails
        </h3>
    </div>
    """, unsafe_allow_html=True)

    guardrails = [
        ("🎯", "#10b981", "Faithfulness ≥ 85%",
         "Ratio of answer claims directly supported by retrieved chunks. Prevents LLM hallucinations."),
        ("🔎", "#f97316", "Context Precision ≥ 80%",
         "Measures whether retrieved chunks are ranked in relevance order. Validates BM25 + dense fusion."),
        ("⚙️", "#fbbf24", "CI/CD Regression Blocking",
         "Each GitHub Actions run executes this runner. PRs that breach SLAs fail the build automatically."),
        ("📊", "#fb923c", "Historical Trend Logging",
         "Every run is persisted with git commit hash, timestamps, and all metric values for audit trails."),
    ]

    for icon, color, title, desc in guardrails:
        col_icon, col_text = st.columns([1, 5], gap="small")
        with col_icon:
            st.markdown(f"""
            <div style="width:38px; height:38px; border-radius:10px;
                        background:linear-gradient(135deg,{color},{color}88);
                        display:flex; align-items:center; justify-content:center;
                        font-size:1.1rem; margin-top:0.15rem;">
                {icon}
            </div>
            """, unsafe_allow_html=True)
        with col_text:
            st.markdown(f"**{title}**")
            st.caption(desc)
        st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)

# ── HISTORY ───────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="section-label">Regression History</div>
<h3 style="font-size:1.3rem; color:#ffffff; margin:0 0 1rem 0;">📈 Quality Trend Over Time</h3>
""", unsafe_allow_html=True)

hist_data = []
with db_manager.get_session() as session:
    history = session.query(EvaluationModel).order_by(EvaluationModel.timestamp.asc()).all()
    if history:
        hist_data = [
            {
                "Timestamp":     h.timestamp,
                "Faithfulness":  h.faithfulness,
                "Ctx Precision": h.context_precision,
                "MRR":           h.retrieval_mrr,
                "Recall@5":      h.retrieval_recall_at_5,
                "Commit":        h.commit_hash or "local",
                "Gate":          "Pass" if h.passed_gate else "Fail",
            }
            for h in history
        ]

if hist_data:
    df_hist = pd.DataFrame(hist_data)
    
    # Melt the history dataframe to prepare it for Altair multi-line encoding
    df_melted = df_hist.melt(
        id_vars=["Timestamp"],
        value_vars=["Faithfulness", "Ctx Precision", "MRR"],
        var_name="Metric",
        value_name="Score"
    )

    # Build a premium smooth line chart in Altair
    line_chart = alt.Chart(df_melted).mark_line(
        strokeWidth=3,
        interpolate='monotone'
    ).encode(
        x=alt.X('Timestamp:T', title=None, axis=alt.Axis(
            grid=False,
            labelColor='#94a3b8',
            tickColor='rgba(255,255,255,0.1)'
        )),
        y=alt.Y('Score:Q', title=None, axis=alt.Axis(
            grid=True,
            gridColor='rgba(255,255,255,0.05)',
            labelColor='#94a3b8',
            tickColor='rgba(255,255,255,0.1)'
        )),
        color=alt.Color('Metric:N', scale=alt.Scale(
            domain=['Faithfulness', 'Ctx Precision', 'MRR'],
            range=['#10b981', '#f97316', '#38bdf8']
        )),
        tooltip=[
            alt.Tooltip('Timestamp:T', title='Time', format='%Y-%m-%d %H:%M:%S'),
            alt.Tooltip('Metric:N', title='Metric'),
            alt.Tooltip('Score:Q', title='Score', format='.2f')
        ]
    ).properties(
        height=300,
        background='transparent'
    ).configure_view(
        strokeWidth=0
    ).configure_legend(
        labelColor='#e2e8f0',
        titleColor='#e2e8f0',
        orient='top-left'
    )

    st.altair_chart(line_chart, use_container_width=True)

    st.markdown('<div class="section-label" style="margin-top:1rem;">Run Log Registry</div>',
                unsafe_allow_html=True)
    st.dataframe(
        df_hist,
        column_order=["Timestamp", "Faithfulness", "Ctx Precision", "MRR", "Recall@5", "Commit", "Gate"],
        column_config={
            "Timestamp": st.column_config.DatetimeColumn(
                "Run Time",
                format="YYYY-MM-DD HH:mm:ss",
            ),
            "Faithfulness": st.column_config.ProgressColumn(
                "Faithfulness",
                help="SLA limit is >= 85%",
                format="%.0f%%",
                min_value=0.0,
                max_value=1.0,
            ),
            "Ctx Precision": st.column_config.ProgressColumn(
                "Ctx Precision",
                help="SLA limit is >= 80%",
                format="%.0f%%",
                min_value=0.0,
                max_value=1.0,
            ),
            "MRR": st.column_config.NumberColumn(
                "MRR",
                format="%.2f",
            ),
            "Recall@5": st.column_config.NumberColumn(
                "Recall@5",
                format="%.2f",
            ),
            "Commit": st.column_config.TextColumn(
                "Git Commit",
            ),
            "Gate": st.column_config.TextColumn(
                "Gate Status",
            ),
        },
        use_container_width=True,
        height=260,
        hide_index=True
    )
else:
    st.markdown("""
    <div style="padding:2.5rem; text-align:center; background:rgba(15,15,25,0.5);
                border:1px dashed rgba(255,255,255,0.07); border-radius:16px;">
        <div style="font-size:2rem; margin-bottom:0.75rem;">📭</div>
        <div style="font-weight:600; color:#475569; margin-bottom:0.3rem;">No evaluation history yet</div>
        <div style="font-size:0.85rem; color:#334155;">
            Trigger a quality gate run above to start recording metrics.
        </div>
    </div>
    """, unsafe_allow_html=True)
