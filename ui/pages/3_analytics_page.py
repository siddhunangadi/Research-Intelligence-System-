"""
Analytics & Telemetry Page — Research Intelligence System
Live operational dashboard with charts, registry table, and delete controls.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import streamlit as st
import altair as alt
from sqlalchemy import func

from ui.style_helper import apply_custom_style, render_header
from indexing.postgres_store import PostgresMetadataStore, PaperModel, ChunkModel
from indexing.index_builder import IndexBuilder
from monitoring.query_tracker import QueryTracker

st.set_page_config(page_title="Analytics — RIS", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")
apply_custom_style()
render_header("analytics")

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:1rem 0 2rem 0;">
    <div class="section-label">Observability</div>
    <h1 class="gradient-text" style="font-size:2.8rem; margin:0 0 0.6rem 0;">
        System Analytics &amp; Telemetry
    </h1>
    <p style="color:#64748b; font-size:1.05rem; margin:0;">
        Monitor repository statistics, index structures, query latencies, and live telemetry logs.
    </p>
</div>
""", unsafe_allow_html=True)

# ── DATA LOAD ─────────────────────────────────────────────────────────────────
pg_store = PostgresMetadataStore()
builder  = IndexBuilder()
tracker  = QueryTracker(store=pg_store)

with pg_store.manager.get_session() as session:
    total_papers = session.query(PaperModel).count()
    total_chunks = session.query(ChunkModel).count()
    sections_raw = session.query(
        ChunkModel.section,
        func.count(ChunkModel.chunk_id).label("count")
    ).group_by(ChunkModel.section).all()

query_logs = tracker.fetch_all_logs()

avg_latency    = 0.0
nli_pass_rate  = 100.0
refusal_rate   = 0.0
total_queries  = 0
df_telemetry   = None

if query_logs:
    df_telemetry  = pd.DataFrame(query_logs)
    total_queries = len(df_telemetry)
    avg_latency   = df_telemetry["latency_ms"].mean()
    refusal_rate  = (df_telemetry["refusal_triggered"].sum() / total_queries) * 100
    nli_pass_rate = (df_telemetry["nli_passed"].sum() / total_queries) * 100

# ── METRIC CARDS ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Operational Snapshot</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    (c1, str(total_papers),        "Papers Indexed",    "#fb923c", "linear-gradient(90deg,#fb923c,#ea580c)"),
    (c2, f"{total_chunks:,}",      "Vector Chunks",     "#f97316", "linear-gradient(90deg,#f97316,#f97316)"),
    (c3, str(total_queries),       "Queries Logged",    "#14b8a6", "linear-gradient(90deg,#14b8a6,#0ea5e9)"),
    (c4, f"{avg_latency:.0f} ms",  "Avg Latency",       "#f59e0b", "linear-gradient(90deg,#f59e0b,#ef4444)"),
    (c5, f"{nli_pass_rate:.0f}%",  "NLI SLA Pass",      "#10b981", "linear-gradient(90deg,#10b981,#34d399)"),
]

for col, val, lbl, color, grad in metrics:
    with col:
        st.markdown(f"""
        <div class="metric-card" style="--accent:{grad};">
            <div class="metric-label">{lbl}</div>
            <div class="metric-value" style="color:{color}; text-shadow:0 0 20px {color}44;">{val}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📂  Document Registry", "📈  Query Telemetry", "📊  Chunk Distribution"])

# ── TAB 1: REGISTRY ──────────────────────────────────────────────────────────
with tab1:
    papers = pg_store.list_all_papers()
    if papers:
        papers_df = pd.DataFrame(papers)
        papers_df.columns = ["Paper ID", "Title", "Authors", "Year", "Venue"]

        st.markdown("""
        <div style="margin-bottom:1rem;">
            <div class="section-label">Registered Documents</div>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(
            papers_df,
            column_config={
                "Paper ID": st.column_config.TextColumn(
                    "ID",
                    width="small",
                    help="Deduplicated paper hash ID",
                ),
                "Title": st.column_config.TextColumn(
                    "Title",
                    width="large",
                ),
                "Authors": st.column_config.TextColumn(
                    "Authors",
                    width="medium",
                ),
                "Year": st.column_config.NumberColumn(
                    "Year",
                    format="%d",
                    width="small",
                ),
                "Venue": st.column_config.TextColumn(
                    "Venue",
                    width="small",
                ),
            },
            use_container_width=True,
            height=280,
            hide_index=True
        )

        # Delete zone
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-card" style="border-left:4px solid rgba(239,68,68,0.5) !important;">
            <div class="section-label" style="color:#ef4444;">Danger Zone</div>
            <h4 style="margin:0 0 0.5rem 0; font-size:1.05rem; color:#fca5a5;">
                🗑️ Remove Paper from All Stores
            </h4>
            <p style="color:#475569; font-size:0.85rem; margin:0 0 1rem 0; line-height:1.5;">
                This permanently removes all text chunks, metadata records, and ChromaDB vectors
                for the selected paper. This action cannot be undone.
            </p>
        </div>
        """, unsafe_allow_html=True)

        del_left, del_right = st.columns([3, 1], gap="medium")
        with del_left:
            paper_to_delete = st.selectbox(
                "Target paper",
                options=[p["paper_id"] for p in papers],
                format_func=lambda x: next(
                    (f"📄 {p['title'][:60]}…" if len(p['title'])>60 else f"📄 {p['title']}"
                     for p in papers if p["paper_id"] == x), x
                ),
                label_visibility="collapsed"
            )
        with del_right:
            if st.button("Delete Index", use_container_width=True):
                with st.spinner("Removing records and vectors…"):
                    ok = builder.delete_paper_index(paper_to_delete)
                if ok:
                    st.markdown("""
                    <span class="status-badge-passed">✓ Index deleted successfully</span>
                    """, unsafe_allow_html=True)
                    st.rerun()
                else:
                    st.markdown("""
                    <span class="status-badge-failed">✗ Deletion failed — check logs</span>
                    """, unsafe_allow_html=True)
    else:
        st.info("No documents registered yet. Upload a PDF from the Upload page.")

# ── TAB 2: TELEMETRY ─────────────────────────────────────────────────────────
with tab2:
    if df_telemetry is not None:
        st.markdown("""
        <div style="margin-bottom:1rem;">
            <div class="section-label">Live Query Log</div>
        </div>
        """, unsafe_allow_html=True)

        # KPI strip
        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(f"""
            <div class="metric-card" style="--accent:linear-gradient(90deg,#ea580c,#f97316);">
                <div class="metric-label">Total Queries</div>
                <div class="metric-value" style="color:#fb923c; font-size:1.8rem;">{total_queries}</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="metric-card" style="--accent:linear-gradient(90deg,#ef4444,#f59e0b);">
                <div class="metric-label">Refusal Rate</div>
                <div class="metric-value" style="color:#f87171; font-size:1.8rem;">{refusal_rate:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="metric-card" style="--accent:linear-gradient(90deg,#10b981,#14b8a6);">
                <div class="metric-label">NLI Pass Rate</div>
                <div class="metric-value" style="color:#34d399; font-size:1.8rem;">{nli_pass_rate:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(
            df_telemetry,
            column_order=["timestamp", "query_text", "latency_ms", "token_count", "nli_passed", "refusal_triggered"],
            column_config={
                "timestamp": st.column_config.DatetimeColumn(
                    "Time",
                    format="YYYY-MM-DD HH:mm:ss",
                ),
                "query_text": st.column_config.TextColumn(
                    "User Query",
                ),
                "latency_ms": st.column_config.NumberColumn(
                    "Latency",
                    format="%d ms",
                ),
                "token_count": st.column_config.NumberColumn(
                    "Tokens",
                    format="%d",
                ),
                "nli_passed": st.column_config.CheckboxColumn(
                    "NLI Passed",
                    help="Response passed NLI hallucination check",
                ),
                "refusal_triggered": st.column_config.CheckboxColumn(
                    "Refusal",
                    help="System refused to answer (low confidence)",
                ),
            },
            use_container_width=True,
            height=320,
            hide_index=True
        )
    else:
        st.info("No queries logged yet. Run searches from the Chat page to populate telemetry.")

# ── TAB 3: CHUNK DISTRIBUTION ─────────────────────────────────────────────────
with tab3:
    if sections_raw:
        st.markdown("""
        <div style="margin-bottom:1rem;">
            <div class="section-label">Chunks per Section</div>
        </div>
        """, unsafe_allow_html=True)
        sec_df = pd.DataFrame(sections_raw, columns=["Section", "Chunk Count"])
        sec_df = sec_df.sort_values("Chunk Count", ascending=False)

        # Premium Altair bar chart with orange rounded bars and clean gridlines
        bar_chart = alt.Chart(sec_df).mark_bar(
            color='#f97316',
            cornerRadiusTopLeft=6,
            cornerRadiusTopRight=6
        ).encode(
            x=alt.X('Section:N', sort='-y', title=None, axis=alt.Axis(
                labelAngle=-45,
                labelColor='#94a3b8',
                tickColor='rgba(255,255,255,0.1)',
                grid=False
            )),
            y=alt.Y('Chunk Count:Q', title=None, axis=alt.Axis(
                grid=True,
                gridColor='rgba(255,255,255,0.05)',
                labelColor='#94a3b8',
                tickColor='rgba(255,255,255,0.1)'
            )),
            tooltip=[
                alt.Tooltip('Section:N', title='Section'),
                alt.Tooltip('Chunk Count:Q', title='Chunks')
            ]
        ).properties(
            height=340,
            background='transparent'
        ).configure_view(
            strokeWidth=0
        )
        st.altair_chart(bar_chart, use_container_width=True)

        # Stats summary
        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f"""
            <div class="metric-card" style="--accent:linear-gradient(90deg,#f97316,#f97316);">
                <div class="metric-label">Unique Sections</div>
                <div class="metric-value" style="color:#fb923c; font-size:1.8rem;">{len(sec_df)}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            top_sec = sec_df.iloc[0]
            st.markdown(f"""
            <div class="metric-card" style="--accent:linear-gradient(90deg,#fb923c,#f97316);">
                <div class="metric-label">Largest Section</div>
                <div class="metric-value" style="color:#c084fc; font-size:1.4rem; padding-top:0.2rem;">
                    {top_sec['Section']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_c:
            avg_chunks = sec_df["Chunk Count"].mean()
            st.markdown(f"""
            <div class="metric-card" style="--accent:linear-gradient(90deg,#0ea5e9,#14b8a6);">
                <div class="metric-label">Avg Chunks/Section</div>
                <div class="metric-value" style="color:#38bdf8; font-size:1.8rem;">{avg_chunks:.0f}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No chunk data yet. Upload a paper to see section distribution.")
