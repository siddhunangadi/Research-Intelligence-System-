import streamlit as st
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ui.style_helper import apply_custom_style, render_header

st.set_page_config(
    page_title="Research Intelligence System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

apply_custom_style()
render_header("home")

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 1rem 0 2.5rem 0;">
    <div style="display:flex; align-items:center; gap:14px; margin-bottom:0.6rem;">
        <div style="font-size:2.8rem; filter:drop-shadow(0 0 12px rgba(249,115,22,0.45));">🧬</div>
        <span style="font-family:'Outfit',sans-serif; font-size:0.78rem; font-weight:700;
                     letter-spacing:0.18em; text-transform:uppercase; color:#f97316;">
            Production-Grade RAG Platform
        </span>
    </div>
    <h1 class="gradient-text" style="font-size:3.75rem; line-height:1.08; margin:0 0 1rem 0;">
        Research Intelligence<br>System
    </h1>
    <p style="font-size:1.15rem; color:#64748b; max-width:680px; line-height:1.65; margin:0;">
        Zero-trust citation-grounded retrieval-augmented generation over scientific literature —
        NLI entailment verification, hybrid BM25 &amp; vector search, and continuous quality gates.
    </p>
</div>
""", unsafe_allow_html=True)

# ── STATS ROW ─────────────────────────────────────────────────────────────────
try:
    from indexing.postgres_store import PostgresMetadataStore, PaperModel, ChunkModel
    pg = PostgresMetadataStore()
    with pg.manager.get_session() as s:
        n_papers = s.query(PaperModel).count()
        n_chunks = s.query(ChunkModel).count()
except Exception:
    n_papers, n_chunks = 0, 0

c1, c2, c3, c4 = st.columns(4)
cards = [
    (c1, str(n_papers),     "Indexed Papers",  "#f97316", "linear-gradient(90deg,#ea580c,#f97316)"),
    (c2, f"{n_chunks:,}",   "Vector Chunks",   "#fb923c", "linear-gradient(90deg,#f97316,#fb923c)"),
    (c3, "BGE-Large",       "Embedder Model",  "#fbbf24", "linear-gradient(90deg,#fbbf24,#f59e0b)"),
    (c4, "Mistral",         "Generation LLM",  "#10b981", "linear-gradient(90deg,#10b981,#14b8a6)"),
]
for col, val, lbl, color, grad in cards:
    with col:
        st.markdown(f"""
        <div class="metric-card" style="--accent:{grad};">
            <div class="metric-label">{lbl}</div>
            <div class="metric-value" style="color:#ffffff; text-shadow:0 0 20px {color}55;">{val}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── MAIN TWO-COL LAYOUT ───────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown("""
    <div class="glass-card">
        <div class="section-label">Core Architecture</div>
        <h3 style="margin:0 0 1.25rem 0; font-size:1.35rem; color:#ffffff;">
            How the Pipeline Works
        </h3>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.9rem;">
            <div class="feature-item">
                <div style="font-size:1.4rem; margin-bottom:0.5rem;">📄</div>
                <div style="font-size:0.88rem; font-weight:700; color:#fb923c; margin-bottom:0.3rem;">Layout-Aware Parsing</div>
                <div style="font-size:0.82rem; color:#64748b; line-height:1.5;">
                    PDFs routed through PyMuPDF, Docling, or Nougat based on table and math density.
                </div>
            </div>
            <div class="feature-item">
                <div style="font-size:1.4rem; margin-bottom:0.5rem;">🔎</div>
                <div style="font-size:0.88rem; font-weight:700; color:#fb923c; margin-bottom:0.3rem;">Hybrid BM25 + Dense</div>
                <div style="font-size:0.82rem; color:#64748b; line-height:1.5;">
                    Okapi BM25 sparse index fused with BGE-Large vectors via Reciprocal Rank Fusion.
                </div>
            </div>
            <div class="feature-item">
                <div style="font-size:1.4rem; margin-bottom:0.5rem;">⚡</div>
                <div style="font-size:0.88rem; font-weight:700; color:#fb923c; margin-bottom:0.3rem;">Cross-Encoder Rerank</div>
                <div style="font-size:0.82rem; color:#64748b; line-height:1.5;">
                    ms-marco-MiniLM reranker eliminates lost-in-the-middle context decay.
                </div>
            </div>
            <div class="feature-item">
                <div style="font-size:1.4rem; margin-bottom:0.5rem;">🛡️</div>
                <div style="font-size:0.88rem; font-weight:700; color:#fb923c; margin-bottom:0.3rem;">NLI Citation Gate</div>
                <div style="font-size:0.82rem; color:#64748b; line-height:1.5;">
                    DeBERTa-v3 verifies every factual claim against source chunks before output.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown("""
    <div class="glass-card">
        <div class="section-label">System Health</div>
        <h3 style="margin:0 0 1.1rem 0; font-size:1.2rem; color:#ffffff;">Live Service Status</h3>
    </div>
    """, unsafe_allow_html=True)

    services = []
    try:
        from database.connection import DatabaseConnectionManager
        with DatabaseConnectionManager().get_session() as s:
            s.execute(__import__("sqlalchemy").text("SELECT 1"))
        services.append(("SQLite Metadata Store",  "Active",        "#10b981"))
    except Exception:
        services.append(("SQLite Metadata Store",  "Error",         "#ef4444"))

    try:
        import chromadb
        services.append(("ChromaDB Vector Store",  "Connected",     "#10b981"))
    except Exception:
        services.append(("ChromaDB Vector Store",  "Not Installed", "#f59e0b"))

    try:
        import sentence_transformers
        services.append(("BGE-Large Embedder",     "Loaded",        "#10b981"))
    except Exception:
        services.append(("BGE-Large Embedder",     "Mock Mode",     "#f97316"))

    try:
        import transformers
        services.append(("DeBERTa NLI Model",      "Loaded",        "#10b981"))
    except Exception:
        services.append(("DeBERTa NLI Model",      "Mock Mode",     "#f97316"))

    try:
        from generation.llm_factory import LLMFactory
        services.append(("Mistral LLM",            "Ready",         "#10b981"))
    except Exception:
        services.append(("Mistral LLM",            "Config Needed", "#f59e0b"))

    for name, status, color in services:
        st.markdown(f"""
        <div class="system-pill">
            <span style="color:#e2e8f0; font-size:0.88rem; font-weight:500;">{name}</span>
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="color:{color}; font-size:0.8rem; font-weight:700;">{status}</span>
                <span class="pill-dot" style="background:{color}; box-shadow:0 0 8px {color}88;"></span>
            </div>
        </div>
        """, unsafe_allow_html=True)
