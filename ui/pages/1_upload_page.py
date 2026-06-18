"""
Upload Page — Research Intelligence System
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
from ui.style_helper import apply_custom_style, render_header
from indexing.index_builder import IndexBuilder

st.set_page_config(page_title="Upload Papers — RIS", page_icon="📤", layout="wide", initial_sidebar_state="collapsed")
apply_custom_style()
render_header("upload")

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:1rem 0 2rem 0;">
    <div class="section-label">Document Ingestion</div>
    <h1 class="gradient-text" style="font-size:2.8rem; margin:0 0 0.6rem 0;">Upload Research Papers</h1>
    <p style="color:#64748b; font-size:1.05rem; margin:0;">
        Submit PDFs to parse layouts, extract canonical sections, generate embeddings,
        and sync into the retrieval index.
    </p>
</div>
""", unsafe_allow_html=True)

# ── DROP ZONE ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="glass-card" style="margin-bottom:1.25rem;">
    <div class="section-label">File Selection</div>
    <h3 style="margin:0 0 0.35rem 0; font-size:1.15rem; color:#ffffff;">
        Drop PDFs here or click to browse
    </h3>
    <p style="color:#475569; font-size:0.85rem; margin:0;">
        Supported: PDF · Max 200 MB per file
    </p>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Select PDF files to upload",
    type=["pdf"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

if uploaded_files:
    st.markdown(f"""
    <div style="margin-top:0.75rem; padding:0.7rem 1rem;
                background:rgba(249,115,22,0.08); border:1px solid rgba(249,115,22,0.22);
                border-radius:12px; color:#fb923c; font-size:0.9rem; font-weight:600;">
        📎 &nbsp;{len(uploaded_files)} file{'s' if len(uploaded_files)>1 else ''} selected
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("⚡  Process & Index Documents", use_container_width=True):
    if not uploaded_files:
        st.warning("Please select one or more PDF documents first.")
    else:
        builder = IndexBuilder()
        os.makedirs("./data/raw_papers", exist_ok=True)

        for uf in uploaded_files:
            cached_path = os.path.join("./data/raw_papers", uf.name)

            st.markdown(f"""
            <div style="margin-top:1.5rem; padding:1rem 1.25rem;
                        background:rgba(15,15,25,0.7); border:1px solid rgba(255,255,255,0.06);
                        border-radius:16px; border-left:4px solid #f97316;">
                <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.1em;
                            color:#f97316; font-weight:700; margin-bottom:0.3rem;">Processing</div>
                <div style="font-size:1rem; font-weight:600; color:#e2e8f0;">📄 {uf.name}</div>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("Writing to disk…"):
                try:
                    with open(cached_path, "wb") as f:
                        f.write(uf.getbuffer())
                except Exception as e:
                    st.error(f"Write failed: {e}")
                    continue

            steps = [
                ("🔍", "Parsing page layouts…"),
                ("📂", "Labelling canonical sections…"),
                ("⚡", "Computing BGE-Large embeddings…"),
                ("💾", "Syncing SQLite + ChromaDB…"),
            ]
            with st.status(f"Ingesting {uf.name}…", expanded=True) as status:
                for icon, msg in steps:
                    st.markdown(f"""
                    <div class="step-item">
                        <span style="font-size:1.1rem;">{icon}</span>
                        <span style="color:#94a3b8; font-size:0.9rem;">{msg}</span>
                    </div>
                    """, unsafe_allow_html=True)

                success = builder.build_index_for_pdf(cached_path)

                if success:
                    status.update(label=f"✓ '{uf.name}' indexed!", state="complete", expanded=False)
                    st.markdown(f"""
                    <div style="margin-top:0.75rem;">
                        <span class="status-badge-passed">✓ Registered: '{uf.name}'</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    status.update(label=f"✗ Failed — '{uf.name}'", state="error", expanded=True)
                    st.markdown(f"""
                    <div style="margin-top:0.75rem;">
                        <span class="status-badge-failed">✗ Indexing failed for '{uf.name}'</span>
                    </div>
                    """, unsafe_allow_html=True)
