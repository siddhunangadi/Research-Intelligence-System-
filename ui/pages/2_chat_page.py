"""
Chat Page — Research Intelligence System
Grounded conversational interface with NLI citation verification.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
from ui.style_helper import apply_custom_style, render_header
from indexing.postgres_store import PostgresMetadataStore
from retrieval.retrieval_pipeline import RetrievalPipeline
from generation.answer_generator import AnswerGenerator
from validation.citation_validator import CitationValidator
from validation.refusal_handler import RefusalHandler

st.set_page_config(page_title="Grounded Chat — RIS", page_icon="💬", layout="wide", initial_sidebar_state="collapsed")
apply_custom_style()
render_header("chat")

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:1rem 0 2rem 0;">
    <div class="section-label">Conversational RAG</div>
    <h1 class="gradient-text" style="font-size:2.8rem; margin:0 0 0.6rem 0;">
        Grounded Research Assistant
    </h1>
    <p style="color:#64748b; font-size:1.05rem; margin:0;">
        Ask questions answered exclusively from indexed scientific papers.
        Every claim is verified by an NLI entailment model before delivery.
    </p>
</div>
""", unsafe_allow_html=True)

# ── INIT PIPELINES ─────────────────────────────────────────────────────────────
pg_store   = PostgresMetadataStore()
pipeline   = RetrievalPipeline(postgres_store=pg_store)
generator  = AnswerGenerator()
validator  = CitationValidator()
refusal    = RefusalHandler()

papers_list = pg_store.list_all_papers()
paper_opts  = {"🔍 Search Across All Papers": None}
for p in papers_list:
    paper_opts[f"📄 {p['title'][:55]}…" if len(p['title'])>55 else f"📄 {p['title']}"] = p["paper_id"]

# ── QUERY CONTROLS ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="glass-card" style="margin-bottom:1.5rem;">
    <div class="section-label">Query Configuration</div>
</div>
""", unsafe_allow_html=True)

ctrl_left, ctrl_right = st.columns([1, 2], gap="medium")

with ctrl_left:
    selected_label  = st.selectbox("Scope", list(paper_opts.keys()))
    paper_id_filter = paper_opts[selected_label]

with ctrl_right:
    query = st.text_input(
        "Research Question",
        placeholder="e.g. What optimizer and learning rate were used in training?"
    )

run_col, _ = st.columns([1, 3])
with run_col:
    run = st.button("🚀  Run Grounded Search", use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── SEARCH EXECUTION ──────────────────────────────────────────────────────────
if run:
    if not query.strip():
        st.warning("Please enter a research question before searching.")
    else:
        with st.spinner("Running hybrid retrieval · reranking · NLI verification…"):
            try:
                context_str, retrieved = pipeline.retrieve_grounded_context(
                    query=query, paper_id=paper_id_filter
                )

                if not retrieved:
                    st.markdown("""
                    <div class="response-bubble-refused">
                        <span class="status-badge-failed" style="margin-bottom:0.75rem; display:inline-flex;">
                            ✗ No Evidence Found
                        </span>
                        <p style="color:#fca5a5; margin:0.75rem 0 0 0; font-size:1rem; line-height:1.65;">
                            No relevant chunks were recovered from the index for this query.
                            The system is enforcing a zero-evidence refusal.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.stop()

                raw_answer   = generator.generate_answer(query, context_str)
                report       = validator.validate_answer(raw_answer, retrieved)
                final_answer = refusal.process_response(raw_answer, report, retrieved)
                faithful     = final_answer != refusal.fallback_message

                # ── ANSWER BUBBLE ─────────────────────────────────────────────
                st.markdown(f"""
                <div class="section-label" style="margin-top:0.5rem;">Generated Answer</div>
                """, unsafe_allow_html=True)

                if faithful:
                    st.markdown(f"""
                    <div class="response-bubble">
                        <div style="margin-bottom:0.85rem;">
                            <span class="status-badge-passed">✓ NLI Verified — Faithfulness {report['faithfulness_score']:.0%}</span>
                        </div>
                        <div style="font-size:1.05rem; color:#e2e8f0; line-height:1.75;
                                    font-family:'Inter',sans-serif; white-space:pre-wrap;">{final_answer}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="response-bubble-refused">
                        <div style="margin-bottom:0.85rem;">
                            <span class="status-badge-failed">✗ Rejected — Faithfulness {report['faithfulness_score']:.0%} (Below 50% SLA)</span>
                        </div>
                        <div style="font-size:1rem; color:#fca5a5; line-height:1.7;
                                    font-style:italic;">{final_answer}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── INSPECTION GRID ───────────────────────────────────────────
                st.markdown("""
                <div class="section-label" style="margin-top:1.75rem;">Diagnostic Inspection</div>
                """, unsafe_allow_html=True)

                audit_col, chunks_col = st.columns(2, gap="large")

                # --- Citation Audit ---
                with audit_col:
                    st.markdown("""
                    <div class="glass-card" style="height:100%;">
                        <div class="section-label">NLI Citation Audit</div>
                        <h4 style="margin:0 0 0.9rem 0; font-size:1.05rem; color:#f1f5f9;">
                            🛡️ Claim Verification Log
                        </h4>
                    </div>
                    """, unsafe_allow_html=True)

                    # Faithfulness score ring (CSS only)
                    score_pct = int(report['faithfulness_score'] * 100)
                    ring_color = "#10b981" if report['is_faithful'] else "#ef4444"
                    badge_cls  = "status-badge-passed" if report['is_faithful'] else "status-badge-failed"
                    badge_txt  = "FAITHFUL" if report['is_faithful'] else "UNFAITHFUL"

                    st.markdown(f"""
                    <div style="display:flex; align-items:center; gap:1.25rem; margin-bottom:1.25rem;
                                padding:1rem; background:rgba(0,0,0,0.2); border-radius:14px;">
                        <div style="position:relative; width:72px; height:72px; flex-shrink:0;">
                            <svg width="72" height="72" viewBox="0 0 72 72">
                                <circle cx="36" cy="36" r="28" fill="none"
                                        stroke="rgba(255,255,255,0.05)" stroke-width="8"/>
                                <circle cx="36" cy="36" r="28" fill="none"
                                        stroke="{ring_color}" stroke-width="8"
                                        stroke-dasharray="{int(score_pct*1.759)} 175.9"
                                        stroke-linecap="round"
                                        transform="rotate(-90 36 36)"/>
                            </svg>
                            <div style="position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
                                        font-family:'Outfit',sans-serif; font-size:0.85rem;
                                        font-weight:800; color:{ring_color};">{score_pct}%</div>
                        </div>
                        <div>
                            <div style="margin-bottom:0.35rem;">
                                <span class="{badge_cls}">{badge_txt}</span>
                            </div>
                            <div style="font-size:0.8rem; color:#475569; line-height:1.5;">
                                {len(report['verified_claims'])} verified &nbsp;·&nbsp;
                                {len(report['unverified_claims'])} unverified
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if report["verified_claims"]:
                        st.markdown("""
                        <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase;
                                    letter-spacing:0.1em; color:#10b981; margin-bottom:0.5rem;">
                            ✓ Verified Claims
                        </div>
                        """, unsafe_allow_html=True)
                        for vc in report["verified_claims"]:
                            st.markdown(f"""
                            <div class="claim-verified">
                                <div style="color:#d1fae5; font-size:0.88rem; line-height:1.5; margin-bottom:0.3rem;">
                                    "{vc['claim']['clean_claim']}"
                                </div>
                                <div style="font-size:0.76rem; color:#6ee7b7;">
                                    Source:&nbsp;<code style="background:rgba(0,0,0,0.3); padding:1px 6px;
                                    border-radius:4px; color:#a7f3d0;">{vc['verified_by']}</code>
                                    &nbsp;·&nbsp; Confidence:&nbsp;<strong style="color:#10b981;">{vc['score']:.2f}</strong>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                    if report["unverified_claims"]:
                        st.markdown("""
                        <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase;
                                    letter-spacing:0.1em; color:#ef4444; margin:0.9rem 0 0.5rem 0;">
                            ✗ Unverified Claims
                        </div>
                        """, unsafe_allow_html=True)
                        for uc in report["unverified_claims"]:
                            st.markdown(f"""
                            <div class="claim-unverified">
                                <div style="color:#fecaca; font-size:0.88rem; line-height:1.5; margin-bottom:0.3rem;">
                                    "{uc['claim']['clean_claim']}"
                                </div>
                                <div style="font-size:0.76rem; color:#f87171;">
                                    Reason: <strong>{uc['reason']}</strong>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                # --- Source Chunks ---
                with chunks_col:
                    st.markdown(f"""
                    <div class="glass-card" style="height:100%;">
                        <div class="section-label">Retrieved Context</div>
                        <h4 style="margin:0 0 0.9rem 0; font-size:1.05rem; color:#f1f5f9;">
                            📖 Source Chunks &nbsp;
                            <span style="font-size:0.8rem; color:#475569; font-weight:400;">
                                ({len(retrieved)} retrieved)
                            </span>
                        </h4>
                    </div>
                    """, unsafe_allow_html=True)

                    for idx, chunk in enumerate(retrieved, 1):
                        cid    = chunk.get("chunk_id", "?")
                        sec    = chunk.get("section", "?")
                        page   = chunk.get("page_number", "?")
                        score  = chunk.get("rerank_score", chunk.get("rrf_score", 0.0))
                        color  = "#14b8a6" if score > 0.05 else "#64748b"

                        st.markdown(f"""
                        <div class="chunk-card">
                            <div style="display:flex; justify-content:space-between;
                                        align-items:center; margin-bottom:0.5rem;">
                                <span style="font-family:'JetBrains Mono',monospace; font-size:0.75rem;
                                             background:rgba(59,130,246,0.12); color:#60a5fa;
                                             padding:2px 8px; border-radius:6px; font-weight:600;">
                                    #{idx} &nbsp; {cid[:24]}
                                </span>
                                <span style="font-size:0.78rem; color:{color}; font-weight:700;">
                                    ↑ {score:.4f}
                                </span>
                            </div>
                            <div style="font-size:0.78rem; color:#475569; margin-bottom:0.5rem;">
                                <span style="background:rgba(255,255,255,0.04); padding:2px 7px;
                                             border-radius:5px;">{sec}</span>
                                &nbsp;·&nbsp; pg&nbsp;{page}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        preview = chunk.get("content", "")[:300]
                        st.markdown(f"""
                        <div style="font-family:'JetBrains Mono',monospace; font-size:0.76rem;
                                    color:#475569; background:rgba(0,0,0,0.3); border-radius:10px;
                                    padding:0.65rem 0.9rem; margin-bottom:0.6rem; line-height:1.6;
                                    border:1px solid rgba(255,255,255,0.04); word-break:break-word;">
                            {preview}{"…" if len(chunk.get("content",""))>300 else ""}
                        </div>
                        """, unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f"""
                <div style="margin-top:1rem; padding:1rem; background:rgba(239,68,68,0.08);
                            border:1px solid rgba(239,68,68,0.25); border-radius:12px;">
                    <span class="status-badge-failed">System Error</span>
                    <p style="color:#fca5a5; margin:0.5rem 0 0 0; font-size:0.9rem;
                               font-family:'JetBrains Mono',monospace;">{e}</p>
                </div>
                """, unsafe_allow_html=True)
