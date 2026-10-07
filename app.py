"""
app.py
------
AI Legal Aid Multi-Agent System — Streamlit Dashboard
"""

from __future__ import annotations

import io
import os
import sys

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Legal Aid — Multi-Agent System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Lazy imports (avoid import errors before requirements are installed)
# ---------------------------------------------------------------------------
try:
    from agents.contract_reviewer import review_contract, RISK_COLORS
    from agents.compliance_checker import check_compliance
    from agents.qa_rag import index_document, answer_question, clear_index, get_indexed_doc_count
    from agents.summarizer import summarize_document
    from utils.granite import get_client
    _IMPORTS_OK = True
except ImportError as _e:
    _IMPORTS_OK = False
    _IMPORT_ERROR = str(_e)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def _init_state() -> None:
    defaults = {
        "document_text": "",
        "document_name": "",
        "review_result": None,
        "compliance_result": None,
        "summary_result": None,
        "analysis_done": False,
        "chat_history": [],
        "indexed": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg",
        width=80,
    )
    st.title("⚖️ AI Legal Aid")
    st.markdown("**Multi-Agent System**")
    st.divider()

    client = get_client() if _IMPORTS_OK else None
    if client and client.is_online:
        st.success("🟢 IBM Granite — Online")
    else:
        st.warning("🟡 IBM Granite — Offline (Mock Mode)")
        st.caption(
            "Add credentials to `.env` to enable real AI responses.\n"
            "See `.env.example` for required variables."
        )

    st.divider()
    st.markdown("**Agents**")
    st.markdown("1. 📄 Contract Reviewer")
    st.markdown("2. ⚖️ Compliance Checker")
    st.markdown("3. 🔍 RAG Q&A Engine")
    st.markdown("4. 📝 Plain-English Summarizer")
    st.divider()
    st.caption("Built with IBM Granite · LangChain · ChromaDB · Streamlit")

# ---------------------------------------------------------------------------
# Guard: check imports
# ---------------------------------------------------------------------------
if not _IMPORTS_OK:
    st.error(f"❌ Import error: {_IMPORT_ERROR}")
    st.info("Run `pip install -r requirements.txt` to install dependencies.")
    st.stop()

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _extract_text_from_file(uploaded_file) -> str:
    """Extract plain text from .txt or .pdf uploads."""
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(uploaded_file.read()))
            return "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
        except Exception as e:
            st.error(f"PDF extraction failed: {e}")
            return ""
    else:
        return uploaded_file.read().decode("utf-8", errors="replace")


def _risk_badge(level: str) -> str:
    colors = {"HIGH": "#d9534f", "MEDIUM": "#f0ad4e", "LOW": "#5cb85c", "INFO": "#5bc0de"}
    color = colors.get(level, "#aaa")
    return (
        f'<span style="background:{color};color:#fff;padding:2px 8px;'
        f'border-radius:4px;font-size:0.75rem;font-weight:bold;">{level}</span>'
    )


def _severity_badge(severity: str) -> str:
    colors = {"VIOLATION": "#d9534f", "WARNING": "#f0ad4e", "NOTE": "#5bc0de"}
    color = colors.get(severity, "#aaa")
    return (
        f'<span style="background:{color};color:#fff;padding:2px 8px;'
        f'border-radius:4px;font-size:0.75rem;font-weight:bold;">{severity}</span>'
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📁 Document Upload", "🤖 Multi-Agent Analysis", "📊 Risk Breakdown", "💬 Legal Q&A"]
)

# ============================================================
# TAB 1 — Document Upload
# ============================================================
with tab1:
    st.header("📁 Document Upload")
    st.markdown(
        "Upload a legal agreement (PDF or TXT) or paste text directly. "
        "A sample NDA is included for immediate testing."
    )

    col_upload, col_sample = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload a legal document",
            type=["txt", "pdf"],
            help="Supported: plain text (.txt) and PDF (.pdf)",
        )

    with col_sample:
        st.markdown("**Or use the sample NDA:**")
        if st.button("📄 Load Sample NDA", use_container_width=True):
            sample_path = os.path.join(os.path.dirname(__file__), "samples", "sample_agreement.txt")
            try:
                with open(sample_path, "r", encoding="utf-8") as f:
                    st.session_state.document_text = f.read()
                    st.session_state.document_name = "sample_agreement.txt"
                    st.session_state.analysis_done = False
                    st.session_state.indexed = False
                st.success("✅ Sample NDA loaded!")
            except FileNotFoundError:
                st.error("Sample file not found.")

    # Handle file upload
    if uploaded_file is not None:
        text = _extract_text_from_file(uploaded_file)
        if text:
            st.session_state.document_text = text
            st.session_state.document_name = uploaded_file.name
            st.session_state.analysis_done = False
            st.session_state.indexed = False
            st.success(f"✅ Uploaded: **{uploaded_file.name}** ({len(text):,} characters)")

    # Text area for direct paste
    st.markdown("**Or paste / edit text directly:**")
    pasted = st.text_area(
        "Contract text",
        value=st.session_state.document_text,
        height=300,
        placeholder="Paste your legal agreement text here…",
        label_visibility="collapsed",
    )
    if pasted != st.session_state.document_text:
        st.session_state.document_text = pasted
        st.session_state.document_name = "pasted_text.txt"
        st.session_state.analysis_done = False
        st.session_state.indexed = False

    # Document stats
    if st.session_state.document_text:
        words = len(st.session_state.document_text.split())
        chars = len(st.session_state.document_text)
        c1, c2, c3 = st.columns(3)
        c1.metric("Characters", f"{chars:,}")
        c2.metric("Words", f"{words:,}")
        c3.metric("Document", st.session_state.document_name or "—")

        st.info(
            "👉 Go to the **Multi-Agent Analysis** tab to run all four AI agents on this document."
        )

# ============================================================
# TAB 2 — Multi-Agent Analysis
# ============================================================
with tab2:
    st.header("🤖 Multi-Agent Analysis")

    if not st.session_state.document_text:
        st.warning("⚠️ No document loaded. Please upload or paste a document in the **Document Upload** tab.")
    else:
        st.markdown(
            f"**Document:** `{st.session_state.document_name}` — "
            f"{len(st.session_state.document_text):,} characters"
        )

        if st.button("🚀 Run All Agents", type="primary", use_container_width=True):
            text = st.session_state.document_text

            with st.status("Running multi-agent analysis…", expanded=True) as status:
                # Agent 1: Contract Reviewer
                st.write("📄 Agent 1: Contract Reviewer — scanning for risky clauses…")
                try:
                    st.session_state.review_result = review_contract(text)
                    st.write(
                        f"   ✅ Found {len(st.session_state.review_result.risky_clauses)} notable clauses "
                        f"(Overall risk: {st.session_state.review_result.overall_risk})"
                    )
                except Exception as e:
                    st.write(f"   ❌ Error: {e}")

                # Agent 2: Compliance Checker
                st.write("⚖️ Agent 2: Compliance Checker — checking statutory compliance…")
                try:
                    st.session_state.compliance_result = check_compliance(text)
                    st.write(
                        f"   ✅ Status: {st.session_state.compliance_result.overall_status} — "
                        f"{len(st.session_state.compliance_result.issues)} issues found"
                    )
                except Exception as e:
                    st.write(f"   ❌ Error: {e}")

                # Agent 3: RAG Indexer
                st.write("🔍 Agent 3: RAG Engine — indexing document into vector store…")
                try:
                    clear_index()
                    n_chunks = index_document(text, source_name=st.session_state.document_name)
                    st.session_state.indexed = True
                    st.write(f"   ✅ Indexed {n_chunks} chunks into ChromaDB")
                except Exception as e:
                    st.write(f"   ❌ Indexing error: {e}")
                    st.session_state.indexed = False

                # Agent 4: Summarizer
                st.write("📝 Agent 4: Summarizer — generating plain-English summary…")
                try:
                    st.session_state.summary_result = summarize_document(text)
                    st.write("   ✅ Summary generated")
                except Exception as e:
                    st.write(f"   ❌ Error: {e}")

                st.session_state.analysis_done = True
                status.update(label="✅ All agents completed!", state="complete", expanded=False)

        # Display results if analysis has been run
        if st.session_state.analysis_done:
            st.divider()

            # ── Summary ──
            if st.session_state.summary_result:
                sr = st.session_state.summary_result
                st.subheader("📝 Plain-English Summary")

                meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
                meta_col1.metric("Document Type", sr.document_type)
                meta_col2.metric("Effective Date", sr.effective_date)
                meta_col3.metric("Term", sr.term_duration)
                meta_col4.metric("Governing Law", sr.governing_law)

                if sr.parties:
                    st.markdown(f"**Parties:** {' · '.join(sr.parties)}")

                st.markdown(sr.plain_english_summary)

                if sr.key_points:
                    st.markdown("**Key Points:**")
                    cat_icons = {
                        "Obligation": "📌",
                        "Restriction": "🚫",
                        "Right": "✅",
                        "Deadline": "⏰",
                        "Risk": "⚠️",
                        "Warning": "⚠️",
                    }
                    for kp in sr.key_points:
                        icon = cat_icons.get(kp.category, "•")
                        st.markdown(f"{icon} **{kp.category}:** {kp.point}")

                st.info(sr.disclaimer)

            st.divider()

            # ── Contract Review quick view ──
            if st.session_state.review_result:
                rr = st.session_state.review_result
                st.subheader("📄 Contract Review — Quick View")
                oc = rr.overall_risk
                oc_color = RISK_COLORS.get(oc, "#aaa")
                st.markdown(
                    f'**Overall Risk:** <span style="color:{oc_color};font-weight:bold;font-size:1.1rem;">'
                    f'{oc}</span>',
                    unsafe_allow_html=True,
                )
                if rr.risky_clauses:
                    for clause in rr.risky_clauses[:5]:  # top 5 in this tab; full list in Risk Breakdown
                        with st.expander(
                            f"{_risk_badge(clause.risk_level)} &nbsp; {clause.clause_title}",
                            expanded=False,
                        ):
                            st.markdown(f"*{clause.excerpt}*")
                            st.markdown(clause.explanation)
                    if len(rr.risky_clauses) > 5:
                        st.caption(
                            f"Showing 5 of {len(rr.risky_clauses)} clauses. "
                            "See **Risk Breakdown** tab for full details."
                        )

            st.divider()

            # ── Compliance quick view ──
            if st.session_state.compliance_result:
                cr = st.session_state.compliance_result
                st.subheader("⚖️ Compliance Check — Quick View")
                status_colors = {
                    "NON_COMPLIANT": "#d9534f",
                    "WARNINGS": "#f0ad4e",
                    "COMPLIANT": "#5cb85c",
                }
                sc = status_colors.get(cr.overall_status, "#aaa")
                st.markdown(
                    f'**Overall Status:** <span style="color:{sc};font-weight:bold;font-size:1.1rem;">'
                    f'{cr.overall_status.replace("_", " ")}</span>',
                    unsafe_allow_html=True,
                )
                if cr.issues:
                    for issue in cr.issues[:4]:
                        with st.expander(
                            f"{_severity_badge(issue.severity)} &nbsp; {issue.rule.statute}",
                            expanded=False,
                        ):
                            st.markdown(f"**Triggered by:** `{issue.triggered_by}`")
                            st.markdown(f"**Guidance:** {issue.rule.guidance}")
                            if issue.llm_analysis:
                                st.markdown(f"**AI Analysis:** {issue.llm_analysis}")


# ============================================================
# TAB 3 — Risk Breakdown
# ============================================================
with tab3:
    st.header("📊 Risk Breakdown")

    if not st.session_state.analysis_done:
        st.info("Run the **Multi-Agent Analysis** first to see the risk breakdown.")
    else:
        rr = st.session_state.review_result
        cr = st.session_state.compliance_result

        if rr:
            st.subheader("Contract Risk Metrics")
            counts = rr.risk_counts()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🔴 HIGH", counts["HIGH"])
            m2.metric("🟠 MEDIUM", counts["MEDIUM"])
            m3.metric("🟢 LOW", counts["LOW"])
            m4.metric("🔵 INFO", counts["INFO"])

            # Risk distribution bar (pure HTML/CSS — no external chart lib needed)
            total = max(sum(counts.values()), 1)
            st.markdown("**Risk Distribution**")
            bar_html = '<div style="display:flex;height:24px;border-radius:4px;overflow:hidden;width:100%;">'
            for level, color in RISK_COLORS.items():
                pct = counts[level] / total * 100
                if pct > 0:
                    bar_html += (
                        f'<div style="width:{pct:.1f}%;background:{color};'
                        f'display:flex;align-items:center;justify-content:center;'
                        f'color:#fff;font-size:0.7rem;font-weight:bold;">'
                        f'{counts[level]} {level}</div>'
                    )
            bar_html += "</div>"
            st.markdown(bar_html, unsafe_allow_html=True)
            st.markdown("")

            # Full clause list
            st.subheader("All Identified Risky Clauses")
            if not rr.risky_clauses:
                st.success("No risky clauses identified.")
            else:
                for i, clause in enumerate(rr.risky_clauses, 1):
                    with st.expander(
                        f"{i}. {_risk_badge(clause.risk_level)} &nbsp; {clause.clause_title}",
                        expanded=clause.risk_level == "HIGH",
                    ):
                        st.markdown(
                            f'<div style="background:#f7f8fa;border-left:4px solid {clause.color};'
                            f'padding:10px 14px;border-radius:4px;font-style:italic;margin-bottom:8px;">'
                            f'{clause.excerpt}</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"**Why it matters:** {clause.explanation}")

        st.divider()

        if cr:
            st.subheader("Compliance Issues")
            icounts = cr.issue_counts()

            c1, c2, c3 = st.columns(3)
            c1.metric("🔴 Violations", icounts["VIOLATION"])
            c2.metric("🟠 Warnings", icounts["WARNING"])
            c3.metric("🔵 Notes", icounts["NOTE"])

            if not cr.issues:
                st.success("No compliance issues found.")
            else:
                for issue in cr.issues:
                    with st.expander(
                        f"{_severity_badge(issue.severity)} &nbsp; {issue.rule.statute} "
                        f"({issue.rule.jurisdiction})",
                        expanded=issue.severity == "VIOLATION",
                    ):
                        st.markdown(f"**Rule ID:** `{issue.rule.rule_id}`")
                        st.markdown(f"**Triggered by:** `{issue.triggered_by}`")
                        st.markdown(f"**Description:** {issue.rule.description}")
                        st.markdown(f"**Guidance:** {issue.rule.guidance}")
                        if issue.llm_analysis:
                            st.markdown(f"**AI Analysis:** {issue.llm_analysis}")

            if cr.compliant_rules:
                with st.expander(f"✅ {len(cr.compliant_rules)} rules with no issues detected"):
                    for rule in cr.compliant_rules:
                        st.markdown(f"- **{rule.rule_id}**: {rule.statute} ({rule.jurisdiction})")

        # Raw LLM outputs (collapsible)
        with st.expander("🔬 Raw LLM Output — Contract Reviewer"):
            st.code(rr.raw_llm_output if rr else "No output available.", language="text")

        with st.expander("🔬 Raw LLM Output — Compliance Checker"):
            st.code(cr.raw_llm_output if cr else "No output available.", language="text")


# ============================================================
# TAB 4 — Legal Q&A
# ============================================================
with tab4:
    st.header("💬 Legal Q&A")
    st.markdown(
        "Ask questions about the uploaded document. Answers are grounded in the "
        "document content via semantic retrieval (RAG)."
    )

    if not st.session_state.indexed:
        if not st.session_state.document_text:
            st.warning("⚠️ No document loaded. Upload a document and run Multi-Agent Analysis first.")
        else:
            st.warning(
                "⚠️ Document has not been indexed yet. "
                "Run **Multi-Agent Analysis** (Tab 2) to index the document for Q&A."
            )
            # Allow manual indexing
            if st.button("🔍 Index Document Now"):
                with st.spinner("Indexing…"):
                    try:
                        clear_index()
                        n = index_document(
                            st.session_state.document_text,
                            source_name=st.session_state.document_name,
                        )
                        st.session_state.indexed = True
                        st.success(f"✅ Indexed {n} chunks. You can now ask questions!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Indexing failed: {e}")
    else:
        indexed_count = get_indexed_doc_count()
        st.success(f"✅ Document indexed — {indexed_count} chunks in vector store.")

        # Pre-built example questions
        st.markdown("**Quick questions:**")
        example_qs = [
            "What are the main obligations of the receiving party?",
            "How long does the non-compete clause last?",
            "What is the governing law for disputes?",
            "Can Acme Corp modify the agreement without notice?",
            "What happens to confidential information after termination?",
        ]
        q_cols = st.columns(len(example_qs))
        for col, q in zip(q_cols, example_qs):
            if col.button(q, key=f"eq_{q[:20]}", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": q})
                with st.spinner("Thinking…"):
                    result = answer_question(q)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": result.answer, "sources": result.sources}
                )
                st.rerun()

        st.divider()

        # Chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources"):
                    with st.expander("📎 Source excerpts used"):
                        for src in msg["sources"]:
                            st.markdown(
                                f'<div style="background:#f7f8fa;border-left:3px solid #3b82d4;'
                                f'padding:8px 12px;margin-bottom:6px;border-radius:3px;font-size:0.85rem;">'
                                f'<strong>Chunk {src.chunk_id}</strong> from <em>{src.source}</em> '
                                f'(similarity: {1 - src.distance:.2%})<br>{src.text[:300]}…</div>',
                                unsafe_allow_html=True,
                            )

        # Input
        user_input = st.chat_input("Ask a question about the document…")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.spinner("Searching document and generating answer…"):
                result = answer_question(user_input)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": result.answer, "sources": result.sources}
            )
            st.rerun()

        # Clear chat
        if st.session_state.chat_history:
            if st.button("🗑️ Clear chat history"):
                st.session_state.chat_history = []
                st.rerun()
