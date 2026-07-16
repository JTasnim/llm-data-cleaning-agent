"""
Layer 3 — Human-in-the-Loop Approval Interface

Streamlit app that wires the full agent pipeline into a single interactive
UI. The user uploads a CSV, the pipeline runs automatically, and each
cleaning proposal is displayed as a card with:
  - Confidence tier badge (High / Medium / Low)
  - Issue type and column
  - Plain-English description and proposed fix
  - Transform code
  - Before/after preview from the dry-run
  - Approve / Edit / Reject buttons

Approved proposals are passed to the Layer 4 transform executor.

Run with:
    streamlit run src/hitl_ui/app.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.domain_inference import infer_domain
from src.agent.propose import propose_transforms
from src.agent.verify import verify_all_proposals
from src.profiler.profile import profile_dataset

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Data Cleaning Agent",
    page_icon="🧹",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.tier-high   { background:#D4EDDA; color:#155724; padding:3px 10px;
               border-radius:12px; font-weight:600; font-size:13px; }
.tier-medium { background:#FFF3CD; color:#856404; padding:3px 10px;
               border-radius:12px; font-weight:600; font-size:13px; }
.tier-low    { background:#F8D7DA; color:#721C24; padding:3px 10px;
               border-radius:12px; font-weight:600; font-size:13px; }
.section-label { font-size:12px; color:#888; font-weight:600;
                 text-transform:uppercase; margin-bottom:4px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧹 LLM-Powered Data Cleaning Agent")
st.caption(
    "Gap 2 — Self-verification with confidence scoring  |  "
    "CS 587/387 Independent Study  |  Jubaida Tasnim (160027)"
)
st.divider()

# ── Session state ─────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "df": None,
        "filename": None,
        "profile": None,
        "domain": None,
        "proposals": [],
        "verifications": [],
        "decisions": {},
        "edited_code": {},
        "pipeline_done": False,
        "pipeline_error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Tier badge ────────────────────────────────────────────────────────────────
def tier_badge(tier: str) -> str:
    if tier == "High":
        return '<span class="tier-high">✅ High confidence</span>'
    if tier == "Medium":
        return '<span class="tier-medium">⚠️ Medium confidence</span>'
    return '<span class="tier-low">❌ Low confidence</span>'

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("1 — Upload dataset")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

    st.header("2 — Settings")
    n_sample = st.slider("Sample rows sent to Gemini", 3, 10, 5)
    show_low = st.checkbox("Show Low-confidence proposals", value=True)

    st.divider()
    st.caption("**Pipeline layers**")
    st.caption("🔵 Layer 1 — Statistical profiler")
    st.caption("🟣 Layer 2a — Domain inference (Gemini)")
    st.caption("🟣 Layer 2b — Transform proposals (Gemini)")
    st.caption("🟣 Layer 2c — Dry-run self-verification")
    st.caption("🟢 Layer 3 — This interface (you are here)")
    st.caption("🟠 Layer 4 — Transform executor (next)")

# ── No file uploaded ──────────────────────────────────────────────────────────
if uploaded is None:
    st.info("👈  Upload a CSV file in the sidebar to begin.")
    st.stop()

# ── Reset state when a new file is uploaded ───────────────────────────────────
if uploaded.name != st.session_state.filename:
    st.session_state.filename = uploaded.name
    st.session_state.pipeline_done = False
    st.session_state.pipeline_error = None
    st.session_state.proposals = []
    st.session_state.verifications = []
    st.session_state.decisions = {}
    st.session_state.edited_code = {}

# ── Dataset preview ───────────────────────────────────────────────────────────
preview_df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
with st.expander(
    f"📄 Dataset preview — {uploaded.name}  "
    f"({preview_df.shape[0]} rows × {preview_df.shape[1]} columns)",
    expanded=False
):
    st.dataframe(preview_df.head(20), use_container_width=True)

# ── Run pipeline button ───────────────────────────────────────────────────────
if not st.session_state.pipeline_done:
    if st.button("🚀  Run cleaning pipeline", type="primary", use_container_width=True):
        try:
            progress = st.progress(0, text="Starting pipeline ...")
            progress.progress(10, text="🔵 Layer 1 — running statistical profiler ...")
            df_tmp = pd.read_csv(io.BytesIO(uploaded.getvalue()))
            profile_tmp = profile_dataset(df_tmp)

            progress.progress(30, text="🟣 Layer 2a — inferring domain (Gemini) ...")
            domain_tmp = infer_domain(df_tmp, n_sample_rows=n_sample)

            progress.progress(55, text="🟣 Layer 2b — generating proposals (Gemini) ...")
            proposals_tmp = propose_transforms(df_tmp, profile_tmp, domain_tmp, n_sample_rows=n_sample)

            progress.progress(80, text="🟣 Layer 2c — dry-run self-verification ...")
            verified_tmp = verify_all_proposals(df_tmp, proposals_tmp)

            progress.progress(100, text="✅ Pipeline complete!")

            st.session_state.df = df_tmp
            st.session_state.profile = profile_tmp
            st.session_state.domain = domain_tmp
            st.session_state.proposals = [p for p, _ in verified_tmp]
            st.session_state.verifications = [r for _, r in verified_tmp]
            for i in range(len(st.session_state.proposals)):
                st.session_state.decisions[i] = "pending"
                st.session_state.edited_code[i] = st.session_state.proposals[i].transform_code
            st.session_state.pipeline_done = True
            st.rerun()
        except Exception as e:
            st.session_state.pipeline_error = str(e)

    if st.session_state.pipeline_error:
        st.error(f"Pipeline error: {st.session_state.pipeline_error}")
    st.stop()

# ── Summary metrics ───────────────────────────────────────────────────────────
profile   = st.session_state.profile
domain    = st.session_state.domain
proposals = st.session_state.proposals
verifs    = st.session_state.verifications

high    = sum(1 for p in proposals if p.confidence_tier == "High")
med     = sum(1 for p in proposals if p.confidence_tier == "Medium")
low     = sum(1 for p in proposals if p.confidence_tier == "Low")
pending  = sum(1 for v in st.session_state.decisions.values() if v == "pending")
approved = sum(1 for v in st.session_state.decisions.values() if v == "approved")
rejected = sum(1 for v in st.session_state.decisions.values() if v == "rejected")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Dataset rows", f"{profile.n_rows:,}")
c2.metric("Inferred domain", domain.domain.split("/")[0].strip())
c2.caption(f"Gemini confidence: {domain.domain_confidence}")
c3.metric("Proposals generated", str(len(proposals)))
c3.caption(f"✅ {high} High  ⚠️ {med} Medium  ❌ {low} Low")
c4.metric("Review progress", f"{len(proposals)-pending} / {len(proposals)}")
c4.caption(f"✅ {approved} approved  ❌ {rejected} rejected")

st.divider()
st.subheader("Proposal Cards — Review each suggestion")
st.caption("Approve, edit, or reject each proposal. Only approved proposals will be applied to your data.")

# ── Proposal cards ────────────────────────────────────────────────────────────
for i, (proposal, result) in enumerate(zip(proposals, verifs)):

    if proposal.confidence_tier == "Low" and not show_low:
        continue

    decision = st.session_state.decisions.get(i, "pending")

    border = (
        "#28A745" if decision == "approved" else
        "#DC3545" if decision == "rejected" else
        "#CCCCCC"
    )

    st.markdown(
        f'<div style="border-left:4px solid {border}; padding:12px 16px; '
        f'margin-bottom:4px; background:#1E1E1E; border-radius:0 8px 8px 0;">',
        unsafe_allow_html=True
    )

    # Card header
    h1, h2 = st.columns([6, 2])
    with h1:
        st.markdown(
            f"**Proposal {i+1}** &nbsp;&nbsp;"
            f"{tier_badge(proposal.confidence_tier)}"
            f"&nbsp; `[{proposal.issue_type}]` &nbsp; column: **{proposal.column}**",
            unsafe_allow_html=True
        )
    with h2:
        st.markdown(
            f"Score: **{result.score:.2f}** &nbsp;|&nbsp; "
            f"Rows changed: **{result.rows_affected}**",
            unsafe_allow_html=True
        )

    # Issue + fix
    d1, d2 = st.columns(2)
    with d1:
        st.markdown('<div class="section-label">Issue detected</div>', unsafe_allow_html=True)
        st.write(proposal.description)
    with d2:
        st.markdown('<div class="section-label">Proposed fix</div>', unsafe_allow_html=True)
        st.write(proposal.proposed_fix)

    # Before / after preview
    if result.before_sample or result.after_sample:
        b_col, a_col = st.columns(2)
        with b_col:
            st.markdown('<div class="section-label">Before (sample)</div>', unsafe_allow_html=True)
            st.code(str(result.before_sample), language=None)
        with a_col:
            st.markdown('<div class="section-label">After (dry-run)</div>', unsafe_allow_html=True)
            st.code(str(result.after_sample), language=None)

    # Transform code
    st.markdown('<div class="section-label">Transform code</div>', unsafe_allow_html=True)
    if decision == "editing":
        new_code = st.text_area(
            "Edit code:",
            value=st.session_state.edited_code[i],
            key=f"edit_area_{i}",
            height=80,
            label_visibility="collapsed"
        )
        st.session_state.edited_code[i] = new_code
    else:
        st.code(st.session_state.edited_code[i], language="python")

    # Warnings
    if not result.executed_successfully:
        st.error(f"Dry-run failed: {(result.error_message or '')[:200]}")
    if result.unexpected_columns_changed:
        st.warning(f"⚠️ Side effects on: {result.unexpected_columns_changed}")

    # Action buttons
    b1, b2, b3, b4, _ = st.columns([1.5, 1.5, 1.5, 1.5, 4])
    with b1:
        if st.button("✅ Approve", key=f"approve_{i}", use_container_width=True,
                     type="primary" if decision != "approved" else "secondary"):
            st.session_state.decisions[i] = "approved"
            st.rerun()
    with b2:
        if st.button("✏️ Edit", key=f"edit_btn_{i}", use_container_width=True):
            st.session_state.decisions[i] = "editing"
            st.rerun()
    with b3:
        if decision == "editing":
            if st.button("💾 Save", key=f"save_{i}", use_container_width=True):
                st.session_state.decisions[i] = "approved"
                st.rerun()
    with b4:
        if st.button("❌ Reject", key=f"reject_{i}", use_container_width=True):
            st.session_state.decisions[i] = "rejected"
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    st.divider()

# ── Apply approved transforms ─────────────────────────────────────────────────
approved_idx = [i for i, d in st.session_state.decisions.items() if d == "approved"]
pending_count = sum(1 for d in st.session_state.decisions.values() if d == "pending")

if pending_count > 0:
    st.info(f"⏳ {pending_count} proposal(s) still need a decision.")
elif not approved_idx:
    st.warning("No proposals were approved. Nothing to apply.")
else:
    st.subheader("📦 Apply approved transforms")
    st.success(f"{len(approved_idx)} proposal(s) approved and ready to apply.")

    if st.button("⚙️  Apply approved transforms & export CSV",
                 type="primary", use_container_width=True):

        cleaned_df = st.session_state.df.copy()
        audit_rows = []

        for i in approved_idx:
            proposal = proposals[i]
            code = st.session_state.edited_code[i]
            namespace = {
                "__builtins__": {
                    "abs": abs, "len": len, "list": list, "dict": dict,
                    "range": range, "min": min, "max": max, "sum": sum,
                    "round": round, "int": int, "float": float,
                    "str": str, "bool": bool,
                    "True": True, "False": False, "None": None,
                },
                "df": cleaned_df,
                "pd": pd,
                "np": np,
            }
            try:
                exec(compile(code, "<transform>", "exec"), namespace)
                cleaned_df = namespace["df"]
                status = "✅ applied"
            except Exception as e:
                status = f"❌ failed: {e}"

            audit_rows.append({
                "Proposal #": i + 1,
                "Issue type": proposal.issue_type,
                "Column": proposal.column,
                "Confidence tier": proposal.confidence_tier,
                "Status": status,
                "Code applied": code,
            })

        # Download cleaned CSV
        csv_out = cleaned_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️  Download cleaned CSV",
            data=csv_out,
            file_name=f"cleaned_{st.session_state.filename}",
            mime="text/csv",
            use_container_width=True,
        )

        # Audit log
        st.subheader("📋 Audit log")
        st.dataframe(pd.DataFrame(audit_rows), use_container_width=True)
