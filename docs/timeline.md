# Timeline

9 weeks at roughly 8–10 hours/week, consistent with a 1-credit independent
study (see `PROPOSAL.md` Section 8 for full detail).

| Phase | Weeks | Focus | Deliverable | Status |
|---|---|---|---|---|
| 1 | 1–2 | Setup & data prep | 3 benchmark datasets curated, repo initialized, profiling scripts working | ✅ Complete |
| 2 | 3–5 | Core agent logic | LLM agent detects issues and proposes transforms; self-verification + confidence scoring | ✅ Complete |
| 3 | 5–7 | UI & human-in-the-loop | Streamlit UI wired to agent backend; full pipeline working | 🟡 In progress |
| 4 | 7–9 | Evaluation & write-up | Benchmark results, confidence-tier confusion matrix, final report | ⬜ Not started |

---

## Phase 1 — Complete ✅

| Deliverable | File | Notes |
|---|---|---|
| 3 benchmark datasets curated | `benchmarks/raw/` | Healthcare (Pima, 768 rows), e-commerce (Superstore, 8,400 rows), government (Census Adult Income, 48,842 rows) |
| Synthetic error-injection pipeline | `src/evaluation/error_injection.py` | Missing values, label inconsistencies, outliers, duplicates — seeded RNG, fully reproducible |
| Ground-truth ledger | `benchmarks/ground_truth/` | One ledger CSV per dataset recording every injected error |
| Layer 1 data profiler | `src/profiler/profile.py` | Combined z-score + IQR outlier detection, zero_count field, mixed-type detection, duplicate detection |
| Benchmark build script | `scripts/build_benchmark.py` | One command corrupts all 3 datasets and profiles the output |

---

## Phase 2 — Complete ✅

| Deliverable | File | Notes |
|---|---|---|
| Domain inference module | `src/agent/domain_inference.py` | Gemini 2.5 Flash infers dataset domain and per-column semantics from column names + sample rows |
| Transform proposal module | `src/agent/propose.py` | Combines Layer 1 profile + domain semantics to generate CleaningProposal objects with executable Pandas code |
| Dry-run self-verification | `src/agent/verify.py` | Sandboxed exec(), measures rows affected, side effects, unexpected nulls; maps outcome to High/Medium/Low tier |
| Confidence scoring | `src/agent/confidence.py` | Threshold-based tier assignment (High ≥ 0.85, Medium ≥ 0.50, Low < 0.50) |
| End-to-end demo | `scripts/demo_pipeline.py` | Runs all four layers in sequence; prints proposal cards with verified confidence tiers |

**Key findings from Phase 2:**
- Domain inference (Gemini) correctly flags `BloodPressure = 0` as clinically implausible — a semantic issue the Layer 1 statistical profiler cannot detect, since zero is not a z-score/IQR outlier in that distribution
- Dry-run verifier caught two real bugs: (1) pandas CoW `inplace` silently doing nothing (`rows_affected = 0` → Low tier correctly assigned); (2) `b != a` raising `TypeError` on NaN columns (fixed with `b.ne(a)`)
- First full pipeline run on corrupted Pima dataset: 12 High, 4 Medium, 0 Low proposals across 10 detected issues
- Remaining Medium proposals are Winsorization proposals that affect more rows than stated — a genuine uncertainty signal, not a scoring artifact

---

## Phase 3 — In Progress 🟡

**Goal:** Wire the verified, confidence-scored proposals into an interactive Streamlit approval interface where a human reviewer approves, edits, or rejects each card before any transform is applied.

**Planned deliverables:**

| Deliverable | File | Status |
|---|---|---|
| Streamlit proposal card UI | `src/hitl_ui/app.py` | ⬜ Not started |
| Approve / edit / reject actions | `src/hitl_ui/app.py` | ⬜ Not started |
| Layer 4 transform executor | `src/executor/executor.py` | ⬜ Not started |
| Audit log export | `src/executor/executor.py` | ⬜ Not started |
| Full pipeline wired into one app | `src/hitl_ui/app.py` | ⬜ Not started |

---

## Phase 4 — Not Started ⬜

**Goal:** Run the full pipeline on all 3 benchmark datasets, score proposals against the ground-truth ledger, and build the confidence-tier confusion matrix (Section 7.2.1 of the proposal).

**Planned deliverables:**

| Deliverable | File | Status |
|---|---|---|
| Confidence-tier confusion matrix | `src/evaluation/confusion_matrix.py` | ⬜ Not started |
| Recall / precision vs. baselines | `src/evaluation/confusion_matrix.py` | ⬜ Not started |
| Lightweight user study (n ≥ 3) | TBD | ⬜ Not started |
| Final project report (12–18 pages) | TBD | ⬜ Not started |
