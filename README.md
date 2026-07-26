# LLM-Powered Data Cleaning Agent with Human-in-the-Loop Approval

> **MS Computer Science Independent Study (CS 587/387) — Summer 2026**
> Student: Jubaida Tasnim (160027) · Supervisor: Dr. Satarupa Mukherjee · SFBU

---

## Research Question

> Can a dry-run self-verification loop produce a calibrated confidence score for LLM-generated data-cleaning proposals — one where high confidence reliably predicts correctness?

**Research focus:** Gap 2 — No self-verification with confidence scoring
**Theme:** Theme 2 — Agentic LLM Systems and Tool Use

---

## Key Results

| Metric                             | Value                                              |
| ---------------------------------- | -------------------------------------------------- |
| Overall precision (strict scoring) | **94.1%** (16/17 scoreable proposals)              |
| High-tier precision                | **100%** (15/15)                                   |
| Low-tier precision                 | **50%** (1/2)                                      |
| Improvement over naive baseline    | **+44.1pp** (94.1% vs 50.0%)                       |
| Natural issues detected beyond GT  | **18** (domain-implausible zeros, type mismatches) |
| Datasets evaluated                 | 3 (healthcare, e-commerce, government)             |

**H₁ ✅** — High confidence reliably predicts correctness (100% vs 50%)
**H₂ ✅** — Dry-run verification outperforms naive baseline by +44.1pp
**H₃ ✅** — LLM domain reasoning catches domain-implausible issues statistics misses

---

## System Architecture

```
Upload CSV
    │
    ▼  Layer 1 ── Statistical Profiler
    │             (z-score + IQR, zero_count, duplicates)
    │
    ▼  Layer 2a ── Domain Inference (Gemini)
    │              (dataset domain + per-column semantics)
    │
    ▼  Layer 2b ── Transform Proposals (Gemini)
    │              (CleaningProposal objects with executable Pandas code)
    │
    ▼  Layer 2c ── Dry-Run Self-Verification
    │              (sandboxed exec → rows_affected, side effects, errors)
    │              → Confidence tier: High ≥ 0.85 / Medium ≥ 0.50 / Low < 0.50
    │
    ▼  Layer 3 ── Streamlit HITL Approval UI
    │             (human approves / edits / rejects each proposal card)
    │
    ▼  Layer 4 ── Transform Executor + Audit Log
                  (applies approved fixes → cleaned CSV)
```

---

## Project Structure

```
llm-data-cleaning-agent/
├── benchmarks/
│   ├── raw/                          # Clean source datasets (committed)
│   ├── corrupted/                    # Synthetically corrupted versions (gitignored)
│   └── ground_truth/                 # Ground-truth error ledgers (committed)
├── src/
│   ├── profiler/
│   │   └── profile.py                # Layer 1 — statistical profiler
│   ├── agent/
│   │   ├── domain_inference.py       # Layer 2a — Gemini domain inference
│   │   ├── propose.py                # Layer 2b — transform proposals
│   │   ├── verify.py                 # Layer 2c — dry-run self-verification
│   │   └── confidence.py             # Confidence tier scoring
│   ├── hitl_ui/
│   │   └── app.py                    # Layer 3 — Streamlit HITL UI
│   ├── executor/
│   │   └── executor.py               # Layer 4 — transform executor
│   └── evaluation/
│       ├── error_injection.py        # Synthetic error injection pipeline
│       └── confusion_matrix.py       # Confidence-tier confusion matrix
├── scripts/
│   ├── build_benchmark.py            # Build all 3 corrupted benchmark datasets
│   ├── demo_pipeline.py              # End-to-end pipeline demo (CLI)
│   ├── demo_baseline.py              # Naive LLM baseline (control condition)
│   ├── score_proposals.py            # Score proposals against ground truth
│   ├── compare_baseline.py           # Full system vs. baseline comparison
│   ├── demo_domain_inference.py      # Domain inference demo
│   ├── demo_propose.py               # Proposal generation demo
│   └── test_gemini_key.py            # API key connectivity test
├── docs/
│   ├── architecture.md
│   ├── timeline.md
│   └── figures/
│       ├── confusion_matrix_precision.png
│       └── reliability_diagram.png
├── tests/                            # Unit tests (36 tests, all passing)
├── .env.example
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/JTasnim/llm-data-cleaning-agent.git
cd llm-data-cleaning-agent

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Add your free Gemini API key from https://aistudio.google.com/app/apikey
```

### Verify API key

```bash
python scripts/test_gemini_key.py
```

---

## Run

### Build the benchmark datasets

```bash
python scripts/build_benchmark.py
```

Corrupts all 3 raw datasets with seed=42 and writes ground-truth ledgers.

### Run the full pipeline (CLI)

```bash
# Healthcare dataset (corrupted version)
python scripts/demo_pipeline.py --corrupted

# E-commerce dataset
python scripts/demo_pipeline.py --corrupted --dataset ecommerce_superstore_sales.csv

# Government dataset
python scripts/demo_pipeline.py --corrupted --dataset government_adult_income.csv

# Save output to file
python scripts/demo_pipeline.py --corrupted | tee outputs/pipeline_run.txt
```

### Run the Streamlit HITL UI

```bash
streamlit run src/hitl_ui/app.py
```

Opens at `http://localhost:8501`. Upload any CSV to start.

### Run the Phase 4 evaluation

```bash
# Score proposals against ground-truth ledger
python scripts/score_proposals.py

# Build confusion matrix + reliability diagram
python src/evaluation/confusion_matrix.py

# Run naive baseline
python scripts/demo_baseline.py --all --corrupted

# Compare full system vs baseline
python scripts/compare_baseline.py
```

---

## Datasets

| File                             | Domain     | Rows   | Source                             |
| -------------------------------- | ---------- | ------ | ---------------------------------- |
| `healthcare_pima_diabetes.csv`   | Healthcare | 768    | UCI ML Repository (NIDDK)          |
| `ecommerce_superstore_sales.csv` | E-commerce | 8,400  | Tableau sample data                |
| `government_adult_income.csv`    | Government | 48,842 | UCI ML Repository (1994 US Census) |

All datasets are publicly available and de-identified. See `benchmarks/SOURCES.md` for full provenance.

---

## Tech Stack (100% free tier)

| Component       | Technology                                        |
| --------------- | ------------------------------------------------- |
| LLM backbone    | Google Gemini 2.5 Flash (`google-genai >= 1.0.0`) |
| Agent framework | LangGraph / LangChain                             |
| Data layer      | Python, pandas 2.2+ (CoW-aware), NumPy            |
| UI              | Streamlit 1.51+                                   |
| Evaluation      | scikit-learn, matplotlib, seaborn                 |
| Version control | GitHub                                            |

> **Note:** `google-generativeai` was deprecated (EOL November 30, 2025). This project uses the current `google-genai` SDK.

---

## Key Technical Findings

- **Pandas 2.0+ Copy-on-Write (CoW):** `df['col'].fillna(x, inplace=True)` silently does nothing — the dry-run verifier catches this as `rows_affected=0` → Low confidence
- **NaN-safe comparison:** `b != a` raises `TypeError` on NaN columns — use `b.ne(a)` instead
- **Zero-count field:** zeros are common domain-implausible placeholders (e.g. `BloodPressure=0`) that don't register as statistical outliers — added `zero_count` to the profiler so Gemini can reason about them
- **Strict non-circular scoring:** correctness is defined as column match only — `rows_affected` is used in confidence scoring but NOT in the correctness label, to avoid circular evaluation

---

## Test Suite

```bash
pytest tests/ -v
```

36 tests across 5 modules, all passing.

---

## Project Status

| Phase   | Focus                 | Status      |
| ------- | --------------------- | ----------- |
| Phase 1 | Setup & data prep     | ✅ Complete |
| Phase 2 | Core agent logic      | ✅ Complete |
| Phase 3 | Streamlit HITL UI     | ✅ Complete |
| Phase 4 | Evaluation & write-up | ✅ Complete |
