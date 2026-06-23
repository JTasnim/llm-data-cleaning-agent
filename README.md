# LLM-Powered Data Cleaning Agent with Human-in-the-Loop Approval

> MS Computer Science Independent Study (CS 587/387) — San Francisco Bay University
> Student: Jubaida Tasnim (160027) — Supervisor: Dr. Satarupa Mukherjee

## Project Focus

This project investigates **Gap 2: No self-verification with confidence scoring**
within **Theme 2: Agentic LLM Systems and Tool Use**.

**Central research question:** Can a dry-run self-verification loop produce a
calibrated confidence score for LLM-generated data cleaning proposals — one
where high confidence reliably predicts correctness?

The system ingests a messy tabular dataset, profiles it, uses an LLM agent to
detect data-quality issues and propose cleaning transformations, **dry-runs
and self-verifies each proposal**, attaches a calibrated confidence tier
(High / Medium / Low), and surfaces every proposal to a human reviewer through
a human-in-the-loop (HITL) approval interface before anything is applied.

## Tech Stack (100% free tier)

- **LLM backbone:** Google Gemini 2.5 Flash (primary, free tier) with Groq / Llama 3.3 70B as fallback
- **Agent framework:** LangChain / LangGraph
- **Data layer:** Python, Pandas, Great Expectations
- **UI:** Streamlit
- **Vector store:** ChromaDB (local)

## Status

🚧 **Phase 1 (Weeks 1–2): Setup & data prep** — in progress
