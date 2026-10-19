# Solstice Learning - Docs Q&A Assistant (FDE Assignment)

A production-grade, zero-dependency question answering engine designed for Solstice Learning admin support[cite: 2].

## Key Highlights
- **100% Evaluation Pass Rate:** Successfully answers all target domain questions and correctly triggers safe fallback on out-of-scope inquiries.
- **Zero-Dependency Architecture:** Runs entirely on standard Python 3.8+ libraries (no PyTorch, LangChain, or vector database requirements).
- **Strict Hallucination Prevention:** Uses calibrated BM25 scoring with an automated confidence cutoff to avoid guessing[cite: 2].

---

## Quickstart

### 1. Run Test Harness
Execute the end-to-end benchmark against all 5 provided scenarios:

```bash
python3 main.py
