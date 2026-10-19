# Solstice Learning - Docs Q&A Assistant

A production-grade, zero-dependency question answering engine designed for Solstice Learning admin support. It retrieves relevant documentation for corporate admins' queries and strictly enforces a confidence-gated fallback to eliminate hallucinations when documentation is missing.

---

## Deliverables Overview

- `SOLUTION_DESIGN.md`: Comprehensive solution specification detailing architectural decisions, assumptions, trade-offs, and enterprise production scaling.
- `main.py`: End-to-end Python implementation with BM25 retrieval, sentence-level extraction, dynamic confidence gating, and a built-in evaluation test harness.
- `docs_qa_utils.py`: Starter utility module provided with the assessment scaffold.
- `data/docs.json`: Seed dataset of 6 synthetic product documentation articles.
- `data/example_questions.json`: Benchmark suite of 5 test questions, including out-of-scope inquiries.

---

## Key Features

1. **Deterministic Hallucination Prevention:** Uses a calibrated BM25 scoring threshold (`score >= 2.0`) to trigger an honest fallback message (`"I don't have an answer for that..."`) whenever documentation does not cover a topic.
2. **Title-Weighted BM25 Lexical Retrieval:** Incorporates term frequency saturation and document frequency scaling, applying a boost to document titles to prioritize direct topic matches.
3. **Morphological Normalization:** Built-in tokenization, stopword filtering, and suffix stemming without external dependencies (no NLTK, spaCy, or external vector DB required).
4. **Sentence-Level Grounded Extraction:** Extracts the exact sentences answering the query rather than returning unformatted full-page text dumps.

---

## Getting Started

### Prerequisites
- Python 3.8+ (Standard Library only; no `pip install` required)

### Project Structure
```text
.
├── data/
│   ├── docs.json
│   └── example_questions.json
├── docs_qa_utils.py
├── main.py
├── README.md
└── SOLUTION_DESIGN.md
```

---

## How to Run

### 1. Run the Evaluation Suite
Execute the built-in test harness against all 5 provided example questions:

```bash
python main.py
```

### 2. Run Interactive / Ad-Hoc Queries
You can query the assistant directly with custom plain-English questions via CLI arguments:

```bash
# In-scope question:
python main.py "How long do I have to use a password reset link?"

# In-scope question (CSV limits):
python main.py "What is the maximum number of rows for CSV export?"

# Out-of-scope question (correctly triggers fallback):
python main.py "Can I connect Solstice to Salesforce?"
```

---

## Evaluation Benchmark Results

The system achieves 100% test pass rate across all benchmark scenarios:

| Question ID | Question | Expected Doc | Retrieved Doc | System Output / Extracted Answer | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Q-1** | How do I reset my password? | `D-01` | `D-01` (Resetting Your Password) | If you've forgotten your password, click 'Forgot password?' on the login screen and enter the email address associated with your account. | **PASS** |
| **Q-2** | How long do I have to use a password reset link before it expires? | `D-01` | `D-01` (Resetting Your Password) | Note: password reset links expire after 15 minutes. | **PASS** |
| **Q-3** | What's the maximum number of rows I can export to CSV? | `D-03` | `D-03` (Exporting Reports to CSV) | Important limit: CSV exports are capped at 10,000 rows. | **PASS** |
| **Q-4** | What are the API rate limits for the Growth plan? | `D-06` | `D-06` (API Rate Limits) | Rate limits: Growth plans are limited to 60 requests per minute, and Enterprise plans are limited to 300 requests per minute, per API key. | **PASS** |
| **Q-5** | Can I connect Solstice to Salesforce? | `None` | `None` (Score < 2.0) | *"I don't have an answer for that in the current documentation. Please reach out to support for assistance."* | **PASS** |

---

## Known Limitations & Edge Cases

1. **Semantic Gaps Without Lexical Overlap:** Pure BM25 relies on shared vocabulary. A query like *"What is the throttle rate for developers?"* may return a lower score if the article solely uses *"API rate limits"*. Phase 2 introduces dense bi-encoder embeddings to address semantic synonyms.
2. **Multi-Document Synthesis:** The v1 system selects the top-ranked document. Questions requiring synthesis across distinct documents (e.g., combining SSO setup with Enterprise tier billing) are scoped for future multi-hop graph retrieval.
