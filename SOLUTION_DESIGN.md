# Solution Design: Solstice Learning Docs Q&A Assistant

**Author:** Forward Deployed Engineer (FDE)  - Thulluri Naga Sai Durga Mounika
**Client:** Solstice Learning (Elena Torres, Head of Support)  
**Deliverable:** System Architecture, Trade-Off Analysis, and Production Roadmap  

---

## 1. Problem Understanding & Customer Objective

Solstice Learning sells an online-course platform to corporate training teams. Elena Torres, Head of Support, identified that new corporate administrators repeatedly submit support requests for administrative tasks already documented in the company's knowledge base (e.g., password resets, CSV export limits, billing cycles, and API rate limits).

Admins avoid manually searching through documentation pages because browsing is slow and friction-heavy. Elena's requirements are:
1. **Natural Language Interface:** Admins must type plain-English questions and receive direct, accurate answers extracted from documentation.
2. **Deterministic Fallback (Zero Hallucinations):** If the documentation does not contain the answer, the assistant must honestly state *"I don't have an answer for that"* rather than guessing or fabricating guidance.

In corporate SaaS, an incorrect answer can cause misconfiguration or customer churn. Therefore, precision, source grounding, and explicit fallback behavior take precedence over conversational embellishment.

---

## 2. Key Assumptions

- **Closed-World Domain:** The ingested documentation (`data/docs.json`) is the single source of truth. External integrations (such as third-party CRM connections like Salesforce) do not exist within the platform scope unless documented.
- **Admin Query Intent:** Admins ask both high-level procedural questions (*"How do I reset my password?"*) and granular parameter questions (*"What is the maximum number of rows I can export to CSV?"*).
- **Corpus Characteristics:** The seed corpus contains 6 short documents. In production, this will scale to hundreds or thousands of articles, requiring sub-linear search complexity.

---

## 3. Proposed Architecture & System Workflow

The architecture follows a modular pipeline designed for deterministic reliability:

```
[ Admin Question ]
        │
        ▼
[ Query Normalizer & Tokenizer ]
  - Punctuation stripping
  - Stopword filtering
  - Suffix-based stemming
        │
        ▼
[ BM25 Retrieval Engine ]
  - Title boosting (weight: 2.0)
  - IDF calculation & TF saturation
        │
        ▼
[ Confidence Gate ] ── (Top Score < 2.0) ──► [ Honest Fallback Message ]
        │                                     "I don't have an answer for that..."
  (Top Score >= 2.0)
        ▼
[ Sentence-Level Extractor ]
  - Scored sentence overlap
  - Grounded passage selection
        │
        ▼
[ Formatted Output with Citation ]
```

### 3.1 Preprocessing and Morphological Normalization
Queries and document bodies are normalized by stripping non-alphanumeric characters, removing common English stopwords, and applying lightweight suffix stemming. This conflates linguistic variants (e.g., "exporting" <-> "export", "expires" <-> "expire") without requiring external dependencies.

### 3.2 Retrieval Engine: BM25 with Title Weighting
Instead of naive Jaccard word-overlap (which treats rare technical terms and common words identically), the retrieval layer implements Okapi BM25:
- **Inverse Document Frequency (IDF):** Penalizes ubiquitous terms and rewards distinctive domain nouns like "CSV", "SAML", or "Growth".
- **Term Frequency (TF) Saturation:** Prevents longer documents from artificially inflating their rank simply by repeating words.
- **Title Weighting:** Document titles are weighted (2.0x) so that explicit subject matches rank higher.

### 3.3 Dynamic Confidence Gating (Zero-Hallucination Guardrail)
To meet Elena's core constraint, retrieval scores pass through a confidence gate:
- Supported questions (Q-1 through Q-4) achieve BM25 scores between 2.8 and 7.2.
- Unsupported questions (e.g., Q-5 regarding Salesforce integration) lack relevant tokens in the corpus and produce scores < 1.2.
- The threshold is calibrated at 2.0. Any query scoring below 2.0 bypasses extraction and triggers:
  > *"I don't have an answer for that in the current documentation. Please reach out to support for assistance."*

### 3.4 Grounded Passage Extraction
Rather than returning raw article blocks, the engine segments the target document into sentences, scores each sentence against the query's informative tokens, and returns the top 1–2 most relevant sentences alongside the document title and ID.

---

## 4. Trade-Offs Evaluated

| Architecture Option | Latency | Cost & Dependencies | Hallucination Risk | Decision & Justification |
| :--- | :--- | :--- | :--- | :--- |
| **Naive Jaccard Overlap (Starter Kit)** | < 1 ms | Zero dependencies | High | **Rejected:** Splits arbitrary chunks midway through words/sentences and cannot distinguish important terms from noise. |
| **BM25 Extractive Retrieval (Adopted v1)** | < 5 ms | Zero dependencies | Near Zero | **Selected:** Fast, fully transparent, zero ongoing API cost, and provides deterministic score gating to prevent hallucinations. |
| **End-to-End Generative RAG (LLM only)** | 1,000–3,000 ms | High (Requires API keys, token cost, network uptime) | Moderate to High | **Deferred to Phase 2:** Pure LLM generation introduces risks of sycophantic guessing on edge cases without strict retrieval gating. |
| **Hybrid (Dense Vector + BM25 + Cross-Encoder)** | 40–100 ms | Moderate (Vector index + embedding models) | Very Low | **Recommended for Production Scale:** Bridges semantic vocabulary mismatch while preserving exact lexical filtering. |

---

## 5. Production & Scaling Roadmap

With additional time, data, and budget, the following enhancements would be deployed:

1. **Hybrid Retrieval (Reciprocal Rank Fusion):**
   Combine BM25 with dense semantic vector embeddings (e.g., `text-embedding-3-small` or `bge-small-en-v1.5`) merged via Reciprocal Rank Fusion (RRF) to capture queries phrased in non-standard vocabulary.
2. **Support Ticket Escalation Hook:**
   When the confidence gate triggers a fallback, provide an interactive one-click button: *"Create a ticket for support"*, pre-filling Elena's team's ticketing queue (Zendesk/Salesforce Service Cloud) with the admin's query.
3. **Telemetry & Documentation Gap Analysis:**
   Log all queries that trigger fallback responses. Aggregating these queries provides Elena with product insights into what features users need or what documentation gaps exist.
4. **Constrained Generative Synthesis (Optional Bonus Track):**
   For clients wanting conversational responses rather than extracted quotes, pass the retrieved passage into a strict LLM prompt with `temperature=0.0`:
   `"Answer using ONLY the facts in the context below. If not fully contained, reply: I don't know."`

