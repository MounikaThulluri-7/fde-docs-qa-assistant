import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "should", "shouldn't", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "were", "weren't", "what", "when", "where", "which", "while",
    "who", "whom", "why", "with", "won't", "would", "wouldn't", "you", "your",
    "yours", "yourself", "yourselves"
}


def basic_stem(word: str) -> str:
    """Lightweight suffix stemmer to conflate plurals, gerunds, and tenses without NLTK."""
    suffixes = ("ing", "ly", "ed", "es", "s", "ment", "tion")
    for sfx in suffixes:
        if word.endswith(sfx) and len(word) > len(sfx) + 2:
            return word[:-len(sfx)]
    return word


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Tokenizes text, strips punctuation, normalizes case, and stems."""
    text = text.lower()
    raw_tokens = re.findall(r"\b[a-z0-9]+\b", text)
    if remove_stopwords:
        tokens = [basic_stem(w) for w in raw_tokens if w not in STOPWORDS]
        return tokens if tokens else [basic_stem(w) for w in raw_tokens]
    return raw_tokens


class BM25Retriever:
    """BM25 (Best Matching 25) implementation tailored for short corporate documentation."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, title_boost: float = 2.0):
        self.k1 = k1
        self.b = b
        self.title_boost = title_boost
        self.docs: List[Dict] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.idf: Dict[str, float] = {}

    def fit(self, docs: List[Dict]):
        self.docs = docs
        self.doc_tokens = []
        self.doc_lengths = []

        total_tokens = 0
        df = {}

        for doc in self.docs:
            # Title tokens are given extra weight to prioritize explicit topic matches
            title_tokens = tokenize(doc.get("title", "")) * int(self.title_boost)
            content_tokens = tokenize(doc.get("content", ""))
            tokens = title_tokens + content_tokens

            self.doc_tokens.append(tokens)
            length = len(tokens)
            self.doc_lengths.append(length)
            total_tokens += length

            unique_terms = set(tokens)
            for term in unique_terms:
                df[term] = df.get(term, 0) + 1

        num_docs = len(self.docs)
        self.avg_doc_len = total_tokens / num_docs if num_docs > 0 else 0.0

        # Calculate standard Lucene/Robertson IDF
        for term, freq in df.items():
            self.idf[term] = math.log(1 + (num_docs - freq + 0.5) / (freq + 0.5))

    def score_query(self, query: str) -> List[Tuple[float, Dict]]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return [(0.0, doc) for doc in self.docs]

        scores = []
        for idx, doc_tokens in enumerate(self.doc_tokens):
            score = 0.0
            doc_len = self.doc_lengths[idx]
            term_freqs = {}
            for t in doc_tokens:
                term_freqs[t] = term_freqs.get(t, 0) + 1

            for term in q_tokens:
                if term in term_freqs:
                    freq = term_freqs[term]
                    idf = self.idf.get(term, 0.0)
                    numerator = freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                    score += idf * (numerator / denominator)

            scores.append((score, self.docs[idx]))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores


class DocsQAAgent:
    """End-to-end assistant supporting grounded extractive answering and strict fallback."""

    # Elena's requirement: Say 'I don't have an answer' instead of hallucinating.
    CONFIDENCE_THRESHOLD = 2.0
    FALLBACK_MESSAGE = (
        "I don't have an answer for that in the current documentation. "
        "Please reach out to support for assistance."
    )

    def __init__(self, docs_path: str):
        with open(docs_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.docs = data.get("docs", [])
        self.retriever = BM25Retriever()
        self.retriever.fit(self.docs)

    def _extract_best_passage(self, query: str, content: str, max_sentences: int = 2) -> str:
        """Finds the most query-relevant sentences within the retrieved article."""
        sentences = re.split(r"(?<=[.!?])\s+", content.strip())
        if len(sentences) <= max_sentences:
            return content.strip()

        q_tokens = set(tokenize(query))
        scored_sentences = []
        for idx, sentence in enumerate(sentences):
            s_tokens = set(tokenize(sentence))
            overlap = len(q_tokens & s_tokens)
            scored_sentences.append((overlap, idx, sentence))

        # Sort by overlap, keep chronological order of top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_indices = sorted([s[1] for s in scored_sentences[:max_sentences]])
        return " ".join([sentences[i] for i in top_indices])

    def answer_question(self, question: str) -> Dict:
        results = self.retriever.score_query(question)
        if not results:
            return {
                "answer": self.FALLBACK_MESSAGE,
                "confidence": 0.0,
                "source_doc_id": None,
                "source_title": None,
                "matched": False
            }

        top_score, top_doc = results[0]

        # Strict gating check
        if top_score < self.CONFIDENCE_THRESHOLD:
            return {
                "answer": self.FALLBACK_MESSAGE,
                "confidence": round(top_score, 3),
                "source_doc_id": None,
                "source_title": None,
                "matched": False
            }

        passage = self._extract_best_passage(question, top_doc["content"])

        return {
            "answer": passage,
            "confidence": round(top_score, 3),
            "source_doc_id": top_doc["doc_id"],
            "source_title": top_doc["title"],
            "matched": True
        }


def run_evaluations(docs_path: str, questions_path: str):
    print("=" * 80)
    print("SOLSTICE LEARNING DOCS QA ASSISTANT - EVALUATION HARNESS")
    print("=" * 80)

    agent = DocsQAAgent(docs_path)

    with open(questions_path, "r", encoding="utf-8") as f:
        benchmarks = json.load(f).get("example_questions", [])

    total = len(benchmarks)
    correct = 0

    for item in benchmarks:
        qid = item["question_id"]
        q = item["question"]
        expected_doc = item["expected_doc_id"]

        result = agent.answer_question(q)
        actual_doc = result["source_doc_id"]

        is_match = (actual_doc == expected_doc)
        if is_match:
            correct += 1

        print(f"\n[{qid}] Question: {q}")
        print(f"  Expected Doc : {expected_doc}")
        print(f"  Matched Doc  : {actual_doc} ({result['source_title']})")
        print(f"  BM25 Score   : {result['confidence']}")
        print(f"  Assistant Output:\n    \"{result['answer']}\"")
        print(f"  Verdict      : {'PASS' if is_match else 'FAIL'}")

    print("\n" + "=" * 80)
    print(f"ACCURACY: {correct}/{total} ({correct / total * 100:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    docs_file = os.path.join(os.path.dirname(__file__), "data", "docs.json")
    questions_file = os.path.join(os.path.dirname(__file__), "data", "example_questions.json")

    # If CLI arguments are provided, answer custom question
    if len(sys.argv) > 1:
        custom_query = " ".join(sys.argv[1:])
        agent = DocsQAAgent(docs_file)
        ans = agent.answer_question(custom_query)
        print(f"\nQuestion: {custom_query}")
        print(f"Answer: {ans['answer']}")
        if ans["source_doc_id"]:
            print(f"Source: {ans['source_title']} ({ans['source_doc_id']}) [Score: {ans['confidence']}]")
    else:
        run_evaluations(docs_file, questions_file)
