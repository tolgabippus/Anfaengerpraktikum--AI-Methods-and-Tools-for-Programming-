"""
Minimal RAG (Retrieval-Augmented Generation) System — built from scratch.

Pipeline:
  1. Load & chunk documents
  2. Index chunks with TF-IDF vectors (no external embedding API needed)
  3. Retrieve top-k most relevant chunks for a query
  4. Pass retrieved context + query to an LLM for a grounded answer
"""

import re
import math
import json
import urllib.request
import urllib.error
from pathlib import Path
from collections import Counter


# ──────────────────────────────────────────────
# 1. Document loading & chunking
# ──────────────────────────────────────────────

def load_documents(sources: list[str]) -> list[dict]:
    """
    Load plain-text documents from file paths or raw strings.
    Returns a list of {source, text} dicts.
    """
    docs = []
    for src in sources:
        p = Path(src)
        if p.exists():
            docs.append({"source": src, "text": p.read_text(encoding="utf-8")})
        else:
            # Treat as raw text
            docs.append({"source": "inline", "text": src})
    return docs


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """
    Split text into overlapping word-level chunks.

    Args:
        chunk_size: number of words per chunk
        overlap:    number of words shared between consecutive chunks
    """
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


# ──────────────────────────────────────────────
# 2. TF-IDF index
# ──────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into tokens."""
    return re.findall(r"[a-z]+", text.lower())


class TFIDFIndex:
    """
    Lightweight TF-IDF retrieval index.
    No external libraries — pure Python math.
    """

    def __init__(self):
        self.chunks: list[str] = []
        self.metadata: list[dict] = []
        self._tf: list[dict] = []      # term frequencies per chunk
        self._idf: dict[str, float] = {}
        self._vocab: set[str] = set()

    def add_documents(self, docs: list[dict], chunk_size: int = 200, overlap: int = 40) -> None:
        """Chunk all documents and add them to the index."""
        for doc in docs:
            for chunk in chunk_text(doc["text"], chunk_size, overlap):
                self.chunks.append(chunk)
                self.metadata.append({"source": doc["source"]})

        self._build_index()

    def _build_index(self) -> None:
        """Compute TF for each chunk and IDF over the corpus."""
        n = len(self.chunks)
        self._tf = []

        for chunk in self.chunks:
            tokens = tokenize(chunk)
            total = len(tokens) or 1
            freq = Counter(tokens)
            tf = {term: count / total for term, count in freq.items()}
            self._tf.append(tf)
            self._vocab.update(freq.keys())

        # IDF: log(N / df) with +1 smoothing
        df: dict[str, int] = Counter()
        for tf in self._tf:
            df.update(tf.keys())

        self._idf = {
            term: math.log(n / (df[term] + 1)) + 1
            for term in self._vocab
        }

    def _tfidf_vector(self, tf: dict) -> dict:
        return {term: tf.get(term, 0) * self._idf.get(term, 0) for term in self._vocab}

    @staticmethod
    def _cosine_similarity(a: dict, b: dict) -> float:
        keys = set(a) & set(b)
        dot = sum(a[k] * b[k] for k in keys)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Return the top-k most relevant chunks for a query."""
        query_tf = {}
        tokens = tokenize(query)
        total = len(tokens) or 1
        freq = Counter(tokens)
        query_tf = {term: count / total for term, count in freq.items()}
        query_vec = self._tfidf_vector(query_tf)

        scores = [
            self._cosine_similarity(query_vec, self._tfidf_vector(tf))
            for tf in self._tf
        ]

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in ranked[:top_k]:
            results.append({
                "chunk": self.chunks[idx],
                "source": self.metadata[idx]["source"],
                "score": round(score, 4),
            })
        return results


# ──────────────────────────────────────────────
# 3. LLM generation
# ──────────────────────────────────────────────

def build_prompt(query: str, context_chunks: list[dict]) -> str:
    context_text = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['chunk']}" for c in context_chunks
    )
    return f"""You are a helpful assistant. Answer the question using ONLY the provided context.
If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context_text}

Question: {query}

Answer:"""


def call_llm(prompt: str, api_key: str, model: str = "claude-haiku-4-5-20251001") -> str:
    payload = json.dumps({
        "model": model,
        "max_tokens": 512,
        "messages": [{"role": "user", "content": prompt}]
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["content"][0]["text"].strip()
    except urllib.error.HTTPError as e:
        return f"[LLM ERROR] {e.code}: {e.read().decode()}"


# ──────────────────────────────────────────────
# 4. RAG pipeline
# ──────────────────────────────────────────────

class RAG:
    """
    End-to-end Retrieval-Augmented Generation pipeline.

    Usage:
        rag = RAG(api_key="sk-ant-...")
        rag.load(["doc1.txt", "doc2.txt"])
        answer = rag.query("What is ...?")
    """

    def __init__(self, api_key: str, top_k: int = 3,
                 chunk_size: int = 200, overlap: int = 40):
        self.api_key = api_key
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.index = TFIDFIndex()

    def load(self, sources: list[str]) -> None:
        """Load documents into the retrieval index."""
        docs = load_documents(sources)
        self.index.add_documents(docs, self.chunk_size, self.overlap)
        total_chunks = len(self.index.chunks)
        print(f"[RAG] Indexed {len(docs)} document(s) → {total_chunks} chunks.")

    def query(self, question: str, verbose: bool = False) -> str:
        """Retrieve relevant chunks and generate a grounded answer."""
        chunks = self.index.retrieve(question, top_k=self.top_k)

        if verbose:
            print(f"\n[Retrieval] Top {self.top_k} chunks for: '{question}'")
            for i, c in enumerate(chunks, 1):
                print(f"  {i}. score={c['score']} | {c['chunk'][:80]}...")

        prompt = build_prompt(question, chunks)
        return call_llm(prompt, self.api_key)


# ──────────────────────────────────────────────
# Demo (no API key needed for retrieval test)
# ──────────────────────────────────────────────

SAMPLE_DOCS = [
    """
    Python is a high-level, interpreted programming language known for its readability.
    It was created by Guido van Rossum and first released in 1991.
    Python supports multiple programming paradigms including procedural, object-oriented,
    and functional programming. It is widely used in data science, machine learning,
    web development, and automation.
    """,
    """
    Machine learning is a subset of artificial intelligence that enables systems to learn
    from data without being explicitly programmed. Key algorithms include linear regression,
    decision trees, support vector machines, and neural networks.
    Deep learning uses multi-layer neural networks to learn representations from raw data.
    Large Language Models (LLMs) are trained on massive text corpora using transformer architectures.
    """,
    """
    Retrieval-Augmented Generation (RAG) combines information retrieval with text generation.
    Instead of relying solely on a model's parametric knowledge, RAG retrieves relevant documents
    at inference time and uses them as context for the language model.
    This reduces hallucinations and allows the model to answer questions about private or
    up-to-date information that was not part of its training data.
    """
]

if __name__ == "__main__":
    print("=== RAG Demo (retrieval only — no API key needed) ===\n")

    index = TFIDFIndex()
    docs = [{"source": f"doc_{i}", "text": t} for i, t in enumerate(SAMPLE_DOCS)]
    index.add_documents(docs, chunk_size=60, overlap=10)

    queries = [
        "What is RAG and why is it useful?",
        "Who created Python?",
        "How do LLMs work?",
    ]

    for q in queries:
        print(f"Query: {q}")
        results = index.retrieve(q, top_k=2)
        for r in results:
            print(f"  score={r['score']} | {r['chunk'][:100]}...")
        print()

    print("To use full RAG with LLM generation:")
    print("  rag = RAG(api_key='sk-ant-...')")
    print("  rag.load(['your_doc.txt'])")
    print("  print(rag.query('Your question here?', verbose=True))")