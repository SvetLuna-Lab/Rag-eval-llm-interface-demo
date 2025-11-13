import math, os
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class Document:
    doc_id: str
    text: str

class SimpleBM25Retriever:
    """Very small BM25-like retriever without external deps (demo-grade)."""
    def __init__(self, corpus_dir: str, k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus_dir, self.k1, self.b = corpus_dir, k1, b
        self.documents: List[Document] = []
        self.doc_lengths: Dict[str, int] = {}
        self.df: Dict[str, int] = {}
        self.N, self.avg_doc_len = 0, 0.0
        self._load_corpus(); self._build_index()

    @staticmethod
    def _tok(s: str) -> List[str]:
        out = []
        for raw in s.lower().split():
            t = "".join(ch for ch in raw if ch.isalnum())
            if t: out.append(t)
        return out

    def _load_corpus(self) -> None:
        for fn in os.listdir(self.corpus_dir):
            if fn.endswith(".txt"):
                with open(os.path.join(self.corpus_dir, fn), "r", encoding="utf-8") as f:
                    self.documents.append(Document(fn, f.read()))
        self.N = len(self.documents)

    def _build_index(self) -> None:
        total = 0
        self.df.clear(); self.doc_lengths.clear()
        for d in self.documents:
            toks = self._tok(d.text)
            total += len(toks)
            self.doc_lengths[d.doc_id] = len(toks)
            for term in set(toks):
                self.df[term] = self.df.get(term, 0) + 1
        self.avg_doc_len = total / self.N if self.N else 0.0

    def _score(self, q_tokens: List[str], d: Document) -> float:
        if not self.N: return 0.0
        toks = self._tok(d.text); L = len(toks) or 1
        tf: Dict[str, int] = {}
        for t in toks: tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for term in q_tokens:
            if term not in tf: continue
            df = self.df.get(term, 0)
            if df == 0: continue
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            freq = tf[term]
            denom = freq + self.k1 * (1 - self.b + self.b * L / (self.avg_doc_len or 1.0))
            s += idf * (freq * (self.k1 + 1) / denom)
        return s

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        q_tokens = self._tok(query)
        scores = [(d, self._score(q_tokens, d)) for d in self.documents]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
