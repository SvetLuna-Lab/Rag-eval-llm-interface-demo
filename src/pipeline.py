from dataclasses import dataclass
from typing import List, Tuple

from retriever import SimpleBM25Retriever, Document
from llm_client import BaseLLMClient, StubLLMClient

@dataclass
class RetrievedContext:
    doc_id: str
    text: str
    score: float

class RagPipeline:
    """retrieve → LLMClient.generate(question, contexts)"""
    def __init__(self, corpus_dir: str, llm: BaseLLMClient | None = None) -> None:
        self.retriever = SimpleBM25Retriever(corpus_dir)
        self.llm = llm or StubLLMClient()

    def retrieve(self, question: str, top_k: int = 3) -> List[RetrievedContext]:
        results: List[Tuple[Document, float]] = self.retriever.retrieve(question, top_k=top_k)
        return [RetrievedContext(d.doc_id, d.text, s) for d, s in results]

    def answer(self, question: str, contexts: List[RetrievedContext], max_chars: int = 400) -> str:
        chunks = []
        remain = max_chars
        for c in contexts:
            t = c.text.strip().replace("\n", " ")
            t = t[:remain]
            chunks.append(t)
            remain -= len(t)
            if remain <= 0: break
        return self.llm.generate(question, chunks, max_tokens=max_chars)

    def run(self, question: str, top_k: int = 3) -> tuple[str, List[RetrievedContext]]:
        ctxs = self.retrieve(question, top_k=top_k)
        ans = self.answer(question, ctxs)
        return ans, ctxs
