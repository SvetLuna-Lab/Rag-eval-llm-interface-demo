# Rag-eval-llm-interface-demo

Minimal **RAG evaluation** harness with a **pluggable LLM client**.  
Pure-Python BM25-like retrieval, simple answer metrics (**keyword coverage**, **context overlap**) with a combined score, and a stub LLM client you can swap for a real provider.

---

> Requires **Python 3.10+**. Pure-Python demo (no external deps).


## Repository structure

```text
rag-eval-llm-interface-demo/
├─ data/
│  ├─ corpus/
│  │  ├─ doc1.txt
│  │  └─ doc2.txt
│  └─ eval_questions.json
├─ src/
│  ├─ __init__.py
│  ├─ retriever.py           # tiny BM25-like retriever (no external deps)
│  ├─ llm_client.py          # BaseLLMClient + StubLLMClient
│  ├─ pipeline.py            # retrieve → LLMClient.generate(...)
│  ├─ eval_metrics.py        # keyword coverage, context overlap, combined score
│  └─ run_eval.py            # run eval set and save JSON results
├─ tests/
│  ├─ __init__.py
│  ├─ test_metrics.py
│  ├─ test_retriever.py
│  ├─ test_pipeline_stub.py
│  ├─ test_llm_client_stub.py
│  └─ test_run_eval_smoke.py
├─ README.md
├─ requirements.txt          # (no external deps required)
└─ .gitignore



Data

data/corpus/*.txt — tiny toy corpus.

data/eval_questions.json — evaluation set with:

id: question id

question: the question text

expected_keywords: terms expected to appear in the answer

must_be_grounded_in: list of supporting docX.txt



Quick start

From the project root:

python src/run_eval.py


The script prints per-question metrics and writes a detailed report to:

rag_eval_results.json


The JSON contains answers, metrics (coverage, overlap, combined score), and retrieved docs for each question.



Tests

Run the full suite:

python -m unittest discover -s tests


Covered modules:

tests/test_metrics.py — keyword coverage & context overlap.

tests/test_retriever.py — BM25-like retriever: top-k and ranking sanity.

tests/test_pipeline_stub.py — pipeline + stub client: answer presence and max-length behavior.

tests/test_llm_client_stub.py — StubLLMClient: truncation by max_tokens.

tests/test_run_eval_smoke.py — E2E smoke: creates rag_eval_results.json and checks structure.



Swap in a real LLM

Implement your client in src/llm_client.py:

from llm_client import BaseLLMClient

class MyLLMClient(BaseLLMClient):
    def generate(self, question, context_chunks, max_tokens=256) -> str:
        # call your provider (prompt = question + condensed context)
        return "...model output..."



Use it in the pipeline:

from pipeline import RagPipeline
from llm_client import MyLLMClient

pipe = RagPipeline(corpus_dir="data/corpus", llm=MyLLMClient())
answer, contexts = pipe.run("What is RAG?")



=== RAG eval with pluggable LLM client ===
q1: score=0.750 (coverage=0.750, overlap=0.750)
q2: score=1.000 (coverage=1.000, overlap=1.000)

Detailed results saved to: rag_eval_results.json



Extending

Replace BM25-like retrieval with embeddings + vector DB.

Add citation marking and stricter grounding checks.

Log per-question traces (retrieved docs, prompts, token counts).

Add new metrics (n-gram recall vs. gold, exact-source coverage).


