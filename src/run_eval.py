import json, os
from typing import Any, Dict, List

from pipeline import RagPipeline
from eval_metrics import evaluate_single, to_dict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CORPUS_DIR = os.path.join(DATA_DIR, "corpus")
EVAL_PATH = os.path.join(DATA_DIR, "eval_questions.json")
OUT_PATH = os.path.join(PROJECT_ROOT, "rag_eval_results.json")

def _load_eval(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_docs(doc_ids: List[str]) -> str:
    parts: List[str] = []
    for doc_id in doc_ids:
        p = os.path.join(CORPUS_DIR, doc_id)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                parts.append(f.read())
    return "\n".join(parts)

def main() -> None:
    qs = _load_eval(EVAL_PATH)
    pipe = RagPipeline(corpus_dir=CORPUS_DIR)

    results: List[Dict[str, Any]] = []
    print("=== RAG eval with pluggable LLM client ===")
    for q in qs:
        qid = q["id"]; question = q["question"]
        expected = q.get("expected_keywords", [])
        gold_docs = q.get("must_be_grounded_in", [])

        answer, ctxs = pipe.run(question, top_k=3)
        gold_ctx = _load_docs(gold_docs)
        res = evaluate_single(qid, answer, expected, gold_ctx, alpha=0.5)

        results.append({
            "id": qid,
            "question": question,
            "answer": answer,
            "expected_keywords": expected,
            "must_be_grounded_in": gold_docs,
            "metrics": to_dict(res),
            "retrieved_docs": [{"doc_id": c.doc_id, "score": c.score} for c in ctxs],
        })

    for r in results:
        m = r["metrics"]
        print(f"{r['id']}: score={m['score']:.3f} (coverage={m['keyword_coverage']:.3f}, overlap={m['context_overlap']:.3f})")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to: {OUT_PATH}")

if __name__ == "__main__":
    main()
