from dataclasses import dataclass
from typing import Dict, List

def _tok(s: str) -> List[str]:
    out = []
    for raw in s.lower().split():
        t = "".join(ch for ch in raw if ch.isalnum())
        if t: out.append(t)
    return out

def keyword_coverage(answer: str, expected_keywords: List[str]) -> float:
    if not expected_keywords: return 0.0
    a = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in a)
    return hits / len(expected_keywords)

def context_overlap(answer: str, context_text: str) -> float:
    ans = _tok(answer)
    ctx = set(_tok(context_text))
    if not ans: return 0.0
    return sum(1 for t in ans if t in ctx) / len(ans)

@dataclass
class EvalResult:
    question_id: str
    coverage: float
    overlap: float
    score: float

def compute_score(coverage: float, overlap: float, alpha: float = 0.5) -> float:
    return alpha * coverage + (1 - alpha) * overlap

def evaluate_single(qid: str, answer: str, expected_keywords: List[str], context_text: str, alpha: float = 0.5) -> EvalResult:
    cov = keyword_coverage(answer, expected_keywords)
    ov = context_overlap(answer, context_text)
    return EvalResult(qid, cov, ov, compute_score(cov, ov, alpha))

def to_dict(res: EvalResult) -> Dict[str, float]:
    return {"question_id": res.question_id, "keyword_coverage": res.coverage, "context_overlap": res.overlap, "score": res.score}
