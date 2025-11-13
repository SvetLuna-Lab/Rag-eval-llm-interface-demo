import unittest
from src.eval_metrics import keyword_coverage, context_overlap, evaluate_single

class TestMetrics(unittest.TestCase):
    def test_keyword_coverage_basic(self):
        ans = "RAG combines retrieval and generation; grounding reduces hallucinations."
        expected = ["retrieval", "generation", "grounding", "hallucinations"]
        cov = keyword_coverage(ans, expected)
        self.assertAlmostEqual(cov, 1.0, places=6)

    def test_context_overlap_basic(self):
        ans = "retrieval and generation reduce hallucinations"
        ctx = "The RAG approach combines information retrieval with generation. Grounding answers reduces hallucinations."
        ov = context_overlap(ans, ctx)
        self.assertGreater(ov, 0.5)

    def test_evaluate_single_shapes(self):
        res = evaluate_single(
            qid="qX",
            answer="keyword coverage and overlap",
            expected_keywords=["keyword coverage", "overlap"],
            context_text="Evaluation of RAG answers can track keyword coverage and overlap."
        )
        self.assertEqual(res.question_id, "qX")
        self.assertGreaterEqual(res.coverage, 0.0)
        self.assertGreaterEqual(res.overlap, 0.0)
        self.assertGreaterEqual(res.score, 0.0)
