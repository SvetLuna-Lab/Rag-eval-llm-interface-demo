import os
import unittest
from src.pipeline import RagPipeline

class TestPipelineStub(unittest.TestCase):
    def setUp(self):
        here = os.path.dirname(os.path.abspath(__file__))
        self.corpus_dir = os.path.join(os.path.dirname(here), "data", "corpus")

    def test_pipeline_answer_and_contexts(self):
        pipe = RagPipeline(corpus_dir=self.corpus_dir)
        ans, ctxs = pipe.run("Name two simple RAG answer metrics")
        self.assertIsInstance(ans, str)
        self.assertGreater(len(ctxs), 0)

    def test_pipeline_respects_max_chars(self):
        pipe = RagPipeline(corpus_dir=self.corpus_dir)
        ctxs = pipe.retrieve("What is RAG?", top_k=2)
        ans = pipe.answer("What is RAG?", ctxs, max_chars=80)
        self.assertLessEqual(len(ans), 85)  
