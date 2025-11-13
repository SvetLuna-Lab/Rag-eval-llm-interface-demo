import os
import unittest
from src.retriever import SimpleBM25Retriever

class TestRetriever(unittest.TestCase):
    def setUp(self):
        here = os.path.dirname(os.path.abspath(__file__))
        self.corpus_dir = os.path.join(os.path.dirname(here), "data", "corpus")

    def test_retrieve_topk(self):
        r = SimpleBM25Retriever(self.corpus_dir)
        results = r.retrieve("What is retrieval and generation?", top_k=2)
        self.assertEqual(len(results), 2)
        # BM25 intuition: doc1 should rank higher for a query about RAG/grounding
        self.assertEqual(results[0][0].doc_id, "doc1.txt")

    def test_retrieve_empty_query(self):
        r = SimpleBM25Retriever(self.corpus_dir)
        results = r.retrieve("", top_k=3)
        # Return up to top_k items, but scores should be zero for an empty query
        self.assertLessEqual(len(results), 3)
