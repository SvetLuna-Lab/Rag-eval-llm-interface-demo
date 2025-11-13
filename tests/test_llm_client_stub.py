import unittest
from src.llm_client import StubLLMClient

class TestLLMClientStub(unittest.TestCase):
    def test_generate_truncation(self):
        client = StubLLMClient()
        chunks = ["A" * 200, "B" * 200]
        out = client.generate("q", chunks, max_tokens=150)
        self.assertLessEqual(len(out), 155)
        self.assertTrue(len(out) > 0)
