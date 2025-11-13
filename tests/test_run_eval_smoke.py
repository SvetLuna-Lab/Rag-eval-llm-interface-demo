import json
import os
import unittest

class TestRunEvalSmoke(unittest.TestCase):
    def test_main_produces_results_file(self):
        # Import inside the test so PROJECT_ROOT and paths are resolved as in the module
        from src import run_eval  # noqa: WPS433

        out_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "rag_eval_results.json",
        )

        # Remove stale artifact from previous runs (if any)
        if os.path.exists(out_path):
            os.remove(out_path)

        # Run the main entrypoint — it should create the results file
        run_eval.main()
        self.assertTrue(os.path.exists(out_path), "rag_eval_results.json should be created")

        # Quick structural sanity check
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        self.assertIn("metrics", data[0])
