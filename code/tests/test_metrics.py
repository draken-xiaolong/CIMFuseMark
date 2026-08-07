import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from run_benchmark import auc_from_scores, eer_from_scores, quantile


class MetricTests(unittest.TestCase):
    def test_auc_perfect_separation(self):
        self.assertEqual(auc_from_scores([0.8, 0.9], [0.1, 0.2]), 1.0)

    def test_eer_perfect_separation(self):
        result = eer_from_scores([0.8, 0.9], [0.1, 0.2])
        self.assertEqual(result["eer"], 0.0)

    def test_quantile_endpoints(self):
        self.assertEqual(quantile([1.0, 2.0, 3.0], 0.0), 1.0)
        self.assertEqual(quantile([1.0, 2.0, 3.0], 1.0), 3.0)


if __name__ == "__main__":
    unittest.main()
