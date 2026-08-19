import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from run_benchmark import auc_from_scores, eer_from_scores, quantile
from cimfusemark.robust_losses import bit_margin_loss, soft_nc_loss


class MetricTests(unittest.TestCase):
    def test_soft_nc_loss_matches_binary_bit_error(self):
        bits = torch.tensor([[[1.0, -1.0, 1.0, -1.0], [1.0, 1.0, 1.0, -1.0]]])
        mean, worst = soft_nc_loss(bits)
        self.assertAlmostEqual(mean.item(), 0.25)
        self.assertAlmostEqual(worst.item(), 0.25)

    def test_bit_margin_penalizes_logits_near_zero(self):
        logits = torch.tensor([0.0, 0.25, 1.0])
        self.assertAlmostEqual(bit_margin_loss(logits, 0.5).item(), (0.25 + 0.0625) / 3)

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
