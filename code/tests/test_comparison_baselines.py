import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from comparison_experiments.baselines import all_baselines, bit_similarity
from comparison_experiments.citygml_embedded import all_embedded_methods, payload
from cimfusemark.robust_losses import embedding_tail_loss, robust_bit_loss

DATA = Path(__file__).parents[1] / "data" / "Building_CityGML3.0_LOD2_with_several_attributes.gml"


class ComparisonBaselineTests(unittest.TestCase):
    def test_clean_only_losses_are_finite(self):
        embeddings = torch.nn.functional.normalize(torch.randn(4, 1, 8), dim=2)
        bits = torch.tanh(torch.randn(4, 1, 16))
        losses = (*robust_bit_loss(bits), embedding_tail_loss(embeddings))
        self.assertTrue(all(torch.isfinite(loss) for loss in losses))
    def test_all_baselines_are_deterministic_binary_fingerprints(self):
        for method in all_baselines():
            left = method.fingerprint(DATA); right = method.fingerprint(DATA)
            self.assertEqual(tuple(left.shape), (method.bits,), method.name)
            self.assertTrue(np.array_equal(left, right), method.name)
            self.assertTrue(set(np.unique(left)).issubset({0, 1}), method.name)
            self.assertEqual(bit_similarity(left, right), 1.0)

    def test_native_embedded_methods_roundtrip_binary_payload(self):
        watermark = payload()
        with tempfile.TemporaryDirectory() as temporary:
            for method in all_embedded_methods():
                target = Path(temporary) / f"{method.name}.gml"
                method.embed(DATA, target, watermark)
                extracted = method.extract(target)
                self.assertEqual(tuple(extracted.shape), (method.bits,), method.name)
                self.assertTrue(set(np.unique(extracted)).issubset({0, 1}), method.name)
                # Small fixtures can lack enough eligible gml:id homographs for all 64 bits.
                self.assertGreaterEqual(bit_similarity(watermark, extracted), .65, method.name)


if __name__ == "__main__":
    unittest.main()
