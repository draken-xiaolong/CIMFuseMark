import unittest
from pathlib import Path

import numpy as np

from comparison_experiments.baselines import all_baselines, bit_similarity

DATA = Path(__file__).parents[1] / "data" / "Building_CityGML3.0_LOD2_with_several_attributes.gml"


class ComparisonBaselineTests(unittest.TestCase):
    def test_all_baselines_are_deterministic_binary_fingerprints(self):
        for method in all_baselines():
            left = method.fingerprint(DATA); right = method.fingerprint(DATA)
            self.assertEqual(tuple(left.shape), (method.bits,), method.name)
            self.assertTrue(np.array_equal(left, right), method.name)
            self.assertTrue(set(np.unique(left)).issubset({0, 1}), method.name)
            self.assertEqual(bit_similarity(left, right), 1.0)


if __name__ == "__main__":
    unittest.main()
