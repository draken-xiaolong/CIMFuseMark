import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from cimfusemark import attack_points, extract_citygml, fingerprint, similarity

DATA = Path(__file__).parents[1] / "data" / "Building_CityGML3.0_LOD2_with_several_attributes.gml"
NEGATIVE_DATA = Path(__file__).parents[1] / "data" / "JeffersonBuilding_CityGML3.0_LOD1_with_xAL3_CommonTypes.gml"


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.points, cls.semantics = extract_citygml(DATA)
        cls.baseline = fingerprint(cls.points, cls.semantics)

    def test_exact_similarity_attacks(self):
        for attack in ("translation", "scale", "rotation_z", "rotation_3d"):
            candidate = fingerprint(attack_points(self.points, attack), self.semantics)
            self.assertGreaterEqual(similarity(self.baseline, candidate), 0.99)

    def test_weak_noise_is_stable(self):
        attacked = fingerprint(attack_points(self.points, "noise", 0.001), self.semantics)
        self.assertGreaterEqual(similarity(self.baseline, attacked), 0.90)

    def test_different_model_is_less_similar(self):
        points, semantics = extract_citygml(NEGATIVE_DATA)
        candidate = fingerprint(points, semantics)
        self.assertLess(similarity(self.baseline, candidate), 0.90)


if __name__ == "__main__":
    unittest.main()
