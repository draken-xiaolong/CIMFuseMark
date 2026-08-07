import unittest
import xml.etree.ElementTree as ET

from prepare_plateau_dataset import is_rich_building, semantic_counts


class PlateauPrepareTests(unittest.TestCase):
    def test_rich_building_requires_semantic_boundaries(self):
        member = ET.fromstring("""
        <cityObjectMember xmlns:bldg="urn:bldg">
          <bldg:Building>
            <bldg:boundedBy><bldg:GroundSurface /></bldg:boundedBy>
            <bldg:boundedBy><bldg:RoofSurface /></bldg:boundedBy>
            <bldg:boundedBy><bldg:WallSurface /></bldg:boundedBy>
          </bldg:Building>
        </cityObjectMember>
        """)
        self.assertTrue(is_rich_building(member, 3))
        self.assertEqual(semantic_counts(member)["WallSurface"], 1)

    def test_lod1_building_is_rejected(self):
        member = ET.fromstring("<cityObjectMember xmlns:bldg='urn:bldg'><bldg:Building /></cityObjectMember>")
        self.assertFalse(is_rich_building(member, 1))


if __name__ == "__main__":
    unittest.main()
