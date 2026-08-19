import tempfile
import unittest
from pathlib import Path

from cimfusemark import attack_citygml_xml, build_citygml_graph, graph_fingerprint, similarity

DATA = Path(__file__).parents[1] / "data" / "Building_CityGML3.0_LOD2_with_several_attributes.gml"


class CityGMLGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = build_citygml_graph(DATA)

    def test_hierarchy_and_relations_exist(self):
        self.assertGreater(len(self.graph.nodes), 1)
        relations = {edge.relation for edge in self.graph.edges}
        self.assertIn("bounded_by", relations)
        self.assertIn("part_of", relations)

    def test_xml_rigid_attacks_are_invariant(self):
        reference = graph_fingerprint(self.graph)
        with tempfile.TemporaryDirectory() as directory:
            for attack in ("translation", "scale", "rotation_z", "object_reorder"):
                path = Path(directory) / f"{attack}.gml"
                attack_citygml_xml(DATA, path, attack, 0.0)
                candidate = graph_fingerprint(build_citygml_graph(path))
                self.assertGreaterEqual(similarity(reference, candidate), 0.99, attack)

    def test_object_delete_removes_real_node(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deleted.gml"
            mutation = attack_citygml_xml(DATA, path, "object_delete", 0.10)
            attacked = build_citygml_graph(path)
            self.assertGreater(mutation["changed_elements"], 0)
            self.assertLess(len(attacked.nodes), len(self.graph.nodes))

    def test_semantic_deletion_attacks_are_distinct_and_parseable(self):
        with tempfile.TemporaryDirectory() as directory:
            for attack in ("building_delete", "surface_delete"):
                path = Path(directory) / f"{attack}.gml"
                mutation = attack_citygml_xml(DATA, path, attack, 0.5)
                attacked = build_citygml_graph(path)
                self.assertGreaterEqual(len(attacked.nodes), 1)
                self.assertGreater(mutation["candidate_elements"], 0)
                self.assertLessEqual(len(attacked.nodes), len(self.graph.nodes))

    def test_unseen_noise_and_sequential_attacks_are_parseable(self):
        with tempfile.TemporaryDirectory() as directory:
            for attack, severity in (("coordinate_noise", 0.001), ("sequential", 0.2)):
                path = Path(directory) / f"{attack}.gml"
                mutation = attack_citygml_xml(DATA, path, attack, severity)
                attacked = build_citygml_graph(path)
                self.assertGreater(len(attacked.nodes), 0)
                self.assertGreater(mutation["changed_elements"], 0)

    def test_cim_specific_attacks_are_parseable(self):
        attacks = (("lod2_to_lod1", 1.0), ("hierarchy_flatten", 0.5),
                   ("relation_delete", 0.5), ("semantic_relabel", 0.5),
                   ("spatial_crop", 0.5), ("building_add", 0.5),
                   ("id_rename", 1.0), ("object_reorder", 1.0),
                   ("coordinate_unit", 0.001), ("cityjson_roundtrip", 1.0))
        with tempfile.TemporaryDirectory() as directory:
            for attack, severity in attacks:
                path = Path(directory) / f"{attack}.gml"
                mutation = attack_citygml_xml(DATA, path, attack, severity)
                attacked = build_citygml_graph(path)
                self.assertGreater(len(attacked.nodes), 0, attack)
                if attack != "spatial_crop":  # The bundled fixture contains one building, which must be retained.
                    self.assertGreater(mutation["changed_elements"], 0, attack)

    def test_lod_downgrade_removes_boundary_nodes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lod1.gml"
            attack_citygml_xml(DATA, path, "lod2_to_lod1", 1.0)
            attacked = build_citygml_graph(path)
            self.assertFalse({node.node_type for node in attacked.nodes} & {
                "WallSurface", "RoofSurface", "GroundSurface"})
            self.assertGreater(attacked.metadata["coordinate_count"], 0)


if __name__ == "__main__":
    unittest.main()
