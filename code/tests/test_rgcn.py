import unittest
from pathlib import Path

import torch

from cimfusemark import build_citygml_graph
from cimfusemark.rgcn import CIMFuseRGCN, graph_tensors, relation_vocabulary

DATA = Path(__file__).parents[1] / "data" / "Building_CityGML3.0_LOD2_with_several_attributes.gml"


class RGCNTests(unittest.TestCase):
    def test_forward_and_fingerprint_shape(self):
        graph = build_citygml_graph(DATA)
        relations = relation_vocabulary([graph])
        model = CIMFuseRGCN(len(graph.nodes[0].features), 16, 24, len(relations), 32)
        embedding = model.encode(*graph_tensors(graph, relations, "cpu"))
        self.assertEqual(tuple(embedding.shape), (24,))
        self.assertTrue(torch.isfinite(embedding).all())
        self.assertEqual(tuple(model.fingerprint(embedding).shape), (32,))


if __name__ == "__main__":
    unittest.main()
