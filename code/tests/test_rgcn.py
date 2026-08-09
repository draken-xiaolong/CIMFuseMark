import unittest
from pathlib import Path

import torch

from cimfusemark import build_citygml_graph
from cimfusemark.rgcn import CIMFuseRGCN, graph_tensors, relation_vocabulary
from cimfusemark.robust_losses import (bit_separation_loss, embedding_tail_loss,
                                       multi_positive_nt_xent, robust_bit_loss)
from cimfusemark.personalization import codebook_similarity, keyed_codebook

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

    def test_robust_losses_prefer_matching_views(self):
        clean = torch.nn.functional.normalize(torch.randn(3, 8), dim=1)
        matching = clean[:, None, :].repeat(1, 3, 1)
        perturbed = matching.clone(); perturbed[:, 1:] = torch.roll(clean, 1, 0)[:, None, :]
        self.assertLess(float(multi_positive_nt_xent(matching)), float(multi_positive_nt_xent(perturbed)))
        self.assertAlmostEqual(float(embedding_tail_loss(matching)), 0.0, places=6)
        soft = torch.tanh(torch.randn(3, 3, 16))
        stability, balance, quantization, tail = robust_bit_loss(soft)
        for value in (stability, balance, quantization, tail):
            self.assertTrue(torch.isfinite(value))

    def test_bit_separation_penalizes_correlated_codes(self):
        codes = torch.tensor([[1.0, 1.0, -1.0, -1.0],
                              [1.0, 1.0, -1.0, -1.0]])
        opposite = codes.clone(); opposite[1] *= -1
        self.assertGreater(float(bit_separation_loss(codes)),
                           float(bit_separation_loss(opposite)))

    def test_keyed_codebook_is_balanced_and_deterministic(self):
        left = keyed_codebook(8, 128, 17); right = keyed_codebook(8, 128, 17)
        self.assertTrue(torch.equal(left, right))
        self.assertTrue(torch.all(left.sum(dim=0) == 0))
        similarities = codebook_similarity(left)
        self.assertLess(float(similarities[~torch.eye(8, dtype=torch.bool)].max()), 0.7)


if __name__ == "__main__":
    unittest.main()
