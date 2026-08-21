import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_experiments import attack, canonical
from hkreal3d.io import load_b3dm_mesh, load_b3dm_vertices, normalized_sample


def test_b3dm_sample_parses_and_samples():
    source = ROOT.parents[0] / "data" / "hk_sample_lod4.b3dm"
    vertices = load_b3dm_vertices(source)
    points = normalized_sample(vertices, 256, 2026)
    assert len(vertices) > 100
    assert points.shape == (256, 3)
    assert np.isfinite(points).all()


def test_all_attack_families_keep_fixed_finite_shape():
    clean = canonical(np.random.default_rng(4).normal(size=(512, 3)).astype(np.float32))
    cases = [("scale", 2.0), ("rotation", 180), ("noise", .01),
             ("quantization", .02), ("point_delete", .8), ("crop", .8),
             ("outliers", .2), ("sequential", .8)]
    for family, level in cases:
        attacked = attack(clean, family, level, np.random.default_rng(5))
        assert attacked.shape == clean.shape
        assert np.isfinite(attacked).all()
