#!/usr/bin/env python3
"""Run the first CIMFuseMark feasibility experiment."""

from __future__ import annotations

import json
from pathlib import Path

from cimfusemark import attack_points, extract_citygml, fingerprint, similarity

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "Building_CityGML3.0_LOD2_with_several_attributes.gml"
NEGATIVE_DATA = ROOT / "data" / "JeffersonBuilding_CityGML3.0_LOD1_with_xAL3_CommonTypes.gml"
OUTPUT = ROOT / "results" / "demo_results.json"


def main() -> None:
    points, semantics = extract_citygml(DATA)
    reference = fingerprint(points, semantics)
    negative_points, negative_semantics = extract_citygml(NEGATIVE_DATA)
    negative_fingerprint = fingerprint(negative_points, negative_semantics)
    cases = [
        ("translation", 0.0),
        ("scale", 0.0),
        ("rotation_z", 0.0),
        ("rotation_3d", 0.0),
        ("noise_0.1pct", 0.001),
        ("noise_0.5pct", 0.005),
        ("crop_5pct", 0.05),
        ("crop_10pct", 0.10),
    ]
    results = []
    for name, severity in cases:
        kind = name if severity == 0 else name.split("_", 1)[0]
        attacked = attack_points(points, kind, severity)
        candidate = fingerprint(attacked, semantics)
        results.append({
            "attack": name,
            "severity": severity,
            "point_count": len(attacked),
            "similarity": round(similarity(reference, candidate), 6),
        })

    report = {
        "dataset": DATA.name,
        "point_count": len(points),
        "semantic_counts": dict(sorted(semantics.items())),
        "fingerprint_bits": len(reference),
        "reference_fingerprint_hex": f"{int(reference, 2):064x}",
        "different_model": {
            "dataset": NEGATIVE_DATA.name,
            "point_count": len(negative_points),
            "similarity": round(similarity(reference, negative_fingerprint), 6),
        },
        "attacks": results,
        "warning": "Feasibility demo only; no decision threshold has been validated.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
