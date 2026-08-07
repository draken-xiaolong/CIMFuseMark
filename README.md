# CIMFuseMark

Minimal feasibility demo for non-invasive zero-watermarking of semantic 3D city models.

The demo parses CityGML 3.0 directly, extracts rotation/translation/scale-invariant geometry features together with semantic tag statistics, and maps them to a deterministic binary fingerprint. It then evaluates the fingerprint under simple synthetic attacks.

## Quick start

```bash
python3 code/run_demo.py
```

Results are written to `code/results/demo_results.json`. The implementation uses only the Python standard library.

## What this demo proves (and does not prove)

This is a feasibility check, not the final paper algorithm. It tests whether a small CityGML model can yield a stable content fingerprint under translation, rotation, uniform scaling, coordinate noise, and partial point deletion. It does not yet model surface connectivity, object-level deletion, cross-format conversion, learned representations, or a formal false-positive threshold.

## Data

The sample is `Building_CityGML3.0_LOD2_with_several_attributes.gml` from the OGC CityGML 3.0 GML Encoding examples:

https://github.com/opengeospatial/CityGML3.0-GML-Encoding/tree/main/resources/examples/Building

The original repository states that the example files are intended for testing implementations of the OGC CityGML standard. Source and attribution are retained in `code/data/SOURCE.md`.

## GPU configuration

The current demo is CPU-only. Connection metadata is kept in `gpu.example.toml`; create `gpu.local.toml` for secrets. The local secret file is excluded by `.gitignore` and must never be committed.

