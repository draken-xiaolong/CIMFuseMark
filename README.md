# CIMFuseMark

Minimal feasibility demo for non-invasive zero-watermarking of semantic 3D city models.

The demo parses CityGML 3.0 directly, extracts rotation/translation/scale-invariant geometry features together with semantic tag statistics, and maps them to a deterministic binary fingerprint. It then evaluates the fingerprint under simple synthetic attacks.

## Quick start

```bash
python3 code/run_demo.py
python3 code/run_benchmark.py
PYTHONPATH=code python3 -m unittest discover -s code/tests -v
```

For the optional relational-GNN prototype:

```bash
python3 -m pip install -r code/requirements-gnn.txt
PYTHONPATH=code python3 code/audit_graphs.py
PYTHONPATH=code python3 code/run_graph_benchmark.py
PYTHONPATH=code python3 code/train_rgcn_demo.py
```

Results are written to `code/results/demo_results.json`. The implementation uses only the Python standard library.

## First result

Run on the bundled OGC LoD2 building sample (77 explicit coordinate triples):

| Case | Fingerprint similarity |
|---|---:|
| Translation | 1.000 |
| Uniform scale | 1.000 |
| Arbitrary 3D rotation | 1.000 |
| Coordinate noise, 0.5% of bounding-box diagonal | 0.996 |
| Random point deletion, 5% | 0.984 |
| Random point deletion, 10% | 0.992 |
| Different building sample | 0.863 |

The transform robustness is encouraging, while the relatively high different-model similarity shows that uniqueness is the next problem to solve. These numbers are descriptive only; two models are not enough to estimate a threshold or false-positive rate.

## v0.2 feasibility benchmark

The benchmark adds 14 traceable OGC CityGML example files. Files believed to represent the same underlying example in different encodings, LoDs, or extensions share a `family` label and are excluded from different-source negatives.

| Statistic | Result |
|---|---:|
| Same-source attack pairs | 98 |
| Different-family pairs | 88 |
| Same-source mean / 5th percentile | 0.995 / 0.980 |
| Different-family mean / 95th percentile | 0.841 / 0.953 |
| ROC AUC | 0.9946 |
| Exploratory EER | 2.67% |
| EER threshold | 0.9766 |
| Observed false accepts at that threshold | 2 |
| Related cross-version/LoD minimum | 0.551 |

The threshold is selected and evaluated on the same small benchmark, so it is optimistic and must not be treated as a deployment threshold. The failures identify two next research problems: collisions between geometrically similar models and cross-LoD/cross-encoding consistency.

## v0.3 semantic relational graph

The v0.3 graph builder preserves CityGML objects instead of reducing the file to an unordered point cloud. Nodes include buildings, parts, rooms, boundary surfaces, roads, bridges, and other supported city objects. Directed typed edges include hierarchy, `bounded_by`, `part_of`, spatial proximity, and resolved XLinks. Coordinates are assigned to the nearest semantic owner and then aggregated through the hierarchy.

The bundled corpus currently produces 243 nodes and 1,157 typed edges. XML attacks edit complete source trees and rebuild the graph afterwards; object deletion therefore removes real CityGML subtrees rather than changing an already-computed counter.

| Method | Evaluation scope | Same-source q05 | Different-source q95 | Related-version minimum |
|---|---|---:|---:|---:|
| v0.2 global handcrafted | all 14 examples | 0.980 | 0.953 | 0.551 |
| v0.3 non-learned relational graph | all 14 examples | 0.980 | 0.936 | 0.762 |
| v0.3 R-GCN prototype | held-out 4-file test split | 0.972 | 0.521 | 0.508 |

The scopes differ, so the table is diagnostic rather than a fair leaderboard. The non-learned graph improves the different-source tail and related-version minimum on the complete sample. The R-GCN strongly separates the tiny held-out test set but does not generalize to an unseen cross-LoD pair; explicit paired cross-LoD training data is required.

The R-GCN uses two relation-specific message-passing layers, mean/max/attention graph readout, and a fixed key-dependent binary projection. It is implemented with plain PyTorch and does not require PyTorch Geometric.

## What this demo proves (and does not prove)

This is a feasibility check, not the final paper algorithm. It tests whether CityGML models can yield stable content fingerprints under translation, rotation, uniform scaling, coordinate noise, coordinate quantization, contiguous spatial cropping, attribute deletion, object reordering, and real XML object-subtree deletion. It does not yet compute exact surface-touch topology, execute real CityGML/CityJSON conversion, train on a representative city-scale corpus, or validate a deployment false-positive threshold.

## Data

All bundled samples are from the OGC CityGML 3.0 GML Encoding examples:

https://github.com/opengeospatial/CityGML3.0-GML-Encoding/tree/main/resources/examples/Building

The original repository states that the example files are intended for testing implementations of the OGC CityGML standard. Source and attribution are retained in `code/data/SOURCE.md`.

The family labels in `code/data/benchmark_manifest.json` are research metadata curated from upstream identifiers and example structure. They are not authoritative identity labels from OGC.

## GPU configuration

The current demo is CPU-only. Connection metadata is kept in `gpu.example.toml`; create `gpu.local.toml` for secrets. The local secret file is excluded by `.gitignore` and must never be committed.
