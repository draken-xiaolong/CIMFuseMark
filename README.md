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

For the geographically isolated Open City Model experiment:

```bash
PYTHONPATH=code python3 code/prepare_ocm_dataset.py
PYTHONPATH=code python3 code/audit_graphs.py --manifest code/data/ocm_manifest.json --output code/results/ocm_graph_audit.json
PYTHONPATH=code python3 code/run_graph_benchmark.py --manifest code/data/ocm_manifest.json --output code/results/ocm_graph_matched_results.json --attacks rotation_z,quantization,attribute_delete,object_delete
PYTHONPATH=code python3 code/train_rgcn_demo.py --manifest code/data/ocm_manifest.json --output-prefix rgcn_ocm
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

## v0.4 geographically isolated benchmark

The scalable experiment downloads three Open City Model CityGML archives and creates 48 non-overlapping tiles containing 1,200 buildings. Geographic splits are fixed before training: Alabama for training, California for validation, and Colorado for testing. Downloaded archives, generated tiles, manifests, checkpoints, and results remain local and are ignored by Git.

On the Colorado split, using the same four attacks and the same 64 positive / 120 negative pairs, the non-learned graph fingerprint obtains AUC 0.9286 and EER 15.26%; the R-GCN obtains AUC 0.9720 and EER 6.46%. This is a useful feasibility result, not a publication-ready claim: the source is shallow LoD1 building data and currently yields only Building nodes and spatial-neighbor edges. It validates scale and geographic generalization, but not the proposed benefit of rich CIM semantics or hierarchy.

See `docs/v0.4_ocm_geographic_benchmark_zh.md` for the protocol, results, limitations, and next experiment.

## v0.5 rich-semantic PLATEAU corpus

The v0.5 preparer selects LoD2 buildings with explicit semantic boundary surfaces from three 2025 Project PLATEAU meshes. Chiyoda, Minato, and Shinjuku are isolated as train, validation, and test regions. The current local corpus contains 24 tiles, 188 buildings, 9,659 boundary-surface nodes, and 48,889 graph edges with no parse failures.

```bash
PYTHONPATH=code python3 code/prepare_plateau_dataset.py
PYTHONPATH=code python3 code/audit_graphs.py --manifest code/data/plateau_manifest.json --output code/results/plateau_graph_audit.json
PYTHONPATH=code python3 code/train_rgcn_demo.py --manifest code/data/plateau_manifest.json --output-prefix rgcn_plateau --device cuda
```

The training script now supports `--relation-mode typed|untyped|no_edges` and `--feature-mode full|geometry` for controlled ablations. Exact sources, years, mesh codes, and license links are versioned in `code/configs/plateau_sources.json`; generated data remains ignored.

The first RTX 4090 run gives the following geographically held-out Shinjuku results: full typed R-GCN AUC 0.9944 / EER 3.35%, geometry-only AUC 0.9688 / EER 6.70%, untyped AUC 0.9939 / EER 3.35%, and no-edge AUC 0.9955 / EER 3.35%. Thus semantic node features help, but this corpus does not yet demonstrate a benefit from relation-aware message passing; the repeated building-to-surface star topology is likely too regular.

## v0.9 Japan-only expanded PLATEAU experiment

The expanded Japan-only experiment uses 31 LoD2 CityGML tiles and 241 buildings. Chiyoda (12 tiles), Minato (8), and Shinjuku (11) remain geographically isolated train, validation, and test regions. A robust 1024-bit typed R-GCN was trained for 800 epochs on an RTX 4090 and evaluated on all 11 unseen Shinjuku tiles.

Rotation through 180 degrees and 80% attribute deletion retain AUC 1.0 / EER 0. Ten-percent building deletion obtains AUC 0.997 / EER 1.82%, while 40% building deletion falls to AUC 0.945 / EER 18.18%. Mixed object deletion, semantic-surface deletion, and sequential attacks deteriorate strongly at high intensity, so 60%--80% structural deletion is a stress test rather than a supported operating range. See `docs/v0.9_japan_plateau_expanded_zh.md` for the complete protocol and fixed-threshold results.

## P0 Japan multicity benchmark

The publication benchmark expands to 143 tiles and 1,137 LoD2 buildings. Tokyo wards and Saitama form the training split, Yokohama and Kawasaki form validation, and Osaka, Fukuoka, Hiroshima, and Sendai form a geographically disjoint 64-tile test split. The audited graphs contain 33,688 nodes and 166,166 edges with zero parse failures. See `docs/p0_multicity_protocol_zh.md` and `docs/data_storage_zh.md`.

## What this demo proves (and does not prove)

This is a feasibility check, not the final paper algorithm. It tests whether CityGML models can yield stable content fingerprints under translation, rotation, uniform scaling, coordinate noise, coordinate quantization, contiguous spatial cropping, attribute deletion, object reordering, and real XML object-subtree deletion. It does not yet compute exact surface-touch topology, execute real CityGML/CityJSON conversion, train on a representative city-scale corpus, or validate a deployment false-positive threshold.

## Data

All bundled samples are from the OGC CityGML 3.0 GML Encoding examples:

https://github.com/opengeospatial/CityGML3.0-GML-Encoding/tree/main/resources/examples/Building

The original repository states that the example files are intended for testing implementations of the OGC CityGML standard. Source and attribution are retained in `code/data/SOURCE.md`.

The family labels in `code/data/benchmark_manifest.json` are research metadata curated from upstream identifiers and example structure. They are not authoritative identity labels from OGC.

The optional v0.4 dataset is downloaded from the [Open City Model AWS Open Data registry](https://registry.opendata.aws/opencitymodel/) and is licensed under ODbL. Exact source URLs, license link, geographic splits, and sampling parameters are versioned in `code/configs/ocm_sources.json`.

## GPU configuration

The current demo is CPU-only. Connection metadata is kept in `gpu.example.toml`; create `gpu.local.toml` for secrets. The local secret file is excluded by `.gitignore` and must never be committed.
