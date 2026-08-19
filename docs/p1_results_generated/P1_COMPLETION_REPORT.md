# P1 mechanism and statistical experiment report

- Completed experiment entries: 24/24.
- Full-model mean core AUC: 0.8866; mean TAR@FAR=5%: 0.6671.
- Best graph ablation by mean AUC: hierarchy flattened (0.9015).
- Bootstrap resamples per attack: 2000; seeds: full_seed2026, full_seed2027, full_seed2028.
- Mean measured training-only duration: 395.58 s over 21 newly measured runs.

## Separated pipeline costs

- Training peak GPU allocation: 18.82 GiB; process peak RSS: 2.32 GiB.
- XML parsing: 3.964 ms/model (mean).
- Graph construction estimate: 29.030 ms/model (mean).
- Uncached dataset download: 11.73 GiB in 1146.93 s (uncached download with 3 concurrent sources; backend=aria2c).
- rgcn: 234,849 parameters, 0.734 ms/model inference, 1363.2 models/s for the measured 64-model pass.
- deepsets: 234,849 parameters, 0.429 ms/model inference, 2331.6 models/s for the measured 64-model pass.
- personalized: 234,849 parameters, 0.727 ms/model inference, 1376.1 models/s for the measured 64-model pass.
- jiang18_citygml_radial_histogram: 6.976 ms/model end-to-end handcrafted fingerprint extraction.
- lee21_spherical_skew: 7.454 ms/model end-to-end handcrafted fingerprint extraction.
- wang19_multifeature_adapted: 7.249 ms/model end-to-end handcrafted fingerprint extraction.
- hu26_radial_fusion_adapted: 7.282 ms/model end-to-end handcrafted fingerprint extraction.
- nonlearned_relation_graph: 31.467 ms/model end-to-end handcrafted fingerprint extraction.
- Personalized registration: 24.973 s for the 64-model registry plus validation background.

The numerical source of every plot and table is retained in the generated CSV/JSON files. Ablation conclusions follow the measured ranking: typed relation propagation is supported for fixed-threshold TAR, where it exceeds no-edge and untyped alternatives, but not as a universal ROC-AUC improvement.
