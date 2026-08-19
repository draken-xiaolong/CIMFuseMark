# CIMFuseMark-Lite personalization exploration

| Variant | Negative mean | Negative q95 | Negative max | AUC | EER | TAR@FAR=5% |
|---|---:|---:|---:|---:|---:|---:|
| Lite Base | 0.5567 | 0.7322 | 0.8789 | 0.8929 | 0.1338 | 0.7079 |
| Lite + projection | 0.4944 | 0.5586 | 0.6836 | 0.9270 | 0.0969 | 0.6683 |
| Lite + partial encoder fine-tuning | 0.4945 | 0.5588 | 0.7236 | 0.9314 | 0.0838 | 0.6947 |

## Selection

The final paper uses **projection-only personalization**. The shared Lite encoder remains frozen, so registration is lightweight, auditable, and does not turn the test registry into encoder training data. Partial-encoder fine-tuning is retained only as a rejected exploratory variant.

Three-seed confirmation (2026--2028) gives Base AUC/EER/TAR of `0.9032±0.0090`, `0.1285±0.0046`, and `0.6975±0.0115`; projection-only personalization gives `0.9317±0.0045`, `0.0950±0.0032`, and `0.6831±0.0172`. Negative q95 falls reproducibly from `0.7170±0.0132` to `0.5565±0.0031`.

Therefore Base remains the inductive default. Projection-only personalization is an optional transductive enrollment step whose stable contribution is registry uniqueness, not a claim of uniformly higher fixed-threshold robustness.
