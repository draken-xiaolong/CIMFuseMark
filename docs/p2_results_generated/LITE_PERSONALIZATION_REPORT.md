# CIMFuseMark-Lite personalization exploration

| Variant | Negative mean | Negative q95 | Negative max | AUC | EER | TAR@FAR=5% |
|---|---:|---:|---:|---:|---:|---:|
| Lite Base | 0.5567 | 0.7322 | 0.8789 | 0.8929 | 0.1338 | 0.7079 |
| Lite + projection | 0.4944 | 0.5586 | 0.6836 | 0.9270 | 0.0969 | 0.6683 |
| Lite + partial encoder fine-tuning | 0.4945 | 0.5588 | 0.7236 | 0.9314 | 0.0838 | 0.6947 |

## Selection

Partial-encoder fine-tuning is the best personalized Lite variant for seed 2026. It preserves the projection-only q95, improves AUC/EER, and recovers most of the Base open-set TAR.
The result remains exploratory until repeated with additional seeds. The Base model remains the non-transductive reference.
