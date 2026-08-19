# P2 encoder, enhanced-feature, and lightweight exploration

LiteGeoFuseMark reference: 87,857 trainable parameters.

| Variant | Params | vs LiteGeo | AUC | EER | TAR@FAR=5% | Negative q95 |
|---|---:|---:|---:|---:|---:|---:|
| R-GCN, G8+D | 234,849 | 2.67x | 0.9097 | 0.1299 | 0.7115 | 0.7395 |
| GCN, G8+D | 161,121 | 1.83x | 0.9098 | 0.1205 | 0.5986 | 0.7454 |
| GraphSAGE, G8+D | 179,553 | 2.04x | 0.9068 | 0.1225 | 0.6454 | 0.7031 |
| GAT, G8+D | 179,937 | 2.05x | 0.9087 | 0.1221 | 0.6911 | 0.7314 |
| Relation-aware GAT, G8+D | 180,897 | 2.06x | 0.8918 | 0.1388 | 0.5397 | 0.7288 |
| R-GCN 64/192, G8+D | 117,057 | 1.33x | 0.8929 | 0.1338 | 0.7079 | 0.7322 |
| R-GCN 64/128, G8+D | 84,033 | 0.96x | 0.8865 | 0.1397 | 0.5036 | 0.7256 |
| R-GCN, G8+D+C | 235,137 | 2.68x | 0.9080 | 0.1331 | 0.6262 | 0.7930 |
| R-GCN, G8+D+C+S | 235,617 | 2.68x | 0.9066 | 0.1353 | 0.6815 | 0.8069 |
| R-GCN, G8+D+C+S+F | 236,001 | 2.69x | 0.9203 | 0.1149 | 0.6238 | 0.7695 |

## Evidence-based selection

- Primary model: full-width R-GCN with G8+D. It retains the best independently calibrated open-set TAR.
- Lightweight model: R-GCN 64/192 (117,057 parameters). It removes 50.2% of the main-model parameters while retaining 99.5% of its open-set TAR.
- GAT is a viable encoder trade-off but does not improve the primary open-set metric.
- DCT radial frequency features improve mean AUC/EER, but their cross-region fixed-threshold TAR is lower; they are not selected for the main model.
- The 84,033-parameter model is smaller than LiteGeoFuseMark but is rejected because the authentication loss is too large.
