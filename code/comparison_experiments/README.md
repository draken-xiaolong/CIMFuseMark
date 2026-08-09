# 三维 CIM 零水印对比实验

本目录在同一 PLATEAU 划分、XML 攻击实现和认证指标下复现传统三维零水印基线。

| 实现 | 论文 | 复现性质 |
|---|---|---|
| `Jiang18` | [Jiang & Kim, *Zero Watermarking Scheme for CityGML*, 2018](https://www.jatit.org/volumes/Vol96No22/13Vol96No22.pdf) | 直接按 CityGML 径向距离直方图步骤复现 |
| `Lee21` | [Lee et al., spherical coordinate and skewness, 2021](https://doi.org/10.1007/s11042-021-10878-0) | CityGML 点集适配；按径向分组统计极角偏度 |
| `Wang19Adapted` | [Wang & Zhan, multi-features, 2019](https://doi.org/10.1007/s11042-017-4666-1) | 适配版；CityGML 缺少原方法所需规则三角网，使用径向稳定性代理替代 OSVETA/SDF |
| `Hu26Adapted` | [Hu et al., feature integration, 2026](https://www.nature.com/articles/s41598-025-28314-w) | 适配版；用多尺度径向残差替代依赖网格有序信号的 EMD |

后两种方法不能标注为论文逐字复现，结果表必须保留 `adapted`。它们的作用是比较人工多特征/径向融合范式，而不是声称完全复制原作者实现。

运行完整实验：

```bash
PYTHONPATH=code python3 code/comparison_experiments/run_comparison.py \
  --manifest code/data/plateau_manifest.json --split test
```

输出位于 `code/results/traditional_baseline_comparison.json`。各方法保留原始设计对应的指纹长度，不通过重复或随机投影强行扩展到 1024 bit；认证比较使用归一化 Hamming 相似度、AUC、EER 和异源 q95 阈值下的 FRR。
