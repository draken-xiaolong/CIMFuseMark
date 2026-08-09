# 三维 CIM 零水印基线对比实验

## 协议

在 PLATEAU 新宿 8 个未参与基础编码器训练的 CityGML 瓦片上统一评测。传统方法直接从干净/攻击后的 XML 提取指纹；CIMFuseMark-personalized 使用干净登记协议。共有 28 个异源对。所有方法使用同一攻击实现、随机种子 2026，以及归一化 Hamming 相似度、AUC、EER 和异源 q95 阈值下 FRR。

传统方法保留其自然指纹长度，不通过重复比特或额外随机投影伪装成 1024 bit。Jiang18 是直接 CityGML 复现；Lee21、Wang19、Hu26 原始对象是三角网格，表中明确标为 CityGML adaptation。

## 实现与唯一性

| 方法 | 指纹长度 | 复现性质 | 异源均值 | 异源 q95 | 异源最大值 |
|---|---:|---|---:|---:|---:|
| Jiang18 radial histogram | 64 | 直接 CityGML 复现 | 0.790 | 0.891 | 0.922 |
| Lee21 spherical skew | 64 | CityGML 点集适配 | 0.501 | 0.588 | 0.688 |
| Wang19 multi-feature | 192 | CityGML 多特征适配 | 0.858 | 0.913 | 0.927 |
| Hu26 radial fusion | 256 | CityGML 径向融合适配 | 0.992 | 1.000 | 1.000 |
| CIMFuseMark-personalized | 1024 | 本文完整方法 | 0.430 | 0.452 | 0.469 |

Hu26 适配版的同源相似度普遍很高，但不同模型之间也几乎生成相同指纹；Jiang18 和 Wang19 同样存在明显的高相关尾部。该结果说明只报告攻击前后 NC 会严重高估传统方法的认证能力。

## AUC/EER 对比

| 方法 | 混合删除20% AUC/EER | 整栋删除40% AUC/EER | 表面删除40% AUC/EER | 属性删除80% AUC/EER | 量化5% AUC/EER |
|---|---:|---:|---:|---:|---:|
| Jiang18 | 0.783/28.57% | 0.694/34.82% | 0.734/34.82% | 1.000/0% | 0.500/53.57% |
| Lee21 | 0.583/50.00% | 0.386/59.82% | 0.473/50.00% | 1.000/0% | 0.641/40.18% |
| Wang19 adapted | 0.844/23.21% | 0.665/32.14% | 0.743/32.14% | 1.000/0% | 0.609/44.64% |
| Hu26 adapted | 0.670/26.79% | 0.634/33.04% | 0.714/26.79% | 0.929/7.14% | 0.598/33.04% |
| CIMFuseMark-personalized | **1.000/0%** | **1.000/0%** | **1.000/0%** | **1.000/0%** | **1.000/0%** |

所有传统几何方法在 Z 轴旋转 180° 下都保持很高的同源一致性，属性删除也不会改变纯几何特征，因此这两类攻击不足以拉开方法差距。对象删除、语义表面删除和坐标量化更能检验真实认证性能。

## 可复现性边界

- **Jiang18**：按论文公开步骤实现质心、归一化顶点范数、等宽分区及“分区数量与均值比较”二值化，可视作严格程度最高的直接复现。
- **Lee21**：按径向距离分组并计算球坐标极角偏度。原文面向三角网格，但该统计只依赖顶点，因此点集适配较直接。
- **Wang19 adapted**：原方法的 OSVETA 和 Shape Diameter Function 依赖规则三角网连接。CityGML 是带语义的多边形边界，本实现以径向局部稳定性、弦长和高度分布替代，不能标为原论文 full reproduction。
- **Hu26 adapted**：原方法对网格径向信号执行 EMD 分段并融合显式/隐式径向特征。本实现使用排序径向信号的多尺度残差完成 CityGML 适配，保留“分段、多尺度径向融合”思想，但不等价于原 Matlab EMD 实现。

因此论文主表可将 Jiang18 列为直接 CityGML baseline，将其余三项写为 `CityGML-adapted 3D zero-watermark baselines`，并在脚注中说明适配规则。不能把 adapted 结果描述为对原论文数值的复现。

## 复现命令

```bash
PYTHONPATH=code python3 code/comparison_experiments/run_comparison.py \
  --manifest code/data/plateau_manifest.json \
  --split test \
  --output code/results/traditional_baseline_comparison.json
```

实现位于 `code/comparison_experiments/`，完整逐样本分数保存在输出 JSON 中。
