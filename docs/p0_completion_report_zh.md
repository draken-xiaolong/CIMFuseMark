# CIMFuseMark P0 完成报告（2026-08-19）

## 完成范围

- P0-1：10 个日本 PLATEAU 地区，143 个 tile，1,137 栋建筑；47/32/64 个 tile 按城市划分为训练/验证/测试，143/143 构图成功。
- P0-2：统一复现 Jiang18、Lee21-adapted、Wang19-adapted、Hu26-adapted、非学习关系图、DeepSets/no-edge 和 CIMFuseMark 系列模型。
- P0-3：以横滨、川崎 32 个未登记模型校准开放集最大匹配阈值；使用严格经验阈值保证验证 FAR 不高于 5% 或 1%，不使用测试攻击重新标定。
- P0-4：完成跨 LoD、CityGML--CityJSON 往返、层级扁平化、关系删除、语义重标记、建筑增量、空间裁剪、对象重排、ID 重命名和坐标单位转换攻击。
- P0-5：在相同 64 个测试 tile、2,016 个异源对和相同攻击文件上比较 Base、Separation、Personalized 与 DeepSets/no-edge。

## 关键结果

| 模型 | 异源均值 | 异源 q95 | 异源最大值 | 互异指纹 | 碰撞对 |
|---|---:|---:|---:|---:|---:|
| Base | 0.577 | 0.758 | 0.896 | 64/64 | 0 |
| Separation | 0.580 | 0.760 | 0.878 | 64/64 | 0 |
| Personalized | **0.495** | **0.564** | **0.688** | 64/64 | 0 |
| DeepSets/no-edge | 0.587 | 0.744 | 0.877 | 64/64 | 0 |

Separation 模型在 180 度旋转、80% 属性删除、ID 重命名、对象重排和 0.001/1000 倍坐标单位转换下取得 AUC 1.000。低强度删除仍可用：5% 混合对象删除 AUC 0.991，10% 建筑删除 AUC 0.967，10% 表面删除 AUC 0.997。跨 LoD、建筑增量及 60%--80% 结构编辑明显失败，不能作为可靠工作区间。

Personalized 将异源最大值从 0.878 降至 0.688，并将建筑删除 40%、表面删除 40%、语义重标记 20% 和 CityJSON 往返的 AUC 分别从 0.898、0.955、0.964、0.671 提升至 0.919、0.965、0.996、0.779；但它属于看过干净登记模型的 transductive 协议。

DeepSets/no-edge 在部分结构删除上不弱于 R-GCN，但语义重标记 20% 的 AUC 为 0.799，低于 Separation 的 0.964。因此现阶段可以强调“学习式语义图表示”和“语义特征有效”，不能宣称关系传播在所有攻击上显著优于集合网络。

## 数据与结果位置

- 原始下载、切片和 manifest：`/Volumes/SANDISK-ELE/CIMFuseMark/CIMFuseMarkData/data`
- 检查点与训练审计：`/Volumes/SANDISK-ELE/CIMFuseMark/CIMFuseMarkData/checkpoints`
- 完整逐模型结果：`/Volumes/SANDISK-ELE/CIMFuseMark/CIMFuseMarkData/results`
- 下载/训练日志及迁移审计：`/Volumes/SANDISK-ELE/CIMFuseMark/CIMFuseMarkData/metadata`
- 可提交仓库的汇总表图：`docs/p0_results_generated`

## GPU 结论

当前 RTX 4090（24 GB）无需更换。typed R-GCN 训练实测峰值约 21.3 GB，能够完成当前 47 个训练 tile，但必须串行训练。外置硬盘上的约 13 GB 主要是可重新下载的 PLATEAU 原始 ZIP，并不需要同时载入显存；GPU 端只同步了约 138 MB 的选定训练切片。若未来直接把十几 GB 解压 CityGML 全量送入训练，现有 eager 多视图缓存会超出 24 GB，需要先改为 CPU/lazy 缓存、mini-batch 身份采样和梯度累积；这属于数据管线变动，不是当前 P0 必须更换 GPU。
