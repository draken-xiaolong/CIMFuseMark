# 数据存储与 GPU 同步

## 本地数据根目录

大规模实验数据存储在：

```text
/Volumes/SANDISK-ELE/CIMFuseMark/CIMFuseMarkData
```

目录结构：

```text
CIMFuseMarkData/
├── data/          # 原始 PLATEAU ZIP、抽取 GML、实验 tile 和 manifest
├── attacks/       # 临时攻击文件工作区
├── graph_cache/   # 可选图缓存
├── checkpoints/   # 本地归档的模型检查点
├── results/       # 完整逐样本结果
└── metadata/      # 迁移审计与训练数据压缩包
```

仓库 `code/data` 中的大数据目录使用绝对符号链接连接到移动硬盘。拔出移动硬盘后这些链接会暂时失效；重新挂载到相同卷名后自动恢复。配置、下载 URL、程序和论文级汇总结果仍由 Git 管理，大文件不进入仓库。

## 迁移审计

原有 PLATEAU、OCM 缓存和实验 tile 已迁移。`metadata/migration_20260819.json` 记录每个迁移项的文件数、总字节数和基于相对路径/文件大小生成的审计摘要。

## GPU 数据策略

租用 GPU 的系统盘 `/` 只有约 13 GB 可用，但高速数据盘 `/root/autodl-tmp` 约有 251 GB 可用。GPU 实验目录为：

```text
/root/autodl-tmp/CIMFuseMark_p0
```

不向 GPU 同步约 12.6 GB 的原始城市 ZIP，只同步约 138 MB 的筛选后训练/验证/测试 tile 和 manifest。神经网络按图处理数据，磁盘数据量不会一次性进入显存。

当前扩大语料在 RTX 4090 上训练时显存峰值约 21.3 GB，能够在 24 GB 显存内运行，但不应并行启动两个训练进程。若继续扩大训练集，应将攻击视图改为按批加载或使用 CPU 图缓存，避免将全部攻击图张量同时驻留 GPU。
