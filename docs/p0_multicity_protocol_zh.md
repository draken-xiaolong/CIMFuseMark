# P0 日本 PLATEAU 跨城市实验协议

## 数据划分

全部数据来自 Project PLATEAU，训练、验证和测试城市互不重叠。

| 划分 | 地区 | tile 数 |
|---|---|---:|
| Train | 东京千代田、港区、新宿，埼玉 | 47 |
| Validation | 横滨、川崎 | 32 |
| Test | 大阪、福冈、广岛、仙台 | 64 |

语料共包含 143 个 tile、1,137 栋建筑、31,793 个语义边界对象。构图后得到 33,688 个节点和 166,166 条边，143 个文件均成功解析。

## 数据来源与规模

完整城市 ZIP 仅用于抽取建筑主题文件，总计约 12.6 GB。程序按文件大小优先扫描中央城区建筑网格，从每个城市选取具有显式 LoD2 BoundarySurface 的建筑，并按 8 栋建筑切分 tile。训练用压缩包约 15 MB，GPU 端解压后约 138 MB。

精确 URL、城市代码、年份、规范版本、Content-Length 和采样上限记录在 `code/configs/plateau_multicity.json`。生成的 manifest 进一步记录原始 ZIP SHA-256、入选 GML 的 SHA-256、建筑数和语义统计。

## P0 评价原则

1. 编码器仅使用 Train 训练；
2. 阈值只由 Validation 确定；
3. Test 不参与训练、模型选择或阈值选择；
4. Base、Separation、Personalized 和传统 baseline 使用相同 Test 模型和攻击定义；
5. 个性化只访问 Test 的干净登记原件，不访问正式攻击样本；
6. 闭集登记认证与开放集拒识分别报告；
7. 高强度删除导致无法构图时按认证失败计入。

## CIM 专用攻击

除通用旋转、量化、噪声、属性和对象删除外，P0 协议增加：

- LoD2 到 LoD1 包围盒实体降级；
- CityGML 到 CityJSON 2.0 再回到 CityGML 的格式往返；
- Building--Surface 层级扁平化；
- XLink/边界关系删除；
- Wall/Roof/GroundSurface 语义标签重标；
- 连续空间裁剪；
- 建筑增量更新；
- XML ID 重命名。

所有攻击均生成独立文件并重新解析构图。CityJSON 往返会保留 MultiSurface 几何与表面语义，但会规范化 XML 结构和属性，是跨编码一致性的压力测试，不等同于验证所有第三方转换器。
