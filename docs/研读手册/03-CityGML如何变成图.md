# 03 CityGML如何变成图

## 1. 为什么要转成图

CityGML本身是一棵XML树，但XML父子关系不完全等于CIM关系。图表示可以同时容纳：

- Building包含RoofSurface；
- Surface从属于Building；
- XLink指向另一个对象；
- 两个对象空间邻近。

图由`节点 + 边 + 节点特征`组成。图神经网络随后让相邻对象交换信息。

## 2. 一个节点到底是什么

一个节点不是单个坐标点，也不是一栋城市的整个模型，而是一个CityGML语义对象。支持的对象包括：

- Building、BuildingPart；
- WallSurface、RoofSurface、GroundSurface；
- Door、Window；
- BuildingInstallation；
- 代码也预留了Road、Bridge、Tunnel等类型，但最终数据集目前只使用建筑主题。

例如，一个含8栋建筑的tile可能形成数百个节点，因为每栋建筑下面有多个墙面和屋顶面。

## 3. 坐标如何归属于节点

解析器遍历XML中的`gml:pos`和`gml:posList`：

1. 找到该坐标最近的上层语义对象；
2. 把坐标记到该对象；
3. 再沿父子层级向上聚合。

因此WallSurface节点只包含自身及子对象坐标，而Building节点会聚合整栋建筑的坐标。这样可以同时描述局部表面和整体建筑。

## 4. 当前边的种类

| 边 | 含义 | 是否有反向信息 |
|---|---|---|
| `contains` | 父对象包含子对象 | 子到父使用`part_of` |
| `bounded_by` | Building由某边界表面围合 | 子到父使用`part_of` |
| `part_of` | 子对象从属于父对象 | 是反向层级边 |
| `spatial_near` | 质心距离最近的对象之一 | 按每节点最近3个对象建立 |
| `xlink:*` | XML中的XLink引用 | 对应`xlink_reverse` |
| `xlink_reverse` | XLink反向边 | 是 |

训练集全部图出现过的关系名会排序并建立`关系名 -> 整数编号`词表。

## 5. 空间邻近边怎么构建

对每个拥有几何坐标的语义对象：

1. 计算对象质心；
2. 计算它和其他对象质心的欧氏距离；
3. 选择最近的3个对象；
4. 建立`spatial_near`有向边。

它让没有直接XML父子关系、但地理位置相近的对象也能交换信息。

## 6. 图构建伪代码

```text
读取GML XML
找到所有支持的语义对象
为每个对象分配节点编号

对每个节点：
    找最近的语义父对象
    建立父 -> 子层级边
    建立子 -> 父 part_of边

遍历所有坐标、Polygon和叶子属性：
    归属到最近语义对象

从子节点向父节点聚合坐标、面数量和子树大小
计算每个节点特征

解析XLink引用并建边
按质心最近邻建立spatial_near边
返回CIMGraph
```

## 7. 图对象在代码里的结构

`CIMGraph`包含：

```text
source: 源GML路径
nodes: GraphNode列表
edges: GraphEdge列表
metadata: 图模式、关系类型、坐标数、尺度等
```

每个`GraphNode`包含：

```text
node_id          对象ID
node_type        Building/WallSurface等
depth            层级深度
parent           父节点编号
point_count      聚合坐标记录数
attribute_count  属性数量
features         基础19维特征
extended_features 扩展12维特征
```

## 8. 图构建中的不变性

节点特征计算前会得到：

- 全模型中心`center`；
- 全模型坐标到中心的均方根半径`scale`。

很多距离都会除以`scale`，因此统一尺度变化影响较小；使用协方差特征值和径向距离，而不是直接使用XYZ坐标，也增强了旋转和平移不变性。

## 9. 相关代码

- 图构建主体：`code/cimfusemark/citygml_graph.py`
- 图张量转换：`code/cimfusemark/rgcn.py`中的`lgfm_graph_tensors`
- 数据清单：`code/data/plateau_multicity_manifest.json`
