# HKReal3DMark

李茂栗老师方向：面向香港实景三维模型的非侵入式零水印研究。

原日本PLATEAU/CityGML版本已在`CIMFuseMark`仓库以标签
`plateau-citygml-final-2026-08-21`冻结，本目录独立开展香港实景三维数据、算法和论文。

## 数据对象

- 来源：香港地政总署Open3Dhk / CSDI 3D Visualisation Map API；
- 产品：方格形式实景三维模型；
- 格式：Cesium 3D Tiles，索引为`.json`，网格payload为`.b3dm`；
- 坐标参考：WGS84；
- 本地数据根目录：`/Volumes/SANDISK-ELE/HKReal3DMarkData`。

## API key配置

```bash
cp .env.example .env
```

编辑`.env`并填写`HK3D_API_KEY`。不要把key提交到Git。

也可以只在当前终端配置：

```bash
export HK3D_API_KEY='你的key'
export HK3D_DATA_ROOT='/Volumes/SANDISK-ELE/HKReal3DMarkData'
```

## 当前阶段

1. 全量遍历官方瓦片树并生成容量清单；
2. 核对硬盘空间和预计请求数；
3. 断点续传下载全港原始3D Tiles；
4. 选择代表区域转换网格，建立实验划分；
5. 重新设计局部网格patch图和攻击协议。

未经全量清单确认，不直接展开全港递归下载，以避免因LoD树重复内容导致不可控占用。

