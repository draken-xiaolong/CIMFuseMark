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
export HK3D_DB_PATH='/Users/wangfugui/Paper/三维CIM水印/HKReal3DMarkState/inventory.sqlite'
```

SQLite清单建议放在Mac内置APFS磁盘，不要直接在exFAT移动盘上使用WAL。
`inventory.pointer`会记录清单的实际位置，B3DM ZIP仍保存在移动盘。
若需先完成全港索引审计、但在确认容量前不下载payload，启动时设置
`HK3D_DISCOVER_ONLY=1`。JSON队列归零后进程会驻留，并生成
`metadata/payload_phase_paused.json`，不会将移动盘写满。

全港索引的最终数量、payload 抽样容量与存储结论见
[`docs/01-全港索引终审-20260821.md`](docs/01-全港索引终审-20260821.md)。

## 当前实现

1. `download_hk3d_packed.py`以SQLite保存层级清单和压缩后的JSON正文，并将B3DM写入每片10,000文件的ZIP64，避免移动硬盘的小文件空间放大；
2. `run_packed_until_complete.sh`负责断点恢复、临时网络错误重试和404缺失资源收敛；
   队列清零后会自动调用`audit_packed_download.py`校验所有ZIP的CRC、B3DM文件头、数量和字节数，
   并生成`metadata/packed_audit.json`；
3. `prepare_hk_dataset.py`建立严格地域隔离的72/24/24实验划分；
4. `run_experiments.py`完成学习式零水印训练、仅投影个性化与九类攻击评估；
5. `generate_paper_artifacts.py`从正式JSON自动生成论文曲线、表格与LaTeX宏。

查看全港打包下载状态：

```bash
python3 code/packed_download_status.py
```

按API URL从打包库导出一个JSON或B3DM（URL中不要包含API key）：

```bash
python3 code/read_packed_resource.py 'https://data.map.gov.hk/.../tile.b3dm' --out /tmp/tile.b3dm
```
