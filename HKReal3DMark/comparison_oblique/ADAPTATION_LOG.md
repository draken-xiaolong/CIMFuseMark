# 适配记录

## PC-Reversible26

- 上游代码固定读取 OBJ、PCD、纹理图和 32×32 BMP；需要改为接收 B3DM/GLB 网格。
- 上游 `keypoint_extract_by_angle` 引用了函数外全局变量 `pcd`；适配版改为显式输入。
- 上游点分组选取 `argsort(...)[-2:]`，实际得到最远点而非注释描述的邻近点。正式实验
  将同时保留 `upstream` 和 `nearest-fixed` 两种模式，主表使用与论文文字一致的
  `nearest-fixed`，附录报告原代码模式。
- 纹理保持原样，PC-Reversible26 的水印只写入几何半径。

其余方法的逐条适配差异将在实现后追加，未完成前不得在论文中称作“精确复现”。
