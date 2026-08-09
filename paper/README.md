# CIMFuseMark 论文初稿

编译：

```bash
cd paper
latexmk -xelatex -interaction=nonstopmode main.tex
```

`main.tex` 是中文第一版研究稿。红色“待补”项表示投稿前仍需用自动化实验替换，不代表已有实验结果。当前稿件明确区分跨区域基础模型和可见干净登记对象的 transductive personalization，也明确记录 typed R-GCN 未优于 no-edge 的负结果。
