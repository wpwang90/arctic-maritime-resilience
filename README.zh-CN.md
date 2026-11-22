# 最终主图和附图计算代码

本仓库仅保留最终 **5 张主图、3 张 Extended Data 图、9 张附图**的计算、
绘图入口和必需依赖。

不含旧版图形方案、无关修复流程、论文主稿或附录正文，也不含 Word、PPT、
Overleaf 写作脚本、原始数据和模型权重。

- 图号与代码入口：`docs/final_figures.json`
- 运行顺序：`docs/PIPELINE.md`
- 输入要求：`docs/DATA.md`
- 方法定义：`docs/METHODS.md`
- 原代码与发布副本校验：`docs/source_manifest.csv`

保留历史编号目录以维持相对路径。依赖模块中的旧文件名不是另一个计算入口。
已清理无关环境文字和写作例程，保留参数、单位、适用范围和数据来源注释。
智能体实验的模型、提示词及方法来源属于复现信息，仍然保留。

```sh
python -m pip install -r requirements-test.txt
python tools/check_release.py
python -m unittest discover -s tests -v
```

小型人工矩阵只用于单元测试，不作为论文研究数据。完整制图需要本地准备具有
使用权限的数据；不能将“通过数值测试”理解为已经从空仓库复算所有图。
