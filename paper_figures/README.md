# 论文绘图包

本目录提供论文正文和附录 **26 张图**的可编辑绘图代码、冻结绘图数据、图号索引，以及两张完整 20-seed 三目标前沿补充图。所有图的文字统一使用 **Times New Roman**；Matplotlib 数学符号使用 STIX。数据对应 2026-11-15 的本地论文快照，具体文件和校验值见 [figure_manifest.json](figure_manifest.json)。

重画只需本目录，不依赖实验服务器、原始大模型输出或相邻论文仓库。数据已包含聚合后的均值、误差、推荐频率和绘图坐标，不需要重新运行实验。

可先查看 [全部图的 PDF 预览](previews/all_figures_times_new_roman.pdf)：前 26 页按论文图号排列，最后两页是补充的三目标前沿图。

## 安装与重画

使用 Python 3.10 或更高版本，在仓库根目录运行：

```sh
python -m pip install -r paper_figures/requirements.txt
python paper_figures/render.py --check-data
python paper_figures/render.py
```

也可以单独复制 `paper_figures` 目录，然后在该目录运行 `python render.py`。本机必须安装 Times New Roman；缺少字体时程序会报错，避免静默替换成其他字体。字体文件不随代码分发。

Figure 2 是 TikZ 示意图，需要安装带 `fontspec`、`standalone` 和 TikZ 的 XeLaTeX。只重画实验图时运行：

```sh
python paper_figures/render.py --skip-diagram
```

输出默认位于 `paper_figures/rendered/`。实验图输出 PDF 和 PNG，部分图还输出 SVG。Figure 2 输出 PDF。指定输出目录、或只重画一组：

```sh
python paper_figures/render.py --list
python paper_figures/render.py --only main_curves three_main --output output/teacher_figures
python paper_figures/render.py --only two_ablations two_prior
```

各图的命令、源码和输出文件名见 [FIGURES.md](FIGURES.md)。`main_curves` 还会输出 10% 预算的辅助版本；论文主图对应 30% 版本。`three_frontiers` 是两张尚未被这份论文快照引用的三目标前沿图。

## 统一格式

[style.json](style.json) 是所有 Matplotlib 图的公共设置入口：

| 参数 | 用途 |
|---|---|
| `font_family` | 所有标题、刻度、图例和注释的字体，默认 Times New Roman |
| `math_fontset` | 数学符号字体，默认 `stix` |
| `font_scale` | 在各图当前字号基础上统一缩放，默认 `1.0` |
| `line_width_scale` | 统一缩放线条宽度，默认 `1.0` |
| `dpi` | PNG 分辨率；`null` 保留每张图的导出分辨率 |

例如，将 `font_scale` 改成 `1.1` 即可把文字统一放大 10%。也可复制配置文件，通过 `--style my_style.json` 指定。放大字号后应检查图例和标签间距；增加 DPI 不会增大论文中的相对字号。

各图的 `figsize`、`subplots_adjust`、图例位置和具体字号保留在对应的 `renderers/*.py` 中，便于按版面调整。公共导出逻辑在 [common.py](renderers/common.py)，会在保存时再次指定 Times New Roman，并检查样式处理没有改变曲线坐标、误差带几何或坐标范围。系统示意图的字体和布局直接编辑 [system_overview.tex](source/system_overview.tex)；该图不读取 `style.json`。

八方法比较的颜色固定为：CC-Gittins 橙、EGE-SH 绿、APE-k 紫、qNEHVI-PC 粉、Random configurations 棕、Random questions 青、CoPSI 蓝、CoHV-UCB 橄榄绿。消融图的颜色表示配置，前沿图的颜色表示推荐频率或 Pareto 分类，含义见各图图例。

## 数据与统计口径

- 曲线和运行时间图保留原有均值及 **±2 SE**；推荐频率图保留 20 个种子的频数。不要把 `two_se` 列改标为 SD 或 95% CI。
- 10% 和 30% 的 CoPSI 来自独立预算实验。随机基线的 `steps-post` 曲线及误差带保持一致。
- 汇总表中的缺失值和 `n_runs` 原样保留。部分基线并非所有数据集、所有预算点都具备 20 个完成种子，不应将缺失值补零。
- 主运行时间图使用已完成任务的 Slurm job elapsed；方向消融图使用 policy wall-clock。两者计时范围不同。
- 前沿图使用原始质量、部署成本和延迟坐标；曲线图读取已计算的 affine 指标。改变格式不需要重新计算评价指标。
- `data_checksums.json` 校验全部绘图输入；`figure_manifest.json` 记录论文资产与图号；`source/provenance.json` 记录来源脚本和输入文件的哈希。来源路径仅用于溯源，运行时不会访问这些路径。

输出目录中的 `font_audit.jsonl` 记录每次 Matplotlib 导出的实际字体路径和数值图元校验值。修改格式后，建议先重画一组并按论文插入宽度检查可读性，再运行全部图。
