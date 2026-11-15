# 论文图号与绘图入口

图号按 `iclr_submission/main.tex` 及其引用章节的顺序排列。具体论文文件和 SHA256 见 `figure_manifest.json`。所有输出均使用 Times New Roman；Matplotlib 数学符号使用 STIX。

| 图号 | 运行组名（`--only`） | 可编辑源码 | 输出文件（PDF；实验图另有 PNG） |
|---|---|---|---|
| 1 | `hero` | [hero.py](renderers/hero.py) | `bird_dev_cost_frontier_20seed_10pct.pdf` |
| 2 | `system_overview` | [system_overview.tex](source/system_overview.tex) | `system_overview.pdf` |
| 3 | `main_curves` | [main_curves.py](renderers/main_curves.py) | `eight_methods_30pct_hv_regret.pdf` |
| 4 | `main_curves` | [main_curves.py](renderers/main_curves.py) | `eight_methods_30pct_generational_distance.pdf` |
| 5 | `main_curves` | [main_curves.py](renderers/main_curves.py) | `eight_methods_30pct_inverted_generational_distance.pdf` |
| 6 | `stackoverflow_frontier` | [stackoverflow_frontier.py](renderers/stackoverflow_frontier.py) | `figure6_larger_fonts.pdf` |
| 7 | `three_main` | [three_main.py](renderers/three_main.py) | `three_objective_eight_methods.pdf` |
| 8 | `job_runtime` | [job_runtime.py](renderers/job_runtime.py) | `figure8_job_elapsed_eight_methods.pdf` |
| 9 | `frequencies` | [frequencies.py](renderers/frequencies.py) | `gittins_20seed_frequency_frontiers_10pct.pdf` |
| 10 | `frequencies` | [frequencies.py](renderers/frequencies.py) | `gittins_20seed_frequency_frontiers_30pct.pdf` |
| 11 | `estimated_actual` | [estimated_actual.py](renderers/estimated_actual.py) | `figure11_eight_methods_10pct.pdf` |
| 12 | `estimated_actual` | [estimated_actual.py](renderers/estimated_actual.py) | `figure12_eight_methods_30pct.pdf` |
| 13 | `two_ablations` | [two_ablations.py](renderers/two_ablations.py) | `curves/cost_mechanism/two_objective_cost_mechanism_ablation_hv_gd_igd_8x3.pdf` |
| 14 | `two_ablations` | [two_ablations.py](renderers/two_ablations.py) | `curves/directions/two_objective_directions_ablation_hv_gd_igd_8x3.pdf` |
| 15 | `quality_frontier` | [quality_frontier.py](renderers/quality_frontier.py) | `cc_vs_quality_only_frontiers_10pct.pdf` |
| 16 | `two_direction_runtime` | [two_direction_runtime.py](renderers/two_direction_runtime.py) | `direction_runtime_comparison_20seed.pdf` |
| 17 | `two_ablations` | [two_ablations.py](renderers/two_ablations.py) | `curves/continuation/two_objective_continuation_ablation_hv_gd_igd_8x3.pdf` |
| 18 | `two_ablations` | [two_ablations.py](renderers/two_ablations.py) | `curves/scheduler/two_objective_scheduler_ablation_hv_gd_igd_8x3.pdf` |
| 19 | `two_prior` | [two_prior.py](renderers/two_prior.py) | `two_objective_prior_variance.pdf` |
| 20 | `recommendation` | [recommendation.py](renderers/recommendation.py) | `recommendation_confidence_2d.pdf` |
| 21 | `three_families` | [three_families.py](renderers/three_families.py) | `cost_mechanism_ablation/three_objective_cost_mechanism_ablation_hv_gd_igd.pdf` |
| 22 | `three_directions` | [three_directions.py](renderers/three_directions.py) | `three_objective_directions_affine_legend.pdf` |
| 23 | `three_direction_runtime` | [three_direction_runtime.py](renderers/three_direction_runtime.py) | `three_objective_direction_runtime_seven_settings.pdf` |
| 24 | `three_families` | [three_families.py](renderers/three_families.py) | `continuation_ablation/three_objective_continuation_ablation_hv_gd_igd.pdf` |
| 25 | `three_prior` | [three_prior.py](renderers/three_prior.py) | `three_objective_prior_variance.pdf` |
| 26 | `recommendation` | [recommendation.py](renderers/recommendation.py) | `recommendation_confidence_3d.pdf` |

另外提供 `three_frontiers`：HotpotQA 和 MathQA 的完整 20-seed、8 方法、30% 预算三目标前沿图。该组尚未被这份论文快照引用。
