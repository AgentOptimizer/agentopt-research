# Combined-objective 实验结果索引

本文件先说明当前结果入口，再保留截至 2026-09-13 整理的历史实验索引。历史部分按实验关系分组，不代表当前工作区的目录清单；许多报告和图表仅保存在本地，不随 Git 分发。

以下历史索引记录的是保存下来的实验协议；当前源码的默认参数和推荐语义已经多次变化。目录名中的 “LCB”“completed-only”“full Pareto” 不能单独用来判断实验内容。

## 当前协议（2026-09-26）

当前论文方法是 **CC-Gittins / two-axis**：G0 使用 `(0,1)`、`(1,0)`，
真实 continuation cost、1:1 round-robin、逐方向 η 衰减和 finite-LCB 推荐。
唯一的消融清单是 [`gittins_ablation_v2.py`](../gittins_ablation_v2.py)：
G0–G9，覆盖 cost、directions、continuation、scheduler 四组。
运行和绘图命令见 [`experiments/README.md`](../../README.md)。

完整消融的新结果写入 `gittins_ablation_v2_8bench_20seed/<configuration>/seed-<seed>/...`。
快速运行的默认结果另存于 `two_axis_finite_lcb_raw_mean_seed42_independent/exact_axes/`，
不直接计入消融汇总；完整消融应通过 task wrapper 运行。
本地有 G0 时优先读本地结果；否则可以复用旧 `g2_exact_axes` 结果。
汇总时会验证配置、输入真值与 arm 顺序，不能把不同矩阵版本混在一组比较里。

Gauss-Radau 已退役。下方是历史结果索引，目录名、指标和结论仅适用于当时的协议；
相关实验命令已清理，不作为当前运行说明。历史结果本身保留用于核对。

## 读结果前统一口径

### 数据、成本和“找到”

下述历史五数据集实验使用所有 arms 都有结果的共同题集：

| 数据集 | Arms | 每个 arm 的共同题数 |
|---|---:|---:|
| HotpotQA | 81 | 190 |
| MathQA | 81 | 135 |
| StackOverflow | 23 | 710 |
| BIRD Dev | 75 | 1,534 |
| restaurant_valid | 972 | 156 |

早期还有 **restaurant_test：36 arms × 86 题**，与 restaurant_valid 是不同实验输入。MathQA 原始 lookup 有 200 个题目 ID、14,961 个可用单元，各 arm 的记录不齐；135 是共同交集，并非公开 MathQA 数据集的总规模。

- 近期报告中的“成本 X%”通常是 **累计实际评估 USD / 完整共同题矩阵的评估 USD**。它不等于题数比例，也不等于程序运行时间；旧 weighted-J 表格的预算定义另看表注。
- **首次追加评估、后验排名第一、完成评估、进入推荐**是不同事件。“持续进入”要求之后直到该次运行结束都保留。
- **全前沿覆盖**允许推荐额外 arms；**精确前沿**要求推荐集合恰好等于完整数据的 Pareto 集合。
- 近期 Pareto snapshots 用完整数据的 accuracy / 每题 mean USD 定位，是离线诊断。实心/空心表示已完成/未完成的推荐；具体图例以该目录的 plot manifest 为准。
- 推荐集合变化产生的 C1、C2… 是事件序号，不是统一预算。不同方法的 C13 不能直接横向比较；使用 matched-budget 表或明确的 USD 节点。
- 这里的绝大多数 radial 实验只使用 seed42。多方向共用缓存、原生采样和推荐重放的 wall time 口径不同，不能直接排成算法速度榜。

### 规则名称随历史变化的含义

| 名称 | 本目录中的含义 |
|---|---|
| 早期 completed-only | 只让 completed arms 参与方向评分，取得各方向 winner 后合并、过滤；通常仍使用 posterior，而非推荐全部 completed 实测前沿。 |
| completed raw Pareto | 从全部 completed arms 的实测 mean accuracy / mean USD 直接取 Pareto frontier，不受方向 winner 限制。 |
| 早期 LCB | 所有 arms，包括 completed，都在 normalized desirability 上使用 posterior mean−β·std，再按方向选 winner。 |
| hybrid-LCB | completed 不减 std，但仍使用 posterior mean；未完成 arms 减 β·std。它不等于后来的 finite-target LCB。 |
| finite-LCB | 预测固定完整题集的最终 mean；accuracy 用预测 mean−β·std，cost 用 mean＋β·std，在所有 arms 的保守坐标上取 Pareto frontier。n=N 时自然等于实测均值、std=0。 |
| finite mean≥32 | 同一个 finite-target mean，β=0；只推荐 n≥32 的 arms。32 是实验门槛，不是统计有效性的普遍界限。 |
| shared questions | 共同 warm-up 后也共享同一题序，各 arm 按自己的进度推进。独立题序实验也有共同 warm-up，不能只看 warm-up 就认定全程共享。 |
| η decay / 旧字段 lambda | 标量搜索费用乘数；旧参数名 `lambda_initial`、`lambda_decay` 不代表方向向量。异步版每个方向独立下降，round-robin 仍保留。 |

Finite-target 预测在每个坐标上使用：

```text
mu = [S + (N - n) * m] / N
variance = [(N - n)^2 * v + (N - n) * tau^2] / N^2
```

其中 S 是已观测结果之和，m、v 是未观测部分所用的潜在均值 posterior，tau² 是冻结的每题噪声参数。β=1 是模型下的保守评分，未提供联合或 anytime 置信保证。

**成本空间也需要区分**：

- 早期 reciprocal 版本对每题费用 c 计算 R/(R+c)，再平均。它通常不等于对 mean(c) 做 reciprocal，可能改变按 mean USD 定义的 Pareto 关系。
- raw_mean 版本在原始 USD 上维护统计模型；acquisition 仍使用仿射坐标 1−c/R。因此它并非“完全没有缩放”。切换时 R、prior/noise 的校准也发生变化。
- 旧 Pareto baselines 的部分诊断用 R/(R+mean(c))，近期 raw-mean 报告又使用各自记录的 metric space。跨目录比较 HV/GD/IGD，先检查 `metric_space`、reference 和输入，不能直接拼数值。

## 实验演进概览

主线是：早期 radial/Gittins → 全局 η decay → 比较 accuracy-last → 每方向异步 η → all-arm LCB → hybrid-LCB → raw-mean cost → completed 实测前沿 → finite-target 推荐 → n≥32 对照 → 共享题序 → 共享轨迹上的 finite-LCB 重放。

其中有三条独立变化轴：

1. **采样**：方向调度、η、成本模型、题序会改变哪些 arm/question 被评估。
2. **推荐**：在固定采样轨迹上更换规则，改变用户看到的集合，通常不改变评估费用。
3. **工程与展示**：轻量事件、延后诊断、缓存、完整 checkpoint 出图，主要用于降低运行开销或便于审计。

### 1. 早期探索、基线和方差门槛

| 目录 | 做了什么 / 相比前版改变什么 | 阅读入口与适用范围 |
|---|---|---|
| [multiobjective/](multiobjective/) | 更早的单个 weighted-J 目标：accuracy 减去 min-max normalized cost 与 latency 的加权项，比较 Matrix UCB、random、hill climbing 等。 | 四个数据集的 LaTeX 表；如 [MathQA 表](multiobjective/multiobjective_mathqa.tex)。表注记录 50 seeds、两项权重均为0.1。“Found Best”是 J 最优，不是完整 Pareto 恢复；数据口径也不同于后来的共同题集。 |
| [radial_gittins_plots/](radial_gittins_plots/) | HotpotQA/MathQA 早期固定 η 的 radial-Gittins 轨迹和 archive 诊断。记录内生停止点，并强制继续到全量以画曲线；含 provisional/deployable 两种 archive scope。 | [summary.json](radial_gittins_plots/summary.json)、cost_trajectory.csv、raw archive comparison 图。记录为九个 interior directions，另有 exact-axis archive anchors；不能当成后来十方向 anytime 版本。 |
| [radial_ucb_vs_gittins/](radial_ucb_vs_gittins/) | 在早期协议下比较 radial-UCB 与 radial-Gittins，检查停止成本、推荐及 HV/GD/IGD。 | summary.json、cost_trajectory.csv、downsampled_trajectories.json 和对照 PDF。含 provisional/deployable/anytime scope；“继续画到100%”与“内生停止时的推荐”须分开。它是历史基线对照，不是异步 η 的消融。 |
| [pareto_baselines/](pareto_baselines/) | HotpotQA/MathQA 上的 EGE-SH、APE-K、qNEHVI Pareto identification 基线，seed42。 | 以 [cost_trajectory.csv](pareto_baselines/cost_trajectory.csv) 为完整方法清单；summary.json 只保留部分汇总，qNEHVI 另有 qnehvi_summary.json / qnehvi_cost_trajectory.csv。绘图入口是 [plot_pareto_identification_baselines.py](../plot_pareto_identification_baselines.py)。 |
| [anytime_radial_gittins/](anytime_radial_gittins/) | 加入真正参与采样的 (1,0)，共十方向；所有方向停止后统一降低 η，继续探索。HotpotQA/MathQA，seed42，旧 completed-only。 | 各数据集 *_run.json 含原生 trace；另有 trajectory.csv、lambda_stops.json、summary.json 和阶段/前沿图。是后续调度及方差门槛实验的历史参照。 |
| [anytime_radial_gittins_variance025_seed42/](anytime_radial_gittins_variance025_seed42/) | 放宽推荐资格：completed，或两个 posterior variance 都降到原 prior variance 的1/4以下；在相同采样轨迹上提前允许推荐。 | [comparison.md](anytime_radial_gittins_variance025_seed42/comparison.md) 记录精确采样一致性、最早 n=20、推荐变化和误推荐增加。**后来撤回该门槛，恢复 completed-only；保留作历史消融。** 文内旧 --recommendation-variance-ratio 命令需要当时实现，当前 CLI 已不提供该参数。 |

### 2. (1,0) 调度与异步 η

| 目录 | 做了什么 / 相比前版改变什么 | 阅读入口与主要结论 |
|---|---|---|
| [direction_schedulers_completed_only_seed42/](direction_schedulers_completed_only_seed42/) | HotpotQA/MathQA 比较所有方向 round-robin 与 accuracy-last：其他方向先运行到停止，再集中运行 (1,0)，复用已有观测。两组仍是全局 η decay、旧 completed-only。 | [summary.csv](direction_schedulers_completed_only_seed42/summary.csv)、comparison.json、matched_budgets.csv。最高 accuracy 首次推荐成本：HotpotQA 48.55%→52.62%，MathQA 61.23%→67.25%；accuracy-last 较晚，未取代 round-robin。 |
| [direction_schedulers_bird_dev_completed_only_seed42/](direction_schedulers_bird_dev_completed_only_seed42/) | 将同一调度消融扩到 BIRD Dev；两种 scheduler 的最高 accuracy 推荐节点均为67.52%。 | comparison.json、bird_dev_pareto_snapshots.pdf、configurations.csv。**diagnosis/accuracy_only/** 是另一个真正只跑 (1,0)、仍保留 η decay 的实验：目标2790在12.59%完成并推荐；看 [result.json](direction_schedulers_bird_dev_completed_only_seed42/diagnosis/accuracy_only/result.json)。diagnosis/ 中还保留采样、后验排名及支出诊断；accuracy-only 与 accuracy-last 不同。 |
| [direction_eta_decay_bird_dev_completed_only_seed42/](direction_eta_decay_bird_dev_completed_only_seed42/) | 保留十方向 round-robin，改为哪个方向停止就只降低自己的 η，无需等待其余方向。复用上一组 global-stop baseline，原生运行 direction-stop。 | [summary.csv](direction_eta_decay_bird_dev_completed_only_seed42/summary.csv)、validation_and_target_timeline.json、direction_stop_trace.json.gz、eta_schedule.pdf。最高 accuracy 完成/推荐由67.52%降到24.99%，最终旧推荐集合相同。**异步 η 机制被后续主线保留**；对应历史 core 提交 d545648。 |

### 3. LCB、hybrid-LCB 与跨数据集扩展

以下六组中的径向方法仍使用历史 reciprocal cost 和独立题序，推荐受方向 winners 限制；random 对照的规则另见对应行。

| 目录 | 做了什么 / 相比前版改变什么 | 阅读入口与主要结论 |
|---|---|---|
| [lcb_recommendations_bird_dev_seed42_beta1/](lcb_recommendations_bird_dev_seed42_beta1/) | 在异步 η 的采样上，对所有 arms 使用 posterior mean−std，completed 也扣 std，提前允许推荐未完成 arms。 | comparison.json、independent_trace_lcb_check.json、lcb_trace.json.gz、lcb_pareto_key_checkpoints_all_pages.pdf。验证与异步 completed 基线的采样/费用/η一致；目标2790首次13.30%、持续13.58%进入推荐，属于推荐时间变化。 |
| [hybrid_lcb_recommendations_bird_dev_seed42_beta1/](hybrid_lcb_recommendations_bird_dev_seed42_beta1/) | completed 不再扣 std，未完成仍扣；尝试解决已完成 arm 反而被不确定性惩罚压低的问题。 | diagnostic.json、hybrid_trace_diagnostic_validation.json、pareto_key_checkpoints_all_pages.pdf。48个快照、6页；pareto_recovery.png 只是局部图。completed 此时仍用 posterior mean；这版仍出现 dominated 推荐。 |
| [recommendation_benchmarks_seed42_beta1/](recommendation_benchmarks_seed42_beta1/) | 在 restaurant_test 和 StackOverflow 比较旧 completed-only、hybrid-LCB、random_configurations。径向两种规则采样相同，随机方法独立抽整行完整评估。 | summary.csv、matched_budgets.csv、各数据集 comparison.json 及两种径向 trace。这里是 restaurant_test，不是后来972-arm的 restaurant_valid；random 没有径向方法的全-arm warm-up。 |
| [hybrid_lcb_benchmarks_seed42_beta1/](hybrid_lcb_benchmarks_seed42_beta1/) | 将 hybrid-LCB 扩到 HotpotQA、MathQA、restaurant_test、StackOverflow，检查跨数据集表现。 | summary.csv、matched_budgets.csv、各数据集 comparison.json / trace。**MathQA 仍含旧 tie/dominance 问题：同 accuracy 时保留76而遗漏更便宜80。** 分析修复后效果应读下一个目录。 |
| [hybrid_lcb_mathqa_dominance_fix_seed42_beta1/](hybrid_lcb_mathqa_dominance_fix_seed42_beta1/) | 单独修正 MathQA 的推荐 tie/dominance 处理，使最终76换为80。 | mathqa/comparison.json 及验证文件。采样、η和实际费用均未改变，最高 accuracy 首次出现仍为82.38%。这是推荐正确性修复，不是探索提速。 |
| [hybrid_lcb_full_pareto_seed42_beta1/](hybrid_lcb_full_pareto_seed42_beta1/) | 汇总 HotpotQA、修复后的 MathQA、restaurant_valid、StackOverflow 的全部推荐变化，并统一导出多页 Pareto 图和支配关系。 | [该目录 README](hybrid_lcb_full_pareto_seed42_beta1/README.md)、各数据集 *_all_pages.pdf、checkpoints.csv、checkpoint_raw_truth_dominance_pairs.csv。**full 指展示完整 checkpoint 序列，不是改为推荐所有 arms 的 Pareto frontier。** 部分结果复用已有完整运行。 |

### 4. 被支配推荐、单方向和 guard 诊断

| 目录 | 做了什么 / 相比前版改变什么 | 阅读入口与范围 |
|---|---|---|
| [hybrid_lcb_dominance_audit_seed42_beta1/](hybrid_lcb_dominance_audit_seed42_beta1/) | 联合检查 StackOverflow 与 restaurant_valid：为什么在线 hybrid 分数下非支配的推荐，在完整数据 raw mean 坐标下会被支配。 | [dominance_examples.md](hybrid_lcb_dominance_audit_seed42_beta1/dominance_examples.md)、validation.json、两数据集 comparison/trace。是诊断及归档；restaurant_valid 只含早期5,000 cells，不能当作完整运行。 |
| [hybrid_lcb_stackoverflow_dominance_audit_seed42_beta1/](hybrid_lcb_stackoverflow_dominance_audit_seed42_beta1/) | 将 StackOverflow 的推荐内部支配事件单独列出，便于按 C 编号定位。 | internal_dominance_events.json、summary.csv、stackoverflow/comparison.json。55个快照的完整轨迹；与综合 dominance audit 中的 StackOverflow 结果重复，不是新算法。 |
| [hybrid_lcb_restaurant_valid_early_audit_seed42_beta1/](hybrid_lcb_restaurant_valid_early_audit_seed42_beta1/) | 专门检查 restaurant_valid 极低成本阶段，记录哪些推荐被其他推荐支配。 | internal_dominance_events.json、restaurant_valid/comparison.json / trace。5,000 cells、约2.877%成本、34个快照；与综合 audit 的对应前缀重复，不是完整 benchmark。 |
| [hybrid_lcb_accuracy_direction_audit_seed42_beta1/](hybrid_lcb_accuracy_direction_audit_seed42_beta1/) | StackOverflow 真正只用 (1,0) 的消融，检查11176等早期推荐是否由多方向引起。 | stackoverflow/audit_summary.json、comparison.json、hybrid_lcb_trace.json.gz。5,444 cells后自然到 η numerical floor；11176仍曾以 n=4 推荐，说明单方向并未消除幸运 warm-up。它不是另一个仓库 BanditGittinsEval 的复现。 |
| [hybrid_lcb_completed_raw_guard_seed42/](hybrid_lcb_completed_raw_guard_seed42/) | 尝试先排除被其他 completed arm 在实测 accuracy / mean USD 下支配的 completed 候选，处理 restaurant_valid 的3904→215等问题。 | 只有 [restaurant_valid_prefix_9900/validation.json](hybrid_lcb_completed_raw_guard_seed42/restaurant_valid_prefix_9900/validation.json) 和 trace：9,900 cells 的前缀验证，C54保留3904、4428。**不是完整数据集对照；后续 raw-mean / finite 主线未开启这个额外 guard。** |

### 5. Raw mean cost、completed frontier、finite target 与共享题序

下列六组保留十方向（含 (1,0)）、round-robin、每方向异步 η、seed42、batch4。采样和推荐的变化分别标明。

| 目录 | 做了什么 / 相比前版改变什么 | 阅读入口与主要结论 |
|---|---|---|
| [hybrid_lcb_raw_mean_seed42/](hybrid_lcb_raw_mean_seed42/) | restaurant_valid 改用 raw USD 统计模型，仍用 hybrid-LCB 和原十个推荐方向；校准 reference、prior/noise 同时改变。 | [experiment_notes.md](hybrid_lcb_raw_mean_seed42/restaurant_valid/experiment_notes.md)、comparison_common_metrics.json、final_direction_ablation.json。去除了3904/215的非线性排序反转，但最终方向推荐只有2个点。同轨迹 completed-frontier 审计发现10个真实前沿 arms 其实已在24.18%成本全部完成；覆盖下降主要包含推荐遗漏，不能据此说这些 arms 没被找到。 |
| [completed_raw_pareto_seed42/](completed_raw_pareto_seed42/) | 推荐直接取所有 completed arms 的实测 raw Pareto frontier，取消方向 winner 对推荐集合的限制。五数据集原生运行，独立题序。 | [four_benchmark_report.md](completed_raw_pareto_seed42/four_benchmark_report.md) 覆盖 HotpotQA/MathQA/StackOverflow/BIRD；restaurant_valid 另看其 experiment_notes.md 与 acquisition_parity.json。与 raw-mean hybrid 的 restaurant 采样相同，推荐更完整。 |
| [finite_lcb_raw_mean_seed42_beta1/](finite_lcb_raw_mean_seed42_beta1/) | 对固定完整题集做 finite-target 预测，所有 arms 都可进入保守 raw Pareto frontier；β=1、无 n_min，completed 自然为实测均值/零预测方差。五组原生运行，独立题序。 | [five_benchmark_report.md](finite_lcb_raw_mean_seed42_beta1/five_benchmark_report.md)、completed_vs_finite_*csv、validate_finite_runs.py。五组自身均有 finite_lcb_trace.json.gz；采样/η与 completed-only 对照一致。图中附带的 completed 对照不构成额外采样实验。 |
| [finite_mean_min32_raw_mean_seed42/](finite_mean_min32_raw_mean_seed42/) | 在同一个 finite-target 定义上去掉 std 惩罚（β=0），改为 n≥32 才允许推荐，独立题序。五组原生运行，采样/η与上述两种独立题序规则一致。 | [five_benchmark_report.md](finite_mean_min32_raw_mean_seed42/five_benchmark_report.md)、three_way_*csv、validate_mean_runs.py。stackoverflow/diagnostics/11169_n52/ 和 diagnostics/shared_question_order/ 是固定样本量诊断，不是完整算法运行。 |
| [finite_mean_min32_raw_mean_seed42_shared_questions/](finite_mean_min32_raw_mean_seed42_shared_questions/) | 保持 finite mean≥32，改成共同全局题序、各 arm 独立进度。warm/prior/noise不变，但后续观察结果、采样选择和η时机会改变。五组原生运行。 | [five_benchmark_question_order_report.md](finite_mean_min32_raw_mean_seed42_shared_questions/five_benchmark_question_order_report.md)、question_order_*csv、各组 validation.json。此目录没有 Pareto PDF；StackOverflow 早期局部报告里“其他四组尚未运行”已过期，以上级五组报告为准。 |
| [finite_lcb_raw_mean_seed42_beta1_shared_questions/](finite_lcb_raw_mean_seed42_beta1_shared_questions/) | 在上一目录的共享采样 trace 上重算 finite-LCB：β=1、min_samples=0。**五组全是 recommendation-only replay，没有新跑 acquisition、DP 或 η 调度。** | [five_benchmark_shared_lcb_report.md](finite_lcb_raw_mean_seed42_beta1_shared_questions/five_benchmark_shared_lcb_report.md)、three_way_*csv、stack_tracked_arm_intervals.csv、各组 Pareto PDF。自身不复制 LCB trace；recommendation_replay 记录来源路径/SHA。controls/independent/ 用旧独立轨迹验证重放器，与旧 native 结果匹配，不能计为新采样实验。 |

同轨迹的主要比较关系：

- 独立题序：completed raw Pareto ↔ finite-LCB ↔ finite mean≥32。
- 共享题序：finite mean≥32 ↔ finite-LCB 重放。
- 跨 shared / independent：是题序实验，不能声称采样一致。
- finite mean≥32 与 finite-LCB 同时改变了样本门槛和 std 惩罚，不能把差异只归因于其中一项。

最新五组报告保留了首次/持续节点和误推荐区间，避免只挑一个漂亮 checkpoint。例如 MathQA 的持续精确前沿：独立 finite-LCB 为68.36%，共享 finite-LCB 为79.03%；共享题序并非在每个数据集都改善。

### 6. 工程回归与缓存

| 目录 | 做了什么 / 相比前版改变什么 | 阅读入口与范围 |
|---|---|---|
| [deferred_checkpoints_bird_dev_seed42/](deferred_checkpoints_bird_dev_seed42/) | 基于 hybrid-LCB 验证“每批只检查推荐集合变化→变化时保存轻量事件→按需计算诊断”。不改变采样或推荐语义。 | validation.json、comparison.json、replay_source.py、trace/CSV。28,724次 membership 检查只保留48个事件；与原 completed/hybrid 关键点、费用和η一致。replay_source.py 是当时完整 engine 的归档，不是专门的一键 driver；此目录没有图。 |
| [split_completed_checkpoints_bird_dev_seed42/](split_completed_checkpoints_bird_dev_seed42/) | 将上述工程优化单独拆回旧 completed-only 的异步 η 分支，排除 LCB/hybrid 推荐改动。 | validation.json、comparison.json、direction_stop_trace.json.gz。115,044个观测和关键点与原异步 completed 基线一致；Warm＋5次变化＋Final共7快照。历史 core 为7d2e82c，LCB额外层随后为8f9fad0。“split”指代码/分支拆分。 |
| [speedup_verify/](speedup_verify/) | GPQA 上验证 Gittins boundary 缓存的冷启动、热启动及全预算运行。 | gpqa_cold/warm/full.json 和对应日志。cold/warm 均20%题数预算、相同356个评估与费用；cold构建35个表，warm从磁盘读35个表。full是另一预算，不与前两组直接比较效果。 |
| [cache_radial_gittins_boundaries/](cache_radial_gittins_boundaries/) | 多个 radial 实验共用的 DP boundary 磁盘缓存，按 schema/key 保存表。 | schema-1/。**缓存资产，不是独立实验或评估轨迹。** 命中缓存影响计算耗时，不代表少花评估 USD。 |
| [cache_radial_gittins_boundaries_speedup_verify/](cache_radial_gittins_boundaries_speedup_verify/) | 为 speedup_verify 单独隔离的缓存，用于明确区分冷/热运行。 | schema-1/。同样不计为算法实验；与通用缓存分开是性能验证设计的一部分。 |

## 哪些目录不能算独立实验

以下关系已由保存的 comparison 或解压后的 trace 核对：

- MathQA：hybrid_lcb_mathqa_dominance_fix 与 hybrid_lcb_full_pareto 的 comparison 相同；修复前后采样轨迹也相同，只改推荐。
- StackOverflow：hybrid_lcb_full_pareto、hybrid_lcb_dominance_audit、hybrid_lcb_stackoverflow_dominance_audit 的 comparison 相同。
- restaurant_valid：综合 dominance audit 与 restaurant_valid_early_audit 保存相同早期 comparison/trace。
- HotpotQA 的 hybrid_benchmarks 与 full_pareto 采样相同；后者用于完整展示和审计。
- shared finite-LCB 使用 shared finite mean≥32 的采样，不应将两个目录当作两条独立获得的采样证据。

## 复现入口

使用当前 [two-axis 与四组 ablations 命令](../../README.md)。
早期推荐规则、accuracy-last 和全局 η 对照的独立命令已退役；
复核历史实验应使用对应的 Git 历史版本与其原始数据。

## 新实验如何归档

新目录至少保存：数据集及共同题集、seed、成本模型/metric space、题序、directions/η、推荐规则及 β/n_min、完整终止条件、原生还是推荐重放、输入与源码 hash。重放应引用原始 trace，复用/诊断应说明来源。

每次新增或改变一个一级目录时，在本 README 补一行“相对哪版改了什么”，并给出报告/验证入口。全量 checkpoint 可以保留用于诊断，论文或对照摘要使用固定 USD 节点及首次/持续恢复指标。
