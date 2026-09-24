# Two-Directional Gauss-Radau Gittins

## 摘要

Two-Directional Gauss-Radau Gittins 是一个用于二维 accuracy-cost Pareto discovery 的低阶、低计算量方法。它使用两个标量化方向

\[
\lambda^{(1)}=\left(\frac13,\frac23\right),
\qquad
\lambda^{(2)}=(1,0),
\]

在同一组 configuration posteriors 上运行两个 cost-aware required-completion Gittins controllers：第一个 controller 覆盖主要的内部 accuracy-cost trade-off，第二个 controller 专门覆盖最高 accuracy 边界。两个 controller 共享所有观测，但拥有各自的 Gittins stopping decision 和搜索成本乘数 \(\eta_q\)。调度器以 1:1 的访问机会交替调用它们；某个方向停止时，只降低该方向的 \(\eta_q\)，然后立即把控制权交给另一方向。

这个方法应被理解为：

> 对理想 hypervolume / radial-envelope discovery problem 的 Gauss-Radau-inspired、index-compatible、separable approximation。

它不是 exact hypervolume-improvement Gittins，也没有整个多目标策略的全局 Bayes-optimality 保证。它的主要价值是仅用两个可预计算的 Gittins 子问题，取得较好的 Pareto coverage、较低的误报和较早的 high-accuracy-region discovery。

---

## 1. 问题设定

每个 configuration（arm）\(i\) 有两个未知 benchmark-level objectives：

\[
\theta_i=(\theta_{i,1},\theta_{i,2}),
\]

其中：

- \(\theta_{i,1}\) 是 accuracy 或 task performance，越大越好；
- \(\theta_{i,2}\) 是 deployment-cost desirability，越大表示部署越便宜。

需要严格区分两种 cost：

1. **Deployment cost** 是 Pareto objective 的一个坐标；
2. **Search/evaluation cost** \(\kappa_i\) 是继续评估 arm \(i\) 一个 batch 所支付的 API cost，并作为 Gittins continuation penalty。

当前 raw-mean 实现直接对 mean accuracy 和 mean USD cost 建模。进入方向标量化前，cost posterior 被映射成冻结的仿射 desirability coordinate

\[
y_{i,2}=1-\frac{c_i}{R},
\]

其中 \(R\) 由 warm-up 数据冻结；它不会使用完整 response matrix 的隐藏真值。accuracy coordinate 为 \(y_{i,1}=\theta_{i,1}\)。因此，`raw_mean` 指 posterior 所在的统计空间，而不表示 radial geometry 完全不做尺度变换。

每个 arm 只有一个共享的二维 posterior。directions 只是该 posterior 的不同标量视图，不会复制实验、posterior 或已经购买的 question-level observation。

---

## 2. 从 hypervolume 到 radial directions

设两个 objectives 都已写成“越大越好”，reference point 为 \(r=(r_1,r_2)\)，已发现的集合为 \(A\)。使用 simplex 参数

\[
\lambda(t)=(t,1-t),\qquad t\in[0,1],
\]

定义沿该方向从 reference point 出发所能到达的 Pareto-envelope radius：

\[
L_A(t)
=
\max_{y\in A}
\min\left\{
\frac{(y_1-r_1)_+}{t},
\frac{(y_2-r_2)_+}{1-t}
\right\},
\]

端点使用相应的连续极限。二维 dominated hypervolume 可以精确写成

\[
HV_r(A)=\frac12\int_0^1 L_A(t)^2\,dt.
\]

这个等式说明，二维 hypervolume 可以看成对连续 radial/Chebyshev envelopes 的积分。它也说明两件重要的事：

- 一个 direction 对应 Pareto frontier 上的一类 preference/trade-off，而不是只对应一个最终推荐 arm；
- exact HV integrand 是 squared radius \(L_A(t)^2\)，不是未平方的 radial utility。

---

## 3. 为什么得到两个 Gauss-Radau nodes

在区间 \([0,1]\) 上固定右端点 \(t=1\) 的两点 Gauss-Radau rule 为

\[
\int_0^1 f(t)\,dt
\approx
\frac34 f\!\left(\frac13\right)
+
\frac14 f(1).
\]

该规则对次数不超过 2 的多项式精确。对应的两个 simplex directions 正好是

\[
\lambda^{(1)}=\left(\frac13,\frac23\right),
\qquad
\lambda^{(2)}=(1,0).
\]

这两个节点具有直接解释：

- \((1/3,2/3)\) 是内部节点，代表主要 accuracy-cost trade-off；
- \((1,0)\) 是 accuracy endpoint，保证最高 accuracy 边界不会仅仅因为它在连续积分中是 measure-zero endpoint 而被低阶离散化完全遗漏。

Gauss-Radau 在这里提供的是一种**最低阶、同时包含一个 interior node 和指定 endpoint 的确定性方向设计**。实际 Pareto envelope 含有 `max` 和 `min`，通常不是低次光滑多项式，因此“两点规则次数 2 精确”不等于它对真实 HV integrand 有严格的小误差保证。它的作用是给出一个可解释的低阶节点选择，而不是证明两个节点已经足够逼近任意 frontier。

### 当前实现与严格 Gauss-Radau quadrature 的区别

当前表现最好的算法使用上述 **node locations**，但没有严格实现完整的两点 quadrature objective：

1. 当前 directional Gittins 使用线性 radial utility \(\rho_\lambda\)，而 exact HV integrand 使用平方后的 radius；
2. 当前 scheduler 给两个 controllers 1:1 的访问机会，而不是把 quadrature weights \(3/4\) 和 \(1/4\) 当作访问频率。

因此，更准确的名称是 **Gauss-Radau-inspired node pair** 或 **Separable Gauss-Radau Gittins**，而不是 exact Gauss-Radau integration of hypervolume。

这一差别不能用“平方是单调函数”完全消除。对于已经完成、没有不确定性的 arms，\(\rho_\lambda\) 和 \(\rho_\lambda^2\) 有相同的 direction winner；但

\[
\mathbb E[\rho_\lambda]-\eta C
\quad\text{和}\quad
\mathbb E[\rho_\lambda^2]-\eta C
\]

一般会产生不同的 stochastic continuation decisions。

---

## 4. 每个 directional Gittins controller 做什么

对于 interior direction \(\lambda=(\lambda_1,\lambda_2)\)，当前实现令

\[
a_\lambda=\max(\lambda_1,\lambda_2),
\]

并使用

\[
\rho_\lambda(y)
=
\min_j
\left\{
\frac{a_\lambda(y_j-r_j)}{\lambda_j}
\right\}.
\]

对于 \((1/3,2/3)\)，这等价于

\[
\rho_{(1/3,2/3)}(y)
=
\min\{2(y_1-r_1),\ y_2-r_2\}.
\]

乘上只依赖 direction 的正常数不会改变 completed-arm winner，但这里的 scaling 会影响 reward 与 search cost 的相对单位，因此它属于算法定义的一部分。

每个 unfinished arm 都使用 finite-horizon required-completion retirement DP。给定 outside terminal reward \(\alpha\)，一次 continuation 支付有效成本

\[
c_{i,q}=\eta_q\kappa_i,
\]

并根据 Gaussian posterior-mean transition 更新。interior direction 使用 radial-Gittins boundary；accuracy endpoint \((1,0)\) 则直接使用 scalar Gaussian Gittins DP，避免除以零并精确表达 accuracy-only terminal utility。

对于固定 direction，策略遵循与 single-objective required-completion Gittins 相同的局部规则：

1. completed arms 的 index 等于其 terminal utility；
2. unfinished arms 的 index包含 posterior mean、uncertainty 和完成它所需的 future evaluation cost；
3. 如果最大 index 由 unfinished arm 获得，则评估该 arm 的下一个 batch；
4. 如果最大 index 已由 completed arm 获得，则该 direction 当前停止。

这里的理论支持只适用于**固定 direction 的 scalar required-completion subproblem**。它不自动推出多个 directions 的调度器是全局最优的。

---

## 5. 1:1 asynchronous coordination

两个 directions 共享 arm posteriors，但分别维护自己的

\[
(\eta_q,\text{decay stage}_q).
\]

调度器按照

```text
interior, accuracy endpoint, interior, accuracy endpoint, ...
```

交替给予决策机会：

1. 当前 direction 重新读取最新的共享 posteriors；
2. 如果它选择 unfinished arm，则购买一个新 batch，并更新该 arm 的唯一共享 posterior；
3. 如果它决定停止，则只把自己的 \(\eta_q\) 减半，然后把控制权交给另一个 direction；
4. 另一个 direction 产生的观测也会改变当前 direction 下一次访问时看到的状态；
5. 运行在预算耗尽、所有 arms 完成，或所有 directions 到达数值 \(\eta\) floor 且没有新观测时结束。

“1:1”只表示**访问机会交替**，不表示：

- 两个 directions 实际选择了相同数量的 batches；
- 两者消耗了相同 USD；
- 每个 observation 只服务触发它的 direction。

某个 direction 的 stop visit 不购买 batch，而任何真实 observation 都会同时改善两个 directional views。因此，把 Gauss-Radau 的 \(3{:}1\) integration weights 直接解释成 \(3{:}1\) sampling frequency 并没有理论依据；已有 weighted-round-robin 实验也不支持这种简单替换。

---

## 6. 为什么它是 separable approximation

一个更理想但更昂贵的 archive-conditioned目标是

\[
\max_\pi
\mathbb E_\pi
\left[
HV(A_T)-\eta C_T
\right].
\]

固定当前 archive \(A\) 时，也可以把完成 arm \(i\) 的 terminal reward 定义成

\[
\Delta HV_i(A)
=
HV(A\cup\{y_i\})-HV(A),
\]

并建立一个 combined HVI-Gittins DP。它在 formulation 上自然，但 terminal reward 同时依赖两个 objectives、outside reward 和不断变化的 archive。每次 archive 更新都会改变所有 arms 的 terminal reward surface，破坏当前实现依赖的平移不变性和可复用 boundary tables。

Two-Directional Gauss-Radau Gittins 对这个耦合问题做了分解：

\[
V^{\mathrm{combined}}
\quad\leadsto\quad
\{V_{(1/3,2/3)},V_{(1,0)}\}.
\]

每个子问题都保留标准 required-completion Gittins 结构，并能独立停止；共享 posterior 再让一次物理评估同时服务两个子问题。从 Bellman operator 的角度，这类似于把共同 stop/continue decision 松弛为 component-wise decisions。局部上有

\[
\max\left\{0,\sum_q w_q z_q\right\}
\le
\sum_q w_q\max\{0,z_q\},
\]

说明独立 positive-part decisions 是一种 optimistic separable relaxation。但由于多个 arms、共享观测、线性 radial surrogate 和 asynchronous \(\eta\) 都进一步改变了问题，当前完整算法不应被宣称为 combined-HV Bellman equation 的严格 upper bound 或精确解。

---

## 7. 为什么它特别经济

### 7.1 二维 direction space 只有一维

两个 objectives 的 simplex direction 可以用单个 \(t\in[0,1]\) 参数表示。一个 interior node 加一个 endpoint 就能覆盖两类最重要的行为。相比原来的 9 个 interior directions 加 accuracy axis，directional controllers 从 10 个降为 2 个。

### 7.2 保留 Gittins boundary precomputation

线性 radial utility 具有平移结构。当前 solver 虽然离线在二维 state grid 上进行 Gaussian convolution，但最终只需要保存每个 stage 的一维 boundary

\[
b_n(\delta).
\]

在线 index evaluation 只进行 posterior update、table lookup 和 arm comparison。boundary tables 还可以按 direction、cost bin、variance schedule 和 horizon 缓存到磁盘并跨运行复用。

### 7.3 Endpoint controller 更便宜

\((1,0)\) 是 scalar accuracy problem，直接复用一维 Gaussian Gittins boundary，不需要建立退化的二维 radial table。

### 7.4 Directions 共享观测

如果两个 directions 都对同一个 arm 感兴趣，第二次选择会评估该 arm 的下一批未观测 questions，而不是重新支付已经评估过的 cells。posterior 数量始终等于 arm 数量，而不是

```text
number of arms x number of directions.
```

### 7.5 推荐数量不受 direction 数量限制

directions 只控制 acquisition。当前 anytime recommendation 使用独立的 `finite_lcb` rule：对所有 arms 的 fixed-test final mean 构造保守 accuracy/cost coordinates，再取它们的 Pareto frontier。因此两个 directions 仍然可以推荐多个 configurations，而不是每个 direction 只能贡献一个 winner。

### 7.6 目标是覆盖而非枚举

如果任务要求找齐每个真实 Pareto arm，两个 directions 很容易遗漏对 HV 或 radial envelope 贡献很小的局部点。当前目标只要求以较低成本覆盖主要 trade-offs、尽量避免 false positives，并在稍高预算下到达 high-accuracy region。对这个目标，低阶节点设计比 dense direction grid 更匹配。

---

## 8. Acquisition 与 recommendation 必须分开描述

当前方法包含两个逻辑层：

### Acquisition layer

- directions：\((1/3,2/3)\) 与 \((1,0)\)；
- 两个 required-completion Gittins controllers；
- shared posterior；
- 1:1 round-robin visit opportunities；
- per-direction asynchronous \(\eta\) decay。

### Anytime recommendation layer

- `finite_lcb`，\(\beta=1\)；
- 预测每个 arm 在固定完整公共题集上的最终 mean；
- accuracy 使用 mean minus one standard deviation；
- cost 使用 mean plus one standard deviation；
- 在所有 eligible arms 的保守坐标上做 Pareto filtering。

`finite_lcb` 是为了减少 early false positives 并允许 unfinished arms 提前进入 recommendation。它不是 Gittins theorem 推导出的 recommendation rule，也不提供 joint 或 anytime confidence guarantee。方向 winners 只是 acquisition diagnostics，不直接决定最终推荐集合。

---

## 9. 当前实验协议

保存下来的五数据集 Gauss-Radau ablation 使用：

- seed：42；
- batch size：4；
- warm-up：每个 arm 4 道题；
- question order：arm-specific independent seeded orders；
- common-question response matrices；
- cost posterior：`raw_mean`；
- recommendation：`finite_lcb`，\(\beta=1\)，无额外最小样本门槛；
- scheduler：`round_robin`；
- \(\eta\) schedule：`direction_stop`；
- 初始 \(\eta=1\)，每次本方向 stop 后乘以 \(0.5\)。

结果目录：

```text
experiments/combined_objective/results/
  two_direction_finite_lcb_raw_mean_seed42_independent/
    gauss_radau_accuracy_endpoint/
```

与 \([(0.1,0.9),(1,0)]\) 相比，Gauss-Radau pair 的总体折中更好，但不在每个 benchmark、每个指标上严格占优：

- HotpotQA：零误报下达到至少 80% recall，从 9.18% BF 提前到 5.96% BF；
- MathQA：多数关键节点小幅提前；
- Restaurant：稳定精确前沿从 74.17% BF 大幅提前到 25.98% BF；
- StackOverflow：首次和稳定 exact recall 反而更晚；
- Bird Dev：最终漏掉一个真实 Pareto arm，没有达到稳定 exact frontier。

既然目标是覆盖主要 trade-offs 而不是枚举全部 Pareto arms，后两项应结合 HV coverage、IGD 和 endpoint-near coverage 判断，不能只用 exact set recall 决定算法优劣。当前这些结果主要是 seed42 证据，不应当被描述为跨 seed 稳定结论。

---

## 10. 适合报告的主要指标

当前研究目标下，建议优先报告：

1. 达到 90% / 95% true-frontier hypervolume coverage 所需的 BF cost percentage；
2. 在零 false positives 条件下达到上述 coverage 的首次和持续成本；
3. IGD 或其他几何 coverage metric；
4. 首次和持续发现 high-accuracy endpoint 或 \(\varepsilon\)-near-endpoint 的成本；
5. 整条 trajectory 上出现 false positives 的评估成本占比；
6. exact frontier recall 作为补充指标，而不是唯一成功标准。

BF percentage 指累计实际 evaluation USD 除以完整公共 response matrix 的 evaluation USD，不是 wall-clock time，也不是用户在线可直接观察的 stopping signal。

---

## 11. 二维特例与更高维推广

两个 directions 的经济性与二维问题密切相关，但不是“一个 objective 对应一个 direction”。对于 \(k\) 个 objectives，direction space 的维度是 \(k-1\)：

\[
\lambda\in S_+^{k-1}
\quad\text{或}\quad
\lambda_j\ge0,\ \sum_j\lambda_j=1.
\]

三个 objectives 时，direction domain 是二维三角形，Pareto frontier 通常也是二维曲面。此时并不自然地得到“恰好三个 directions”；节点数量应由 simplex cubature accuracy、需要保护的 endpoints 和经验 coverage 决定。一个低阶起点可以是三个 symmetric interior nodes，再加入一个重要 endpoint，但它已经不再具有一维两点 Gauss-Radau 的唯一性和低成本。

同时，三目标 radial DP 的 value grid 会从二维上升到三维，平移后的 boundary 也从曲线变成曲面。因此更高维的主要困难不仅是 directions 增多，也包括每个 directional Gittins table 本身更昂贵。

---

## 12. 可以和不可以声称什么

### 可以声称

- 每个固定 direction 都是一个 cost-aware required-completion Gittins subproblem；
- 两个 node locations 来自最低阶、固定 accuracy endpoint 的两点 Gauss-Radau rule；
- shared posterior 避免了 directions 之间的重复数据采集；
- separable design 保留了 boundary precomputation 和低在线计算成本；
- 当前实验中，该 pair 在低误报、主要 Pareto coverage 和 high-accuracy discovery 之间给出了较好的经验折中。

### 不可以声称

- 当前 1:1 async policy 精确最大化 expected hypervolume；
- quadrature weights \(3{:}1\) 推导出了 \(3{:}1\) sampling frequency；
- 线性 radial utility 和 squared HV scalarization 产生相同的 stochastic Gittins policy；
- 两个 directions 对任意二维 Pareto frontier 都足够；
- `finite_lcb` 保证 trajectory 上绝不出现 false positive；
- 多方向调度器继承了单方向 Markov-chain selection problem 的全局 Bayes optimality。

---

## 13. 建议的论文定位

建议使用下面的逻辑顺序：

1. 以 cost-aware hypervolume/radial-envelope discovery 作为理想集合目标；
2. 使用二维 radial integral 建立 hypervolume 与方向标量化的联系；
3. 使用包含 accuracy endpoint 的两点 Gauss-Radau rule 选择最低阶方向集合；
4. 解释 exact combined HVI-Gittins 会产生 archive-dependent nonlinear terminal reward，并破坏可复用的一维 boundary structure；
5. 提出 Separable Gauss-Radau Gittins：用两个可预计算的 required-completion Gittins controllers 近似 coupled problem；
6. 用 balanced asynchronous scheduler 协调 controllers，并通过 shared posterior 回收跨方向信息；
7. 将 `finite_lcb` 明确定位成独立的 anytime recommendation heuristic；
8. 用 direction-pair、1:1 vs weighted scheduling、推荐规则和多 seed 实验分别验证每个设计选择。

一个简洁但不过度承诺的描述是：

> We derive an ideal archive-conditioned hypervolume-improvement objective, whose exact Gittins solution loses the translation-invariant precomputation available in scalar required-completion problems. We therefore introduce Separable Gauss-Radau Gittins, a two-node, endpoint-aware approximation that preserves precomputable scalar Gittins subproblems, shares observations across directions, and coordinates them through a balanced asynchronous scheduler.

---

## 14. 代码、结果与参考资料

- 实验 runner：[run_two_direction_ablation.py](run_two_direction_ablation.py)
- offline acquisition 与 recommendation engine：[offline_radial_gittins.py](offline_radial_gittins.py)
- radial-Gittins DP：[radial_gittins_dp.py](../../src/agentopt/model_selection/radial_gittins_dp.py)
- five-benchmark ablation summary：[summary.json](results/two_direction_finite_lcb_raw_mean_seed42_independent/summary.json)
- single-objective required-completion 理论基础：[BanditGittinsEval.pdf](../../BanditGittinsEval.pdf)
- Daniel Golovin and Qiuyi Zhang, [Random Hypervolume Scalarizations for Provable Multi-Objective Black Box Optimization](https://proceedings.mlr.press/v119/zhang20i.html), ICML 2020.
