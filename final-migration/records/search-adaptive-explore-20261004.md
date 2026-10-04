# 高 po 下的搜索宽度自适应 —— 分析与设计

- 日期：2026-10-04
- 分支：`search-adaptive-explore`（基线 `0cca6be`，即 `prune-width-per-gpu-sched` 头）
- 状态：**分析与设计，尚未实现任何代码**
- 起因：8 卡实测 ~29000 nnEvals/s、`avgBatchSize` 有提升，但实战中
  「超长考时算力高度集中、死磕一两个点，算了几百 M，被对手一子炸树」。

---

## 0. 问题陈述

算力从 3.1k nnEvals/s（T36 单卡生产值）提到 29k nnEvals/s（8 卡）约 **9.3×**。
但搜索**宽度**没有同步增长：根节点访问分布仍高度集中在一两个点上。
对手下出一手不在树内的棋时，可复用的子树几乎归零（"炸树"），
此前几亿次访问全部作废。

这正好对应参考材料里那句话：

> 低 po（每秒小一两千）、中等 po（每秒 1 万）和高 po（每秒 10 万+）时候，
> mcts 引擎的工作参数、搜索树的形状、神经网络输出的调整，截然不同。

也就是说：**参数必须随算力量级变化，而当前引擎用的是单一常数。**

---

## 1. 现状核查（可复现）

### 1.1 生产 cfg 就是官方模板，搜索参数一个都没调

```bash
diff <(grep -vE '^\s*#|^\s*$' cpp/configs/gtp_example.cfg) \
     <(grep -vE '^\s*#|^\s*$' ../../config/gtp_sm89_plan_pruned.cfg)
```

输出只有两处差异：

```
14c14
< numSearchThreads = 6
---
> numSearchThreads = 36
18a19,25
> cudaTacticPlanFile = ...
> cudaTacticPlanBatch = 12
> numNNServerThreadsPerModel = 2
> cudaDeviceToUseThread0 = 0
> cudaDeviceToUseThread1 = 0
> cudaAsyncInferPipeline = true
> cudaEventPipelineUseGraph = false
```

⇒ **`gtp_sm89_plan_pruned.cfg` = 官方 `gtp_example.cfg` 原样 + `numSearchThreads` 6→36 + 追加 CUDA 段。**
所有搜索行为参数保持库默认值，没有为 29k nnEvals/s 做过任何调整。

### 1.2 本 fork 从未改过 `cpp/search/`

```bash
git log --oneline -- cpp/search/     # 只有 a3f54463（fork 的导入提交）
```

我们自己的三个提交（`e8737bd` / `fe73c39` / `ae85f08`）全在 CUDA/plan 侧。
**搜索树代码 = 原版 KataGo 1.17.2。**

### 1.3 因此实际生效的搜索参数（全部库默认）

| 参数 | 生效值 | 含义 |
|---|---|---|
| `cpuctExploration` | 1.0 | 探索常数 |
| `cpuctExplorationLog` | **0.0** | ⇒ cpuct 与访问量**无关**，恒为 1.0 |
| `cpuctExplorationBase` | 500 | log 项的基准（因上项为 0 而未启用） |
| `cpuctUtilityStdevScale` | **0.0** | ⇒ `parentUtilityStdevFactor ≡ 1.0`，**不确定性缩放被关闭** |
| `fpuReductionMax` / `fpuLossProp` | 0.2 / 0.0 | 首访紧急度 |
| `valueWeightExponent` | 0.5 | 劣质子节点降权 |
| `useNoisePruning` | false | 按先验剪除超额权重 |
| `useUncertainty` | **false** | **不确定性加权关闭** |
| `rootPolicyTemperature` | 1.0 | 根先验温度 |
| `rootDesiredPerChildVisitsCoeff` | **0.0** | **根节点"漏斗"关闭**（见 §4.2） |
| `rootNumSymmetriesToSample` | **1** | 根只采 1 个对称（无集成） |
| `wideRootNoise` | 0.0 | 根加宽噪声关闭 |
| `subtreeValueBiasFactor` | **0.0** | **子树价值偏差校正关闭**（见 §4.5） |
| `useLcbForSelection` | false | 不用 LCB 选点 |
| `futileVisitsThreshold` | 0.0 | 徒劳访问剪枝关闭 |
| `maxVisits` | **500** | 见 §8，疑似模板残留 |
| `ponderingEnabled` | **false** | 对手回合不思考 |

### 1.4 对照：KataGo 自己的高算力配置用的是另一套

KataGo 里真正要求「下出强而稳健的棋」的配置 —— 自对弈数据生成、门控、定式库生成 ——
用的参数与 `gtp_example.cfg` 的系统性不同：

| 参数 | gtp_example（= 我们） | selfplay8b20 | gatekeeper2b/c | genbook7tt | task_example |
|---|---|---|---|---|---|
| `cpuctExploration` | 1.0 | 1.1 | 1.1 | **1.25** | 1.1 |
| `cpuctExplorationLog` | 0.0 | 0.0 | 0.0 | **0.5** | 0.0 |
| `rootDesiredPerChildVisitsCoeff` | 0.0 | **2** | — | — | **2** |
| `rootNumSymmetriesToSample` | 1 | **4** | — | **8** | **4** |
| `rootPolicyTemperature` | 1.0 | **1.1** | — | — | **1.1** |
| `subtreeValueBiasFactor` | 0.0 | **0.30** | **0.35** | — | **0.45** |
| `useUncertainty` | false | — | **true** | **true** | — |
| `useLcbForSelection` | false | **true** | **true** | **true** | **true** |
| `wideRootNoise` | 0.0 | — | — | **0.05** | — |
| `rootSymmetryPruning` | false | — | — | **true** | — |
| `valueWeightExponent` | 0.5 | 0.5 | 0.5 | **0.0** | 0.25 |

方向高度一致：**更宽的探索、更保守的选点、显式的不确定性/偏差校正、根节点多对称集成。**
我们一个都没开。

---

## 2. 机理：为什么 N 一大就必然「死磕一两个点」

### 2.1 选择公式

`cpp/search/searchexplorehelpers.cpp:9-54`

```cpp
static double cpuctExploration(double totalChildWeight, const SearchParams& p) {
  return p.cpuctExploration +
    p.cpuctExplorationLog * log((totalChildWeight + p.cpuctExplorationBase) / p.cpuctExplorationBase);
}

double Search::getExploreScaling(double totalChildWeight, double parentUtilityStdevFactor) const {
  return cpuctExploration(totalChildWeight, searchParams)
       * sqrt(totalChildWeight + 0.01)
       * parentUtilityStdevFactor;
}

// 子节点 i 的选择值：
//   V_i = exploreScaling * P_i / (1 + n_i) + u_i
```

因为 `cpuctExplorationLog = 0.0`，`cpuctExploration(N) ≡ 1.0`；
又因为 `cpuctUtilityStdevScale = 0.0`，`parentUtilityStdevFactor ≡ 1.0`。
于是：

```
V_i = 1.0 · √N · P_i / (1 + n_i) + u_i
```

### 2.2 隐含的「分辨率」随 N 单调下降，且没有地板

子节点 i 还能继续拿到访问的条件是

```
√N · P_i / (1 + n_i)  >  Δu_i        （Δu_i = 它与当前最优的价值差）
```

即 `n_i < √N · P_i / Δu_i`。要让 i 在 N→∞ 时**仍然存活**（n_i 随 N 一起涨），
必须 `√N · P_i / Δu_i ≳ N`，即

```
Δu_i  ≲  P_i / √N
```

⇒ **搜索能分辨的最小价值差 ~ 1/√N，随算力单调变细，永不设下限。**
N=10⁴ 时约 10⁻²，N=10⁶ 时约 10⁻³，N=3×10⁸ 时约 6×10⁻⁵。

### 2.3 而神经网络的价值误差并不随 N 缩小

utility 半径 = `winLossUtilityFactor 1.0 + staticScoreUtilityFactor 0.3` = 1.3。
NN 的胜率预测误差量级 0.03–0.1，折到 utility 约 **0.03–0.13**。
这是一个**系统性偏差**（bias，不是 variance），**平均多少次都不会消失** ——
只有 `subtreeValueBiasFactor` 那种「用搜索自己的证据反推偏差」的机制才能修正它。

### 2.4 结论

- 访问量越过约 `(1/σ)²` 量级之后，搜索是在**自身噪声地板以下做无意义的细分**：
  继续加深同一条 PV，得不到任何新信息。
- 由于平均效用收敛、噪声被平均掉，树**只会越来越窄**，最终收敛到
  「NN 系统性高估的那条线」上 —— 这正是「死磕一两个点」。
- 换句话说：**低 po 时搜索宽度由价值噪声撑开（看起来正常）；
  高 po 时噪声被平均掉，宽度由真实价值差决定，而真实价值差又分不出来 ⇒ 树塌成一条线。**
  这就是参考材料说的「搜索树的形状截然不同」。

---

## 3. 为什么「炸树」

KataGo 的树复用是「对手下完一手后，保留以该手为根的子树」。
既然根节点访问高度集中在 1–2 个点：

- 对手**下在树内那 1–2 个点之一** → 复用率高，一切正常；
- 对手**下在别处** → 复用率≈0，几百 M 访问清零，且剩余时间不变。

所以「炸树」不是独立问题，它就是 §2 的直接后果：
**树里根本没有关于备选点的信息。**

---

## 4. 关键代码位置（改动锚点）

| # | 位置 | 内容 |
|---|---|---|
| 4.1 | `cpp/search/searchexplorehelpers.cpp:9-29` | `cpuctExploration()` / `getExploreScaling()` —— **唯一收口点**，所有探索缩放都过这里 |
| 4.2 | `cpp/search/searchexplorehelpers.cpp:166-170` | `rootDesiredPerChildVisitsCoeff` 漏斗：`childWeight < sqrt(P*N*coeff)` ⇒ 返回 `1e20` 强制访问 |
| 4.3 | `cpp/search/searchexplorehelpers.cpp:222-224` | `wideRootNoise` 根加宽 |
| 4.4 | `cpp/search/searchexplorehelpers.cpp:265-321` | `getFpuValueForChildrenAssumeVisited()` —— FPU / `cpuctUtilityStdevFactor` 计算处 |
| 4.5 | `cpp/search/searchupdatehelpers.cpp:26-36, 273-308` | `subtreeValueBiasFactor` 施加点：把节点自身效用往「子节点均值 − 自身」方向拉，权重 `origTotalChildWeight^subtreeValueBiasWeightExponent` |
| 4.6 | `cpp/search/searchresults.cpp:198-236` | LCB 选点（`useLcbForSelection` / `lcbStdevs` / `minVisitPropForLCB`） |
| 4.7 | `cpp/search/search.cpp:497-598` | 上限计算、`upperBoundVisitsLeft`（futile 剪枝依赖它） |
| 4.8 | `cpp/search/search.cpp:650-720` | `beginSearch()` 树复用与偏差表初始化 |
| 4.9 | `cpp/command/gtp.cpp:2182` | 只有 cfg 里**没有** `maxVisits/maxPlayouts/maxTime` 时才套默认时间控制 |

⚠️ 注意 §4.2 的漏斗阈值是 `sqrt(P·N·coeff)`，而 `childWeight ≈ n`，
所以它要求的**访问份额**是 `sqrt(P·coeff/N)` —— **同样随 N 衰减**。
`cpuctExplorationLog` 也只在 cpuct 上加了 `log` 项，`c(N)/√N ~ log(N)/√N` 仍然 →0。
⇒ **现有旋钮全都无法阻止「相对宽度随 N 衰减」，必须新增一个显式的地板项。**

---

## 5. 方案

### Tier 0 —— 只改配置，零代码风险（先做这个）

把 KataGo 自对弈/门控/定式库那一档参数搬进生产 cfg：

```cfg
# ---- 高 po 搜索参数（对照 selfplay8b20 / gatekeeper2b / genbook7tt / task_example）----
cpuctExploration = 1.1
cpuctExplorationLog = 0.0          # 先保持 0；Tier 1 再引入自适应项
rootPolicyTemperature = 1.1
rootNumSymmetriesToSample = 4      # 根多对称集成，近乎免费地降噪
rootDesiredPerChildVisitsCoeff = 2 # 根子节点访问漏斗（直接对抗"死磕一两个点"）
subtreeValueBiasFactor = 0.35      # 修正 NN 系统性价值偏差 —— 本方案最关键的一项
subtreeValueBiasWeightExponent = 0.5
useUncertainty = true
uncertaintyCoeff = 0.2
useLcbForSelection = true
lcbStdevs = 4.0
minVisitPropForLCB = 0.05
useNoisePruning = true
ponderingEnabled = true            # 对手回合也搜 —— 直接缓解"炸树"
```

预期：根访问分布明显变宽、深线价值偏差被校正、树复用率上升。
**但不要只看 nnEvals/s 或单点访问数 —— 必须用对局胜率/门控来判定**（见 §6）。

### Tier 1 —— 代码：显式探索分辨率地板（本分支的核心）

在 `getExploreScaling()` 里加一项，让 cpuct 的下限随 N 增长，
使「最小可分辨价值差」不再无限变细：

```
新增参数：cpuctExplorationFloorCoeff（默认 0.0 = 关闭，行为与现在逐位一致）

cpuct(N) = cpuctExploration + cpuctExplorationLog·log((N+base)/base)
effCpuct(N) = max( cpuct(N), cpuctExplorationFloorCoeff · sqrt(N) )
```

因为分辨率 ≈ `effCpuct/√N`，取 `max(·, k√N)` 恰好让分辨率**止跌于 k**。
k 的物理含义就是「我拒绝在低于这个价值差上做区分」，应设为 NN 价值噪声量级
（0.02–0.05，需实测标定）。默认 0 保证向后兼容。

- 实现点：仅 `searchexplorehelpers.cpp:9-29` + `searchparams.{h,cpp}` + `setup.cpp` 解析 + 文档
- 风险：k 过大会让树变成均匀灌木、单点精度下降 ⇒ 必须扫 k ∈ {0, 0.01, 0.02, 0.05} 并对局验证

### Tier 2 —— 代码：根子节点访问份额地板（更直接、更可控）

Tier 1 改的是全局探索；Tier 2 直接对根做「份额保底」：

```
新增参数：rootMinVisitShare（默认 0.0 = 关闭）

在根节点选择时：若某子节点 n_i / N < rootMinVisitShare
                且该点先验 P_i 超过某阈值（避免给垃圾点保底）
                ⇒ 返回 1e20 强制访问
```

与 §4.2 的区别：阈值是 **∝ N 的绝对份额**，而不是 ∝√N 的衰减量，
所以能真正做到「N 再大也保底」。这是对参考材料那句话最直白的实现。

### Tier 3 —— 代码：偏离感知再展宽（正对「炸树」）

记录上一回合的树复用率（复用子节点数 / 上回合树规模）。
若复用率低于阈值（对手下在树外），在接下来若干回合把
`rootMinVisitShare` / `cpuctExplorationFloorCoeff` 临时抬高一个倍数。

依据：`search.cpp:650` 的 `beginSearch()` 已经知道复用情况，且
`treeReuseCarryOverTimeFactor` 已有「复用折成时间」的先例，可类比加「复用折成宽度」。

---

## 6. 测量与验收纪律（沿用既有约定）

- 连续测量有 **1.7–1.9% 单调热漂移** ⇒ 必须同轮配对、每轮轮换臂顺序、
  报告漂移量作噪声标尺，`|Δ| < drift` 判为等价；用配对中位差而非两组中位数相减。
- 性能指标（`nnEvals/s`、`avgBatchSize`）**不能**作为本方案的验收指标 ——
  本方案改的是搜索质量，不是吞吐。应新增：
  - **根访问熵** `H = −Σ p_i log p_i`（p_i = 根子节点访问份额）
  - **top1 份额**、**有效宽度**（如参与 90% 访问的点数）
  - **树复用率**（对手落子后存活访问 / 上回合总访问）
  - **对局胜率 / 门控 Elo**（最终判据）
- 每个 Tier 单独 A/B，不与其它 Tier 混在一起上。

---

## 7. 风险与边界

1. **展宽 ≠ 变强。** 宽度指标变好而胜率下降是完全可能的（尤其 Tier 2/3 硬保底）。
   最终判据只能是胜率；宽度指标只用于诊断与调参。
2. **参数耦合。** `cpuct × fpu × rootPolicyTemperature × noisePruning` 相互影响，
   必须一次只动一组。
3. **上游化难度。** Tier 1/2/3 都会引入新参数，与上游 `SearchParams` 有冲突面；
   若只自用则无妨。
4. **与已有剪枝/plan 工作的关系。** 本分支只改 `cpp/search/`，
   与 `prune-width-per-gpu-sched` 的 CUDA 改动**无代码重叠**，
   但二进制需要重新编译（同一条 `build_sm89_b11.sh` 路径）。
5. **注意**：搜索参数不进 plan 的四道护栏（硬件身份/模型哈希/batch 认证/精度门），
   所以本分支不触碰精度门那笔账，也不减轻它的必要性。

---

## 8. 一个需要先确认的小问题：`maxVisits = 500`

生产 cfg 第 313 行是 `maxVisits = 500`，与官方 `gtp_example.cfg` 第 313 行**完全一致**
（连行号都对齐）⇒ 高度怀疑是模板残留、从未按 8 卡算力调整。

影响：`gtp.cpp:2182` 只在 cfg 里**没有** `maxVisits/maxPlayouts/maxTime` 时才套默认时间控制；
我们的 cfg 有 `maxVisits`，所以默认 TC 不生效。而 `search.cpp:563-565` 是

```cpp
bool shouldStop = (numPlayouts >= maxPlayouts) || (numPlayouts + numNonPlayoutVisits >= maxVisits);
```

⇒ 若前端没有通过 `kata-set-param maxVisits <N>` 覆盖，引擎会**硬停在 500 访问**。

「算了几百 M」与之矛盾，所以**大概率前端确实下发了 `kata-set-param`**，
但这一点必须核实 —— 如果是靠时间控制（`time_settings`）而不是显式设 `maxVisits`，
那 500 这个上限会一直压着，一切高 po 优化都无从谈起。

核实方法：对局中发 `kata-get-param maxVisits`，或看 GTP 日志里的
`numPlayouts` / 搜索结束时的访问数。

---

## 9. 立即可执行的最小实验

1. `kata-get-param maxVisits` 核实上限（§8）。
2. Tier 0 全套参数 → 同一 harness 同轮配对 A/B，看根访问熵 / top1 份额 / 树复用率 / 胜率。
3. 若 Tier 0 已显著改善，再上 Tier 1 扫 `cpuctExplorationFloorCoeff`。
