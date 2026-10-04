# 高 po 下的搜索宽度自适应 —— 分析与设计

- 日期：2026-10-04（v2，按 8 卡生产 cfg 重做）
- 分支：`search-adaptive-explore`（基线 `0cca6be`）
- 状态：**分析与设计，尚未实现任何代码**
- 起因：8 卡实测 ~29000 nnEvals/s、`avgBatchSize` 有提升，但实战中
  「超长考时算力高度集中、死磕一两个点，算了几百 M，被对手一子炸树」。

> **v1 → v2 更正**：v1 只核对了单卡用的 `config/gtp_sm89_plan_pruned.cfg`，
> 因而得出「搜索参数一个都没调」的结论。**该结论对 8 卡生产 cfg `b11_8.cfg` 不成立** ——
> 那份确实调过。本版按 `b11_8.cfg` 重做全部核对，并给出真正的缺口。

---

## 0. 问题陈述

算力从 3.1k nnEvals/s（单卡 T36 生产值）提到 29k nnEvals/s（8 卡）约 **9.3×**，
但搜索**宽度**没有同步增长：根节点访问分布仍高度集中在一两个点上。
对手下出一手不在树内的棋时，可复用的子树几乎归零（"炸树"），
此前几亿次访问全部作废。

对应参考材料那句话：

> 低 po（每秒小一两千）、中等 po（每秒 1 万）和高 po（每秒 10 万+）时候，
> mcts 引擎的工作参数、搜索树的形状、神经网络输出的调整，截然不同。

⇒ **参数必须随算力量级变化。** 本节要回答的是：引擎里到底有没有这种机制、开了没有。

---

## 1. 现状核查（可复现）

### 1.1 有两份 cfg，用途不同

| 文件 | 用途 | `numSearchThreads` | NN lane | 搜索参数 |
|---|---|---|---|---|
| `config/gtp_sm89_plan_pruned.cfg` | 单卡（device 0） | 36 | 2 | = 官方 `gtp_example.cfg` 原样 |
| **`b11_8.cfg`** | **8 卡生产** | **324** | **16（2 × 8 卡）** | **已做调整** |

单卡那份的核对（留作对照）：

```bash
diff <(grep -vE '^\s*#|^\s*$' cpp/configs/gtp_example.cfg) \
     <(grep -vE '^\s*#|^\s*$' config/gtp_sm89_plan_pruned.cfg)
# → 只有 numSearchThreads 6→36，以及追加的 CUDA 段
```

### 1.2 `b11_8.cfg` 已经调过的部分

吞吐 / 后端：

- `numSearchThreads = 324`、`numNNServerThreadsPerModel = 16`（每卡 2 lane）
- `nnCacheSizePowerOfTwo = 24`、`nnMutexPoolSizePowerOfTwo = 18`、`nodeTableShardsPowerOfTwo = 18`
- `cudaEventPipelineBatchFillMicros = 100`（本 fork 的 fill-grace，提交 `ae85f08`）
- `nnBatchAwareDispatch = true`、`cudaAsyncInferPipeline = true`、`cudaEventPipelineUseGraph = false`
- `useEvalCache = true` + `evalCacheMinVisits = 256`
  → **确实生效**：GTP 路径下 `useGraphSearch` 默认即为 `true`（`setup.cpp:573`：
  `params.useGraphSearch = (setupFor != SETUP_FOR_DISTRIBUTED)`），
  而 `useEvalCache` 的 8 个使用点全部以 `useEvalCache && useGraphSearch` 为门。

搜索行为：

- `rootNumSymmetriesToSample = 4`（库默认 1 —— 比建议更宽，方向正确）
- `rootFpuReductionMax = 0.08`、`fpuReductionMax = 0.18`（库默认 0.2）
- `cpuctExplorationLog = 0.40`（库默认 **0.0**）
- `numVirtualLossesPerThread = 1.5`（库默认 1.0）
- `rules = japanese`

### 1.3 与引擎自带的「认真下棋」参数集对照

引擎里有一份权威的推荐集：`SearchParams::basicDecentParams()`
（`cpp/search/searchparams.cpp:331-366`），
官方 cfg 模板里那批注释掉的建议值就是这一档。逐项对照 `b11_8.cfg`：

| 参数 | 库默认 | `basicDecentParams()` 建议 | `b11_8.cfg` 实际 | 状态 |
|---|---|---|---|---|
| `cpuctExploration` | 1.0 | 1.0 | 未设 → 1.0 | ✅ |
| `cpuctExplorationLog` | 0.0 | 0.45 | **0.40** | ✅ 近似 |
| `cpuctUtilityStdevPrior` | 0.25 | **0.40** | 未设 → 0.25 | ❌ |
| `cpuctUtilityStdevPriorWeight` | 1.0 | **2.0** | 未设 → 1.0 | ❌ |
| **`cpuctUtilityStdevScale`** | **0.0** | **0.85** | 未设 → **0.0** | ❌ **关键** |
| `fpuReductionMax` | 0.2 | 0.2 | **0.18** | ✅ |
| `rootFpuReductionMax` | 0.2 | 0.1 | **0.08** | ✅ |
| `fpuParentWeightByVisitedPolicy` | false | **true** | 未设 → false | ❌ |
| `valueWeightExponent` | 0.5 | **0.25** | 未设 → 0.5 | ❌ |
| `useNoisePruning` | false | **true** | 未设 → false | ❌ |
| **`useUncertainty`** | **false** | **true** | 未设 → **false** | ❌ **关键** |
| `uncertaintyCoeff` | 0.2 | 0.25 | 未设 → 0.2 | ❌ |
| `uncertaintyExponent` / `MaxWeight` | 1.0 / 8.0 | 1.0 / 8.0 | 同默认 | ✅ |
| **`subtreeValueBiasFactor`** | **0.0** | **0.45** | 未设 → **0.0** | ❌ **关键** |
| `subtreeValueBiasFreeProp` | 0.8 | 0.8 | 同默认 | ✅ |
| `subtreeValueBiasWeightExponent` | 0.5 | 0.85 | 未设 → 0.5 | ❌（因子为 0 ⇒ 目前无作用） |
| **`useLcbForSelection`** | **false** | **true** | 未设 → **false** | ❌ **关键** |
| `lcbStdevs` | 4.0 | 5 | 未设 → 4.0 | ❌（联动） |
| `minVisitPropForLCB` | 0.05 | 0.20 | 未设 → 0.05 | ❌（联动） |
| `useNonBuggyLcb` | false | true | 未设 → false | ❌（联动） |
| `rootPolicyTemperature` | 1.0 | 1.0 | 同默认 | ✅ |
| `rootEndingBonusPoints` | 0.0 | 0.5 | 未设 → 0.0 | ❌ |
| `rootPruneUselessMoves` | false | true | 未设 → false | ❌ |
| `conservativePass` | false | true | 未设 → false | ❌ |
| `enablePassingHacks` | false | true | 未设 → false | ❌ |
| `fillDameBeforePass` | false | true | 未设 → false | ❌ |
| `dynamicScoreUtilityFactor` | 0.0 | 0.3 | 未设 → 0.0 | ❌ |
| `staticScoreUtilityFactor` | 0.3 | 0.1 | 未设 → 0.3 | ❌ |
| `useGraphSearch` | false(库) / **true(GTP)** | true | 未设 → **true** | ✅ |
| `rootNumSymmetriesToSample` | 1 | 1 | **4** | ✅ 更宽 |
| `numVirtualLossesPerThread` | 1.0 | 1.0 | **1.5** | ✅ 更宽 |
| `wideRootNoise` | 0.0 | — | 未设 → **0.0** | ❌ 见 §1.4 |
| `ponderingEnabled` | false | — | **false** | ❌（取决于对局环境） |

### 1.4 三个「设了但没打到点上」的地方

1. **`analysisWideRootNoise = 0.08` 对实战无效。**
   `setup.cpp:681-683`：实战只读 `wideRootNoise`；`analysisWideRootNoise` 由
   `gtp.cpp:2068` 读进 `analysisOut`，只作用于 `kata-analyze` / `lz-analyze`。
   ⇒ 实战中 `wideRootNoise` 仍是 **0.0**。cfg 自带注释也写明「Affects analysis only, does not affect play」。
   若意图是实战加宽根节点，需要的是 `wideRootNoise`。
2. **`cpuctExplorationLog = 0.40` 把衰减从 `1/√N` 缓和成 `log(N)/√N`，但仍然没有地板**（见 §2）。
   这是**方向上正确的一步**，只是量级不够。
3. **`subtreeValueBiasWeightExponent`（0.5，默认）目前完全不起作用** ——
   它只在 `subtreeValueBiasFactor != 0` 时才有意义，而因子是 0。

### 1.5 结论

8 卡 cfg 调过，但**只覆盖了吞吐 / 后端 / FPU / 对称**这几类。
真正决定「高算力下树有多宽」的四项 ——
**`subtreeValueBiasFactor`、`useUncertainty`、`cpuctUtilityStdevScale`、`useLcbForSelection`** ——
**全部还是关的**，且这四项恰好都在引擎自己的 `basicDecentParams()` 推荐集里。

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
       * parentUtilityStdevFactor;      // ← 因 cpuctUtilityStdevScale = 0，恒为 1.0
}
// 子节点 i：  V_i = exploreScaling * P_i / (1 + n_i) + u_i
```

代入 `b11_8.cfg` 的实际值（`cpuctExploration = 1.0`、`cpuctExplorationLog = 0.40`、
`cpuctExplorationBase = 500`）：

```
cpuct(N) = 1.0 + 0.40 · ln((N + 500) / 500)
V_i      = cpuct(N) · √N · P_i / (1 + n_i) + u_i
```

### 2.2 隐含「分辨率」仍随 N 单调下降，没有地板

子节点 i 能继续拿到访问的条件是 `cpuct(N)·√N·P_i/(1+n_i) ≳ Δu_i`。
对所有子节点求和并令 Σn_i = N（且 ΣP ≈ 1），得

```
Δu_min(N)  ≈  cpuct(N) / √N
```

即**搜索能分辨的最小价值差**。代入实际参数：

| N | 无 log 项（c≡1） | **当前实际（log=0.40）** | 倍数 |
|---|---|---|---|
| 10³ | 3.16e-2 | 4.55e-2 | — |
| 10⁴ | 1.00e-2 | 2.22e-2 | 2.2× |
| 10⁵ | 3.16e-3 | 9.87e-3 | 3.1× |
| 10⁶ | 1.00e-3 | 4.04e-3 | 4.0× |
| 10⁷ | 3.16e-4 | 1.57e-3 | 5.0× |
| 10⁸ | 1.00e-4 | **5.88e-4** | 5.9× |

⇒ `cpuctExplorationLog = 0.40` **确实有效**（在 10⁸ 处把分辨率放宽了约 6 倍），
**但只是把 `1/√N` 缓和成 `log(N)/√N`，仍然单调趋向 0，没有地板。**

### 2.3 而神经网络的价值误差并不随 N 缩小

utility 半径 = `winLossUtilityFactor 1.0 + staticScoreUtilityFactor 0.3` = 1.3
（`b11_8.cfg` 未改这两项）。NN 胜率预测误差量级 0.03–0.1，
折到 utility 约 **0.03–0.13**。这是**系统性偏差（bias）**，不是 variance，
**平均多少次都不会消失** —— 只有 `subtreeValueBiasFactor` 那类
「用搜索自己的证据反推偏差」的机制能修正它（`searchupdatehelpers.cpp:26-36, 273-308`）。

### 2.4 结论

- 在 N ≈ 10⁸ 时，实际分辨率 **5.88e-4**，比 NN 噪声地板 **0.03** 还低 **约 50 倍**。
  也就是在自身噪声地板以下又细分了约 1.7 个数量级。
- 由于平均效用收敛、噪声被平均掉，树**只会越来越窄**，
  最终收敛到「NN 系统性高估的那条线」上 —— 这正是「死磕一两个点」。
- 低 po 时宽度由价值噪声撑开（看起来正常）；高 po 时噪声被平均掉，
  宽度由真实价值差决定，而真实价值差又分不出来 ⇒ 树塌成一条线。
  这就是参考材料说的「搜索树的形状截然不同」。

---

## 3. 为什么「炸树」

KataGo 的树复用是「对手下完一手后，保留以该手为根的子树」。
既然根节点访问高度集中在 1–2 个点：

- 对手**下在树内那 1–2 个点之一** → 复用率高，一切正常；
- 对手**下在别处** → 复用率≈0，几百 M 访问清零，且剩余时间不变。

⇒ 「炸树」不是独立问题，它就是 §2 的直接后果：**树里根本没有备选点的信息。**

---

## 4. 关键代码位置（改动锚点）

| # | 位置 | 内容 |
|---|---|---|
| 4.1 | `searchexplorehelpers.cpp:9-29` | `cpuctExploration()` / `getExploreScaling()` —— **唯一收口点** |
| 4.2 | `searchexplorehelpers.cpp:166-170` | `rootDesiredPerChildVisitsCoeff` 漏斗：`childWeight < sqrt(P·N·coeff)` ⇒ 返回 `1e20` |
| 4.3 | `searchexplorehelpers.cpp:82-98, 188, 222` | `wideRootNoise` 根加宽（**注意只认 `wideRootNoise`**） |
| 4.4 | `searchexplorehelpers.cpp:265-321` | `getFpuValueForChildrenAssumeVisited()` —— FPU / `cpuctUtilityStdevFactor` 计算处 |
| 4.5 | `searchupdatehelpers.cpp:26-36, 273-308` | `subtreeValueBiasFactor` 施加点 |
| 4.6 | `searchresults.cpp:198-236` | LCB 选点（`useLcbForSelection` / `lcbStdevs` / `minVisitPropForLCB` / `useNonBuggyLcb`） |
| 4.7 | `search.cpp:497-598` | 上限计算、`upperBoundVisitsLeft` |
| 4.8 | `search.cpp:650-720` | `beginSearch()` 树复用与偏差表初始化 |
| 4.9 | `setup.cpp:571-573` | `useGraphSearch` 在 GTP 下默认 **true** |
| 4.10 | `setup.cpp:681-683` | `wideRootNoise` 只在实战读；`analysisWideRootNoise` 另走一条路 |
| 4.11 | `searchparams.cpp:331-366` | `basicDecentParams()` —— 引擎自带的推荐参数集 |

⚠️ 注意 §4.2 的漏斗阈值是 `sqrt(P·N·coeff)`，而 `childWeight ≈ n`，
所以它要求的**访问份额**是 `sqrt(P·coeff/N)` —— **同样随 N 衰减**。
`cpuctExplorationLog` 也只是 `log(N)/√N`。
⇒ **现有旋钮全都无法阻止「相对宽度随 N 衰减」，必须新增一个显式地板项。**

---

## 5. 方案

### Tier 0 —— 只改配置，零代码风险（先做这个）

#### 5.1 补齐 `basicDecentParams()` 里还缺的部分

```cfg
# ---- 高 po 搜索参数：补齐引擎自带推荐集 ----
subtreeValueBiasFactor = 0.45          # ★ 修正 NN 系统性价值偏差 —— 最关键
subtreeValueBiasWeightExponent = 0.85
subtreeValueBiasFreeProp = 0.8
useUncertainty = true                  # ★ 按 NN 自报不确定性加权访问
uncertaintyCoeff = 0.25
uncertaintyExponent = 1.0
uncertaintyMaxWeight = 8.0
cpuctUtilityStdevPrior = 0.40          # ★ 波动局面多探、稳定局面少探
cpuctUtilityStdevPriorWeight = 2.0
cpuctUtilityStdevScale = 0.85
useLcbForSelection = true              # ★ 保守选点，避免追一个均值虚高的点
lcbStdevs = 5.0
minVisitPropForLCB = 0.20
useNonBuggyLcb = true
fpuParentWeightByVisitedPolicy = true
valueWeightExponent = 0.25
useNoisePruning = true
```

#### 5.2 修掉 §1.4 的三处

```cfg
wideRootNoise = 0.03        # 实战加宽根节点（原 analysisWideRootNoise 只管分析）
cpuctExplorationLog = 0.45  # 对齐推荐值（当前 0.40，差异很小，可选）
```

#### 5.3 视对局环境决定

```cfg
ponderingEnabled = true     # 对手回合也搜 —— 直接缓解"炸树"
```

预期：根访问分布变宽、深线价值偏差被校正、树复用率上升。
**不要只看 nnEvals/s 或单点访问数 —— 必须用对局胜率/门控判定**（见 §6）。

### Tier 1 —— 代码：显式探索分辨率地板（本分支的核心）

在 `getExploreScaling()` 加一项，让 cpuct 的下限随 N 增长，
使「最小可分辨价值差」不再无限变细：

```
新增参数：cpuctExplorationFloorCoeff（默认 0.0 = 关闭，行为与现在逐位一致）

cpuct(N)     = cpuctExploration + cpuctExplorationLog·ln((N+base)/base)
effCpuct(N)  = max( cpuct(N), cpuctExplorationFloorCoeff · √N )
```

因为分辨率 ≈ `effCpuct/√N`，取 `max(·, k√N)` 恰好让分辨率**止跌于 k**。
k 的物理含义是「我拒绝在低于这个价值差上做区分」，应设在 NN 价值噪声量级
（0.02–0.05，需实测标定）。默认 0 保证向后兼容。

- 实现点：仅 `searchexplorehelpers.cpp:9-29` + `searchparams.{h,cpp}` + `setup.cpp` 解析 + 文档
- 风险：k 过大会让树变成均匀灌木、单点精度下降 ⇒ 扫 k ∈ {0, 0.01, 0.02, 0.05} 并对局验证

### Tier 2 —— 代码：根子节点访问份额地板（更直接、更可控）

```
新增参数：rootMinVisitShare（默认 0.0 = 关闭）

根节点选择时：若 n_i / N < rootMinVisitShare 且 P_i 超过某阈值
              ⇒ 返回 1e20 强制访问
```

与 §4.2 的区别：阈值是 **∝ N 的绝对份额**，不是 ∝√N 的衰减量，
所以能真正做到「N 再大也保底」。这是对参考材料那句话最直白的实现。

### Tier 3 —— 代码：偏离感知再展宽（正对「炸树」）

记录上一回合的树复用率（复用子节点数 / 上回合树规模）。
若复用率低于阈值（对手下在树外），在接下来若干回合把
`rootMinVisitShare` / `cpuctExplorationFloorCoeff` 临时抬高一个倍数。

依据：`search.cpp:650` 的 `beginSearch()` 已经知道复用情况，且
`treeReuseCarryOverTimeFactor` 已有「复用折成时间」的先例，可类比加「复用折成宽度」。

---

## 6. 测量与验收纪律

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

1. **展宽 ≠ 变强。** 宽度指标变好而胜率下降完全可能（尤其 Tier 2/3 硬保底）。
   最终判据只能是胜率；宽度指标只用于诊断与调参。
2. **参数耦合。** `cpuct × fpu × rootPolicyTemperature × noisePruning × uncertainty` 相互影响，
   必须一次只动一组。
3. **上游化难度。** Tier 1/2/3 都会引入新参数，与上游 `SearchParams` 有冲突面；自用无妨。
4. **与已有剪枝/plan 工作的关系。** 本分支只改 `cpp/search/`，
   与 `prune-width-per-gpu-sched` 的 CUDA 改动**无代码重叠**，
   但二进制需重新编译（同一条 `build_sm89_b11.sh` 路径）。
5. **搜索参数不进 plan 的四道护栏**（硬件身份/模型哈希/batch 认证/精度门），
   所以本分支不触碰精度门那笔账，也不减轻它的必要性。

---

## 8. `maxVisits = 500`

已确认**不是问题**：连线器通过时间控制下发，不会在 500 处停止。
本节保留作为记录 —— 判断依据是 `gtp.cpp:2182` 只在 cfg **完全没有**
`maxVisits/maxPlayouts/maxTime` 时才套默认时间控制，
而实际对局由连线器下发时间控制，两者取先到者，实际访问量远高于 500。

---

## 9. 立即可执行的最小实验

1. **Tier 0 的 5.1 + 5.2** → 同一 harness 同轮配对 A/B，看
   根访问熵 / top1 份额 / 树复用率 / 胜率。这一档零代码风险，优先做。
2. 若 Tier 0 已显著改善，再上 Tier 1 扫 `cpuctExplorationFloorCoeff`。
3. 每一步都记录漂移量，作为噪声标尺。
