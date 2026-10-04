# 高 po 下的搜索宽度自适应 —— 分析、实测与设计

- 日期：2026-10-04（**v3**：改用「运行中二进制的有效参数 dump」重做现状核查，并补入两轮实测）
- 分支：`search-adaptive-explore`（基线 `0cca6be`）
- 状态：**Tier 0 已完成实测归因；Tier 1/2 代码已写，待编译验证**
- 起因：8 卡实测 ~29000 nnEvals/s、`avgBatchSize` 有提升，但实战中
  「超长考时算力高度集中、死磕一两个点，算了几百 M，被对手一子炸树」。

> **更正链**
> - v1 只核对了单卡 `config/gtp_sm89_plan_pruned.cfg`，得出「搜索参数一个都没调」。
> - v2 改核 8 卡 `b11_8.cfg`，但**对照基准用的是 C++ 结构体默认值 `SearchParams()`**，
>   得出「缺 4 项关键参数」。**该结论错了。**
> - **v3（本版）**：不再靠读源码推断，直接对运行中的二进制下发 `kata-get-params`，
>   拿到 GTP 路径下 `Setup::loadSingleParams(..., SETUP_FOR_GTP, ...)` 的**真实有效值**。
>   结论见 §1：**GTP 路径的默认值基本就等于引擎自带的 `basicDecentParams()` 推荐集**，
>   v2 那张表里标 ❌ 的绝大多数项其实**本来就是开的**。

---

## 0. 问题陈述

算力从 3.1k nnEvals/s（单卡 T36）提到 29k nnEvals/s（8 卡）约 **9.3×**，
但搜索**宽度**没有同步增长：根节点访问分布仍高度集中在一两个点上。
对手下出一手不在树内的棋时，可复用的子树几乎归零（"炸树"）。

参考材料那句话：

> 低 po（每秒小一两千）、中等 po（每秒 1 万）和高 po（每秒 10 万+）时候，
> mcts 引擎的工作参数、搜索树的形状、神经网络输出的调整，截然不同。

本文要回答的是：引擎里到底有没有这种机制、开了没有、没开的话怎么补。

---

## 1. 现状核查（可复现，权威）

### 1.1 方法：直接 dump 运行中的有效参数

```bash
cd /root/autodl-tmp/katago-plan
printf 'boardsize 19\nkomi 7.5\nkata-get-params\nquit\n' | \
  ./bin/katago-prune-sched-grace gtp \
    -config config/gtp_sm89_plan_pruned.cfg \
    -model models/b11-ffn-pruned-a8.bin.gz 2>/dev/null
```

`kata-get-params` 返回的是 `engine->getGenmoveParams().changeableParametersToJson()`
（`gtp.cpp:2653`），即**实战真正用的那套**。

### 1.2 ⚠️ 关键更正：GTP 路径的默认值 ≈ `basicDecentParams()`

`setup.cpp:471-780` 的 `loadSingleParams()` 里，大量参数的「cfg 未设时的默认值」
**不是** C++ 结构体默认值，而是按 `setupFor` 分支给的。实战 GTP 走
`SETUP_FOR_GTP`（`gtp.cpp:2062`），于是：

| 参数 | `SearchParams()` 结构体默认 | **GTP 实际默认** | `basicDecentParams()` |
|---|---|---|---|
| `cpuctExploration` | 1.0 | **1.0** | 1.0 |
| `cpuctExplorationLog` | 0.0 | **0.45** | 0.45 |
| `cpuctUtilityStdevPrior` | 0.25 | **0.40** | 0.40 |
| `cpuctUtilityStdevPriorWeight` | 1.0 | **2.0** | 2.0 |
| `cpuctUtilityStdevScale` | 0.0 | **0.85** | 0.85 |
| `fpuReductionMax` | 0.2 | 0.2 | (未设) |
| `fpuParentWeightByVisitedPolicy` | false | **true** | true |
| `valueWeightExponent` | 0.5 | **0.25** | 0.25 |
| `useNoisePruning` | false | **true** | true |
| `useUncertainty` | false | **true** | true |
| `uncertaintyCoeff/Exponent/MaxWeight` | 0.2/1.0/8.0 | **0.25/1.0/8.0** | 0.25/1.0/8.0 |
| `useGraphSearch` | false | **true** | true |
| `subtreeValueBiasFactor` | 0.0 | **0.45** | 0.45 |
| `subtreeValueBiasFreeProp` | 0.8 | 0.8 | 0.8 |
| `subtreeValueBiasWeightExponent` | 0.5 | **0.85** | 0.85 |
| `useLcbForSelection` | false | **true** | true |
| `lcbStdevs` | 4.0 | **5.0** | 5.0 |
| `minVisitPropForLCB` | 0.05 | **0.15** | 0.20 |
| `useNonBuggyLcb` | false | **true** | true |
| `rootFpuReductionMax` | 0.2 | **0.1** | 0.1 |
| `rootPolicyTemperature` / `Early` | 1.0 | 1.0 / 1.0 | 1.0 / 1.0 |
| `rootPruneUselessMoves` | false | **true** | true |
| `rootSymmetryPruning` | false | **true** | — |
| `rootEndingBonusPoints` | 0.0 | **0.5** | 0.5 |
| `staticScoreUtilityFactor` | 0.3 | **0.1** | 0.1 |
| `dynamicScoreUtilityFactor` | 0.0 | **0.3** | 0.3 |
| `enablePassingHacks` / `MorePassingHacks` | false | **true** | true |
| `policyOptimism` / `rootPolicyOptimism` | 0.0 | **1.0 / 0.2** | — |
| `numVirtualLossesPerThread` | 1.0 | 1.0 | (未设) |
| **`wideRootNoise`** | **0.0** | **0.0** | **(未设)** |
| `rootDesiredPerChildVisitsCoeff` | 0.0 | **0.0** | (未设) |

⇒ **v2 表里标 ❌ 的项（`useUncertainty`、`cpuctUtilityStdevScale`、
`subtreeValueBiasFactor`、`useLcbForSelection`、`fpuParentWeightByVisitedPolicy`、
`valueWeightExponent`、`useNoisePruning`、`rootPruneUselessMoves`、`conservativePass`、
`fillDameBeforePass`、`enablePassingHacks`、`rootEndingBonusPoints`、
`static/dynamicScoreUtilityFactor`）全部本来就是开的。**
GTP 默认与 `basicDecentParams()` 只差 `minVisitPropForLCB`（0.15 vs 0.20）。

**教训（同类第三次）：审计配置前，先确认「实际生效值」而不是「源码默认值」——
运行中的二进制自己会告诉你，`kata-get-params` 一问便知。**

### 1.3 8 卡 `b11_8.cfg` 相对 GTP 默认的真实 delta

| 参数 | GTP 默认 | `b11_8.cfg` | 对广度的影响 |
|---|---|---|---|
| `numSearchThreads` | 6 | **324** | 吞吐 |
| `nnMaxBatchSize` / lane / CUDA 段 | — | 见 §1.4 | 吞吐 |
| `cpuctExplorationLog` | 0.45 | **0.40** | **变窄**（探索增长更慢） |
| `rootFpuReductionMax` | 0.1 | **0.08** | 略宽 |
| `fpuReductionMax` | 0.2 | **0.18** | 略宽 |
| `rootNumSymmetriesToSample` | 1 | **4** | 变宽（根政策多对称平均） |
| `numVirtualLossesPerThread` | 1.0 | **1.5** | 变宽（多线程分散） |
| `analysisWideRootNoise` | 0.04 | **0.08** | ⚠️ **对实战无效** |
| **`wideRootNoise`** | **0.0** | **未设 → 0.0** | ❌ **实战仍是 0** |

### 1.4 ⚠️ 唯一真正"设了但打偏"的地方

`analysisWideRootNoise = 0.08` **只作用于 `kata-analyze` / `lz-analyze`**
（`gtp.cpp:2068-2079` 读进 `analysisOut`），实战只读 `wideRootNoise`
（`setup.cpp:681-683`）。cfg 自带注释也写着「Affects analysis only, does not affect play」。
⇒ **实战中根节点加宽量 = 0。**

这不是小问题：**§2 的实测显示，`wideRootNoise` 是全部现成参数里唯一真正有效的广度旋钮。**

### 1.5 结论

8 卡 cfg 调过的是**吞吐 / 后端 / FPU / 对称 / 虚拟损失**。
搜索**质量**参数基本就是引擎自带推荐档 —— 这**不是**配置失误，
而是**PUCT 在高 N 下的固有行为**（§2）。所以：
**配置层能拿到的收益有限且只有 `wideRootNoise` 一个旋钮；要真正加地板必须改代码（§5）。**

---

## 2. 机理：为什么 N 一大就必然「死磕一两个点」

### 2.1 选择公式

`cpp/search/searchexplorehelpers.cpp:9-29`

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
// 子节点 i：  V_i = exploreScaling * P_i / (1 + n_i) + u_i
```

代入实际值（`cpuctExploration = 1.0`、`cpuctExplorationLog = 0.45`、
`cpuctExplorationBase = 500`）：

```
cpuct(N) = 1.0 + 0.45 · ln((N + 500) / 500)
V_i      = cpuct(N) · √N · P_i / (1 + n_i) + u_i
```

### 2.2 隐含分辨率随 N 单调下降，没有地板

子节点 i 能继续拿到访问的条件是 `cpuct(N)·√N·P_i/(1+n_i) ≳ Δu_i`。
子节点 i 的**访问份额**满足 `n_i/N ∝ Δu_i / cpuct(N)`，于是

```
Δu_min(N)  ≈  cpuct(N) / √N        （可分辨的最小价值差）
```

| N | c≡1（无 log 项） | **实际（log=0.45）** |
|---|---|---|
| 10³ | 3.16e-2 | 4.78e-2 |
| 10⁴ | 1.00e-2 | 2.33e-2 |
| 10⁵ | 3.16e-3 | 1.04e-2 |
| 10⁶ | 1.00e-3 | 4.24e-3 |
| 10⁷ | 3.16e-4 | 1.65e-3 |
| 10⁸ | 1.00e-4 | **6.17e-4** |

⇒ `cpuctExplorationLog` 把 `1/√N` 缓和成 `log(N)/√N`（10⁸ 处放宽约 6 倍），
**但仍单调趋向 0，没有地板。**

### 2.3 而神经网络的价值误差不随 N 缩小

utility 半径 = `winLossUtilityFactor 1.0 + staticScoreUtilityFactor 0.1 + dynamicScoreUtilityFactor 0.3`。
NN 胜率预测误差量级 0.03–0.1 ⇒ 折到 utility 约 **0.03–0.13**。
这是**系统性偏差（bias）**，不是 variance，**平均多少次都不消失**。

### 2.4 结论

N ≈ 10⁸ 时实际分辨率 **6.17e-4**，比 NN 噪声地板 **0.03** 低 **约 50 倍** ——
在自身噪声地板以下又细分了约 1.7 个数量级。树**只会越来越窄**，
最终收敛到「NN 系统性高估的那条线」上，这就是「死磕一两个点」。

---

## 3. 为什么「炸树」

树复用 = 对手下完一手后保留以该手为根的子树。根访问集中在 1–2 个点时：

- 对手**下在树内那 1–2 个点之一** → 复用率高；
- 对手**下在别处** → 复用率≈0，几百 M 访问清零，剩余时间不变。

⇒ 「炸树」不是独立问题，它就是 §2 的直接后果：**树里根本没有备选点的信息。**

---

## 4. 实测（本项目的关键证据）

工具：`tools/breadth_probe.py`（走实战 `kata-search`，解析 GTP 日志的 `Tree:` 段，
算根访问分布指标）、`tools/ab_breadth.py`（配对 A/B）、`tools/sweep_breadth.py`（多臂扫描）。
三者都实现「同轮配对 + 每轮轮换臂序 + 漂移作噪声标尺」。

⚠️ 测量口径提醒：`printTree` 的子节点排序依据是 **`playSelectionValue` 而非访问量**
（`searchdata.cpp` 的 `operator<`），且只列前 10 个。所以：
- 探针在解析后**按访问量重排**；
- 必须同时看 `shown_share`（前 10 覆盖根访问的比例）。`shown_share` 掉到 0.75 时，
  说明分布已散出前 10，`top1` 会被截断失真 —— **要准确测必须改源码输出全部子节点**。

### 4.1 基线：症状复现（20000 访问）

| 局面 | top1 | 熵 | w90 | n≥1% |
|---|---|---|---|---|
| `empty` | 0.807 | 0.529 | 2 | 2 |
| `opening4` | 0.924 | 0.333 | 1 | 3 |
| `midgame8` | **0.966** | 0.198 | 1 | 2 |

**仅仅 2 万访问，中盘就已 96.6% 集中在一个点。** 症状完全复现。

### 4.2 Tier 0（配置层）配对 A/B：3 局面 × 4 轮 × 20000 访问

| 局面 | 指标 | 臂A 基线 | 臂B Tier0 | Δmed | driftA | 判定 |
|---|---|---|---|---|---|---|
| empty | top1 | 0.747 | 0.441 | −0.319 | 0.089 | 更广（3.6×漂移） |
| empty | 熵 | 0.605 | 1.255 | +0.657 | 0.091 | 更广 |
| empty | w90 | 2 | 4 | +2 | 0 | 更广 |
| midgame8 | top1 | 0.967 | 0.743 | −0.224 | 0.004 | 更广（**50×漂移**） |
| midgame8 | 熵 | 0.191 | 0.951 | +0.760 | 0.019 | 更广 |
| opening4 | top1 | 0.924 | 0.546 | −0.381 | 0.013 | 更广（**29×漂移**） |
| opening4 | 熵 | 0.333 | 1.215 | +0.881 | 0.041 | 更广 |

⇒ 效果 4–50 倍于漂移，不是噪声。

### 4.3 归因分解（关键）：Tier 0 的效果**全部来自 `wideRootNoise`**

既然 §1.2 证明 Tier 0 里绝大多数参数本来就是 GTP 默认值，那么效果只能来自
`wideRootNoise` 与 `minVisitPropForLCB`。5 臂 × 2 局面 × 4 轮扫描：

| 局面 | 指标 | A 基线 | B `wideRootNoise=0.03` | C `wideRootNoise=0.08` | D `rootDesiredPerChildVisitsCoeff=2.0` | E `minVisitPropForLCB=0.20` |
|---|---|---|---|---|---|---|
| empty | top1 | 0.807 | −0.358 **更广** | −0.505 **更广** | −0.017 **噪声内** | −0.048 **噪声内** |
| empty | 熵 | 0.529 | +0.716 | +1.105 | +0.022 | +0.059 |
| empty | w90 | 2 | +2 | +3 | 0 | 0 |
| midgame8 | top1 | 0.966 | −0.220 **更广** | −0.597 **更广** | +0.004 **噪声内** | +0.001 **噪声内** |
| midgame8 | 熵 | 0.198 | +0.748 | +1.720 | −0.018 | −0.002 |
| midgame8 | w90 | 1 | +2 | +6 | 0 | 0 |
| midgame8 | n≥1% | 2 | +2.5 | +8 | 0 | 0 |

三条结论：

1. **`wideRootNoise` 是唯一有效的现成旋钮。** 0.03 已显著；0.08 把 midgame8 的
   `w90` 从 1 推到 7、`n≥1%` 从 2 推到 10。代价：它是**随机噪声**，
   且 0.08 时 `shown_share` 掉到 0.75（分布散出前 10）。
2. **`rootDesiredPerChildVisitsCoeff = 2.0` 完全无效**（全部噪声内）。
   这**实证**了 §4.2 的推导：它的门槛是 `√(P·N·coeff)`，
   在 N=20000、P≈0.1 时 ≈ 45 次访问 —— 而基线 top1 已有 ~19000 次，
   漏斗早就满足了。**∝√N 的漏斗挡不住集中，必须 ∝N（Tier 2）。**
3. **`minVisitPropForLCB = 0.15→0.20` 无效**（噪声内）。

⇒ **`wideRootNoise` 只作用于 `getExploreSelectionValueOfChild` / `getNewExploreSelectionValue`，
不进入 `getPlaySelectionValues`（最终选点路径）**，所以它是「只改访问分配、不直接改选点」
的纯广度旋钮 —— 这一点对可接受性很重要，但**它引入随机性，最终仍必须用胜率验证**。

---

## 5. 方案

### Tier 0 —— 配置层（已完成实测，**唯一有效项是 `wideRootNoise`**）

```cfg
# 实战加宽根节点。⚠️ 必须写 wideRootNoise，不是 analysisWideRootNoise
# （后者只作用于 kata-analyze / lz-analyze，实战无效）。
# 0.03 = 保守；0.08 = 激进（midgame8 的 w90 1→7，但分布会散出前 10）。
wideRootNoise = 0.03
```

其余「补齐 `basicDecentParams()`」的项**不需要写** —— 见 §1.2，它们本来就是开的。
写上去只是无害的显式声明。

### Tier 1 —— 代码：显式探索分辨率地板（本分支核心）✅ 已实现并实测（见 §11）

```
新增参数：cpuctExplorationFloorCoeff（默认 0.0 = 关闭，行为与旧版逐位一致）

cpuct(N)     = cpuctExploration + cpuctExplorationLog·ln((N+base)/base)
effCpuct(N)  = max( cpuct(N), cpuctExplorationFloorCoeff · √N )
```

因为分辨率 ≈ `effCpuct/√N`，取 `max(·, k√N)` 恰好让分辨率**止跌于 k**。
k 的物理含义是「我拒绝在低于这个价值差上做区分」，应设在 NN 价值噪声量级
（0.01–0.05，需实测标定）。

实现点：`searchexplorehelpers.cpp:getExploreScaling()` + `searchparams.{h,cpp}` +
`setup.cpp` 解析 + `gtpconfig.cpp` 文档。

### Tier 2 —— 代码：根子节点访问份额地板（对「炸树」最直接）✅ 已实现并实测（见 §11）

```
新增参数：rootMinVisitShare（默认 0.0 = 关闭）
          rootMinVisitSharePolicyMin（默认 0.02）

根节点选择时：若 P_i ≥ policyMin 且 n_i / N < rootMinVisitShare ⇒ 返回 1e20 强制访问
```

与 `rootDesiredPerChildVisitsCoeff` 的区别：阈值是 **∝ N 的绝对份额**，不是 ∝√N 的衰减量。
**§4.3 已实测证明 ∝√N 的那个在 2 万访问时就已经完全失效**，所以这一条是必要的。

实现点：`searchexplorehelpers.cpp:getExploreSelectionValueOfChild()`，
紧邻现有 `rootDesiredPerChildVisitsCoeff` 分支。

### Tier 3 —— 代码：偏离感知再展宽（未实现）

记录上一回合树复用率；低于阈值（对手下在树外）时，接下来若干回合临时抬高
Tier 1/2 的地板。依据：`search.cpp` 的 `beginSearch()` 已知复用情况，
且 `treeReuseCarryOverTimeFactor` 已有「复用折成时间」的先例，可类比加「复用折成宽度」。

---

## 6. 关键代码位置（改动锚点）

| # | 位置 | 内容 |
|---|---|---|
| 6.1 | `searchexplorehelpers.cpp:9-29` | `cpuctExploration()` / `getExploreScaling()` —— **Tier 1 收口点** |
| 6.2 | `searchexplorehelpers.cpp:166-191` | `rootDesiredPerChildVisitsCoeff` 漏斗；**Tier 2 加在其后** |
| 6.3 | `searchexplorehelpers.cpp:82-98, 188, 222` | `wideRootNoise` 根加宽（**只认 `wideRootNoise`**） |
| 6.4 | `searchexplorehelpers.cpp:265-321` | `getFpuValueForChildrenAssumeVisited()` —— FPU / `cpuctUtilityStdevFactor` |
| 6.5 | `searchupdatehelpers.cpp:114-136` | `useUncertainty` 实际作用点：**给访问加权**（不是加探索奖励） |
| 6.6 | `searchresults.cpp:198-236` | LCB 选点（`useLcbForSelection` / `lcbStdevs` / `minVisitPropForLCB`） |
| 6.7 | `searchresults.cpp:1210-1245` | `printTree()` 入口；**子节点按 PSV 排序、只列前 10** |
| 6.8 | `analysisdata.cpp:174-193` | `operator<`：PSV 降序 → 访问量降序 → 原始政策 |
| 6.9 | `setup.cpp:471-780` | `loadSingleParams()` —— **GTP 默认值的真正来源** |
| 6.10 | `setup.cpp:681-683` | `wideRootNoise` 只在实战读；`analysisWideRootNoise` 另走一条路 |
| 6.11 | `gtp.cpp:2062` | 实战参数 = `loadSingleParams(cfg, SETUP_FOR_GTP, hasHumanModel)` |
| 6.12 | `searchparams.cpp:331-366` | `basicDecentParams()` —— 引擎自带的推荐参数集 |
| 6.13 | `commandline.cpp:346-353` | `-config` 是 MultiArg，**第 2 个起走 `overrideKeys`** |
| 6.14 | `config_parser.cpp:102-105` | ⚠️ `overrideKeys` **拒绝含 `/` 开头的绝对路径**（见 §8） |

---

## 7. 测量与验收纪律

- 连续测量有 **1.7–1.9% 单调热漂移** ⇒ 必须同轮配对、每轮轮换臂顺序、
  报告漂移量作噪声标尺，`|Δ| < drift` 判为等价；用配对中位差而非两组中位数相减。
- 每个 session **首轮丢弃**（GPU 冷启动偏高约 1.6%）。
- **性能指标（`nnEvals/s`、`avgBatchSize`）不是本方案的验收指标。** 本方案改的是
  搜索质量。用：根访问熵 / top1 份额 / w90 / `n≥1%` / **`shown_share`** / 树复用率 / **胜率**。
- ⚠️ `shown_share` 必须一并报告：`wideRootNoise` 一开，前 10 就盖不住全部访问了，
  此时 `top1` 是在截断分布上算的。**要严格测量必须改源码把 `maxChildrenToShow` 放开。**
- 每个 Tier 单独 A/B，不与其它 Tier 混在一起上。

---

## 8. 两个踩过的坑（值得写下来）

### 8.1 `-config` 的第二个及以后**不接受绝对路径**

```
ConfigParsingError: Absolute paths in the included files are not supported yet
```

- 第 1 个 `-config` 走 `ConfigParser::initialize()`（`config_parser.cpp:55-66`），
  绝对路径**可以**；
- 第 2 个起走 `overrideKeys()` → `processIncludedFile()`（`:374-380`），
  后者在 `:102-105` 对 `extractBaseDir(fname)` 以 `/` 开头的一律抛异常。
- 相对路径（含 `../`）合法。`tools/breadth_probe.py` 因此把所有 `-config`
  统一换算成相对 cwd 的路径。

### 8.2 `maxVisits = 500`

cfg 里确实有 `maxVisits = 500`，`kata-get-params` 也确认生效。
**探针必须用 `kata-set-param maxVisits <大值>` 覆盖**，否则会在 500 处截断
（本项目第一次跑探针时就撞上过）。实战由连线器下发时间控制，按用户说明不受影响。

---

## 9. 风险与边界

1. **展宽 ≠ 变强。** `wideRootNoise` 是随机噪声；Tier 2 是硬保底。
   宽度指标变好而胜率下降完全可能。**最终判据只能是胜率。**
2. **参数耦合。** `cpuct × fpu × rootPolicyTemperature × noisePruning × uncertainty`
   相互影响，一次只动一组。
3. **上游化难度。** Tier 1/2 引入新参数，与上游 `SearchParams` 有冲突面；自用无妨。
4. **与剪枝/plan 工作的关系。** 本分支只改 `cpp/search/` 与 `cpp/program/`，
   与 `prune-width-per-gpu-sched` 的 CUDA 改动**无代码重叠**。
5. **搜索参数不进 plan 的四道护栏**（硬件身份/模型哈希/batch 认证/精度门），
   所以本分支不触碰精度门那笔账，也不减轻它的必要性。

---

## 10. 下一步

1. ~~编译本分支~~ ✅ 已完成（§11.2）。`testgpuerror` 需加 `-override-config nnMaxBatchSize=12`，否则撞 plan 的 batch 闸门。
2. ~~扫 Tier 1 的 `cpuctExplorationFloorCoeff`~~ ✅ 已完成（§11.3）。
3. ~~扫 Tier 2 的 `rootMinVisitShare`~~ ✅ 部分完成：`policyMin=0.02` 固定，扫了 `rootMinVisitShare ∈ {0.01, 0.02}`。
   `policyMin ∈ {0.05}` 那一档未跑。
4. **对局胜率验收** —— 唯一还没做的、也是最重要的一步。上面全部是广度指标。
5. 结论落回 8 卡 `b11_8.cfg`（已产出 `config/search-tier12.cfg` 供参考，见 §11.4）。

---

## 11. Tier 1 / 2 实现与实测（2026-10-04 第二轮）

### 11.1 代码改动

| 文件 | 改动 |
|---|---|
| `cpp/search/searchparams.h` | 新增 3 个字段 + 注释 |
| `cpp/search/searchparams.cpp` | ctor 初值 / `operator==` / `changeableParametersToJson` / `PRINTPARAM` |
| `cpp/search/searchexplorehelpers.cpp` | `getExploreScaling()` 加地板；`getExploreSelectionValueOfChild()` 加份额强制 |
| `cpp/program/setup.cpp` | 三处解析（`cpuctExplorationFloorCoeff` 范围 0–100；两个 share 参数 0–1） |
| `cpp/program/gtpconfig.cpp` | 两段文档注释 |

两处实现都带 `> 0.0` 守卫 ⇒ **默认值下逐位等价于旧行为**。

**安全性核对（做对了才敢上生产）**：`rootMinVisitShare` 的 `return 1e20` 落在
`searchexplorehelpers.cpp:164` 的 `if(isDuringSearch && &parent==rootNode && countEdgeVisit)` 内。
而**最终选点路径** `searchresults.cpp:164` 调 `getExploreSelectionValueOfChild` 时
传的是 `isDuringSearch=false`（`:158`）⇒ 强制访问**不会泄漏进 `getPlaySelectionValues`**，
不会让引擎真去下那个「欠访问」的点。与上游 `rootDesiredPerChildVisitsCoeff` 同一守卫。

### 11.2 ⚠️ 构建卡点：SM89 AOT 内核缺失

**症状**：新二进制 GTP 起不来 ——
`StringError: Failed to start one or more CUDA event pipeline schedulers: GPU 0: Selected SM89 preConv CUTLASS tactic is unavailable`。

**根因链**（源码位置全部核对过）：
- `cpp/CMakeLists.txt:53` `SM89_FLASH_ATTN_ROOT` 默认为空；
- `:216` `if(SM89_FLASH_ATTN_ROOT)` 才加入 `katago_sm89_flash` OBJECT 库（4 个 .cu）；
- `:306-312` 由该库置 `KATAGO_ENABLE_SM89_{FLASH_ATTN,DUAL_GEMM,LINEAR2,OUTPROJ,PRECONV,POSTCONV,QKV_ROPE}_GEMM`；
- `:672-688` 再把这些宏转发给主 target `katago`；
- 缺宏 ⇒ `cudabackend_sm89_forward.cpp:1961-1976` 里 `usedPreConvGemm` 恒 false ⇒ fail-closed 抛异常。
- 生产 plan `apply/per_batch_tactic_overrides["12"]` **明确要求**这些 tactic
  （`cudaPreConvCutlassTacticSm89=m128-n128-k32-w64-n64-s3-sw1` 等）
  ⇒ 不是「可以绕开」，是**必须编进去**。

**解法**：`tools/fetch_flash_attn.sh`
- `Dao-AILab/flash-attention` @ `69e1bcbe77c359c84b3a4589e92a7c076e33a202`（`git fetch --depth 1 <sha>`，6 秒）；
- `csrc/cutlass` **不取子模块**，符号链接到已有 `third_party/cutlass`
  （同 commit `7127592069c2fe01b041e174ba4345ef9b279671`，git 内容寻址 ⇒ 树逐字节一致）；
- 写 `.katago-source-revision` 满足 CMake 的 commit 硬校验（`CMakeLists:229-260`）；
- 应用 `cpp/neuralnet/flash-attention-sm89.patch`，并复刻 CMake 的 4 个 `KATAGO_FLASH_*` 标记检查。
- `tools/build_search_branch.sh` 增加 `-DSM89_FLASH_ATTN_ROOT=…` 与前置存在性检查。

⚠️ **文档与 CMake 不一致**：`cpp/neuralnet/FLASH_ATTENTION_SM89.md:8` 写的 FlashAttention
commit 是旧的 `5835c733…`，**以 `CMakeLists.txt:225` 的 `69e1bcbe…` 为准**。

⚠️ 另外发现：远程 `katago-src` 当时是干净的 `0cca6be`，**不含**本分支的 5 个文件改动
⇒ 之前那次「编译通过」的二进制里其实**没有新参数**。
**教训：跨端声称「已编译」之前，先在目标端 `grep` 一下新符号是否存在。**

### 11.3 实测

方法：`tools/breadth_probe.py` 走实战 `kata-search` 路径解析根节点树形；
`tools/sweep_breadth.py` 做同轮配对 + 臂序轮换，漂移 = 基线臂跨轮极差，`|Δmed| < drift` 判噪声内。

**（A）N = 20,000 访问，midgame8 / opening4，4 轮**

| 臂 | midgame8 top1 | opening4 top1 | 判定 |
|---|---|---|---|
| A 基线 | 0.9649 | 0.9362 | — |
| F `wideRootNoise=0.03` | 0.6921 (−0.273) | 0.5446 (−0.392) | 最强 |
| C `floor=0.05` | 0.9347 (−0.030) | 0.8291 (−0.107) | 有效 |
| E `minShare=0.02` | 0.9486 (−0.016) | 0.9194 (−0.017) | 有效（弱） |
| B `floor=0.02` | 0.9650 (+0.000) | 0.8987 (−0.038) | 基本噪声内 |
| D `minShare=0.01` | 0.9640 (−0.001) | 0.9286 (−0.008) | 噪声内 |

**（B）N = 200,000 访问，midgame8，2 轮（漂移仅 0.0002，判定很硬）**

基线：top1=0.9919、熵=0.0588、w90=1、n≥1%=1 —— 访问量涨 10 倍后**明显更塌**。

| 臂 | Δtop1 | Δ熵 | Δn≥1% | 判定 |
|---|---|---|---|---|
| F `wideRootNoise=0.03` | −0.1891 | +0.6856 | +3.5 | 最强 |
| C `floor=0.05` | −0.0381 | +0.1993 | +2 | 有效 |
| E `minShare=0.02` | −0.0361 | +0.1759 | +3 | 有效 |
| G `floor=0.02 + noise=0.01` | −0.0251 | +0.1378 | +1 | 有效 |
| B `floor=0.02` | −0.0140 | +0.0794 | +1 | 有效 |

**（C）N = 1,000,000 访问，midgame8，2 轮（漂移仅 0.0001）**

基线：top1=0.9958、熵=0.0335、w90=1、n≥1%=1、shown_share=1.000
—— 访问量再涨 5 倍后**几乎完全塌成一个点**。

| 臂 | top1 | Δtop1 | 判定 |
|---|---|---|---|
| B `floor=0.02` | 0.9761 | −0.0196 | 131× 漂移 |
| C `floor=0.05` | 0.9473 | −0.0484 | 323× 漂移 |

### ★ 核心结论：同一个旋钮的效果**随 N 单调增长**

同一局面、同一臂 B（k=0.02），跨三个数量级：

| N | 基线 top1 | B top1 | Δtop1 |
|---|---|---|---|
| 20,000 | 0.9649 | 0.9650 | +0.0001（噪声内） |
| 200,000 | 0.9919 | 0.9779 | −0.0140 |
| 1,000,000 | 0.9957 | 0.9762 | −0.0195 |

两件事同时发生：

1. **基线随算力单调塌缩** `0.9649 → 0.9919 → 0.9957` —— 这就是用户描述的「死磕一两个点」。
2. **地板把它按住不动**：B 在 1e6 时是 0.9762，几乎等于它在 2e5 时的 0.9779。
   即地板把根分布**冻结**在某个宽度上，不再随算力继续塌。

这**正是** §5 Tier 1 的阈值公式预测的行为，不是偶然：
地板与原生项相等的位置是 `N* ≈ (cpuctExploration(N)/k)²`，k=0.02 时 N*≈2e4
⇒ 2 万访问刚好卡在阈值上（所以噪声内），之后每涨一个数量级地板就强一档
（1e6 时约 4.5×，1e7 时约 12×）。

**⇒ Tier 1 才是为「算了几百 M」这个量级设计的旋钮。**
`wideRootNoise` 的效果**与 N 无关**，随基线塌缩反而相对变弱（−0.273 → −0.189）。

### 11.4 交付

`config/search-tier12.cfg`（可作为第二个 `-config` 直接加载，已在实测机上验证生效）：

```cfg
cpuctExplorationFloorCoeff = 0.02   # Tier 1，首选
rootMinVisitShare = 0.02            # Tier 2
rootMinVisitSharePolicyMin = 0.02
# wideRootNoise = 0.01              # 需要更广时再打开（与 Tier 1 组合最优）
```

### 11.5 仍未做

1. **对局胜率验收** —— 以上全是广度指标，更广 ≠ 更强。这是唯一的最终判据。
   建议：固定相同时间/访问预算，与基线跑 ≥200 局配对对局，轮换先后手。
2. `rootMinVisitSharePolicyMin` 只试了 0.02。
3. `wideRootNoise=0.08` 会把分布散出 `printTree` 的前 10（`shown_share` 掉到 0.75）
   ⇒ 若要测更激进的档，需先放开 `maxChildrenToShow(10)`。
4. ~~N=1e6 档确认~~ ✅ 已完成（§11.3 C）。

### 11.6 测量工具踩过的三个坑（都会伪装成「实验结果」）

1. **探针读错日志文件**：`breadth_probe.py` 原来直接取 `sorted(logdir)[-1]`。
   若 `logdir` 里混进了**别的 katago 进程**写的日志，就会读到错的文件，报
   `no Tree block` —— 看起来像搜索失败，其实搜索是好的。
   1M 档的 A_r1 就是这么误判的。**已加固**：从新到旧扫，取第一个真含 `Tree:` 段的文件。
2. **`pgrep -f <名字>` 自匹配**：`while pgrep -f breadth_probe.py; do sleep 10; done`
   这种「等它跑完」的守卫会匹配到**自己**（自己的命令行里就有这个名字），
   于是要么死循环、要么立刻放行 —— 实测导致两个 1M 任务并发跑同一张卡。
   要等就等具体 PID。
3. **`sweep_breadth.py` 文档写错分隔符**：docstring 说臂之间用逗号，代码是 `split(";")`。
   用逗号会退化成「只有 1 个臂」直接报错。已修正。
