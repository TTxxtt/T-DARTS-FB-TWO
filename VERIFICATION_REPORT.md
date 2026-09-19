# 报告：Operator Separability V2 + searched/random 比较的修正

生成于 2026-09-18。本文是可直接复制的完整版；细节另见
`docs/operator_separability_v2.md`。

---

## 一、Operator Separability V2

### 文件改动

**新增 6 个文件，未修改任何现有文件：**

| 文件 | 内容 |
|---|---|
| `tdarts/operator_v2.py` | 5 个算子 + 独立注册表 + 审计（params/MACs/support） |
| `tdarts/operator_v2_network.py` | `OperatorV2Cell` / `OperatorV2Net` |
| `train_operator_v2.py` | standalone 训练入口 |
| `tools/analyze_operator_v2.py` | 方差分解 + 判定规则 |
| `tests/test_operator_v2.py` | 21 个测试 |
| `docs/operator_separability_v2.md` | 审计 + 结构 + 表格 + 结果 |

**刻意没碰的地方**：`tdarts/temporal_ops.OPERATOR_REGISTRY` 一个条目都没加——
加进去会改变 `canonical_candidates` → 改变搜索空间 → 让所有归档 search 和从它解出的
genotype 失效。V2 用自己的注册表，搜索路径不 import 这个模块。
`FBNAS/**`、`codes_new/**`、现有 DARTS / hierarchical / anchored / RF-only 逻辑、
现有结果目录，全部未动。

### 已核对的事实（不是凭印象）

| 项 | 值 | 出处 |
|---|---|---|
| 每个 Low/Mid/High cell 输入 | `[B,3,C,T]` | `config.IN_CHANNELS = 3` |
| 原 FBNAS 每 cell 输出 | `[B,12,C,T]` | `PATH_CHANNELS=6` × `NUM_PATHS=2` |
| 三 cell concat 后 | `[B,36,C,T]` | `NUM_FEAT = 3×2×6 = 36` |
| RF ladder | 15 / 29 / 57 / 113 | `RF_SPACE` |
| temporal kernel | `k=15`，dilation 1/2/4/8 | `BASE_KERNEL = 15` |
| SCB / BN / Swish / LogVar / classifier | 未改 | `backbone.py` 原样使用 |

### 五个算子

pilot 固定 **RF=57** → 由 `k=15` 得 `dilation=4` → **15 个间隔为 4 的位置**，
跨度 `1 + (15-1) × 4 = 57`。五个算子读取**完全相同的这 15 个位置**。

| 算子 | 机制 | 内部宽度 |
|---|---|---|
| `dilated` | FBNAS 原始 anchor：单个膨胀卷积 | 12 |
| `gated` | `tanh(f(x)) * sigmoid(g(x))`，再 1×1 projection | 6（瓶颈） |
| `local_attention` | 只对这 15 个位置做 attention（unfold 同款 gather + softmax），**非 global** | head_dim 3 |
| `dynamic` | K=2 个 depthwise basis kernel，由全局池化特征产生 softmax gating 组合 | 2 basis |
| `band_gated` | 对 cell 内已有 3 个 filter-bank 通道做逐位置 sigmoid gating，再时间卷积 | 3 |

未加入：GCN / electrode attention / Mamba / full Transformer / SincConv。

参数公平性上的取舍点都写在 `operator_v2.AUDIT_NOTES`：

- `gated` 的瓶颈 6 —— 宽度取 12 时 gating 需要两个时间卷积，参数量 1224（2.3× anchor）
- `local_attention` 的 head_dim 3 —— 见下
- `dynamic` 用 depthwise basis —— 两个 dense basis 加 projection 会是 1224

### attention 的返工：2.27× → 1.272×

初版 attention 是 2.27× MACs。作为正式主 candidate 会留下「是机制更合适，还是算力更多」的
混淆，所以返工。两处改动都是**精确等价**——不是近似，也没有把机制削弱：

**一、head_dim 24 → 3。** `q`/`k` 是从 cell 的 3 个输入通道出的线性映射，于是
`q(x_t)·k(x_{t+dk}) = x_tᵀ M x_{t+dk}`，其中 `M = W_qᵀ W_k` 是**任意 3×3 矩阵**——9 个自由度。
head_dim 超过 3 覆盖的是**完全同一组** score 函数：原来的 24 是 8 倍冗余，不是 8 倍表达力。
`HEAD_DIM` 现在钉死为 `IN_CHANNELS`，构造器在两者不等时报错。

**二、输出投影挪到加权归约之前。** 投影线性、权重只依赖 x，所以
`W_o Σa_k v_k = Σa_k (W_o v_k)` 恒等。归约宽度从 20 变成 12，2.27× 的主要来源在此消失。
**函数没变，变的是结合顺序。**

对拍验证（同权重、另写一份「投影在后」的参考实现）：max abs diff `6.4e-6`，输出尺度 `1.28`，
相对 ~`5e-6` ≈ 50×float32 eps，即重结合舍入。该断言已进测试。

**边界说明：参数匹配的窗口注意力到不了 1.0×。** softmax 加权归约里每个参数被用 K=15 次，
卷积权重每个位置只用 1 次。1.27× 是这套机制的下限附近，不是「还没优化干净」。

### 参数量 / MACs / support

`operator_audit(band="Low", target_rf=57)` 实测（Low/Mid/High 三 band 数值相同）：

| operator | params | vs 锚 | MACs | vs 锚 | support |
|---|---:|---:|---:|---:|---:|
| dilated | 540 | 1.000× | 11,880,000 | 1.000× | 15×4 = 57 |
| gated | 612 | 1.133× | 13,464,000 | 1.133× | 15×4 = 57 |
| local_attention | 462 | 0.856× | 15,114,000 | **1.272×** | 15×4 = 57 |
| dynamic | 512 | 0.948× | 11,616,006 | 0.978× | 15×4 = 57 |
| band_gated | 552 | 1.022× | 12,078,000 | 1.017× | 15×4 = 57 |

- **参数量全部落在 ±20% 内**（0.856 ~ 1.133×），无违例
- **算力全部 ≤1.3×**（0.978 ~ 1.272×），无违例
- **support 全部等于 57**，无越界偷看
- 两条带**同时**成立。

#### 一个必须说明的测量修正

MACs 计数器最初只钩 `nn.Conv2d` / `nn.Linear`，结果：

- `dynamic` 报了 **0.27×**（假象）—— 它的 basis 原本是裸 `F.conv2d`，hook 看不见
- `local_attention` 报了 **0.93×**（假象）—— scores 与加权的逐元素运算不计

两处都修了：`dynamic` 的 basis 改成 `nn.Conv2d` 模块；两个算子通过 `extra_macs()`
显式申报 hook 看不到的逐元素算力。**修正前 attention 看起来比卷积还便宜，这是错的。**

### 单元测试

```
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m unittest discover -s tests -t .
Ran 298 tests   OK
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m unittest tests.test_fbnas_compatibility
Ran 22 tests   OK                     FBNAS/ 下无新增或修改文件
```

`tests/test_operator_v2.py` 的 24 个测试覆盖：

- **形状契约**：五个算子都是 `[B,3,C,T] → [B,12,C,T]`，C 与 T 不变
- **几何契约**：五个算子的 support 恰等于 RF57；三个 band 均如此；1×1 的 gate/projection 不扩大窗口
- **公平性**：参数全部在 ±20% 内；anchor 恰为 `3×12×15 = 540`；**五个算子的 MACs 全部 ≤1.3×**
- **attention 不下穿 anchor**：另一头也断言——逐元素算力若不申报，attention 会被报成比卷积还便宜，让算力重的机制看起来免费
- **head 宽度不变量**：`HEAD_DIM == IN_CHANNELS`，不等则构造器报错
- **重结合等价性**：同权重对拍「投影在后」的参考实现，只允许舍入级差异；并把「归约宽度换回 `VALUE_WIDTH` 就重新出界」钉成断言
- **梯度**：五个算子 forward/backward 有限、无 NaN、梯度非零
- **attention 局部性**：断言 softmax 覆盖的是 15 个 support 位置，而不是整条时间轴
- **网络**：backbone 参数量不随算子变化；接受 loader 的 `[B,1,E,T,9]` 布局；输出是 log-probability
- **入口护栏**：`--read-session1` 默认关闭；`load_subject_session` 只出现一次且在该 flag 之下；叶子名必须带算子名；默认 RF=57

### Smoke 结果

`subject003 + seed20250901 + RF57`，每个算子 3 epoch（CPU）：

| operator | opParams | vs 锚 | opMACs | vs 锚 | support | valAcc | S1 |
|---|---:|---:|---:|---:|---:|---:|---|
| dilated | 540 | 1.000× | 11,880,000 | 1.000× | 15×4 | 33.33% | closed |
| gated | 612 | 1.133× | 13,464,000 | 1.133× | 15×4 | 40.35% | closed |
| local_attention | 462 | 0.856× | 15,114,000 | **1.272×** | 15×4 | 29.82% | closed |
| dynamic | 512 | 0.948× | 11,616,006 | 0.978× | 15×4 | 40.35% | closed |
| band_gated | 552 | 1.022× | 12,078,000 | 1.017× | 15×4 | 40.35% | closed |

**valAcc 接近随机是预期的** —— 只跑 3 epoch、从随机初始化开始，smoke 的目的是验证
管线跑通，不是比较性能。

逐项检查：

- ✅ forward shape 正确
- ✅ backward 正常，梯度非零
- ✅ 无 NaN
- ✅ 参数量 / MACs / support 如上报
- ✅ val 能正常跑（3 轮都产出读数）
- ✅ **Session 1 未读取**：五个 run 的 `metrics.jsonl` 都没有任何 `test` 字段，
  `config.session1_opened=false`、`session1_test_size=null`、`final_summary.test=null`

### 分析工具已验证的行为

`tools/analyze_operator_v2.py` 输出：逐 subject×operator 的 mean/std、主效应、交互效应、
seed/residual 方差，以及比值 `V_subject×operator / V_seed`。

**主分析指标是 `val_best_nll`**（`--metric` 默认值），accuracy 辅助报告 —— validation 只有
57 个 trial，一个样本就移动 `1.75pp`，在那个粒度上做方差分解量到的是量化误差。

判定三分支：

- `V_subject×operator > V_seed` 且不同 subject 的最优 family 确实不同 → **PROCEED**
- `> V_seed` 但所有 subject 的 argmax 同一 family → **PROCEED_WEAK**
- `V_subject×operator <= V_seed` → **WEAK_GLOBAL**

⚠️ 第三条**不再**打印「STOP，不要进入 NAS」。每个 run 把同一个 family 用在三个频段上，
任何**频段特异**的偏好会在三个频段之间被平均掉，在这个设计里结构上不可见。所以
WEAK_GLOBAL 只能支持「没有 family 能**全局**分离」这一条结论，**不能**否定 per-band
family search。详细的 probe 设计见 `docs/operator_separability_v2.md` 第十节。

- 空目录 → `no runs found`，不崩
- smoke 数据（1 subject × 1 seed × 5 operators）→ 正确报出"设计不平衡"、列出薄 cell，
  并给出 `verdict: not computable -- the design has no residual degrees of freedom`。
  **它拒绝在自由度不足时硬编一个结论**，这是它最该有的性质。

---

## 二、searched vs random 比较的修正

### 背景：一个未提交的改动悄悄改变了初始权重

归档的 searched 臂（`retrain_lr1e3`，09-15/16，跑在 `T-DARTS-FB`）与随机臂
（`random_k5`，09-18，跑在 `T-DARTS-FB-TWO`）用的**不是同一套 seed 语义**。

根因：工作区里一处**未提交**的改动，把基因型解析从 `set_seed` **之后**挪到了**之前**。

```diff
     set_seed(args.seed)
     ...
-    genotype = extract_genotype(metrics_path, epoch=search_epochs)
     model = TemporalDiscreteNet(genotype).to(device)
```

而基因型解析会构建候选算子池，`nn.Conv2d.__init__` 从全局 torch RNG 取数。
实测（同一 genotype、同一 seed）：

| 顺序 | 初始权重 |
|---|---|
| 旧（`set_seed` 在解析之前） | 与新版最大差 **0.285** |
| 新（解析在 `set_seed` 之前） | — |

s009 在 epoch 1 的表现因此是 `train_nll` **2.78 vs 2.19**、`train_acc` 不同。

证据链：

1. 当前代码**逐位复现**了随机臂（`train_acc`/`val_acc` 完全相同，`train_nll` 只差浮点尾数）
2. 当前代码**复现不了**归档的 searched run
3. 归档时期两个 commit（`7c869ee`、`e6751a6`）都能**精确复现**归档值 → 元凶是工作区未提交的改动
4. 微型实验确认 `load_genotype` / `extract_genotype` / `path_structure_keys` /
   `duplicate_structure_bands` **四个函数全部消耗全局 torch RNG**

### 修复

`train_retrain.py` 里把 `set_seed` 移到**紧贴模型构造之前**，两条路径（主路径与
`--stage2-only`）都改。修复后 `--search-dir` 与 `--genotype-json` 给出**相同**的初始权重，
而随机臂的行为一个浮点数都没变（它本来就是"解析后 set_seed"）。

### 重跑与结果

重跑 13 个任务：9 个 searched（seed 修复）+ 4 个早期停止的随机 run（应用下限 100）。

| | searched 均值 | 随机均值 | Δ | 95% CI | Δ>0 | 平均分位 P |
|---|---:|---:|---:|---|---:|---:|
| 修正前 | 78.47% | 78.33% | **+0.15** | [−1.94, +2.23] | 4/9 | 0.42 |
| 修正后 | 77.85% | 78.38% | **−0.52** | [−3.10, +2.05] | 5/9 | 0.44 |

逐受试者的 init 影响：

| subject | 旧 searched | 新 searched | Δinit (pp) | 随机均值 | Δ (旧) | Δ (新) |
|---|---:|---:|---:|---:|---:|---:|
| 001 | 85.76% | 83.33% | −2.43 | 82.85% | +2.57 | +0.49 |
| 002 | 59.03% | 59.38% | +0.35 | 61.18% | −2.15 | −1.81 |
| 003 | 92.01% | 93.75% | +1.74 | 92.64% | −0.62 | +1.11 |
| 004 | 71.18% | 76.04% | +4.86 | 71.11% | +0.49 | +4.93 |
| 005 | 81.25% | 77.43% | −3.82 | 76.11% | +5.14 | +1.32 |
| 006 | 60.42% | 57.29% | −3.12 | 62.78% | −2.36 | −5.49 |
| 007 | 89.58% | 81.94% | −7.64 | 87.50% | +2.08 | −5.56 |
| 008 | 85.07% | 85.42% | +0.35 | 85.62% | −0.56 | −0.21 |
| 009 | 81.94% | 86.11% | +4.17 | 85.62% | −3.26 | +0.49 |
| **均值** | **78.47%** | **77.85%** | **−0.62** | **78.38%** | **+0.15** | **−0.52** |

**结论没变**：Δ 仍然跨 0，标准 DARTS@1e-3 与随机架构在统计上仍无法区分。
但点估计从 +0.15 翻到 −0.52，CI 从 ±2.1 展宽到 ±2.6，逐受试者位移范围
−7.64 ~ +4.86 pp、sd ≈ 4pp —— **原来的数字确实比当时报告的更脏**。

---

## 三、Stage-2 早停下限

### 问题

Stage-2 的停止条件是 `val_nll < stage1_terminal_train_nll`，它只是**上限 600 轮**，
不是固定轮数。有些 run 极早停：

```
   7  s009 g003
   9  s009 searched
  20  s001 g001
  20  s004 g003
  22  s005 g000
 148  s001 g000        <- 第 6 短的直接跳到这里
```

7–22 轮和其余 49 个（≥148 轮）明显不成比例。

### 新增 `--stage2-min-epochs N`

下限，不是固定轮数：到达 N 轮之前不允许阈值 break；到 N 之后阈值规则照常生效。
`--stage2-fixed-epochs` 保留但与之互斥（给两个会报错）。

下限取 **100** 的理由：分布里 5 个 run 停在 7/9/20/20/22，第 6 短的直接跳到 148，
所以 **23–148 之间任意值效果完全相同**，100 离两边都最远。代价：54 个 run 总共只多
422 轮（原 19923 轮），约 2%。

### s009

同为**新 init 顺序**的三个 run 可直接比较：

| 配置 | Stage-2 轮数 | test acc | kappa | macro-F1 |
|---|---:|---:|---:|---:|
| 归档（旧代码，阈值停） | 9 | 81.94% | 0.7593 | 0.8205 |
| 固定 200 轮 | 200 | 85.76% | 0.8102 | 0.8544 |
| **下限 100 + 阈值停** | 100 | **86.11%** | — | — |

固定 200 轮（85.76%）与下限 100（86.11%）几乎一致 → **下限就够用，不需要固定轮数**。

---

## 四、Tier-1 定位与后续路线

45 个正式任务（003/005/006 × 20250901/02/03 × 5 operators）定义为 **Tier-1
global-family pilot**：每个 run 把**一个** family 同时用在三个频段上。它回答
「某种机制全局使用时 subject 是否偏好不同」，**不等价于**「不同 subject 是否在不同
frequency group 上偏好不同 mechanism」。后者才是 Phase A 的 `(f_L, f_M, f_H)`。

路线：

```
Tier-1（45 runs，global family）
  ├─ PROCEED / PROCEED_WEAK  → 有信号，进 Phase A 的 per-band 设计
  └─ WEAK_GLOBAL             → 只判 global separability weak
                               下一步跑 band-specific probe，不是放弃 V2
```

band-specific probe：一次放开一个频段，另两个钉在 anchor。`1 + 3×4 = 13` 个配置 ×
3 subjects × 3 seeds = **117 runs**。这一维与未来 `5³` 的 per-band family search 完全对齐。
设计细节见 `docs/operator_separability_v2.md` 第十节。

## 五、待决定

1. ~~`local_attention` 的 2.27× MACs：接受，还是改成更省的变体？~~ → 已返工到 **1.272×**。
2. 修正后的 searched 臂是否作为新基准？旧的 `retrain_lr1e3` 已保留未删。
