# Operator Separability V2 — 审计、实现与 smoke 结果

本阶段**只做 standalone pilot 的基础设施**，不碰完整 NAS、不调 DARTS/alpha。
45 个正式任务**未提交**，等确认。

## 一、审计：哪些复用、哪些新增

先读了现有代码再动手，结论如下。

### 可直接复用（未改动一行）

| 模块 | 用途 |
|---|---|
| `tdarts/backbone.py` `TemporalBackbone` | SCB / LogVar / classifier，V2 原样使用 |
| `tdarts/search_data.py` `load_session0_search_split` | FBNAS 兼容的 231/57 Session-0 划分 |
| `tdarts/search_data.py` `load_subject_session` | Session 1（screening 下不调用） |
| `tdarts/temporal_ops.py` `resolve_kernel_dilation` / `_same_length_padding` / `count_parameters` | 几何解析与同长 padding |
| `run_layout.allocate` | run 目录与日志的落盘约定 |
| `train_retrain.py` 的 `train_epoch` / `evaluate` | 直接照搬，保证读数口径一致 |

### 新增（4 个文件）

| 文件 | 内容 |
|---|---|
| `tdarts/operator_v2.py` | 5 个算子 + 独立注册表 + 审计（params/MACs/support） |
| `tdarts/operator_v2_network.py` | `OperatorV2Cell` / `OperatorV2Net` |
| `train_operator_v2.py` | standalone 训练入口 |
| `tools/analyze_operator_v2.py` | 方差分解与判定规则 |
| `tests/test_operator_v2.py` | 21 个测试 |

### 刻意**不**改的地方

- **`tdarts/temporal_ops.OPERATOR_REGISTRY` 一个条目都没加。** 加进去会改变
  `canonical_candidates` → 改变搜索空间 → 让所有归档 search 和从它解出的
  genotype 失效。V2 用自己的注册表，搜索路径不 import 这个模块。
- `FBNAS/**`、`codes_new/**`、现有 DARTS / hierarchical / anchored / RF-only 逻辑、
  现有结果目录，全部未动。

## 二、已核对的事实（不是凭印象）

| 项 | 值 | 出处 |
|---|---|---|
| 每个 Low/Mid/High cell 输入 | `[B,3,C,T]` | `config.IN_CHANNELS=3` |
| 原 FBNAS 每 cell 输出 | `[B,12,C,T]` | `PATH_CHANNELS=6` × `NUM_PATHS=2` |
| 三 cell concat 后 | `[B,36,C,T]` | `NUM_FEAT = 3×2×6 = 36` |
| RF ladder | 15 / 29 / 57 / 113 | `RF_SPACE` |
| temporal kernel | `k=15`，dilation 1/2/4/8 | `BASE_KERNEL=15` |
| SCB / BN / Swish / LogVar / classifier | 未改 | `backbone.py` 原样 |

## 三、五个算子

pilot 固定 **RF=57** → 由 `k=15` 得 `dilation=4` → **15 个间隔为 4 的位置**，
跨度 `1+(15-1)*4 = 57`。五个算子都读**完全相同的这 15 个位置**。

| 算子 | 机制 | 内部宽度 |
|---|---|---|
| `dilated` | FBNAS 原始 anchor：单个膨胀卷积 | 12 |
| `gated` | `tanh(f(x)) * sigmoid(g(x))`，再 1×1 projection | 6（瓶颈） |
| `local_attention` | 只对这 15 个位置做 attention（unfold 同款 gather + softmax），**非 global** | head_dim 3 |
| `dynamic` | K=2 个 depthwise basis kernel，由全局池化特征产生 softmax gating 组合 | 2 basis |
| `band_gated` | 对 cell 内已有 3 个 filter-bank 通道做逐位置 sigmoid gating，再时间卷积 | 3 |

未加入：GCN / electrode attention / Mamba / full Transformer / SincConv。

**参数公平性是为公平性做了取舍的**，取舍点都写在 `operator_v2.AUDIT_NOTES` 里：

- `gated` 的瓶颈 6：宽度取 12 时 gating 需要两个时间卷积，参数量 1224（2.3× anchor）。
- `local_attention` 的 head_dim 3：见下面的「attention 为什么不贵」。
- `dynamic` 用 depthwise basis：两个 dense basis 加 projection 会是 1224。

### attention 为什么不贵（以及为什么 head_dim 只能是 3）

初版 attention 是 **2.27× MACs**，作为正式主 candidate 会引入「是机制好还是算力多」的混淆，
所以返工。改完是 **1.272×**。改的两点都是**精确等价**，不是近似、不是把机制削弱：

**一、head_dim 24 → 3。** `q`/`k` 是从 cell 的 3 个输入通道出的线性映射，于是

$$
q(x_t)\cdot k(x_{t+dk}) \;=\; x_t^\top M\,x_{t+dk},\qquad M = W_q^\top W_k
$$

`M` 是**任意 3×3 矩阵**——9 个自由度。head_dim 取 3 以上覆盖的是**完全同一组** score 函数，
原来的 24 是 8 倍冗余，不是 8 倍表达力。所以 `HEAD_DIM` 现在钉死为 `IN_CHANNELS`，
构造器会在两者不等时直接报错——这个等式是「head 可以这么窄」的唯一依据。

**二、输出投影挪到加权归约之前。** 投影是线性的，且 attention 权重只依赖 x、不依赖 value：

$$
W_o \sum_k a_k v_k \;=\; \sum_k a_k \left(W_o v_k\right)
$$

归约从 20 通道变成 12 通道，2.27× 的主要来源就在这里消失。**函数没变，变的是结合顺序。**

对拍验证（同权重，写了一份「投影在后」的参考实现）：max abs diff `6.4e-6`，输出尺度 `1.28`，
即相对 ~`5e-6` ≈ 50×float32 eps，就是重结合的舍入。这个断言进了测试。

**一个必须说明的边界：参数匹配的窗口注意力到不了 1.0×。**
softmax 加权归约里每个参数被用 K=15 次，而卷积权重每个位置只用 1 次。
所以 1.27× 不是「还没优化干净」，是这套机制的下限附近。这句话写进了 `AUDIT_NOTES`。

## 四、参数量 / MACs / support

`operator_audit(band="Low", target_rf=57)` 的实测输出（Low/Mid/High 三 band 数值相同，
因为 RF ladder 三 band 一致）：

| operator | params | vs 锚 | MACs | vs 锚 | support |
|---|---:|---:|---:|---:|---:|
| dilated | 540 | 1.000× | 11,880,000 | 1.000× | 15×4 = 57 |
| gated | 612 | 1.133× | 13,464,000 | 1.133× | 15×4 = 57 |
| local_attention | 462 | 0.856× | 15,114,000 | **1.272×** | 15×4 = 57 |
| dynamic | 512 | 0.948× | 11,616,006 | 0.978× | 15×4 = 57 |
| band_gated | 552 | 1.022× | 12,078,000 | 1.017× | 15×4 = 57 |

- **参数量全部落在 ±20% 内**（0.856 ~ 1.133×），无违例。
- **算力全部 ≤1.3×**（0.978 ~ 1.272×），无违例。
- **support 全部等于 57**，无越界偷看。
- 这两条带现在**同时**成立。初版 attention 是 2.27×，见上一节的返工说明。

### 一个必须说明的测量修正

MACs 计数器最初只钩 `nn.Conv2d` / `nn.Linear`，结果：
- `dynamic` 报了 **0.27×**（假象）——它的 basis 原本是裸 `F.conv2d`，hook 看不见；
- `local_attention` 报了 **0.93×**（假象）——scores 与加权的逐元素运算不计。

两处都修了：`dynamic` 的 basis 改成 `nn.Conv2d` 模块；两个算子通过 `extra_macs()`
显式申报 hook 看不到的逐元素算力。**修正前 attention 看起来比卷积还便宜，这是错的。**

## 五、单元测试

`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m unittest discover -s tests -t .`

```
Ran 298 tests   OK
```

其中 `tests/test_operator_v2.py` 24 个（新增部分）：

- **形状契约**：五个算子都是 `[B,3,C,T] → [B,12,C,T]`，C 与 T 不变
- **几何契约**：五个算子的 support 都恰好等于 RF57；三 band 均如此；1×1 的 gate/projection 不扩大窗口
- **公平性**：参数全部在 ±20% 内；anchor 恰为 `3×12×15=540`；**五个算子的 MACs 全部 ≤1.3×**
- **attention 不下穿 anchor**：另一头也断言了——计数器看不见的逐元素算力若不申报，attention 会被报成比卷积还便宜，让一个算力重的机制看起来是免费的
- **head 宽度不变量**：`HEAD_DIM == IN_CHANNELS`；不等则构造器报错
- **重结合等价性**：同权重对拍「投影在后」的参考实现，只允许舍入级差异；并把「把归约宽度换回 `VALUE_WIDTH` 就会重新出界」钉成断言
- **梯度**：五个算子 forward/backward 有限、无 NaN、梯度非零
- **attention 局部性**：断言 softmax 覆盖的是 15 个 support 位置，而不是整条时间轴
- **网络**：backbone 参数量不随算子变化；接受 loader 的 `[B,1,E,T,9]` 布局；输出是 log-probability
- **入口护栏**：`--read-session1` 默认关闭；`load_subject_session` 只出现一次且在该 flag 之下；叶子名必须带算子名；默认 RF=57

FBNAS 完整性单独跑 `tests/test_fbnas_compatibility.py`：22 项 OK，`FBNAS/` 下无新增文件。

## 六、Smoke 结果

`subject003 + seed20250901 + RF57`，每个算子 3 epoch（CPU）：

| operator | opParams | vs 锚 | opMACs | vs 锚 | support | valAcc | S1 |
|---|---:|---:|---:|---:|---:|---:|---|
| dilated | 540 | 1.000× | 11,880,000 | 1.000× | 15×4 | 33.33% | closed |
| gated | 612 | 1.133× | 13,464,000 | 1.133× | 15×4 | 40.35% | closed |
| local_attention | 462 | 0.856× | 15,114,000 | **1.272×** | 15×4 | 29.82% | closed |
| dynamic | 512 | 0.948× | 11,616,006 | 0.978× | 15×4 | 40.35% | closed |
| band_gated | 552 | 1.022× | 12,078,000 | 1.017× | 15×4 | 40.35% | closed |

**valAcc 接近随机是预期的**——只跑 3 epoch、从随机初始化开始，smoke 的目的是验证
管线能跑通，不是比较性能。

**产物位置**：`run/outputs/operator_v2_smoke/bci42a/`（**不是** `run/outputs/operator_v2/`）。

刻意分开：`train_operator_v2.py` 的 run 目录用 `exist_ok=False` 创建，smoke 若写进正式
namespace，正式任务会直接 `FileExistsError`；而且分析工具会把 3-epoch 的 smoke 当成一个
真实 run 读进方差分解，污染 residual。上表由**当前代码**重跑得到（`--epochs 3 --patience 3
--device cpu`），五个 run 的 operator params / MACs 与返工后的值逐项一致。

⚠️ **历史坑**：返工前留下的那批 smoke 产物（`local_attention` 记的是 504 params /
26,928,000 MACs / 2.267×）曾被误当成现行证据 —— 那是 attention 返工**之前**的旧代码跑出来的。
四个未受返工影响的 family 数值恰好相同，所以只看那四个是发现不了的。分析工具现在会对
这种产物打 `dev 2.27x` 标记。

逐项检查：

- ✅ forward shape 正确
- ✅ backward 正常，梯度非零
- ✅ 无 NaN
- ✅ 参数量 / MACs / support 如上报
- ✅ val 能正常跑（3 轮都产出读数）
- ✅ **Session 1 未读取**：五个 run 的 `metrics.jsonl` 都没有任何 `test` 字段，
  `config.session1_opened=false`、`session1_test_size=null`、`final_summary.test=null`

## 七、分析工具

`tools/analyze_operator_v2.py`，输出：逐 subject×operator 的 mean/std、主效应、
交互效应、seed/residual 方差，以及比值 `V_subject×operator / V_seed`。

**主分析指标是 `val_best_nll`**（`--metric` 的默认值），accuracy 作为辅助指标同时报告。
理由：validation 只有 57 个 trial，一个样本就对/错就移动 `1/57 ≈ 1.75pp`，在这个粒度上做
方差分解，量到的一大半是量化误差而不是信号；NLL 连续，且本来就是训练循环的选点依据。

判定三分支：

- `V_subject×operator > V_seed` 且不同 subject 的最优 family 确实不同 → **PROCEED**
- `> V_seed` 但所有 subject 的 argmax 是同一个 family → **PROCEED_WEAK**（无法按 subject 特化）
- `V_subject×operator <= V_seed` → **WEAK_GLOBAL**，见下

### 为什么第三条不再打印「STOP，不要进入 NAS」

原来的措辞把这件事写成了对 NAS 的**终审否决**，这是错的，已删除。

每个 run 把**同一个** family 同时用在 Low/Mid/High 上。于是任何**频段特异**的偏好——
比如 Low 想要 attention、High 想要别的——会在三个频段之间**被平均掉**，在这个设计里
**结构上就看不见**。所以 WEAK_GLOBAL 支持的结论只有一条：

> 没有 family 能够**全局**分离。

它**不能**证明 per-band family search 没有价值。原规则把这两件事混为一谈，
会让一个本可以救回来的方向被一句话判死。

**若 Tier-1 不通过，下一步是 band-specific probe，不是放弃 V2。** 见第十节。

已验证的行为：

- 空目录 → `no runs found`，不崩
- 只有 1 subject × 1 seed × 5 operators 的 smoke 数据 → 正确报出"设计不平衡"、
  列出薄 cell，并给出 **`verdict: not computable -- the design has no residual degrees of freedom`**。
  它**拒绝在自由度不足时硬编一个结论**，这是它最该有的性质。

## 八、必须先解决的一个问题：seed 流不一致

写 `train_operator_v2.py` 时确认了一件影响整个实验设计的事：

**构建网络会消耗全局 torch RNG**（每个 `nn.Conv2d.__init__` 都取数），
`load_genotype` / `extract_genotype` / `path_structure_keys` / `duplicate_structure_bands`
四个函数也都会（它们构建候选算子池）。

于是 `set_seed` 相对于"基因型解析"的位置会改变初始权重。实测：
同一个 genotype、同一个 seed，`set_seed` 在解析之前 vs 之后，初始权重最大差
**0.285**；s009 在 epoch 1 的 `train_nll` 因此是 **2.78 vs 2.19**。

这对 V2 的直接意义：**`seed` 是这次实验唯一的噪声标尺**。若 seed 的效应里混进了
"RNG 流被前面的构造调用挪动了多少"，`V_seed` 就不再是干净的噪声估计，
`V_subject×operator / V_seed` 这个判定比值也就不可信。

`train_operator_v2.py` 因此把 `set_seed` 放在**紧贴建模之前**，中间不夹任何消耗
RNG 的调用。该约束写在模块 docstring 里，并有测试看着。

（同一问题在 `train_retrain.py` 里已修：`--search-dir` 与 `--genotype-json` 两条
路径现在会给出相同的初始权重。修之前 searched 臂是旧顺序、随机臂是新顺序，
两臂的 "seed 20250901" 不对应同一组初始权重。）

## 九、Tier-1 定位与网格

这一轮 45 个任务定义为 **Tier-1 global-family pilot**：每个 run 把**一个** family 同时用于
三个频段，回答的是「某一种机制全局使用时，subject 是否表现出不同偏好」。

网格：

```
subjects 003 / 005 / 006  ×  seeds 20250901 / 20250902 / 20250903  ×  5 operators  =  45
```

namespace：`run/outputs/operator_v2/bci42a/train_s003_seed20250901_operator_v2_dilated/`。

`--read-session1` 保持**关闭**，`val_best_nll` 为唯一选点指标（accuracy 辅助报告）。

## 十、后续：band-specific probe（Tier-1 不通过时）

Tier-1 说的是「一个 family 打满三个频段」。真正要做的是 per-band 选择，即
`(f_L, f_M, f_H)` 三个位置各自挑 family，共 `5³ = 125` 种组合。这两件事不等价，
所以要有一个能看见**频段特异**信号的中间实验。

设计：**一次只放开一个频段**，另外两个钉在 anchor 上。

```
Baseline:   Low=dilated, Mid=dilated, High=dilated

Low probe:  Low ∈ {gated, local_attention, dynamic, band_gated}
            Mid=dilated, High=dilated

Mid probe:  Low=dilated
            Mid ∈ {gated, local_attention, dynamic, band_gated}
            High=dilated

High probe: Low=dilated, Mid=dilated
            High ∈ {gated, local_attention, dynamic, band_gated}
```

配置数不是 `3×4 = 12` 而是 `1 + 3×4 = 13`——baseline 三条 probe 共用一条，只跑一次。

$$
13\ \text{configs} \times 3\ \text{subjects} \times 3\ \text{seeds} = 117\ \text{runs}
$$

这 117 个 run 的每一维都与未来 `5³` 的 per-band family search 对齐：probe 测的就是
搜索要利用的那一维。判据仍然是 `V_subject×operator / V_seed`，但现在是**逐频段**算的。

**触发条件**：Tier-1 打印 `WEAK_GLOBAL`。此时只能判定 global family separability weak，
**不得**据此否定 per-band family search。

## 十一、未做 / 待确认

1. `local_attention` 的 2.27× MACs → 已返工到 **1.272×**，见第三节。
2. `--read-session1`：pilot 阶段**保持关闭**。
3. Tier-1 若通过（PROCEED / PROCEED_WEAK）→ 直接进 Phase A 的 per-band 设计；
   若 WEAK_GLOBAL → 按第十节跑 117 个 band-specific probe，**不放弃 V2**。

## 十二、结果已冻结：本世代改称 Matched-V2

45 个 run 已跑完并冻结在 `run/outputs/operator_v2/`，**不再修改、不覆盖**。

**判定：`WEAK_GLOBAL`（raw `val_best_nll` 尺度，ratio 0.833）。** 该读数维持原样。

```
V_operator / V_seed          = 4.754     机制确实拉开了
V_subject×operator / V_seed  = 0.833     但没有按受试者特化
```

9/9 个 subject×seed 的排序完全一致（top-2 恒为 `{dilated, dynamic}`，
bottom-2 恒为 `{local_attention, band_gated}`），三个受试者的 argmax 全部跨 seed 翻转。

**关键推论：这一批说明的是「有明显机制差异，但精简版 operator 的优劣是全局性的」，
不等于「新 operator 没区别」。** 因此它**不构成**对 per-band family search 的否定。

⚠️ **两代尺度不同。** 本世代的记录用 **raw** `val_best_nll`；第二世代
（Expressive-V2，放开了参数匹配）预注册 **`log(val_best_nll)`**。同一份数据 raw 下
ratio 0.833、log 下 1.187 —— 结论跨过阈值，这正是判据从「单一 `>1` 开关」改掉的直接原因。
两代的方差分量**不可数值比较**。

第二世代见 **`docs/operator_separability_v2e.md`**。
