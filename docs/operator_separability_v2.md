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

判定规则是硬性的：

- `V_subject×operator <= V_seed` → 打印 **STOP，不要进入 NAS**
- `> V_seed` 且不同 subject 的最优 family 确实不同 → 打印 PROCEED

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

## 九、未做 / 待确认

1. **45 个正式任务未提交**，按约定停在 smoke 之后。
2. 第一轮网格：`003/005/006 × 20250901/20250902/20250903 × 5 operators = 45`。
   跑之前需要先决定 namespace（`run/outputs/operator_v2/bci42a/train_s003_seed20250901_operator_v2_dilated/`）。
3. ~~`local_attention` 的 2.27× MACs 是否接受~~ → 已返工到 **1.272×**，见第三节。
4. 是否要 `--read-session1`：pilot 阶段**保持关闭**，`val_best_acc` 作为唯一指标。
