# Expressive-V2 — 审计、设计与预注册判据

第二世代 temporal operator families。**代码完成、smoke 通过、45 个正式任务按约定未提交。**

## 一、为什么要有第二世代

Matched-V2（`docs/operator_separability_v2.md`，结果冻结在 `run/outputs/operator_v2/`）的
45 个 run 给出：

```
V_operator / V_seed          = 4.754     机制确实拉开了
V_subject×operator / V_seed  = 0.833     但没有按受试者特化
```

而且 9/9 个 subject×seed 的大排序完全一致 —— top-2 恒为 `{dilated, dynamic}`，
bottom-2 恒为 `{local_attention, band_gated}`。

**一个从不重排的全局排序，与「预算把机制压平了」至少同样吻合，而不只是「没有受试者偏好不同机制」。**
Matched 世代的三个妥协是明摆着的：`gated` 的 gate 只在 6 宽上跑、`local_attention` 的 head 被钉成 3 维、
`dynamic` 只有 2 个 depthwise basis 且带一个可被吸收的 proj。

所以公平性的定义改了：

> **公平 = 相同的张量契约、相同的 RF/支撑位置、相同的 backbone、相同的训练协议。**
> **参数量自由，但必须完整上报，并用容量对照来回答「是不是只是参数更多」。**

## 二、必须仍然成立的不变量

| 项 | 要求 |
|---|---|
| 张量 | `[B, 3, C, T] -> [B, 12, C, T]`，C、T 不变 |
| 时间支撑 | RF57，kernel 15，dilation 4，**完全相同的 15 个位置** |
| downstream | FBNAS backbone 一字不改 |
| 训练协议 | 与 Matched 世代逐项相同 |

**任何容量增加都不得靠加宽窗口买。** 两个 wide 对照也断言 span == 57 —— 这是测试里
唯一一条防止「对照悄悄变成另一个实验」的断言。

## 三、容量表（RF57，probe `[1,3,22,1000]`）

`operator_capacity_audit(band="Low", target_rf=57)` 实测：

| family | role | params | ×anchor | MACs | ×anchor | 申报逐元素 | flag |
|---|---|---:|---:|---:|---:|---:|---|
| `dilated_e` | candidate | 540 | 1.000 | 11,880,000 | 1.000 | 0 | |
| `gated_e` | candidate | 1260 | 2.333 | 28,248,000 | 2.378 | 528,000 | |
| `local_attention_e` | candidate | 624 | 1.156 | 21,384,000 | 1.800 | 7,920,000 | |
| `dynamic_e` | candidate | 2176 | 4.030 | 48,576,012 | 4.089 | 1,056,000 | |
| `band_gated_e` | candidate | 627 | 1.161 | 13,530,000 | 1.139 | 66,000 | |
| `wide_dilated_2p5_e` | control | 1368 | 2.533 | 30,096,000 | 2.533 | 0 | |
| `wide_dilated_5_e` | control | 2736 | 5.067 | 60,192,000 | 5.067 | 0 | `control` |

`dynamic_e` 的 MACs 比 `4 × 11,880,000 + 1,056,000` 多 12，那是 gate 的 `Linear(3→4)`
在 batch 1 下被计数器看到的 12 次乘加。精确记下，不四舍五入。

**超过 5× 的只有 `wide_dilated_5_e`，且是设计如此。** 最接近的候选是 `dynamic_e`（≈4.0×）。

**5× 上限是 advisory、只报告。** `operator_capacity_audit` 打 flag 但**从不**把它变成失败，
审计记录里**没有** `within_tolerance` 键 —— Matched 审计有那个键是因为那里 ±20% 是**要求**，
这里不是。有测试专门断言这个键**不存在**（而不是断言它为真），这样没人能悄悄把预算规则接回来。

## 四、五个 family 的设计要点

三个必须写下来的取舍（完整版在 `tdarts/operator_v2e.CAPACITY_NOTES`）：

1. **`dynamic_e` 不带输出投影。** dense basis 下
   `proj(Σᵢ wᵢBᵢ(x)) = Σᵢ wᵢ(proj∘Bᵢ)(x)` 可被吸收，加它只多 144 参数和 3.17M MACs 而
   **不增加函数类**。Matched 版是 depthwise，proj 真的在混通道，所以那里必须留。
   这正是「表达性世代」要停止背负的假预算。
2. **`local_attention_e` 的 head_dim 不冗余，但 GELU 是承重的。** Matched 版把 `HEAD_DIM`
   钉成 `in_channels=3`，因为 3 通道的双线性 score 只有 3×3 = 9 个自由度。有了非线性 `3→12`
   embedding，q/k 读的是 12 通道，6 宽 head 不再秩冗余，2 个 head 给出两组独立的双线性型。
   **若去掉 GELU，`q∘embed` 退化回 3 通道的线性映射，这个 family 就退回 Matched 版的
   score 类别** —— 所以「激活不是 Identity」是一条测试，不是注释。
3. **测量不对称是刻意的。** `gated_e` / `band_gated_e` 申报了逐元素乘法，Matched 的
   `gated` / `band_gated` 没有。审计表同时打印 `macs` 与 `macs_hook_only` 让不对称**在表里可见**，
   但**不回填 Matched 的数字**（那会移动已记录的比值）。跨世代的 MACs 因此不可逐字节比较，
   参数与 MACs 的**比值**仍然可比。

### 两个 wide 对照怎么用

`wide_dilated_2p5_e` / `wide_dilated_5_e` 保留 anchor 的机制与 RF，只加宽通道。如果之后
出现「attention 赢了」这类结果，它们回答的就是那句必答题：

> 是机制，还是参数更多？

**它们不是 NAS candidate，也不在这个网格里。** 结构性隔离：它们在
`WIDE_CONTROL_REGISTRY`，`E_OPERATOR_NAMES` 里没有它们；`build_e_operator("wide_dilated_5_e")`
抛错；CLI 用互斥的 `--capacity-control` 单独开出。有测试断言两个 registry 不相交。

## 五、两代尺度不同（任何并列处必须注明）

| | 世代 | 记录尺度 | 已记录的判定 |
|---|---|---|---|
| `run/outputs/operator_v2/` | Matched | **raw** `val_best_nll` | `WEAK_GLOBAL`（**维持原样**） |
| `run/outputs/operator_v2e/` | Expressive | **`log(val_best_nll)`** | 待跑 |

**两代的方差分量不可数值比较，只有定性方向可比。**

log 不是在两个尺度里挑一个更顺眼的：Matched 数据的残差 sd 随均值近似成比例
（003 / 005 / 006 的 sd/mean 分别是 0.062 / 0.091 / 0.102），这是乘性噪声的特征，log 是它的
自然尺度。而同一份 Matched 数据在 raw 下 ratio 是 0.833、log 下是 1.187 —— **结论跨过阈值**。
这正说明「一条 `>1` 的机械开关」不该继续当判据。

## 六、世代隔离：默认禁止跨世代 pooling

不靠命名或目录，而是**从 operator 名派生 generation**（`_e` 后缀 ⇒ Expressive），
因此一个落错目录的 run 仍会被正确归类。

- 一次分析里出现多于一个 generation 且**未传** `--combine-generations`
  → **报错并 exit 2**，逐条列出违规 run。不静默丢弃，不静默合并。
- 传了 `--combine-generations` → 允许，但必须显式给 `--metric`（两代主尺度不同，
  不允许走默认），否则同样 exit 2。
- 冻结工具 `tools/analyze_operator_v2.py` 带**同一条守卫**。它在冻结 root 上永不触发
  （那里只有 Matched 算子），所以**可证明不改变已记录的读数** —— 这条已用「重构前后
  逐字节 diff 输出」验证过，不是靠检查代码得出的。

## 七、预注册判据：三条，不是一个阈值

E 工具（`tools/analyze_operator_v2e.py`）主尺度 `log(val_best_nll)`，raw NLL 与
rank stability 作为 sensitivity。

**关键性质：log 严格单调，所以逐 seed 的 operator 排序与逐 subject 的 argmax 在两把尺子上
完全相同。** 变换只能移动方差分解（C1），**动不了 C2 和 C3** —— 所以决策由后两条承担。

- **C1 交互够大吗**：`V_S×O / V_seed >= 1.0` **或** `p(interaction) <= 0.10`（scipy 可用时）。
  刻意是两个软信号的 OR，单条不决定结论。
- **C2 不同 subject 真的不同吗**：`>= 2` 个互异 winner，**且** `>= 2` 个 subject 的
  best−worst 间距超过**它自己的** cell 内 seed sd。后半条挡住「两个 subject 在噪声内换位」。
- **C3 winner 跨 seed 稳定吗**：没有 subject 的 per-seed argmax 发生变化，**且**
  逐 subject Kendall tau 最小值 `>= 0.5`。

映射：`C1∧C2∧C3 → PROCEED`；`C1∧(¬C2∨¬C3) → AMBIGUOUS`；`¬C1 → WEAK_GLOBAL`。
三条**各自**打印数字与是否达标。`assess()` 是纯函数，有单测覆盖 —— 其中一条专门验证
**ratio 很大但 winner 不稳时不得判 PROCEED**，即它不是比值的纯函数。

`AMBIGUOUS` 的含义：交互是真的，但当前不支持按受试者特化。它与 `WEAK_GLOBAL` 的区别是
**可被更多 seed 分开** —— 不稳定会随 seed 收敛，真正的不存在不会。

## 八、测试

```
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m unittest discover -s tests -t .
Ran 366 tests   OK
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m unittest tests.test_fbnas_compatibility
Ran 22 tests   OK
```

`tests/test_operator_v2e.py` 67 条，其中：

- **形状 / 几何契约**：五个 family × 三个 band 的 support 恒为 `(15, 4, 57)`；
  1×1 的 gate/projection 不扩大窗口；**两个 wide 对照也是 span 57**；
  `local_attention_e._patches` 与冻结版 `SparseDilatedLocalAttention._patches`
  在**同一输入上逐元素相等**。
- **精确算术**：上表七个 (params, MACs, 申报量) 三元组逐一钉死 —— 架构漂移必须是测试红，
  而不是让上报比值悄悄移动。
- **容量不变量**：审计记录不含 `within_tolerance`；`exceeds_advisory` 只对
  `wide_dilated_5_e` 为真；attention 的 `macs > macs_hook_only`（防止「attention 看起来
  比卷积便宜」那类漏计）；attention 确实是 2 heads × 6（断言 score 张量的 head 维，
  防有人悄悄塌成 1×12）。
- **跨世代护栏**：两代名字不相交且都是 5；`_e` 后缀覆盖两个对照
  （早期版本漏了后缀，会让对照被归为 Matched 从而绕过 pooling 守卫 —— 就是这么发现的）；
  五个冻结 family 的 params/MACs 仍等于归档整数，且与磁盘上 45 个 `final_summary.json` 对拍；
  `temporal_ops.canonical_candidates_all()` 每 band 仍 14、总数仍 42
  （证明 E 从未泄漏进 DARTS 搜索池）。

## 九、Smoke

`subject003 + seed20250901 + RF57`，2 epoch（CPU），产物在
**`run/outputs/operator_v2e_smoke/`**（**不是** `run/outputs/operator_v2e/` ——
提交脚本把任何已存在的 run 目录当作已占位，smoke 写进去会永久毒化那个格子）。

| operator | opParams | opMACs | s1 | test | valNLL |
|---|---:|---:|---|---|---:|
| dilated_e | 540 | 11,880,000 | closed | null | 1.3766 |
| gated_e | 1260 | 28,248,000 | closed | null | 1.3139 |
| local_attention_e | 624 | 21,384,000 | closed | null | 2.0536 |
| dynamic_e | 2176 | 48,576,012 | closed | null | 1.6679 |
| band_gated_e | 627 | 13,530,000 | closed | null | 1.1452 |
| wide_dilated_5_e | 2736 | 60,192,000 | closed | null | — |

五个都 forward/backward 正常、无 NaN、params/MACs/support 与容量表一致、
`metrics.jsonl` 无任何 `test` 字段。另单独跑了一次 `wide_dilated_5_e` 对照，
`config.json` 记 `operator_role=capacity_control`、`operator_exceeds_advisory=true`。

**valNLL 不要拿来分析** —— 只 2 epoch、从随机初始化开始。

## 十、未做 / 待确认

1. **45 个正式任务未提交**，按约定停在 smoke 之后。
2. 网格：`003/005/006 × 20250901/02/03 × 5 families = 45`；
   `run/bin/submit_operator_v2e.sh`（可续跑，ledger `sh_log/opv2e_submitted.txt`）。
3. 两个 wide 对照**不在这个网格里**，需要时用 job 脚本的 `CAPACITY_CONTROL=` 单独开。
4. 跑完 45 个之后**先看结果再决定**是否进入 band-specific probe
   （见 `docs/operator_separability_v2.md` 第十节）。
