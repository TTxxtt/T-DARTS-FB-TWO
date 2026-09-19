# Band-specific Mechanism Probe — 设计、实现与验证

**本阶段仍属于 Session-0 architecture-space diagnosis / screening；Session 1 始终保持关闭。
Stage2 cross-session evaluation 只会在最终搜索方法和架构选择规则完全冻结后执行。**

全局算子实验已结束：Matched-V2 与 Expressive-V2 共 90 runs 已冻结。本阶段不修改任何已有算子结构，
不新增 temporal operator，不调 `local_attention_e`。

## 一、这一阶段回答什么

前两代都把一个 family **同时**用在 Low/Mid/High 三个频段上。任何**频段特异**的偏好
——Low 想要门控而 High 想要别的——会在三个频段之间被平均掉，在这个设计里**结构上不可见**。
两代的判定（强全局主效应、无 subject×operator 交互）都与此一致。

本阶段问的是那个更窄的问题：

> $$\boxed{\text{不同 subject 在 Low/Mid/High 不同频带上，是否偏好不同 temporal mechanism？}}$$

## 二、设计：controlled single-band replacement

基准架构：

```
Low = dilated_e, Mid = dilated_e, High = dilated_e
```

每个配置只替换**一个**频段，其余两个钉在 `dilated_e`。

| 频段 | 可替换 family |
|---|---|
| Low | `dynamic_e` / `gated_e` / `band_gated_e` |
| Mid | `dynamic_e` / `gated_e` / `band_gated_e` |
| High | `dynamic_e` / `gated_e` / `band_gated_e` |

$$
9\ \text{configs} \times 3\ \text{subjects}\ (003/005/006) \times 3\ \text{seeds}\ (20250901\text{–}03) = 81\ \text{runs}
$$

**两个或三个频段同时替换会被拒绝**（`validate_band_families`）。同时改两个频段无法把效应归给其中任何一个。

### 候选 family 冻结

保留了 4 个（含锚）：

```text
dilated_e      锚
dynamic_e
gated_e
band_gated_e
```

`local_attention_e` **从后续搜索空间淘汰**，不参加本阶段。理由来自前两代的正式读数：验证性能最差，
且早过拟合率最高（这一代 5/9，最好轮次中位数 14）。它训练集能拟合满（9/9 到 100% 训练准确率），
所以这是**泛化失败**而不是调参没调好，band probe 不是能区分这两者的实验。
理由写在 `tdarts/operator_v2e_band.EXCLUDED_FAMILIES`，CLI 拒绝时会原样打印出来。

### all-dilated 锚：复用，不重训

锚的 3 subjects × 3 seeds = **9 runs 已存在于 Expressive-V2**（`run/outputs/operator_v2e/`），
本阶段**不重新训练**，直接配对读取。重训会把同一配置的第二份、不同 seed 的副本塞进配对比较里。

## 三、红线（已逐条验证）

| 红线 | 实现方式 | 验证 |
|---|---|---|
| Session1 默认且实际关闭 | `--read-session1` 默认关；job 脚本里不存在该 flag | smoke 产物 `session1_opened=false` |
| 提交脚本不出现 `--read-session1` | 源码断言 | `SubmitChainTests.test_the_driver_never_opens_session1` |
| `config.json` 记录 subject/seed/三个 band family/target_rf/`session1_opened=false`/`session1_test_size=null` | 顶层键，不在嵌套 `args` 里 | smoke 逐 run 审计，0 violation |
| `final_summary.json` 中 `test=null` | 同上 | 同上 |
| `metrics.jsonl` 无任何 test 字段 | 记录里从不写 test | 同上 |
| 不覆盖 Matched-V2 / Expressive-V2 | **对已有文件零改动**（见第五节） | `git status`：8 个文件全部为新增 |
| smoke 用单独 namespace | `run/outputs/operator_v2e_band_smoke/`，与正式 `operator_v2e_band/` 同级不同名 | 正式提交脚本的 \"leftover run dir\" 检查只看 `outputs/operator_v2e_band/` |
| 81 个跑完即停 | 本阶段不自动进入 Phase A NAS | — |

## 四、实现：8 个新文件，零改动已有文件

```
tdarts/operator_v2e_band.py            配置词汇表：9 个配置、slug 拼写与解析、单频段校验
tdarts/operator_v2e_band_network.py    OperatorV2EBandNet：每 band 一个 family
train_operator_v2e_band.py             训练入口
tools/analyze_operator_v2e_band.py     配对分析
tests/test_operator_v2e_band.py        54 个测试
run/bin/run_operator_v2e_band_job.sh   作业脚本
run/bin/submit_operator_v2e_band.sh    提交驱动（81 runs，可续跑）
docs/operator_separability_v2e_band.md 本文
```

**`OperatorV2Net` / `OperatorV2ENet` / `operator_v2_network.py` / `operator_v2e.py` 一行都没有改。**
上一代（Expressive-V2）对冻结文件做了两处带默认值的改动；这一代**一处都不需要**。
`git status` 里没有任何 modified 条目，`FBNAS/` 与 `code/FBNAS-master-main/codes_new/` 逐字节不变
（`tests/test_fbnas_compatibility.py` 22 项 OK）。

### 锚等价性：为什么可以直接和 Expressive-V2 的 `dilated_e` 配对

复用锚的前提是：**新入口在全锚配置下就是旧入口**。这一点是机械验证的，不是靠注释保证的：

1. `OperatorV2EBandNet({"Low":dilated_e, "Mid":dilated_e, "High":dilated_e})` 与
   `OperatorV2ENet("dilated_e")` 在同一个 seed 下 `state_dict()` **逐位相同**（36 个张量，0 个不匹配），
   同一输入的 logits **逐位相同**（max abs diff = 0.0）。
2. 端到端：两个入口跑同一 subject/seed/数据/轮数，`metrics.jsonl` 的 9 个训练字段 × 3 轮的
   **差异数 = 0**。这条现在是一个常驻测试（`DatasetBackedEquivalenceTests`）。

第 2 条覆盖第 1 条之后的一切——loader 顺序、优化器步、评估——而第 1 条覆盖初始权重。

> `OperatorV2EBandNet.__init__` 刻意**不调用**父类构造函数（父类只接一个 family），
> 而是直接建三个 cell 与 backbone。这份重复是这里唯一可能漂移的地方，所以由上面第 1 条测试守住：
> cell 顺序、builder、通道数、backbone 参数任一变化都会改变 RNG 抽样顺序，测试立刻变红。
> 反过来若先调用父类再用占位 family 重建两个 cell，初始权重会被**抽两次**，网络就不再是它自己
> `--seed` 描述的那个——这正是两个入口的 seeding 条款要防的那类 bug。

### 协议不变

SCB / BN / Swish / LogVar / classifier、Session-0 231/57 划分、optimizer、LR、
batch size、early stopping、checkpoint metric（`val_nll`）全部沿用 Expressive-V2。
**没有任何 family 有专属训练超参**——`test_the_protocol_constants_match_the_expressive_entry_point`
逐字段比对新旧入口的 `batch_size / lr / epochs / patience / rf / num_workers`。

RF 固定：`RF = 57`、`kernel = 15`、`dilation = 4`、15 个支撑位置。本阶段**不搜索 RF**。
守卫：每个配置的**三个 band** 的 support span 都断言等于 57。

张量契约不变：每 cell `[B,3,C,T] → [B,12,C,T]`，三 cell concat `[B,36,C,T]`，backbone 未变。

## 五、主分析：paired analysis

**不再把绝对 NLL 作为核心统计量。** 对每个相同的 subject 和 seed，与复用的 all-dilated 锚配对：

$$
\Delta_{s,b,f} = \log(NLL_{s,b,f}) - \log(NLL_{s,\text{all-dilated}})
$$

只有**一个**频段变了，锚又被减掉，所以 subject 难度、频段难度、seed 的初始权重抽样三者**同时抵消**。

```text
Δ < 0 = 替换这个 band 后优于全膨胀卷积
Δ = 0 = 无变化（锚自身）
Δ > 0 = 替换后变差
```

**`log(val_best_nll)` 是预先固定的 primary scale。** raw NLL 与 accuracy 辅助报告，不取代 primary。

每个 `subject × band × family` cell 有 3 个 paired Δ，输出：

```text
mean ΔlogNLL
sd
median
3 个 seed 的原始 paired Δ
改善方向一致次数（3/3、2/3、1/3）
```

### 报告的四个量（判定不再是单一 ratio 开关）

| 量 | 含义 |
|---|---|
| effect magnitude | 每个 cell 的 mean Δ，以及 band spread |
| winner diversity | 9 个 subject×band cell 的 winner 分布与各 subject 的 signature |
| winner stability | 每个 cell 的 winner 是否跨 3 个 seed 保持 |
| paired Δ direction consistency | 改善方向在 3 个 seed 里出现几次 |

### 三个条件

- **C1 band effect**：同一个 family 放在 Low/Mid/High，**spread 必须超过 band 内部的 seed 噪声**。
  至少 2 个 subject 满足，且其最好 band 跨 seed 稳定。没有这个分母，
  "Low 比 High 差"和"这两个 seed 不一样"是同一个数。
- **C2 subject-specific**：不同 subject 的 **`Low/Mid/High → winner family` signature 存在真实差异**
  （至少存在一个 band，不同 subject 的 winner 不同），且至少 2 个 subject 存在**跨 seed 稳定的非锚 winner**。

  差异**既允许 anchor vs non-anchor，也允许 non-anchor vs another non-anchor**。
  后者是本阶段最有价值的结果之一：

  ```
  003: Low → 动态卷积
  005: Low → 门控卷积
  006: Low → 动态卷积
  ```

  这里没有任何一边是膨胀卷积，但它显然就是 subject-specific mechanism preference。
  要求"必须一边是锚"会把这种结果漏掉。

  只数 distinct winner 不够——"所有 subject 的 Low 都要 dynamic_e"已经产生两个 distinct winner
  （`dynamic_e` 与锚），但那是**没有 subject 的 band 效应**；三个 signature 完全相同 → C2 false。

- **C3 stability**：**同时输出两个指标**，判定只看第二个。

  | 指标 | 是否判定 | 含义 |
  |---|---|---|
  | overall winner stability | 否，描述用 | 全部 cell 里 winner 跨 seed 稳定的比例 |
  | **non-anchor contested-cell stability** | **是** | 只看 replacement 赢下的 cell |

  只看 contested cell 的理由：锚赢的 cell 天然稳定，把 9 个"没结果"的 cell 平均进来，
  会让一个什么都没判的网格报出 100% 稳定性。
  保留 overall 的理由：**"某个 band 稳定不需要替换"本身是有价值的信息**，
  完全丢掉它会让报告看不见这件事。两个数故意说不同的话。

### 判决分支

C1/C2/C3 各答一个问题：

- **C1** 有没有 band 结构？（**任意 1 个** subject 的某种 family 的 band spread 超过 seed 噪声且最好 band 跨 seed 稳定）
- **C2** subject 之间是否真的不同？（`band→winner signature` 存在真实差异）
- **C3** 它是否扛得住 seed？（非锚 contested cell 全部跨 seed 稳定）

第四个问题**不是条件，是标签**：这个信号覆盖**几个**被试。

| 条件 | verdict |
|---|---|
| C1 ∧ C2 ∧ C3，且 **≥2** 个 subject 有稳定非锚偏好 | `PERSONALIZED_BAND_MECHANISM` |
| C1 ∧ C2 ∧ C3，但**只有 1 个** | `SINGLE_SUBJECT_PERSONALIZATION_SIGNAL` |
| C1 ∧ (¬C2 ∨ ¬C3) | `BAND_EFFECT_WITHOUT_STABLE_PERSONALIZATION` |
| ¬C1 ∧ C2 ∧ C3 | `SUBJECT_PREFERENCE_WITHOUT_BAND_STRUCTURE` |
| 否则 | `WEAK_PERSONALIZATION` |
| 网格不完整 | `NOT_COMPUTABLE`（拒绝在自由度不足时给结论） |

**为什么「几个被试」是标签而不是条件。** 3-subject pilot 太小，"信号只出现在一个被试身上"
是一个**证据单薄的真实发现**，不是"没有发现"。这正是本阶段要找的形状：

```
003: Low → 动态卷积，3/3 seeds 都改善
005: 全部维持膨胀卷积
006: 全部维持膨胀卷积
```

即 $003 \neq 005 = 006$。把它归进「无 personalization」会让一个有希望的方向被一句判死；
单独给它一个标签，下一步是**扩展 subject 验证**，而不是否定该方向。

> ⚠️ **这个阈值只算一次。** 早期版本把它同时写进 C1 和 C2（"至少 2 个 subject"），
> 结果是同一个数字在两个条件里各判一次，且会把你上面那个例子判成
> `BAND_EFFECT_WITHOUT_STABLE_PERSONALIZATION`——因为 005/006 连 band effect 都没有，C1 也过不了。
> 现在它只出现在标签映射里（`MULTI_SUBJECT_EVIDENCE`），C1/C2/C3 各管一件事。

**判定是纯函数**（`assess_band_probe`），可以脱离磁盘上的 run 在合成网格上测。
每种 verdict 携带自己的解释文案（`VERDICT_NOTICES`），打印时一起输出，
所以结论不会被单独摘出来当成另一句话使用。

### 方差分解

对配对差做 subject / band / subject×band / seed 的两路分解（复用 `tools/operator_anova.two_way_anova`，
自由度与 EMS 公式是已经被测过的那套）。配对已经把 subject 与 seed 主效应消掉，
所以它们不会**再次**被当成结构计入。

### 结果判读原则（已冻结）

真正支持后续 Phase A family search 的结果应该长这样：

```
s003 Low  -> dynamic 稳定改善
s003 Mid  -> dilated 最好
s003 High -> gated 稳定改善
...
```

即出现稳定的 **subject × band × mechanism preference**。

若所有 subject、所有 band 仍然主要偏好 `dilated_e`/`dynamic_e`，且不存在稳定的个体化
band-family preference，则记录为 **temporal mechanism personalization evidence weak**，
**不要继续强行做 temporal-family NAS**；下一阶段再考虑 spatial / electrode / graph 方向。

## 六、early-overfit 仅作诊断

每个 run 继续记录：

```text
best_epoch
stop_epoch
train_nll_at_best
final_train_acc
early_overfit_flag
```

但 `< 50 epoch` 只是基于前 90 runs 得到的 **post-hoc diagnostic threshold**：

- **不作为本阶段正式 selection criterion**；
- **不能用于删 run**；
- **全部 81 个 run 都必须进入正式分析**。

分析工具在打印这张表时会把这段说明一起打出来，所以它无法被单独摘出来当作筛选规则使用。

## 七、验证状态

```
python -m unittest discover -s tests -t .            →  427 tests   OK
python -m unittest tests.test_fbnas_compatibility    →   22 tests   OK
```

其中本阶段新增 54 个（`tests/test_operator_v2e_band.py`）：

- **词汇表**：9 个配置唯一；每个都是单频段改动；slug 往返；两频段被拒；
  `local_attention_e` 被拒且理由非空；畸形 slug 被拒；锚不是可替换 family。
- **形状契约**：9 个配置 + 锚都保持 `[B,3,C,T]→[B,12,C,T]` 与 `[B,36,E,T]`；
  backbone 参数名与冻结 backbone 完全一致；缺 band 被拒。
- **几何契约**：所有配置所有 band 的 support 恒为 `(15,4,57)`；每个可替换 family 单独再查一次。
- **锚等价性**：三个不同 seed 下 `state_dict` 逐位相同；logits 逐位相同；参数量相同；
  归档的 9 个锚 run 存在且 Session1 关闭、`test=null`、`target_rf=57`。
- **冻结守卫**：Matched 五个 family 的 params/MACs 仍等于归档整数；Expressive 五个 family 的
  params 仍等于归档整数；`canonical_candidates_all()` 无 `_e` 泄漏；本阶段不写冻结树。
- **入口护栏**：`--read-session1` 默认关；`load_subject_session(` 全文件只出现一次且在该 flag 的
  ±200 字符内；leaf 带 slug；`DEFAULT_RF=57`；`--arm` 与 `--output-root` 两道树守卫；
  超参逐字段与旧入口相同；config/summary 的红线键都在。
- **分析**：`assess_band_probe` 的四种判决 + 不完整网格拒绝；锚作为 Δ≡0 的候选且平局归锚；
  band effect 必须超过 seed 噪声；winner 跨 seed 翻转不算稳定；
  **C2 的两类差异各自有测试**——non-anchor vs non-anchor 判 true（且断言 Low 的三个 winner 里
  不含锚），全 subject 同 band 同 family 判 false（断言 `distinct_signatures == 1`）；
  **C3 双指标有测试**——全是锚 cell 的网格 overall 读 100% 而 C3 判 false，
  contested cell 翻转时两个数字分别读 `0/1` 与 `8/9`；
  **标签边界有测试**——单被试信号判 `SINGLE_SUBJECT_PERSONALIZATION_SIGNAL`（断言 C1/C2/C3 全 PASS、
  `needs_at_least == 1`、文案含"扩展 subject"）；再补一个被试就升为 `PERSONALIZED_BAND_MECHANISM`，
  两个网格的三个条件都成立，差别只在标签；单被试信号**不得**落入 `WEAK_PERSONALIZATION`；
  **完整 81-run 合成网格端到端跑通并还原植入的 ground truth**（`PERSONALIZED_BAND_MECHANISM`，
  `paired_runs=81`，三族分解均有残差自由度）。

### smoke

`subject003 + seed20250901 + RF57`，每个配置 3 epoch（CPU），覆盖 Low / Mid / High 三种替换 + 全锚：

| 配置 | 变化频段 | support | params | Session1 |
|---|---|---|---:|---|
| `all_dilated_e` | — | 57/57/57 | 18112 | closed |
| `low_dynamic_e` | Low | 57/57/57 | 19748 | closed |
| `mid_gated_e` | Mid | 57/57/57 | 18832 | closed |
| `high_band_gated_e` | High | 57/57/57 | 18199 | closed |

产物位置：`run/outputs/operator_v2e_band_smoke/bci42a/`（**不是** `run/outputs/operator_v2e_band/`）。
smoke 的 accuracy 不要用来分析——只跑 3 epoch、从随机初始化开始，接近随机是预期的。

## 八、提交

```bash
cd run/
bash bin/submit_operator_v2e_band.sh dry     # 预览 81 行，不写文件、不投任务
bash bin/submit_operator_v2e_band.sh         # 投一轮
bash bin/submit_operator_v2e_band.sh drain   # 轮询直到 81 个全部投出
```

- 网格：`SUBJECTS=2 4 5` × 9 configurations × `SEEDS=20250901 20250902 20250903`。
- 分区：`GPUFEE04 GPUFEE05 GPUFEE02`（GPUFEE06/08 按要求排除），**投之前逐分区查空卡**。
- ledger：`run/sh_log/operator_v2e_band_submitted.txt`，可续跑，不会重复投。
- 残留 run 目录会被跳过并记账，不会让 job 在跑起来之后才 `FileExistsError`。
- 单次 QOS 上限 30，超出部分重跑脚本即可。

分析：

```bash
PYTHONPATH=. python tools/analyze_operator_v2e_band.py
# 默认 --probe-root run/outputs/operator_v2e_band --anchor-root run/outputs/operator_v2e
```

## 九、跑完即停

81 个实验完成后停止，**不自动进入 Phase A NAS**。
下一步怎么走取决于上面的判决分支，而不是取决于"跑完了就继续"。
