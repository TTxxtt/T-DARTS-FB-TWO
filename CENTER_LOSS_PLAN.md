# 中心损失移植方案

把 `code/FBNAS-master-main/codes_new/centralRepo/` 里已实现的中心损失（+ 拉姆达调度）
移植进 `T-DARTS-FB-TWO` 的**重训路径**。

只改代码、不动 `FBNAS/`。本文只写方案，尚未改动任何文件。

## 范围

| 位置 | 是否改 |
|---|---|
| `train_retrain.py`（Stage 1 + Stage 2） | **改** |
| `tdarts/center_loss.py` | **新增** |
| `train_search.py` / `tdarts/search.py` / `tdarts/architect.py` | 不改 |
| 任何 `FBNAS/` 下的文件 | 不改（52 个 blob 的 manifest 必须保持逐字节一致） |
| `code/FBNAS-master-main/codes_new/` | 只读 |

不改搜索路径的理由：`CenterlossFunc` 是自定义 autograd，二阶导数不成立，
而 DARTS 的架构梯度依赖对 `w` 的微分。搜索阶段引入会静默破坏 alpha 的梯度。

## 参考实现的行为（照抄，含 5 个易错点）

### 易错点 1：中心梯度不是 autograd 能给出来的

`CenterlossFunc.backward` 里：

```python
counts = centers.new_ones(centers.size(0))          # 注意是从 1 开始，不是 0
counts = counts.scatter_add_(0, label.long(), ones) # 所以 counts[c] = 1 + n_c
grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
grad_centers = grad_centers / counts.view(-1, 1)
return -grad_output * diff / batch_size, None, grad_centers / batch_size, None
```

即 `dL/dc_c = n_c·(c_c − mean_c) / ((1+n_c)·B)`。

若用普通 autograd 写 `L = Σ‖f_i − c_{y_i}‖²/(2B)`，得到的是 `n_c(c_c − mean_c)/B`，
差一个 `(1+n_c)` 因子。**这是 Wen 等人原始 center loss 的写法，必须用自定义 Function 复刻，
不能图省事用 autograd 重写。**

顺带：`counts` 从 1 起算还有个副作用——某个类在 batch 里完全缺席时 `grad=0/1=0`，
不会 0/0=NaN。batch=16、4 类时这类空 batch 并不罕见，所以这个 +1 是必需的，不是笔误。

feature 的梯度两边一致（都是 `(f−c)/B`），只有中心那一路不同。

### 易错点 2：`forward` 参数顺序是反的

`CenterLoss.forward(self, label, feat)` —— 调用处必须写 `centerloss(label, feature)`，
不是 `(feature, label)`。抄的时候按签名走。

### 易错点 3：中心初始化是确定性的，但与 feat_dim 耦合

`CenterLoss.py` 顶部是 `from numpy import random`，所以：

```python
random.seed(19981127)                    # np.random.seed
centers = random.randn(num_classes, feat_dim)
```

是**固定种子**的 numpy 抽样，不是 `np.random.rand`（那是均匀分布）。两点后果：

- 同样 feat_dim 下每次运行中心初值完全一致，可复现；
- `randn(4, feat_dim)` 的抽样顺序随 feat_dim 变化，**换网络就换中心初值**，
  不存在"同一组中心"的说法。

移植时用局部 `np.random.RandomState(19981127)` 取同样的数，避免 `np.random.seed`
写全局状态——否则会污染 `set_seed()` 之后所有 numpy 抽样（本仓库的 DataLoader
用 torch Generator，暂时没冲突，但这是个定时炸弹）。

另外参考实现 `torch.from_numpy(centers)` 出来是 **float64**，我们的模型是 float32。
移植时 `.float()` 对齐 dtype（与参考有微小数值差异，是可接受的偏离，要记录）。

### 易错点 4：延迟模式下中心损失是"单独 backward"，不是"不加"

`baseModel.py` 的 `trainOneEpoch`：

```python
if centerLossApplyToTotal:
    total_loss = loss + Lambda * closs
    total_loss.backward()
else:
    total_loss = loss
    total_loss.backward()
    detached_closs = centerloss(label, feature.detach())   # 中心照常更新
    detached_closs.backward()
    closs = detached_closs
optimizer.step()
optimzer4center.step()
```

`centerLossApplyToTotal=False` 时，中心**仍然在更新**，只是梯度不从中心回流到网络。
这和"这一轮不算中心损失"是两回事，必须照搬。

### 易错点 5：分类损失先除了 batch size

```python
loss = lossFn(output, label)      # lossFn 是 reduction='sum'
loss = loss / data.shape[0]
```

本仓库 `criterion = nn.NLLLoss()` 已经是 mean，**这一行不要重复除**。

### 调度

- `get_lambda(epoch, max_epoch) = 2/(1+exp(-10p)) − 1`：在 `codes_new` 里**定义了但从未被调用**（死代码），
  所以 Stage 1 实际用的是**常数** `Lambda`（默认 0.0005）。移植时按现状照搬，不要"顺手接上"。
- Stage 2 用 `_get_stage2_center_loss_weight`：
  - `smooth_decay_enable=False`（默认 True 的反面）→ 返回 0.0，等于关掉
  - `decay_epochs <= 0` → 返回 0.0
  - `linear`：`start_lambda · max(0, 1 − stage2_epoch/decay_epochs)`
  - `cosine`：`start_lambda · 0.5 · (1 + cos(π · min(1, max(0, epoch/decay_epochs))))`
  - 其它 mode → `raise ValueError`
- 延迟启用（默认关）：`stage2CenterLossDelayedEnable` + `stage2CenterLossStartEpoch=100`，
  且**只在非 smooth decay 分支**生效。这段逻辑分支很绕（`baseModel.py:494-530`），移植时逐行对照。

## 挂载点（`train_retrain.py`，行号为当前版本）

| 行 | 现状 | 改动 |
|---|---|---|
| 146 `train_epoch` | `logits, _ = model(x)` | `logits, features = model(x)`，把 features 和 lambda 透传 |
| 152-157 | `optimizer.zero_grad` / `loss.backward` / `step` | 加 `center_optimizer.zero_grad()`，按 `apply_to_total` 走两条分支，末尾 `center_optimizer.step()` |
| 547-548 | `criterion` + `Adam` | 之后构造 `CenterLoss` + `SGD(lr=0.01)` |
| 594 | Stage-1 `record` | 加 `center_lambda`、`center_loss` |
| 604-615 | 存 `best.pt` | 加 `center_loss_state_dict` + `center_optimizer_state_dict` |
| 642-644 | 恢复 `best.pt` | 同步恢复两样，否则 Stage 2 从随机中心重新开始 |
| 684-690 | Stage-2 `record` | 同 594 |
| 711 | 存 `stage2_final.pt` | 同 604 |
| 713-727 | `final_summary.json` | 加 `center_loss` 配置块（enable / lambda / decay_mode / decay_epochs / start_lambda） |

`evaluate()`（165 行）**不加**中心损失——它只算 NLL/acc，中心损失不参与选点。
这点和参考一致：`baseModel.predict()` 里也没有中心损失。

feature 从 `backbone.py:283-307` 出来：`features = flatten(LogVar输出, 1)`，
是 classifier **之前**的那一层，和 FBNAS 的 `(c, f)` 约定一致。
`feat_dim` 按 `_inferCenterLossFeatDim` 的做法实测一次前向得到，不硬编码。

## 参数（默认值取 `codes_new` 现状）

按已确认的"默认开启"：

```
--center-loss-enable / --no-center-loss        默认 ON
--center-loss-lambda            0.0005         Stage 1 常数
--center-loss-lr                0.01           中心的独立 SGD
--stage2-center-loss-smooth-decay-enable       True
--stage2-center-loss-decay-epochs      50
--stage2-center-loss-decay-mode       linear | cosine，默认 linear
--stage2-center-loss-start-lambda     0.0005
--stage2-center-loss-delayed-enable   False
--stage2-center-loss-start-epoch      100
```

## 与 Stage 2 停止条件的相互作用（重要）

Stage 2 的停止规则是 `val["nll"] < stage1_terminal_train_nll`，而阈值本身
取自 Stage 1 终止轮的 `train_nll_eval`。

开了中心损失之后，两条曲线都会变，**阈值也跟着变**。所以：

- 开/关两臂的 Stage-2 轮数**不可比**，只能比 Session-1 的 test 读数；
- 现有 s009 那个只有 9 轮的退化 Stage-2（阈值过低导致）在这个改动下可能变化，
  但这属于副作用，不是本次目的，不要顺手去改停止规则。

## 风险

1. **默认开启会改变旧命令语义**。已在跑的 45 个 random 基线任务会读到新代码——
   经查队列已空、45 个叶子全部落盘，没有排队任务会受影响，但这条约束对以后仍然成立。
2. `np.random.seed` 全局污染 → 用 `RandomState` 局部化规避。
3. float64 → float32 的 dtype 偏离（见易错点 3），需在文档里写明。
4. 自定义 Function 不支持二次求导 / `torch.compile`，只用于重训不影响。

## 测试

在 `tests/` 下新增，并跑全量（当前 267 个用例全绿）：

- `CenterlossFunc` 的前向值和反向梯度和参考实现逐元素一致（把参考代码抄进测试做对照）；
- 中心梯度**不等于**普通 autograd 的结果（把 `(1+n_c)` 这个差异钉成断言）；
- 某个类在 batch 中缺席时不产生 NaN；
- `decay_mode` 取 `linear` / `cosine` / 非法值三种情况；
- `smooth_decay_enable=False` 时 Stage-2 lambda 恒为 0；
- `--no-center-loss` 时 `train_retrain.py` 的数值与改动前**完全一致**（回归保护）；
- 跑完检查 `FBNAS/` 无任何新增或修改文件（`_MANIFEST.json` 52 个 blob 校验）。

## 执行顺序

1. 写 `tdarts/center_loss.py` + 单测，先不动 `train_retrain.py`；
2. 加 CLI 参数与 `train_epoch` 接线，`--no-center-loss` 路径必须与现状逐位一致；
3. 跑全量测试 + FBNAS manifest 校验；
4. 提交 3 个受试者的小规模对照（开/关各一次），确认 Session-1 读数能出来、
   `center_lambda` 按预期衰减，再决定是否铺 9 个受试者。

## 需要确认

1. 第 4 步的对照只跑 3 个受试者（s001/s005/s009）够不够，还是直接铺满 9 个？
2. 中心的 `--center-loss-lr 0.01` 是参考实现的取值，它配的是参考实现的 batch 划分。
   我们 batch=16、231 个训练样本，是否原样照搬？
