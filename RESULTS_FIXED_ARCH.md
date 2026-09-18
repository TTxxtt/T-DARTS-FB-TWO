# 固定架构（按受试者手工指定）从头训练：Stage-1 + Stage-2 结果

> 最后更新：2026-09-18
> 状态：**全部完成**（Stage-1 9/9，Stage-2 9/9）
>
> **结论速览**：固定架构在 Session-1 测试集上 **78.63%**（线 B 口径），与两条搜索臂
> （层级 NAS 78.33%、RF-only 78.73%）**处在同一水平**——手工指定的这张表没有带来优势。
> 唯一的例外是受试者 006（70.83%，比 RF-only 高 7.7 个百分点）。

## 1. 实验设置

- 数据集：BCI-IV-2a，4 类运动想象，9 个受试者（001–009）
- 数据划分：Session 0（231 train / 57 val），Session 1（288）作为**留出测试集**
- Seed：`20190821`（单 seed，FBNAS 上游所用 seed）
- 架构来源：用户按受试者手工给出的固定双路径表（见第 2 节），**不经任何搜索**
- 与搜索臂的区别：`hier` / `rf` 两条臂用的是搜索出来的 genotype、seed 为 20250901–03；
  本臂是**单 seed、非配对**，与它们比较时只能看量级，不能做配对检验

## 2. 固定架构表

每个 band 两条固定路径，均为 `dilated` 算子；`dil1→RF15, dil2→RF29, dil4→RF57, dil8→RF113`
（`RF = 1 + (kernel-1) × dil = 1 + 14 × dil`）。

| Subj | Low | Mid | High |
|---|---|---|---|
| 001 | dil8, dil2 | dil1, dil4 | dil4, dil8 |
| 002 | dil4, dil8 | dil1, dil2 | dil1, dil8 |
| 003 | dil1, dil8 | dil4, dil8 | dil1, dil2 |
| 004 | dil2, dil8 | dil1, dil8 | dil2, dil4 |
| 005 | dil4, dil8 | dil4, dil8 | dil2, dil8 |
| 006 | dil2, dil4 | dil2, dil8 | dil2, dil8 |
| 007 | dil2, dil8 | dil2, dil8 | dil1, dil4 |
| 008 | dil4, dil8 | dil4, dil8 | dil4, dil8 |
| 009 | dil1, dil4 | dil2, dil4 | dil4, dil8 |

全部 18112 参数、41.99M MACs，与 RF-only 臂的模型规模一致。

## 3. 协议（与 `hier` / `rf` 臂完全一致）

**Stage-1（筛选）**：`max_epochs=1500, patience=200`，按**验证集 NLL** 存最优 checkpoint。
只跑 Session 0，Session 1 保持关闭（`--screening-only`），不产生测试读数。

**Stage-2（再训练）**：从 Stage-1 最优 checkpoint 恢复；训练集换成 **Session 0 的
train+val 合并（288 条）**；优化器状态一并恢复。停止条件是两条线**都**被跨过：

- **线 A** ＝ Stage-1 验证集最好那次的 NLL（`stage1.best_score`）—— 历史口径
- **线 B** ＝ Stage-1 结束那一轮、**训练集**上的 NLL —— FBNAS 原协议（`baseModel.py:409-422`）

每跨过一条线就存一张快照，循环结束后分别在 Session 1 上读出测试准确率；
`final_summary.json` 里的 `test` 取线 B 的读数（`test_threshold: terminal_train_nll`）。

> 注意：Stage-2 的「验证集」就是被合并进训练集的那 57 条，所以它的 `val_nll` 会迅速
> 掉到接近 0（各受试者 `val_acc` 均到 1.000）。这是 FBNAS 原协议的设计，不是 bug。

## 4. Stage-1 结果（Session 0 验证集，57 trials）

| Subj | 验证集准确率 | 停止轮次 | 最优 NLL（线A） | 收尾训练 loss（线B） |
|---|---|---|---|---|
| 001 | 0.8421 | 475 | 0.4529 | 0.0018 |
| 002 | 0.6491 | 581 | 0.8785 | 0.0403 |
| 003 | 0.8772 | 638 | 0.3742 | 0.0015 |
| 004 | **0.4912** | 463 | 1.1073 | 0.0052 |
| 005 | 0.7719 | 422 | 0.6155 | 0.0018 |
| 006 | 0.7895 | 1038 | 0.7160 | 0.0025 |
| 007 | **0.9474** | 514 | 0.2321 | 0.0011 |
| 008 | 0.8070 | 697 | 0.5900 | 0.0014 |
| 009 | 0.9123 | 618 | 0.2759 | 0.0007 |
| **均值** | **0.7875** | | | |

只有 57 个样本，抖动极大（004 与 007 差了 45 个百分点），这一列不能当作结论。

## 5. Stage-2 结果（Session 1 留出测试集，288 trials）

| Subj | Stage-2 轮次 | 停止原因 | 测试准确率（线B） | 参照：线A 口径 |
|---|---|---|---|---|
| 001 | 233 | all_thresholds_crossed | 0.8368 | 0.6076 |
| 002 | **6** | all_thresholds_crossed | **0.5417** | 0.5590 |
| 003 | 345 | all_thresholds_crossed | 0.9097 | 0.8715 |
| 004 | 359 | all_thresholds_crossed | 0.7188 | 0.4931 |
| 005 | 383 | all_thresholds_crossed | 0.7917 | 0.5660 |
| 006 | 412 | all_thresholds_crossed | 0.7083 | 0.5312 |
| 007 | 285 | all_thresholds_crossed | 0.8750 | 0.8507 |
| 008 | 346 | all_thresholds_crossed | 0.8472 | 0.7396 |
| 009 | 282 | all_thresholds_crossed | 0.8472 | 0.7604 |
| **均值** | | | **0.7863** | 0.6644 |

9 个受试者的停止原因全部是 `all_thresholds_crossed`（没有一个是跑满 600 轮的）。
线 A 口径那一列一并列出，但它只在 1–7 轮就停，模型远未收敛，不应作为结论
（这正是为什么必须用线 B）。

### 逐 run 汇总（一行一个受试者）

```
sub   s1_ep  s2_ep  crossA@ep  crossB@ep    thrA     thrB    testAcc
s001    475    233         2       233    0.4529   0.0018   0.8368
s002    581      6         1         6    0.8785   0.0403   0.5417
s003    638    345         2       345    0.3742   0.0015   0.9097
s004    463    359         1       359    1.1073   0.0052   0.7188
s005    422    383         1       383    0.6155   0.0018   0.7917
s006   1038    412         1       412    0.7160   0.0025   0.7083
s007    514    285         2       285    0.2321   0.0011   0.8750
s008    697    346         1       346    0.5900   0.0014   0.8472
s009    618    282         2       282    0.2759   0.0007   0.8472
------------------------------------------------------------------------
ACC 均值 = 0.7863   (n=9)
```

`s1_ep` = 阶段1 停止轮次（= 最优轮 + patience 200）；`s2_ep` = 阶段2 停止轮次；
`crossA@ep` / `crossB@ep` = 两条阈值线各自被跨过的轮次；`thrA` = 阶段1 验证集最优 NLL；
`thrB` = 阶段1 收尾那轮的训练集 NLL；`testAcc` = Session-1 测试准确率，取 `thrB` 快照。

**线 A 全部在 1–2 轮就触发**，说明线 A 口径下模型完全没收敛；线 B 才是真正的停止线。

### 线 B 的特性（为什么 Stage-2 会很快停）

Stage-1 普遍训了 400–1000 轮，训练集 loss 已贴到 0，所以线 B 极低（0.0007–0.0052，
s002 例外为 0.0403）。Stage-2 的目标是在合并后的 288 条上把 loss 压到这个水平，
实际用了 233–412 轮，与 `hier` / `rf` 臂的 280–330 轮量级相当。

对比同一批受试者在线 A 口径下的读数（均值仅 66.44%），可见**线 B 才是有效的停止口径**：
线 A 在 1–7 轮就触发，模型远未收敛。

**s002 是唯一的退化个例**：它 Stage-1 结束时的训练集 loss 是 0.0403（9 个里最松），
Stage-2 第 6 轮就跨过了这条线，测试只拿到 54.17%。这与它 Stage-1 验证集本来只有
64.91%（第二差）是一致的，属于 FBNAS 原协议在「Stage-1 过拟合严重」受试者上的固有弱点。

## 6. 与两条搜索臂的对比（Session 1 测试，线 B 口径）

| 方案 | seed | 测试准确率 |
|---|---|---|
| 固定架构（本臂，手工表） | 20190821（n=9） | **0.7863** |
| 层级 NAS 搜索 | 20250901–03（3 seed，n=27） | 0.7833 |
| RF-only 搜索 | 20250901–03（3 seed，n=27） | 0.7873 |

参考：`hier` 每 seed 均值 0.7797 / 0.7905 / 0.7797；`rf` 为 0.7824 / 0.7909 / 0.7886。

**seed 不同，不是配对比较**，只能看量级。若要严格比较，需把固定架构也跑 3 个 seed
（20250901–03）。

### 逐受试者对比（固定架构 vs 两条搜索臂的 3-seed 均值）

| Subj | 固定架构 | 层级 NAS | RF-only | 固定 − RF-only |
|---|---|---|---|---|
| 001 | 0.8368 | 0.830 | 0.817 | +0.020 |
| 002 | 0.5417 | 0.618 | 0.606 | −0.064 |
| 003 | 0.9097 | 0.922 | 0.921 | −0.011 |
| 004 | 0.7188 | 0.711 | 0.742 | −0.023 |
| 005 | 0.7917 | 0.762 | 0.772 | +0.020 |
| 006 | **0.7083** | 0.611 | 0.631 | **+0.077** |
| 007 | 0.8750 | 0.885 | 0.883 | −0.008 |
| 008 | 0.8472 | 0.845 | 0.840 | +0.007 |
| 009 | 0.8472 | 0.866 | 0.873 | −0.026 |

9 个受试者里 4 个变好、5 个变差，均值几乎相同。**唯一值得单独一提的是 006**：
固定架构给它选的 Low[dil2,dil4] / Mid[dil2,dil8] / High[dil2,dil8] 比两条搜索臂
都好 7–10 个百分点。但这是单 seed、单受试者的孤例，在 9 个受试者里属于离群点，
不足以支撑「手工表更优」的结论。

## 7. 复现命令

### 实际训练命令（作业脚本内 `srun` 的那一行）

`$REPO` = `/gpfs/home/W125221190/code/T-DARTS-FB-TWO`，`$RUNS` = `$REPO/run`

**Stage-1**（`run/bin/run_fixed_retrain_job.sh`）：

```bash
srun python "$REPO/train_retrain.py" \
  --genotype-json "$RUNS/genotypes/fixeddil/s${SUBJ}.json" \
  --data-root  "$REPO/../FBNAS-master-main/data/bci42a/multiviewPython" \
  --output-root "$RUNS/outputs/fixeddil" --log-root "$RUNS/logs/fixeddil" \
  --dataset bci42a --arm fixeddil --subject "$SUBJ" --seed 20190821 \
  --initialization random \
  --max-epochs 1500 --patience 200 --best-metric val_nll \
  --num-workers 0 --screening-only --no-duplicate-paths --preload-data
```

**Stage-2**（`run/bin/run_stage2_job.sh`）：

```bash
srun python "$REPO/train_retrain.py" \
  --genotype-json "$RUNS/genotypes/fixeddil/s${SUBJ}.json" \
  --data-root  "$REPO/../FBNAS-master-main/data/bci42a/multiviewPython" \
  --output-root "$RUNS/outputs/fixeddil" --log-root "$RUNS/logs/fixeddil" \
  --dataset bci42a --arm fixeddil --subject "$SUBJ" --seed 20190821 \
  --stage2-only --stage2-epochs 600 --batch-size 16 \
  --num-workers 0 --preload-data
```

训练实现均在 `train_retrain.py`：Stage-1 循环在 `train_retrain.py:575` 附近，
Stage-2 循环在 `train_retrain.py:365-401`，阈值判定 `train_retrain.py:380-391`，
Session-1 读出 `train_retrain.py:408-415`，阈值 B 的取值 `train_retrain.py:246`。

### 提交命令

```bash
cd run
# Stage-1（9 个受试者）
for sub in 0 1 2 3 4 5 6 7 8; do
  subj=$(printf "%03d" $((sub+1)))
  sbatch --export=ALL,SUB=$sub,SEED=20190821,ARM=fixeddil,OUTARM=fixeddil \
    --job-name="fixdil_s${subj}" bin/run_fixed_retrain_job.sh
done

# Stage-2 —— 必须显式传 GENOTYPE_JSON，否则脚本会回退到 hier 搜索目录而失败
for sub in 0 1 2 3 4 5 6 7 8; do
  subj=$(printf "%03d" $((sub+1)))
  sbatch --export=ALL,SUB=$sub,SEED=20190821,ARM=fixeddil,OUTARM=fixeddil,STAGE2_EPOCHS=600,\
GENOTYPE_JSON=$(pwd)/genotypes/fixeddil/s${subj}.json \
    --job-name="stage2_fixeddil_s${subj}" bin/run_stage2_job.sh
done
```

> 必须在 `run/` 目录下提交：`run_stage2_job.sh` 用 `SLURM_SUBMIT_DIR` 定位 `run/`，
> 从仓库根目录提交会让它解析错路径并立刻失败。

## 8. 产物位置

- 架构定义：`run/genotypes/fixeddil/s001.json` … `s009.json`
- 训练输出：`run/outputs/fixeddil/bci42a/train_s<subj>_seed20190821_fixeddil/`
  - `final_summary.json` —— 汇总（`stage1` / `stage2` / `test` / `test_threshold`）
  - `metrics.jsonl` —— 逐轮指标（`stage1` 与 `stage2` 混写，用 `stage` 字段区分）
  - `best.pt`（Stage-1 最优）、`stage2_final.pt`（Stage-2 终态）
- 作业日志：`run/sh_log/fixeddil/*.log`（Stage-1）、`run/sh_log/stage2/*.log`（Stage-2）

## 9. 已知问题 / 待办

- [ ] **seed 不配对**：本臂单 seed 20190821，搜索臂 3 seed 20250901–03。要做配对检验，
      需把固定架构补跑 seed 20250902 / 20250903。
- [ ] （可选）补 FBNAS baseline 臂在 Session 1 上的测试读数，凑齐三臂对比。
- [x] Stage-1 与 Stage-2 全部完成（2026-09-18）

### 提交时踩过的坑（别再犯）

1. **Stage-2 必须显式传 `GENOTYPE_JSON`**。`run_stage2_job.sh` 的兜底逻辑是
   「非 rf 臂就去找 `hier` 搜索目录」，而 `hier` 用 seed 20250901–03，
   `hier_search_sXXX_seed20190821/genotype.json` 根本不存在 → 预检 `exit 1`，
   作业 3 秒内失败。第一次提交的 9 个作业全部因此挂掉。
2. **必须在 `run/` 目录下提交**。脚本用 `SLURM_SUBMIT_DIR` 定位 `run/`；
   从仓库根目录提交会解析出少一层 `run/` 的路径，全部立刻失败。
   （不能用 `${BASH_SOURCE[0]}` 兜底：sbatch 执行的是 `/var/spool/slurm/...` 下的副本。）
3. **提交后要确认作业真的活过预检**，不能只看 `sbatch` 返回的作业号——
   `squeue` 里消失 + `sacct` 显示 `FAILED` 才是真相。
