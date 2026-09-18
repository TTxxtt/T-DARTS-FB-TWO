# Stage 2 补充方案

## 问题

当前所有重训（hier_retrain × 27 + rf_retrain_nll × 27）都用了 `--screening-only`，只有 Session 0 的 validation-best 读数，没有 Session 1 的 test 结果。要跟原始 FBNAS 对比，需要 Stage 2。

## Stage 2 协议

原始 FBNAS 的两阶段协议：
- **Stage 1**：Session 0 训练，Session 0 验证（已完成）
- **Stage 2**：Session 0 全量（train+val 合并）继续训，直到 val NLL < Stage 1 的 terminal train NLL，然后在 Session 1 上测

## 修复内容

### 1. 代码改动（train_retrain.py）

已加 `--stage2-only` 参数，还需修一个 bug：

`load_genotype()` 接收文件路径，不能直接传 dict。修法：当 genotype 已是 dict（从 final_summary.json 读出）时，直接 `Genotype(**genotype)` 构造，不走 `load_genotype()`。

### 2. 提交 27 个 Stage 2 任务

对每个已有 best.pt 的 run：

```bash
python train_retrain.py --stage2-only \
  --genotype-json <run_dir>/genotype.json \
  --output-root <arm_dir> --log-root <log_dir> \
  --dataset bci42a --arm <hier|rf> --subject <subj> --seed <seed> \
  --stage2-epochs 600 --batch-size 16 --preload-data
```

### 3. 耗时估算

- 每个 Stage 2 最多 600 epoch，通常几十轮就停
- 27 个并行约 10–20 分钟

### 4. 产出

更新每个 `final_summary.json`，加上 `test` 字段（acc/nll/kappa/macro_f1/confusion_matrix），可做最终三臂对比。

## 需要确认

是否现在修 bug + 提交 27 个 Stage 2？
