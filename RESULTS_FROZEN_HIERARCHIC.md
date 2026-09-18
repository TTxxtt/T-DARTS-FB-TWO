# Frozen Hierarchical NAS vs RF-only: 9-Subject Results

## Experimental Setup

- Dataset: BCI-42a (4-class motor imagery, EEG)
- Subjects: 001–009 (all 9 subjects)
- Seeds: 20250901, 20250902, 20250903 (3 seeds per subject)
- Total chains: 9 × 3 = 27 per arm
- Split: Session 0 (231 trials train / 57 trials val), Session 1 closed (screening-only)
- Protocol: max 1500 epochs, patience 200, checkpoint by **validation NLL**

## Arms

### Full Hierarchical (operator→RF)
- Phase A: hard one-hot operator search at RF57 (Gumbel-ST, 300 epochs, τ 1.0→0.3)
- Phase B: RF search over frozen operators (200 epochs, 4 RFs × 2 paths × 3 bands = 24 γ)
- Phase C: from-scratch retrain of the genotype (1500/200, val_nll)

### RF-only Control
- Both paths fixed to `dilated`; only RF searched (200 epochs, same γ as Phase B)
- Phase C: same retrain protocol (1500/200, val_nll)
- Genotype: all-dilated_rfXX (18112 params, 41.99M MACs)

## Results

### Per-subject paired ΔNLL (full − RF-only, negative = full better)

| Subject | mean ΔNLL | std | mean ΔAcc | std |
|---------|-----------|------|-----------|------|
| 001 | −0.0480 | 0.0602 | +0.0234 | 0.0268 |
| 002 | −0.0689 | 0.0898 | −0.0175 | 0.0351 |
| 003 | −0.0345 | 0.0634 | +0.0117 | 0.0101 |
| 004 | +0.0373 | 0.1326 | −0.0234 | 0.0810 |
| 005 | +0.0031 | 0.0212 | −0.0409 | 0.0442 |
| 006 | +0.0395 | 0.0535 | +0.0468 | 0.0564 |
| 007 | −0.0301 | 0.0569 | −0.0117 | 0.0101 |
| 008 | −0.0019 | 0.0607 | +0.0234 | 0.0405 |
| 009 | −0.0098 | 0.0567 | +0.0175 | 0.0351 |
| **ALL** | **−0.0126** | **0.0695** | **+0.0032** | **0.0449** |

### Summary
- **15/27 pairs (56%)** favor the full method on NLL
- **Overall ΔNLL = −0.013 ± 0.070** (effect within noise)
- **Overall ΔAcc = +0.003 ± 0.045** (no meaningful accuracy gain)
- **Full method costs +30.6M MACs** (72.6M vs 42.0M)

### Search diagnostics
- Phase A operator margins: 0.009–0.035 (near-uniform softmax(β))
- Operator selections: inconsistent across seeds (no stable per-subject preference)
- Phase A selection counts: ~2100–2240 per operator (balanced training, no starvation)
- Soft→hard gap: full method 0.166 vs RF-only 0.218 (improved but still large)

## Verdict

The hierarchical operator search does **not** produce a reliable final benefit:
1. The NLL advantage (−0.013) is smaller than seed-level noise (σ = 0.070)
2. Accuracy is essentially tied (+0.3pp, within noise)
3. Cost is 73% higher MACs
4. Phase A margins remain tiny (near-uniform), confirming operator differences at RF57 are too small to resolve with 57 validation samples

Per the pre-registered decision rule: **no convincing overall benefit → operator search does not improve final performance**.

## Artifacts
- Search genotypes: `run/outputs/hier/bci42a/hier_search_s*_seed*/`
- Full method retrains: `run/outputs/hier_retrain/bci42a/train_s*_seed*_hier/`
- RF-only matched retrains: `run/outputs/rf_retrain_nll/bci42a/train_s*_seed*_rf/`
- Paired comparison JSON: `run/outputs/nine_subject_hier_vs_rf.json`
- Phase A results: `run/outputs/hier/bci42a/*/phase_a_result.json`
- Logs: `run/sh_log/remaining_subjects.log`
