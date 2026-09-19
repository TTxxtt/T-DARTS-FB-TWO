# Final cross-session protocol: dilated-only search vs four-mechanism search

This file freezes the rules of the final Session1 → Session2 comparison before
Arm B's held-out session is read. It is the document that says what was decided
in advance and what was not.

**Status at the time of writing: Arm B has not yet read Session 2.** Its search
and its Stage 1/2 retrain are implemented, tested, and smoke-verified with the
second session closed. Arm A's numbers, however, already exist — see
[Honest limitations](#honest-limitations), which is not a formality.

## Session numbering (the thing most easily confused)

BCIC-IV-2a has two recording sessions per subject. This repo indexes them by the
`session` column of `dataLabels.csv`, which is derived from the raw file prefix
(`FBNAS/codes/centralRepo/saveData.py:438-443`).

| | raw file | code | role here |
|---|---|---|---|
| dataset Session 1 | `s0NN.mat` ("T") | `session=0` | 231 train / 57 validation |
| dataset Session 2 | `se0NN.mat` ("E") | `session=1` | 288 trials, held out |

Every log line and document writes **both** numbers. "Session 2" in this file
always means the dataset's second recording session, code `session=1`.

## The two arms

### Arm A — dilated-only FBNAS search

The vendored upstream chain, driven by `run/py/run_fbnas_subject.py`, which
calls `ho.py`. Nothing about it is reimplemented.

* **Space.** Each band picks one or two of four dilated convolutions — kernel 15
  at dilation 1/2/4/8, i.e. effective receptive fields 15/29/57/113
  (`FBNAS/codes/centralRepo/networks.py:452-456`). `C(4,1) + C(4,2) = 10` per
  band, so `10**3 = 1000` architectures.
* **Search.** 200 epochs of random-subnet ("MixPath") training with weight
  sharing — one subnet per batch, no architecture parameter, no architecture
  gradient.
* **Selection.** Enumerate all 1000 candidates; for each, restore the trained
  weights, run one train-mode pass over the validation split so the BatchNorms
  recalibrate, and take the argmax of **calibrated validation accuracy**
  (`NAS.py:308-311`). The per-candidate vector is archived as `cali_bn_acc.npy`.
* **Stage 2.** Upstream `baseModel.train` with `continueAfterEarlystop=True`:
  ordered Session-0 80/20 training stopped on validation accuracy (max 1500,
  patience 200), restore the best validation checkpoint, merge the validation
  trials back into training, stop as soon as validation loss drops below Stage
  1's terminal train loss, cap 600 epochs.
* **Where it lives on disk.** `run/outputs/fbnas/bci42a/ses2Test/<timestamp>/sub0..sub8`.

### Arm B — four-mechanism hierarchical search

Implemented in this repository: `train_arm_b_search.py` over
`tdarts/fbnas_sampler.py`, `tdarts/band_supernet.py`, `tdarts/band_discrete.py`.

* **Phase A — family search.** Every band runs a single path at RF 57 and picks
  one of four families: `dilated_e`, `dynamic_e`, `gated_e`, `band_gated_e`.
  `4**3 = 64` configurations. `local_attention_e` is excluded — it had the worst
  validation NLL and the highest early-overfit rate in the band probe.
* **Phase B — RF search.** Each band's family is frozen and its receptive field
  is searched over the same ladder 15/29/57/113, taking one or two RFs:
  `C(4,1) + C(4,2) = 10` per band, `10**3 = 1000` — **the same space Arm A
  searches**. That is the point: the only variable is whether the mechanism is
  searched too.
* **Search.** The same FBNAS protocol: one subnet per batch, weights only, no
  architecture gradient. 200 epochs per phase.
* **Selection.** Identical to Arm A: enumerate every candidate, restore the
  weights, recalibrate BatchNorm on the validation split, argmax of calibrated
  validation accuracy. Validation NLL is recorded alongside and is **never** used
  to select.
* **Channel arithmetic.** A band holding `k` paths runs each at `F1 // k`
  channels and concatenates, so it always emits `F1 = 12` and the backbone is
  bit-identical across all 1000 candidates — upstream's `F1//(i+1)` scheme,
  which is what makes a path-count choice a structural rather than a capacity
  choice.
* **Stage 2.** The same `train_retrain.py` two-stage protocol, with
  `--best-metric val_inacc` — the default, and the setting that matches
  upstream's `bestVarToCheck: 'valInacc'` / `NoDecrease: numEpochs 200`. (The
  frozen repo arms pass `val_nll`; that would *not* match Arm A.)

## Frozen settings

| | value |
|---|---|
| subjects | 001–009 (all nine) |
| seed | **20190821**, both arms, single seed |
| Session-1 split | contiguous 231 train / 57 validation of the 288 Session-0 trials |
| Session-2 | 288 trials, read once per run at the end |
| search budget | Arm A 200 epochs; Arm B 200 + 200 |
| selection metric | calibrated validation accuracy, both arms |
| Stage-1 stop | validation accuracy, max 1500 epochs, patience 200 |
| Stage-2 stop | validation NLL < Stage 1 terminal train NLL, min 0, cap 600 |
| Stage-2 training set | all 288 Session-0 trials |
| evaluation metrics | accuracy, macro-F1, Cohen's kappa, NLL, confusion matrix |
| pairing | per subject, `Arm B − Arm A`, all nine reported; never "the best seed" |
| `--observe-test` | **on** for Arm B's retrain (observation only) |

**Why one seed.** Arm A is the vendored baseline and its `randSeed` is hardcoded
to 20190821 inside the frozen `ho.py` (`:57`, `:236`, `:307`) and again as
`baseModel`'s default. Varying it would mean monkeypatching the baseline's RNG
from outside. The user chose an unpatched baseline and an exactly paired
comparison over three seeds with a patched one.

## What decides the outcome

Per subject, `ΔAcc = Acc_ArmB − Acc_ArmA` on Session 2, and likewise for F1,
kappa and NLL. Reported as the mean ± sd of the nine paired differences, with
the standard error and its interval — never as a best-subject figure.

Both arms run at one seed, so there is no seed-level variance to average away
and **n = 9 subjects is the entire evidence**. A within-noise Δ is a null
result, not a trend.

## Honest limitations

1. **Arm A's Session-2 numbers were already unblinded.** They were produced on
   2026-09-15 and read during protocol design. So the claim "both arms were
   frozen before either was unblinded" is **false** and is not made here. What
   *is* defensible: Arm B's architecture is the argmax of 1000 candidates'
   calibrated validation accuracy — a mechanical rule with no tunable knob, no
   human in the loop, and no dependence on Arm A's numbers. Nothing in Arm B's
   search or selection can read the test set, and there is no hyperparameter
   that could have been chosen to make Arm B look better.
2. **The arms differ in more than the search space.** Arm A runs the upstream
   `SuperNet`/`FBNASNet` code path and `baseModel.train`; Arm B runs the
   `tdarts` backbone and `train_retrain.py`. Both are documented as
   FBNAS-compatible and the backbones are shape-identical, but they are not the
   same code. **ΔAcc therefore cannot be attributed cleanly to the search space
   alone** — it is the difference between two complete pipelines whose search
   spaces differ.
3. **Arm B's search budget is 400 epochs against Arm A's 200.** Phase B matches
   Arm A exactly (200 epochs on the same 1000-candidate space); Phase A is the
   extra step this method proposes. Arm B is best described as *Arm A plus a
   family pre-selection stage*, not as an equal-budget alternative.
4. **`--observe-test` exists only on Arm B.** It logs Session-2 metrics every
   epoch for observation; no stop, checkpoint or selection decision reads them
   (`evaluate` is `torch.no_grad()` and the Stage-2 threshold reads validation
   NLL). The upstream chain reads Session 2 once at the end and offers no
   per-epoch curve, so the two arms are paired on final values only.
5. **Two families are extrapolated to width 6.** `gated_e` and `band_gated_e`
   were probed at width 12 only. `WidthGeneralGatedE` and
   `WidthGeneralBandGatedE` are bit-identical to the probed operators at width
   12 (pinned by test) and well defined at width 6, but width 6 is a documented
   extrapolation, not a measured configuration.
6. **Capacity is not perfectly invariant across path counts.** A two-path band
   differs from a one-path band by +16 params (`dynamic_e`), −72 (`gated_e`),
   +3 (`band_gated_e`), 0 (`dilated_e`) — at most 0.3% of the model. These are
   per-path 1×1/gate overheads, not width artefacts, and they are pinned by
   test so they cannot drift unnoticed.

## Reproducing

```bash
# Arm A (read-only; writes nothing into the archived tree)
python tools/extract_arm_a_session2.py

# Arm B: 9 searches + 9 retrains
cd run && bash bin/submit_armB.sh dry     # preview
cd run && bash bin/submit_armB.sh drain   # submit, resumable via two ledgers

# the comparison
python tools/compare_session2_arms.py
```
