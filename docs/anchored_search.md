# Anchored search: operator first, RF second

This document describes the **anchored** search arm: path 0 keeps the FBNAS
dilated mechanism, path 1 is searched for a complementary operator, and each
path's temporal scale is decided afterwards. It is a separate search space from
the 14-candidate joint DARTS search in `train_search.py`; neither replaces the
other.

## Research question

> Given one dilated path, which operator and time scale make the best
> complementary second path?

It is **not** "what is the best pair of arbitrary paths". Answering that would
need both paths free (`14^6`; see the free-search arm) and is kept as a separate
ablation.

## Space

Each band has two paths of 6 channels each, concatenated to 12. A path can be
one of 14 canonical structures (4 operators x 4 receptive fields, minus the two
RF-15 aliases -- see the README table). The anchored scheme fixes path 0's
operator to `dilated`, so per band the reachable set is

```
path 0: 1 operator  x 4 RFs      =  4
path 1: 14 canonical structures  = 14
                                    56 per band      ->  56^3 = 175,616
```

## Phase A -- search path 1's operator

* Path 0 is fixed to the `dilated` mechanism and **stays active** in the forward
  pass; switching it off would select a path 1 that is best in isolation rather
  than best as a complement.
* Path 1 is a differentiable mixture over the four operator families with one
  `beta` container of 4 logits per band (12 numbers total).
* Both paths use the **same RF within a step**, drawn from the deterministic
  balanced cycle `29 -> 57 -> 113 -> 29 -> ...` (`rf_schedule` in
  `tdarts/anchored.py`). Giving each operator its own random RF would let
  operator and scale covary again, which is exactly what this phase exists to
  avoid.

**Path 0's RF is not a parameter in phase A.** It merely follows the same
schedule so that the four operators of path 1 are compared at one common scale
per step; the schedule marginalises the anchor's scale, it does not search it.
The anchor's actual RF decision is made in phase B, where `gamma_anchor` exists.

### Why RF 15 is excluded from phase A

At RF 15 the dilation is 1, so `dilated == normal` (dense k15) and
`dwsep == lkdw` (separable k15 + 1x1). A four-way softmax would then compare two
structures under four names: each structure would hold two shares of the
probability mass, and `argmax` would reflect naming rather than mechanism. The
scheme therefore compares operators only at 29/57/113, where all four families
are distinct.

RF 15 is **not removed from the scheme**: phase B searches the full ladder
15/29/57/113 for both paths.

## Phase B -- search the RF of both paths

* Path 1's operator is frozen to the phase A argmax (per band).
* Each path has its own `gamma` container of 4 logits over 15/29/57/113
  (24 numbers total for the model).
* Default initialisation is **fresh**; `--inherit-weights` optionally copies
  the phase A weights for the geometries that already exist (RF 15 is never
  trained in phase A, so those modules stay fresh either way).

`dilated`/`dwsep` realise their RF through dilation at kernel 15;
`normal`/`lkdw` use the kernel itself. That mapping lives in
`tdarts/temporal_ops.py` and is not duplicated here.

### `--no-duplicate-paths`

With this flag the two paths of a band are decoded **jointly**: the exported
pair maximises `log p_anchor(i) + log p_searched(j)` over all 4x4 combinations
whose structures differ. Duplicate detection uses the actual structure key

```
structure_key = (kernel, dilation, separable)
```

never the candidate name, because at RF 15 `normal` *is* `dilated_rf15` under a
different name. Cross-family RF-15 pairs such as `dilated_rf15` + `dwsep_rf15`
(dense vs separable) remain allowed. Hard-mode evaluation uses the same
constrained pair, so the reported hard subnet is exactly the exported genotype.

Without the flag both paths may choose the same structure; the two paths still
own independent weights, so the network can learn two different filter sets.

## Pipeline

```bash
# 1. search: phase A -> phase B -> genotype.json
python train_anchor_search.py \
    --data-root /path/to/bci42a/multiviewPython \
    --output-root outputs --log-root logs \
    --subject 003 --seed 20250901 \
    --operator-epochs 200 --rf-epochs 200 --warmup-epochs 20 \
    --no-duplicate-paths --preload-data

# 2. final training: fresh discrete network from the exported genotype,
#    FBNAS two-stage protocol, Session 1 read only at the end
python train_retrain.py \
    --genotype-json outputs/bci42a/anchor_search_s003_seed20250901_anchored/genotype.json \
    --data-root /path/to/bci42a/multiviewPython \
    --subject 003 --seed 20250901 --initialization random \
    --max-epochs 1500 --patience 200 --stage2-epochs 600 --observe-test
```

`--genotype-json` and `--search-dir` are mutually exclusive genotype sources.
The anchored genotype has no compatible supernet checkpoint, so
`--initialization transfer` is rejected for it; the run always starts from a
fresh initialisation.

## Artifacts

| file | contents |
|---|---|
| `config.json` | arguments, phase lengths, RF schedules, `no_duplicate_paths` |
| `manifest.json` | model descriptions, band split sizes, inherited prefixes |
| `metrics.jsonl` | one row per epoch on a continuous axis; phase A rows carry operator probabilities, RF counts and per-RF validation; phase B rows carry soft/hard validation and the gap |
| `phase_a_result.json` | frozen operators and phase A probabilities |
| `genotype.json` | `scheme`, `subject`, `seed`, `fold`, `no_duplicate_paths`, `selected_operators`, `phase_a`, `rf_mixtures`, six canonical `genes` |
| `checkpoint_epoch_*.pt` | phase A and phase B checkpoints (`stage` field distinguishes them) |

`genotype.json` is validated when loaded: the scheme tag must match, every gene
must be a canonical candidate with a consistent index, path 0 must be the
dilated anchor, and -- when the file claims `no_duplicate_paths` -- the two
structures of every band must really differ.

## Current protocol scope

* One FBNAS-ordered Session-0 80/20 split and one seed. The `stability` field in
  `genotype.json` records `single_run_no_cross_run_vote`: with a single run
  there is no cross-run vote, so the export is a genotype, not a stability
  claim.
* K-fold search and cross-seed stability analysis are not implemented yet; the
  run layout and genotype format carry `fold` so they can be added without
  changing the artifact schema.
* Session 1 is never loaded during search. Final training reads it only through
  `--observe-test`, whose metrics never influence stopping, checkpoints or
  selection.

## Code map

| symbol | role |
|---|---|
| `tdarts/anchored.py:rf_schedule` | deterministic balanced phase A RF cycle |
| `AnchoredOperatorCell` | phase A cell: anchor + 4-way operator mixture at the scheduled RF |
| `AnchoredRFCell.select_rf_indices` | phase B decode, joint and constrained when `no_duplicate_paths` |
| `build_anchored_genotype` / `load_anchored_genotype` | export and validated reload |
| `run_anchored_epoch` | alternating first-order epoch with a per-step RF hook |
| `tests/test_anchored.py` | schedule, parameter counts, isolation, no-duplicate, genotype round trip |
