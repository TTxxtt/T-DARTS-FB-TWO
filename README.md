# T-DARTS-FB

Research codebase for temporal EEG decoding. The official **FBNAS** baseline is
vendored unmodified under `FBNAS/`, and new work lives alongside it in `tdarts/`
so the two can be diffed file by file.

**Stage 2 is complete**: the 14-candidate `MixedTemporalOp`, the two-path
temporal cell, and the full `TemporalDARTSNet` supernet. Stage 3 adds the
first-order DARTS search (`train_search.py`) with its fixed-genotype retrain
(`train_retrain.py`), plus a **separate anchored search arm**
(`train_anchor_search.py`, `tdarts/anchored.py`) that keeps one path dilated and
searches the complement; see [docs/anchored_search.md](docs/anchored_search.md).

## Repository layout

```
T-DARTS-FB/
├── FBNAS/                     # official baseline, byte-identical, do not edit
│   ├── codes/
│   │   ├── centralRepo/       # networks.py, NAS.py, eegDataset.py, ...
│   │   └── classify/ho.py     # training entry point
│   ├── requirements.txt
│   └── _MANIFEST.json         # 52/52 blobs with git SHAs
├── tdarts/                    # new code
│   ├── config.py              # machine profiles + search-space constants
│   ├── temporal_ops.py        # 4 operators x 4 receptive fields
│   ├── mixed_op.py            # MixedTemporalOp, TwoPathTemporalCell, TemporalDARTSNet
│   ├── architect.py           # first-order DARTS steps with BN/w isolation
│   ├── search.py              # search epoch + evaluation primitives
│   ├── anchored.py            # anchored arm: dilated anchor -> operator -> RF
│   ├── backbone.py            # SCB / LogVar / classifier, unchanged from FBNAS
│   ├── genotype.py            # genotype extraction
│   ├── discrete_network.py    # fixed-genotype network
│   ├── init_utils.py          # name-addressed deterministic initialisation
│   └── cli.py                 # `t-darts-profile`
├── tests/
│   ├── test_temporal_ops.py          # 48-candidate audit + collapse
│   ├── test_mixed_op.py              # stage-2 architecture audit
│   ├── test_fbnas_compatibility.py   # new ops vs untouched SCB / LogVar
│   ├── test_backbone.py              # fixed backbone shape chain
│   ├── test_architect.py             # w/alpha isolation and BN freezing
│   ├── test_anchored.py              # anchored schedule, no-duplicate, genotype
│   ├── test_init_utils.py            # deterministic init
│   ├── test_config.py
│   └── test_cli.py
├── configs/servers.yaml       # machine profiles -- edit this one
├── train_search.py            # 14-candidate joint DARTS search
├── train_anchor_search.py     # anchored: phase A (operator) -> phase B (RF)
├── train_retrain.py           # final training of one genotype
├── docs/fbnas_audit.md        # what the baseline actually does
├── docs/anchored_search.md    # the anchored arm, phase by phase
├── scripts/smoke_stage1.py    # quick end-to-end sanity check
└── tools/                     # mirror / config-sync utilities
```

## Setup

```bash
pip install -e .          # only runtime dependency is PyYAML
```

## Temporal operators (stage 1)

`tdarts/temporal_ops.py` builds four operator families across four receptive
fields. Every operator maps `[B, 3, 22, 1000] -> [B, 6, 22, 1000]`: the time
length and the electrode axis are preserved exactly.

```python
from tdarts.temporal_ops import build_temporal_op, build_all_candidates

op = build_temporal_op("dilated", band="Low", target_rf=57)
op = build_temporal_op("lkdw", "High", 113, use_norm=False)   # for RF auditing

pool = build_all_candidates()      # 48 candidates
```

Geometry, with `kernel = 15` and `RF = 1 + (kernel - 1) * dilation` for all
three bands in this stage:

| target RF | dilation | `dilated` | `normal` | `dwsep` | `lkdw` |
|---|---|---|---|---|---|
| 15  | 1 | k15 d1 | k15 d1  | dw k15 d1 + 1x1 | dw k15 d1 + 1x1 |
| 29  | 2 | k15 d2 | k29 d1  | dw k15 d2 + 1x1 | dw k29 d1 + 1x1 |
| 57  | 4 | k15 d4 | k57 d1  | dw k15 d4 + 1x1 | dw k57 d1 + 1x1 |
| 113 | 8 | k15 d8 | k113 d1 | dw k15 d8 + 1x1 | dw k113 d1 + 1x1 |

`dilated`/`dwsep` reach the target receptive field with only 15 taps spread
across it (sparse sampling); `normal`/`lkdw` fill the same span densely. That
equality of receptive field with a difference in sampling mechanism is the
property the operator comparison rests on, and it is measured rather than
asserted.

**RF15 produces duplicate candidates.** At `target_rf = 15` the dilation is 1, so
`dilated` coincides with `normal` and `dwsep` with `lkdw`. The declared grid is
therefore 16 names but only **14 distinct structures** per band:

```
RF15   : dilated (= normal), dwsep (= lkdw)      2
RF29   : all four                                 4
RF57   : all four                                 4
RF113  : all four                                 4
                                         total   14
```

`canonical_candidates(band)` returns those 14, derived from each candidate's
`structure_key = (kernel, dilation, separable)` rather than hard-coded, so it
stays correct if the ladder or base kernel changes. Stage 2's `MixedTemporalOp`
ranges over the canonical 14, not the declared 16.

This matters for a gradient-based search: keeping both names for one family
would give it two independent sets of logits and therefore two shares of the
softmax mass, while `argmax` sees only one of them. A family that is genuinely
preferred could split its probability across two aliases and lose the argmax to
a weaker but uniquely-named candidate — the decision would reflect naming, not
structure.

Note that `separable` is part of the key on purpose: at RF15 `dilated` and
`dwsep` share the geometry `(k=15, d=1)` but are different operators (270 vs 126
parameters), so geometry alone would over-collapse.


## Auditing

Each audit is runnable on its own and exits non-zero on any failure — failures
are not warnings.

```bash
python tests/test_temporal_ops.py          # 48 / 48 PASS
python tests/test_mixed_op.py              # 39 / 39 PASS
python tests/test_fbnas_compatibility.py   # COMPATIBLE
python tests/test_backbone.py              # 16 / 16 PASS
python tests/test_init_utils.py            # 18 / 18 PASS
python tests/test_anchored.py              # 33 / 33 PASS
python -m unittest discover -s tests -t .  # 236 tests
```

The temporal audit checks, per candidate: construction, forward, backward,
input/output shape, time preservation, electrode-axis preservation, finiteness,
gradient on every trainable parameter, and the **measured** effective receptive
field via gradient support (in float64, with candidate normalisation disabled —
BatchNorm is affine and would distort the measurement).

## Stage 2: the supernet architecture

`tdarts/mixed_op.py` composes the pool into the architecture a DARTS search will
later optimise.

```
[B, 9, 22, 1000]
   split into Low / Mid / High, 3 filter-bank channels each
   per band:  Path A  MixedTemporalOp(14)  -> [B, 6, 22, 1000]
              Path B  MixedTemporalOp(14)  -> [B, 6, 22, 1000]
              concat + BatchNorm2d(12)      -> [B, 12, 22, 1000]
   concat the three bands                   -> [B, 36, 22, 1000]
   unchanged backbone                       -> [B, 4]
```

```python
from tdarts.mixed_op import TemporalDARTSNet

net = TemporalDARTSNet()
logits, features = net(torch.randn(2, 9, 22, 1000))
len(net.arch_parameters())        # 6 containers: 2 paths x 3 bands
net.num_arch_parameters()         # 84 numbers = 6 x 14
```

Six independent `alpha` containers is the point: paths must be able to select
different operators, so a shared container would make `A == B` structural rather
than incidental. The same reasoning applies across bands.

`network_parameters()` / `arch_parameters()` partition the model exactly, with no
overlap — verified by a test, because an `alpha` leaking into the weight
optimiser is a silent failure mode in DARTS.

The mixture is `y = sum_i softmax(alpha)_i * O_i(x)`. At initialisation `alpha`
is small, so the mixture is near-uniform by design; stage 2 never updates it.

**One-hot equivalence** is verified for all 14 candidates: forcing `alpha` to a
one-hot vector makes the mixed output identical to that single operator, so the
discrete model is an exact special case of the supernet. (A large finite logit is
used rather than `inf`, since `softmax` with an infinite logit produces NaN
gradients and would make the check pass for the wrong reason.)


## Anchored search (`path0 = dilated`, then RF)

A second, separate search space lives in `tdarts/anchored.py`. Path 0 keeps the
FBNAS `dilated` mechanism; phase A searches path 1's operator family under a
balanced 29/57/113 schedule, then phase B freezes the operator and searches the
RF of both paths over 15/29/57/113. RF 15 is excluded from phase A only, because
`dilated == normal` and `dwsep == lkdw` there.

Phase A's schedule **marginalises** path 0's scale -- it does not search it.
Path 0's RF is decided by `gamma_anchor` in phase B; the phase A anchor simply
follows the same schedule so all four operators are compared at one common
scale per step.

```bash
python train_anchor_search.py --help          # phase A -> phase B -> genotype.json
python train_retrain.py --genotype-json <run>/genotype.json ...   # from scratch
```

`--no-duplicate-paths` decodes the two paths of each band jointly and forbids
equal `structure_key = (kernel, dilation, separable)` values, so RF-15 aliases
count as duplicates while dense-vs-separable pairs do not. The phase B hard
subnet is decoded with the same constrained pair.

See [docs/anchored_search.md](docs/anchored_search.md) for the full protocol,
artifacts and code map.


## Candidate normalisation

Each candidate is `operator -> BatchNorm2d(6, affine=False)` by default, so
differently-scaled operators cannot skew a future softmax architecture gradient.
`affine=False` keeps it parameter-free. Pass `use_norm=False` to get
`nn.Identity` instead, which is what the receptive-field audit requires.

## Fixed backbone

`tdarts/backbone.py` reimplements only the three fixed modules — SCB, LogVar and
the classifier — copied from the official `networks.py` with source attribution
and **unchanged mathematics**. It does not import `FBNASCell`, `SuperNet` or the
`nodes[(len(path_ids)-1)*4 + id]` addressing.

```
[B, 36, 22, 1000]  ->  SCB  ->  [B, 288, 1, 1000]
                   ->  reshape -> [B, 288, 8, 125]
                   ->  LogVar  -> [B, 288, 8, 1]
                   ->  flatten -> [B, 2304]
                   ->  classifier -> [B, 4]
```

The new temporal operators feed this backbone **without any modification to
SCB, LogVar or the classifier**: two paths per band give `2 x 6 = 12` channels,
which is exactly the official per-band width `F1 = num_Feat // 3 = 12`, so three
bands concatenate to 36 channels and the SCB accepts them unchanged.

## Deterministic initialisation

`tdarts/init_utils.py` seeds each parameter from `crc32(f"{run_seed}:{name}")`
rather than from module creation order, so two search variants that share a
parameter name start bit-identical. Python's built-in `hash()` is deliberately
not used — it is salted per process. Not yet wired into any model; it is a
prepared tool.

## Configuration

Machine profiles live in `configs/servers.yaml`; search-space constants live in
`tdarts/config.py`. Both are single-source: do not re-declare kernel sizes or RF
ladders at call sites.

```python
from tdarts.config import get_profile, BASE_KERNEL, RF_SPACE, OPERATORS
prof = get_profile()          # auto-detect; or get_profile("serverA")
```

```bash
t-darts-profile --list
```

## Baseline policy

`FBNAS/` is reference-only and must not be edited. Its integrity is enforced by
a test that recomputes each file's git blob SHA-1 against `_MANIFEST.json`. New
code may copy from it with attribution (as `tdarts/backbone.py` does) or import
from it by putting `FBNAS/codes/centralRepo` on `sys.path` (as the
compatibility test does), but must not take a dependency on the path/node cell.

Read `docs/fbnas_audit.md` before building against the baseline. The headline
finding: **the upstream search is not DARTS.** It trains a supernet with randomly
sampled paths, then enumerates every admissible path subset and selects by
validation accuracy after recomputing BatchNorm statistics per candidate.

## License

Not yet chosen. Until one is added the code is all rights reserved by default.
The vendored upstream baseline ships **with no LICENSE file at all** (verified
against the GitHub API), so its redistribution terms are unstated; check with
the upstream author before publishing.
