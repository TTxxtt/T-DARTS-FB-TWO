# Upstream audit: `wang1239435478/FBNAS-master`

Purpose: record what the frozen baseline actually does before any new code is
written against it, and flag where a planned design would rely on a wrong
assumption.

- Repository: <https://github.com/wang1239435478/FBNAS-master>
- Branch: `main`
- Commit audited: `acd1fd9fa5251903b3ed35a804ee8cd1a25df57b`
- Local mirror: `FBNAS/` (see `_MANIFEST.json` for the exact blob SHAs)
- Refetch with: `python tools/fetch_upstream.py wang1239435478/FBNAS-master main FBNAS --all`
- Mirror is **complete**: 52/52 blobs, 0 skipped, 0 failed. Verified byte-for-byte
  against the recorded git blob SHAs by
  `tests/test_fbnas_compatibility.py::BaselineIntegrity`.

The paper is *Subject-Adaptive EEG Decoding via Filter-Bank Neural Architecture
Search for BCI Applications*, so **FBNAS = Filter-Bank NAS**, built on top of
the FBCNet multi-view representation.

## 0. Confirmed temporal architecture (measured, not read from prose)

Measured by instantiating the official modules; see
`tests/test_fbnas_compatibility.py`.

| item | value |
|---|---|
| input | `[B, 9, 22, 1000]` (`ho.py`: `torch.randn(1, 9, 22, 1000)`) |
| per-band input | `[B, 3, 22, 1000]`, bands grouped 0-2 / 3-5 / 6-8 |
| temporal conv kernel | `(1, 15)` — **the same for all three bands** |
| dilation | 1 / 2 / 4 / 8, selected by the path/node index |
| padding | `(0, 7·dilation)` — preserves `T` exactly |
| groups | **1** (dense 3→6 conv, *not* depthwise) |
| bias | `bias=False` |
| path widths | length-1 path → `F1` channels; length-2 paths → `F1 // 2` each |
| cell output | `[B, F1, 22, 1000]`; with the official `num_Feat=36`, `F1 = 12` |
| SCB input | `num_Feat = 36` channels, grouped `groups=36`, kernel `(22, 1)` |
| SCB output | `[B, 288, 1, 1000]` (288 = 36 · dilatability 8) |
| LogVar | `dim=3` on the reshaped `[B, 288, 8, 125]` → `[B, 288, 8, 1]` |
| classifier | `LinearWithConstraint(2304, 4, max_norm=0.5)` + `LogSoftmax` |

Two details worth remembering because they are easy to get wrong:

- the official constraint convolutions (SCB) and the classifier linear **do**
  carry a bias, whereas the temporal `ConvBn` uses `bias=False`. Pinned by
  `test_official_constraint_convs_carry_a_bias`.
- `torch.renorm(..., maxnorm=2)` on the SCB weight is a **no-op at
  initialisation**, because the default init already gives every row a norm
  below 2. It only starts clamping once training pushes rows past 2.

## 0b. Candidate collapse in the extended search space

The extended space crosses 4 mechanisms with the 4 official RF scales, which
nominally gives 16 candidates per band. Two of them are aliases, because the
RF15 scale resolves to `dilation = 1`:

| structure key `(kernel, dilation, separable)` | names at RF15 | count |
|---|---|---|
| `(15, 1, dense)` | `dilated`, `normal` | 2 names, 1 structure |
| `(15, 1, depthwise)` | `dwsep`, `lkdw` | 2 names, 1 structure |

So the distinct structure count per band is `2 + 4 + 4 + 4 = 14`, and 42 across
three bands. `tdarts.temporal_ops.canonical_candidates` derives this set from the
built geometry and `duplicate_groups` exposes the aliases.

`separable` is part of the key deliberately: at RF15 `dilated` and `dwsep` share
the geometry `(15, 1)` but are different operators -- a dense `3->6` convolution
(270 parameters) versus a depthwise `3->6` plus pointwise projection (126
parameters). Keying on geometry alone would collapse them wrongly and produce 7
structures instead of 14.



## 1. File inventory

`codes/centralRepo/` holds `networks.py`, `NAS.py`, `eegDataset.py`,
`stopCriteria.py`, `utils.py`, `baseModel.py`, `saveData.py`, `samplers.py`,
`transforms.py`, `CenterLoss.py`, `testFilterDiff.py`. Training entry point is
`codes/classify/ho.py`. This matches what the implementation plan assumed.

`requirements.txt` pins `torch==2.5.1+cu118`, `torchvision==0.20.1+cu118`,
`torchaudio==2.5.1+cu118`, plus `numpy==2.0.2`, `mne==1.8.0`,
`scikit-learn==1.5.2`. Also confirmed.

## 2. The search is NOT DARTS — this is the important finding

`NAS.py::nas_phase` does not relax the architecture into continuous weights and
optimise it with gradients. It is a **train-once, enumerate-and-select**
procedure:

1. Train `SuperNet` for `--epochs` (default 200), drawing a *random* path subset
   every step via `utils.random_choice(m=2)`.
2. Save the supernet.
3. `utils.traverse_choice(m=2)` enumerates every admissible path combination.
4. For each candidate: reload the supernet, run the validation set through it
   with `net.train()` so **BatchNorm running statistics are recomputed for that
   specific path**, and score it.
5. Pick the candidate with the highest validation accuracy.

Supporting details:

- `traverse_choice(2)` yields all combinations of size 1 *and* 2 from
  `{0,1,2,3}` per band, i.e. 4 + 6 = 10 options per band, then takes the full
  cross product over three bands: **10³ = 1000 candidates**.
- `find_choice_index` builds the same enumeration, so index ↔ subset mapping is
  consistent, but it is unused by `nas_phase`.
- There is a `while choice in check_dict` duplicate guard in
  `validate_search` — guarded by a module-level list that is never cleared.
- `ho.py` builds a `SuperNet` for the search phase and then instantiates
  `FBNASNet` for retraining, copying across `state_dict` entries whose keys
  match.

## 3. Cell structure and the `nodes` indexing scheme

`FBNASCell.__init__` (and the near-duplicate `Cell`) creates exactly 8
convolution modules:

```python
for i in range(2):                      # i = 0, 1  -> path length 1 or 2
    self.nodes.append(ConvBn(3, F1//(i+1), 15, (1,1), (0,7)))
    self.nodes.append(ConvBn(3, F1//(i+1), 15, (1,2), (0,14)))
    self.nodes.append(ConvBn(3, F1//(i+1), 15, (1,4), (0,28)))
    self.nodes.append(ConvBn(3, F1//(i+1), 15, (1,8), (0,56)))
```

`forward` indexes them as `self.nodes[(len(path_ids) - 1) * 4 + id]`. So the
layout is:

| path length | channel width | indices | dilations available |
|---|---|---|---|
| 1 | `F1`      | 0–3 | 1, 2, 4, 8 |
| 2 | `F1 // 2` | 4–7 | 1, 2, 4, 8 |

`ConvBn` is `Conv2d(..., kernel_size=(1, k), dilation=(1, d), padding=(0, p))`
— time-axis convolution, channel dimension untouched. The four kernels **all
have `k=15`**; only the dilation changes. With `padding = d * 7` the temporal
length is preserved exactly.

So the implementation plan's description of the coupling is accurate: the
index depends on how many paths are selected, and `F1 // 2` for two paths is
what makes the concatenation channel count come out right.

Channel arithmetic for BCI-IV-2a (`nChan=22`, `num_Feat=36`):

- `F1 = num_Feat // 3 = 12` per band.
- 2 paths → `2 × (12 // 2) = 12` channels per band → `3 × 12 = 36` after concat.
- `shadow_bn=True` creates `bn_list = [BatchNorm2d(F1) for _ in range(2)]`, i.e.
  two BatchNorm2d(12) layers, selected by `bn_list[len(path_ids) - 1]` — one per
  path length.
- `SCB`: `Conv2dWithConstraint(36, 288, (22, 1), groups=36)` → `[B, 288, 1, 1000]`.
- `reshape` to `strideFactor=8` → `[B, 288, 8, 125]`.
- `LogVarLayer(dim=3)` → `[B, 288, 8, 1]` → flatten → **2304**.
- `LastBlock(2304, nClass)` = `LinearWithConstraint(..., max_norm=0.5)` then
  `LogSoftmax`.

The `[B, 9, 22, 1000]` input shape is confirmed directly in `ho.py`
(`torch.randn(1, 9, 22, 1000)`) and the `permute` in `SuperNet.forward`. The
`LogVarLayer` definition is

```python
torch.log(torch.clamp(x.var(dim=self.dim, keepdim=True), 1e-6, 1e6))
```

## 4. Where the plan's framing needs care

**(a) "First-order DARTS" has no counterpart upstream.** The plan's core
contribution — relaxing each edge to `softmax(alpha)` over 16 candidates and
differentiating the architecture — is new construction, not a change to the
existing search. That is fine, but it means "keep the official FBNAS as a
frozen baseline for comparison" compares two different search paradigms, and
any claim about the 4-operator × 4-RF space should be scoped to the new method.

**(b) The 4 "RF" values are dilations, and their effective receptive fields are
not any of the numbers in the plan.** For `k=15` on the time axis, effective RF
is `1 + d·(k−1) = 1 + 14d`:

| dilation | effective RF | plan's `RF_SPACE` entry (Low) |
|---|---|---|
| 1 | 15 | 25 |
| 2 | 29 | 49 |
| 4 | 57 | 97 |
| 8 | 113 | 193 |

Note `1 + 14d` is always **odd**, while `RF_SPACE` entries are `2k+1` for
`k ∈ {12, 24, 48, 96}` — i.e. the plan's numbers assume a kernel of `2d+1`
applied at unit stride. The plan's `BASE_KERNEL`/`RF_SPACE` relationship is
internally consistent as *design intent* (`dilated k=25 d=4` and `normal k=97
d=1` both give RF 97), but it is a **new operator set**, not a relabelling of
the upstream convs. Worth stating explicitly in the write-up so the receptive
field claim is defensible.

**(c) The upstream search space is size-1-or-2 *subsets*, not 16 independent
candidates.** Because a size-2 path concatenates two convs (12 channels total),
the supernet's 16-candidate relaxation per edge is a genuinely different
parameterisation from upstream's 4 + 6 = 10 subsets per band.

**(d) Class/docstring drift upstream.** `Conv2dWithConstraint` documents
`max_norm=1` but every call site passes `max_norm=2`; `doWeightNorm` is
implemented by mutating `weight.data` inside `forward`, with no
`torch.no_grad()` guard. Worth not copying that pattern into new code.

**(e) Official code depends on the working directory.** `networks.py` does
`from utils import ...` (flat import) and `ho.py` inserts `centralRepo` onto
`sys.path`. Importing the frozen baseline from new code will need the same
manipulation or a shim; it is not importable as a package.

## 5. Data split, confirmed

From `ho.py`: `config['validationSet'] = 0.2`, the split is a **contiguous**
slice (`range(0, ceil(n*0.8))` train, `range(ceil(n*0.8), n)` val), session 0 is
train and session 1 is test, and the search validation DataLoader uses
`batch_size=len(valData)`. `maxEpochs=1500` with `NoDecrease: numEpochs=200`
matches the plan's retrain settings.
