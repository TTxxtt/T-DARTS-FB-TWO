# Arm B implementation record

Companion to [FINAL_SESSION2_PROTOCOL.md](../FINAL_SESSION2_PROTOCOL.md), which
freezes the *rules*. This file records what was *built*, what was verified, and
what was deliberately left alone.

## The question

Matched-V2, Expressive-V2 and the band probe all measured differences **inside
Session 1** — they say which temporal mechanism fits the 57-trial validation
split better, and nothing about whether that choice generalises. This stage
stops designing operators and asks the only question that matters:

> Does searching over temporal mechanisms improve Session-1 → Session-2
> generalisation, compared with searching only the receptive field of a dilated
> convolution?

## Two arms

**Arm A — dilated-only FBNAS search.** The vendored upstream chain, unchanged:
`run/py/run_fbnas_subject.py` → `ho.py`. Its four candidates per band are kernel
15 at dilation 1/2/4/8 — receptive fields 15/29/57/113 — and one or two are
selected, so `10**3 = 1000` architectures. Already run: nine subjects, seed
20190821, with `opt_choice.csv`, `cali_bn_acc.npy` and Session-2 results under
`run/outputs/fbnas/bci42a/ses2Test/`. **No new Arm A jobs exist in this stage.**

**Arm B — four-mechanism hierarchical search.** New code, three search spaces
served by one engine:

| phase | per-band candidate | `m` | space |
|---|---|---|---|
| A | one of four families at RF 57 | 1 | `4**3 = 64` |
| B | the frozen family at one of four RFs | 2 | `10**3 = 1000` |

Phase B is *the same space Arm A searches*, which is what makes "searching the
mechanism too" the only difference between the arms.

## What was built

| file | role |
|---|---|
| `tdarts/fbnas_sampler.py` | `random_choice` / `traverse_choices` — a behavioural port of the baseline's subnet sampler and candidate enumeration |
| `tdarts/band_supernet.py` | the FBNAS-idiom supernet, both width-general family variants, calibrated traversal |
| `tdarts/band_discrete.py` | Arm B's genotype dialect (`band_family_rf`) and its discrete network |
| `train_arm_b_search.py` | Phase A → freeze → Phase B → `genotype.json` |
| `train_retrain.py` | **additive** third `scheme` branch (the only edit to an existing entry point) |
| `run/bin/run_arm_b_{search,retrain}_job.sh`, `run/bin/submit_armB.sh` | 9 searches + 9 retrains, two ledgers |
| `tools/extract_arm_a_session2.py` | read-only normaliser for Arm A's archive |
| `tools/compare_session2_arms.py` | the paired table |
| `tests/test_arm_b_search.py` | 60 tests |

### Why the search engine is new

`tdarts/` had no random-subnet sampler: the only discrete draw in the tree is
`gumbel_one_hot` (`tdarts/hierarchical.py:43`), and the baseline's MixPath
sampling is bound to the frozen `SuperNet`/`FBNASNet` classes. Arm B needs the
same *flow* over the `tdarts` backbone, so `tdarts/fbnas_sampler.py` ports it —
and is then checked **against the original** rather than against a restatement
of it: same enumeration order for `m=1..4`, and identical draws for 80
`(m, seed)` pairs with the same RNG consumption order.

### The channel arithmetic, and the two width-general families

A band holding `k` paths runs each at `F1 // k` channels and concatenates, so it
always emits `F1 = 12` and the backbone never sees a different width. That is
upstream's `F1//(i+1)` trick, and it makes a path-count choice structural rather
than a capacity choice.

Two families could not be built at width 6: `GatedTemporalConvE` and
`BandGatedConvE` pin an internal width to the module constant 12
(`tdarts/operator_v2e.py:188`, enforced at `:190-198`). Rather than edit the
frozen Expressive generation, `band_supernet.py` defines width-general
subclasses with that width tied to `out_channels`, injected through
`V2TemporalOp`'s existing `registry=` keyword — the same extension point
`build_e_operator` already uses.

**At `out_channels == 12` they are the probed operators, bit for bit** — same
state-dict keys, same shapes, same values under the same seed, same reported
support. A test pins that, because Phase A's family comparison is only
meaningful if the families are the ones that were measured.

Residual capacity variation across path counts (a two-path band minus a
one-path band): `dilated_e` 0, `dynamic_e` +16, `gated_e` −72, `band_gated_e`
+3. These are per-path 1×1/gate overheads, not width artefacts — two paths
carry two of them where one path carried one — and they are pinned by test so
they cannot drift unnoticed. The worst is 0.3% of the model.

## How the reuse works

`train_retrain.py` already dispatched `genotype.json` on a `scheme` tag (plain
six-gene vs `anchored_operator_then_rf`). Arm B adds a third branch rather than
a second trainer, so the two-stage protocol, the Stage-2 stopping rule, the
params/MACs accounting and the `--observe-test` guard are shared verbatim. Four
sites needed handling:

1. the dispatch itself (`train_retrain.py:504-521`);
2. `path_structure_keys` / `duplicate_structure_bands`, which parse a
   `"<op>_rf<int>"` candidate string from the 14-name registry and are called
   unconditionally before `allocate()` — the band dialect gets its own reader
   built from real operators;
3. model construction, which now picks `BandDiscreteNet` or
   `TemporalDiscreteNet`;
4. `--stage2-only`, which reconstructs through `load_genotype` and would
   silently reject a band file — it now **refuses explicitly**. This path is not
   used here anyway (`--observe-test` and `--stage2-only` are mutually
   exclusive), so a refusal is the honest outcome rather than a half-working
   resume.

The two reference dialects are unchanged; the plain branch is still the
fallback for an untagged file, and a test asserts that ordering so a future
condition cannot swallow it.

## Verification

```
494 tests pass            (439 before this stage, +55 here)
0 red-line violations     session1_opened=false, session2_test=null,
                          screening_only=true, session1_used=false
```

`tests/test_arm_b_search.py` covers, beyond the sampler agreement above:

* **Bit-identity** of both width-general families at the probed width.
* **Capacity invariance** — the exact cases and the pinned residuals.
* **Candidate-space size** — 64 and 1000, against the baseline's own
  `traverse_choice`.
* **Genotype dialect** — round-trip, the human-readable export, and eight
  rejection paths (wrong scheme, excluded family, 3 RFs, no RFs, repeated RF,
  off-ladder RF, missing band, mixed families).
* **Traversal semantics** — restoring the weights is what makes a candidate's
  score independent of its position in the sweep; asserted both positively
  (reverse order gives the same numbers) and structurally.
* **Stage-2 protocol parity** — Arm A's own archived `config.csv` is parsed and
  matched against Arm B's job script: `maxEpochs 1500`, patience `200` on
  `valInacc`, `continueAfterEarlystop`, `lr 1e-3`, `validationSet 0.2`,
  `batchSize 16`, `randSeed 20190821`. This is the check that would otherwise be
  a reviewer's act of faith.
* **Frozen guards** — the archived Expressive capacities, the width-6 guard that
  motivated the subclasses, and that Arm B writes nowhere near a frozen root.

Smoke run (subject 003, seed 20190821, one epoch per phase, Session 2 closed):
search produced a valid `band_family_rf` genotype, and `train_retrain.py`
consumed it through the new branch — `genotype_source.kind =
"band_genotype_json"`, `session1_test_size = null`, `test = null`,
`stage2_final.pt` absent. Both smoke directories were removed afterwards so the
submission driver's `allocate(exist_ok=False)` sees a clean tree.

## Running it

```bash
python tools/extract_arm_a_session2.py          # read-only; Arm A
cd run && bash bin/submit_armB.sh dry           # preview the 18 jobs
cd run && bash bin/submit_armB.sh drain         # resumable, two ledgers
python tools/compare_session2_arms.py           # the paired table
```

`submit_armB.sh` submits a subject's retrain **only** once that subject's search
has written both `genotype.json` and `final_summary.json`, so the architecture is
provably frozen on disk before the second session is ever read.

## What this stage does not claim

See [FINAL_SESSION2_PROTOCOL.md § Honest limitations](../FINAL_SESSION2_PROTOCOL.md#honest-limitations).
In short: Arm A's Session-2 numbers were already unblinded; the arms differ in
more than their search spaces; Arm B's search budget is 400 epochs against Arm
A's 200; `--observe-test` exists only on Arm B; and two families are extrapolated
to width 6.
