# Session-2 results: searching the mechanism does not improve cross-session transfer

Rules were frozen in [FINAL_SESSION2_PROTOCOL.md](../FINAL_SESSION2_PROTOCOL.md).
Implementation is in [session2_arm_comparison.md](session2_arm_comparison.md).
Artifacts: `run/outputs/operator_armB_comparison/session2_paired.json`.

**Arm A** = dilated-only FBNAS search (vendored upstream chain).
**Arm B** = four-mechanism hierarchical search (family at RF57, then RF within it).
Nine subjects, seed 20190821, evaluated once on the dataset's second recording
session (code `session=1`).

## Verdict

**The mechanism search does not improve Session1 → Session2 generalisation. The
point estimate is negative, and only 1 of 9 subjects favours it.**

## Paired Session-2 results

| subject | Arm A acc | Arm B acc | ΔAcc | Arm A F1 | Arm B F1 | ΔF1 |
|---|---|---|---|---|---|---|
| 001 | 0.8438 | 0.8229 | −0.0208 | 0.8443 | 0.8235 | −0.0208 |
| 002 | 0.6007 | 0.5868 | −0.0139 | 0.5931 | 0.5794 | −0.0138 |
| 003 | 0.9236 | 0.9097 | −0.0139 | 0.9233 | 0.9098 | −0.0135 |
| 004 | 0.7535 | 0.5972 | **−0.1562** | 0.7521 | 0.5941 | −0.1580 |
| 005 | 0.7361 | 0.7812 | **+0.0451** | 0.7163 | 0.7763 | +0.0600 |
| 006 | 0.6493 | 0.5660 | −0.0833 | 0.6507 | 0.5669 | −0.0837 |
| 007 | 0.8924 | 0.8715 | −0.0208 | 0.8923 | 0.8698 | −0.0226 |
| 008 | 0.8333 | 0.8299 | −0.0035 | 0.8328 | 0.8303 | −0.0025 |
| 009 | 0.8299 | 0.8056 | −0.0243 | 0.8277 | 0.8031 | −0.0246 |

| metric | Arm A | Arm B | paired Δ | sd | Arm B wins |
|---|---|---|---|---|---|
| accuracy | 0.7847 ± 0.1086 | 0.7523 ± 0.1322 | **−0.0324** | 0.0568 | 1/9 |
| macro-F1 | 0.7814 ± 0.1111 | 0.7504 ± 0.1333 | −0.0311 | 0.0600 | 1/9 |
| kappa | 0.7130 ± 0.1448 | 0.6698 ± 0.1763 | −0.0432 | 0.0757 | 1/9 |

### The two tests disagree, and that is the honest summary

* **Paired t**: t(8) = −1.712, 95% CI **−0.0761 .. +0.0112**. Spans zero — the
  magnitude is too variable to call.
* **Sign test**: 1/9 in Arm B's favour, two-sided **p = 0.039**. The *direction*
  is consistent.

So: no significant difference in magnitude, but a consistent direction against
Arm B. Reading this as "no difference" would be wrong; reading it as "Arm B is
significantly worse" would also be wrong.

**Leave-one-out** (does one subject carry it?): dropping the −15.6pp outlier
(004) moves the mean from −0.032 to −0.017. All nine leave-one-out means are
negative, spanning −0.017 .. −0.042. The direction is robust; the magnitude is
not.

## Why: the family freeze, not overfitting

The obvious hypothesis — that Arm B's *second* selection step (Phase A over 64,
then Phase B over 1000) buys extra overfitting to the 57-trial validation split
— is **refuted by the data**:

| | selected validation accuracy |
|---|---|
| Arm A | 0.7863 ± 0.1286 |
| Arm B | 0.7297 ± 0.1456 |
| mean(Arm B − Arm A) | **−0.0565** |

Arm B's chosen architectures fit Session-0 validation *worse*, not better. And
their Session-2 accuracy is also worse. So the search is not finding
better-overfit architectures; it is finding **worse ones**.

The mechanism that fits: **Phase A freezes each band's family, and Phase B can
only tune the receptive field inside that frozen choice.** A bad family decision
is unrecoverable. Arm A has no such commitment — every band searches all four
RF structures at both path counts directly.

## Exploratory: more substitution, worse transfer

*Post hoc and hypothesis-generating — not pre-registered, n is small, and the
worst group holds only two subjects. Reported because it is the coherent story,
not because it is confirmatory.*

| bands switched away from `dilated_e` | subjects | mean ΔAcc |
|---|---|---|
| 1 | 3 | −0.0127 |
| 2 | 4 | −0.0035 |
| 3 | 2 | **−0.1198** |

`corr(#bands switched away from dilated_e, ΔAcc) = −0.658`

The two subjects where Arm B replaced all three bands (004, 006) are the two
worst results. Across the 27 frozen band-slots the search chose `gated_e` 9
times, `dilated_e` 10, `band_gated_e` 6, `dynamic_e` 2 — so Session-0 validation
*did* prefer non-dilated mechanisms, and those are exactly the substitutions
that transferred worst.

## How this fits the earlier stages

| stage | finding |
|---|---|
| Matched-V2 | strong global operator effect, no subject×operator interaction; `dilated`/`dynamic` on top |
| Expressive-V2 | same, with capacity released |
| Band probe | band effect, but no stable per-subject band preference (C3 failed 1/4) |
| **This stage** | **the first held-out cross-session test: mechanism search does not help, and substituting more mechanism tracks worse transfer** |

Four stages, three of them Session-1-internal and one genuinely cross-session,
now point the same way. The band probe's verdict was "temporal mechanism
personalization evidence weak"; this is the cross-session confirmation of it.

## What this does not establish

The five limitations in the protocol file all still apply and are not repeated
here in full. The two that bear on this result:

1. **The arms differ in more than their search spaces** — Arm A runs the
   upstream `SuperNet`/`FBNASNet` and `baseModel.train`; Arm B runs the `tdarts`
   backbone and `train_retrain.py`. `ΔAcc` therefore cannot be attributed
   cleanly to the search space alone.
2. **n = 9, one seed, and the selection metric resolves to a single validation
   trial** (protocol limitation #7). Neither arm's architecture was sharply
   resolved on Session 0, so this compares two pipelines more than it compares
   two precisely-chosen architectures.

The substitution correlation in the section above is exploratory. It would need
its own pre-registered design — more subjects, and a controlled family
assignment rather than one read off the search — before it could carry weight.

---

# Frozen outcome (2026-09-19)

Per the review decision, this line is **closed**. The conclusion is frozen as:

> **扩大 temporal mechanism 搜索空间，在当前协议下反而降低跨-session 泛化。**

Expanding the temporal-mechanism search space, under the current protocol,
*reduces* cross-session generalisation.

## Why it is frozen rather than investigated further

An implementation audit was run before freezing, because a negative result from
buggy code is not a negative result. It found no defect:

| check | result |
|---|---|
| Does the searched architecture execute identically in the search stage and the retrain stage? | **max abs difference = 0.0** (weights transplanted, outputs compared element-wise) |
| Stage-2 rule, split, batch size, selection variable, seed vs. Arm A's own archived `config.csv` | matched, pinned by `Stage2ProtocolParityTests` |
| `genotype.json` vs `final_summary.json` vs `phase_a_result.json` (families and RFs) | self-consistent on every subject checked |
| `set_seed` placement relative to both model constructions | immediately before, both phases |
| params / MACs | Arm A 26284 / 40.0M; Arm B 18292–20396 / 45.2–91.9M |

**Arm B is not to be tuned against Session 2.** Adjusting it now, having seen
these numbers, would be test-set tuning, and the comparison would stop meaning
anything. The negative result stands as recorded.

## What is *not* claimed

The frozen claim is scoped to *this protocol and these arms*. It is not the
claim that temporal-mechanism search is useless in general, and it is not a
verdict on the operators themselves: Arm B differs from Arm A in its search
space, its search implementation and its network implementation at once, so the
3.2-point gap cannot be attributed to any one of them. What the audit
establishes is only that the gap is not an execution defect.

The one thing that did point at the operators — more substituted bands tracking
worse transfer (`corr = -0.658`) — is post hoc, is confounded with the same
three-way difference, and rests on two subjects in its worst group. It is
recorded as a hypothesis, not a finding.

## Methodological note carried forward

The dataset's second session has now been read and its results are known. **Any
frequency-frontend method designed in response to this cannot claim a
never-seen test on BCIC-IV-2a Session 2.** Development for the next stage stays
locked to Session 1; final reporting on 2a Session 2 is a retrospective
comparison, and an independent cross-session dataset (e.g. OpenBMI) is needed
for genuine external confirmation.
