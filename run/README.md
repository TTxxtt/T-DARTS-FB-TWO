# run/ — reproduction harness

Driver scripts for running the FBNAS and T-DARTS arms over the nine BCI-IV-2a
subjects.  Nothing here is part of the library; it exists to launch jobs and to
keep the vendored baseline untouched.

```
run/
├── py/run_fbnas_subject.py   # the only place that imports the frozen baseline
├── bin/run_fbnas_job.sh      # one subject, official FBNAS pipeline
├── bin/run_darts_job.sh      # one subject, DARTS search
├── bin/submit_fbnas.sh       # quota-aware submitter, 9 subjects
├── bin/submit_darts.sh       # quota-aware submitter, 9 subjects
├── sh_log/                   # sbatch stdout/stderr + agenda files
├── outputs/                  # run artefacts (git-ignored)
└── logs/                     # per-epoch progress logs (git-ignored)
```

## Three constraints that shape everything

**C1 — `FBNAS/**` must not change by a single byte.**
`tests/test_fbnas_compatibility.py:465-482` recomputes the git blob SHA-1 of
every file under `FBNAS/` and compares it against `FBNAS/_MANIFEST.json` (52
blobs).  Any extra file — including bytecode — fails
`test_no_extra_files_were_written_into_the_baseline`.  Therefore paths that the
official code needs can only be supplied as symlinks, never as real directories
and never by editing `ho.py`.

**C2 — `ho.py` derives every path from `__file__`.**
`ho.py:18` computes `masterPath` from its own location, `:96-97` builds the data
path as `<repo>/FBNAS/data/<dataset>/multiviewPython`, and `:113-115` builds the
output path as `<repo>/FBNAS/output/<dataset>/ses2Test`.  It reads no environment
variable and takes no path argument, so those two names are the only hooks.
`ho.py:57` also hardcodes `randSeed = 20190821`, so the FBNAS arm is
single-seed by construction and cannot be made otherwise without violating C1.

**C3 — `python ho.py ...` cannot be run directly.**
`ho.py:431` reads positional arguments from `sys.argv[1:]`.  Those positionals
are still in `sys.argv` when `nas_phase()` reaches `NAS.get_args()` →
`argparse.parse_args()` (`NAS.py:55`), which rejects them with `SystemExit(2)`.
`py/run_fbnas_subject.py` imports `ho` instead of executing it, which sidesteps
the `__main__` block entirely.

## One-time setup: two symlinks

```
cd <repo>/FBNAS
ln -s <data-root>  data      # ho.py:97 expects FBNAS/data/bci42a/multiviewPython
ln -s <repo>/run/outputs/fbnas  output   # ho.py:114 expects FBNAS/output
```

`output` is not optional.  A real directory there would be picked up by the
integrity test's recursive scan and counted as baseline pollution.  A symlink is
safe: `Path.rglob` does not descend symlinked directories (verified on 3.10 —
`is_dir(follow_symlinks=False)` in the recursive selector), so neither the
symlink nor anything behind it is ever collected.

**These two symlinks are not covered by `.gitignore`.**  The `data/` pattern has
a trailing slash, which matches directories only, and git treats a symlink as a
non-directory; `output` does not match `outputs/` either.  Add them to
`.git/info/exclude` (local, uncommitted) so the repo stays clean:

```
/FBNAS/data
/FBNAS/output
```

## Running

One-time, so that `sbatch`'s `-o sh_log/...` has somewhere to land when you
submit a job by hand (the submit scripts do this themselves):

```bash
mkdir -p sh_log/fbnas sh_log/darts outputs/fbnas outputs/darts logs/fbnas logs/darts
```

Pilot first, to measure wall clock:

```bash
SUB=0 sbatch -p GPUFEE04 bin/run_fbnas_job.sh
SUB=0 SEED=20250901 sbatch -p GPUFEE04 bin/run_darts_job.sh
```

Then the full sets.  The DARTS arm needs both stages, and every retrain job reads
what its own search job wrote, so submit them only after the searches finish:

```bash
bash bin/submit_fbnas.sh
bash bin/submit_darts.sh
bash bin/submit_darts_retrain.sh    # after the searches above are done
```

Set `OBSERVE_TEST=1` in the environment to add `--observe-test` to a retrain job:
Session 1 metrics are then logged every epoch so you can watch the trend, and
they are still read by no stop, checkpoint or selection decision.

`sh_log/` holds the sbatch logs; the per-epoch progress logs land under
`logs/<arm>/bci42a/`.

## Layout

There are two trees, because the two arms write their results differently.

**The DARTS arm** goes through `run_layout.py`, which assembles
`<root>/<dataset>/<phase>_s<subject>_seed<seed>[_<arm>]`.  The phase sits inside
the leaf rather than in a directory level of its own, so a search directory
keeps the plain `search_s003_seed20250901` spelling:

```
run/outputs/darts/bci42a/search_s003_seed20250901/   # DARTS search
run/outputs/darts/bci42a/train_s003_seed20250901/    # DARTS retrain
run/logs/darts/bci42a/search_s003_seed20250901.log
run/logs/darts/bci42a/train_s003_seed20250901.log
```

**The FBNAS arm** is driven by the authors' `ho.py`, which builds its own output
path from `__file__` (`ho.py:114-115`).  Everything lands under
`FBNAS/output/<dataset>/ses2Test/<timestamp>/sub<i>/`, which the `FBNAS/output`
symlink redirects into `run/outputs/fbnas/`.  That tree is shaped by `ho.py`,
not by `run_layout.py`, and renaming it would violate C1:

```
run/outputs/fbnas/bci42a/ses2Test/<timestamp>/sub<i>/ # FBNAS search + training
```

**The cross-check tool** `train_fbnas_baseline.py` does use `run_layout.py`, and
writes beside the DARTS tree.  It is deliberately *not* wired into the harness —
see its module docstring and the note below:

```
run/outputs/fbnas/bci42a/search_s003_seed20190821/   # cross-check search
run/outputs/fbnas/bci42a/train_s003_seed20190821/    # cross-check training
```

Leaves must keep carrying `seed<digits>`: `tdarts/genotype.py` recovers the
search seed from the directory name, so `train_retrain.py` cannot read a search
directory that does not spell it out.

Because a `run_layout` leaf has no timestamp, a second run with the same subject
and seed stops at `mkdir(exist_ok=False)` rather than silently overwriting.  That
is deliberate — `run/outputs/` is git-ignored, so nothing there is recoverable
from git, and a collision is worth surfacing rather than clobbering.  The
`ho.py` tree is timestamped by upstream and does not have this property.

## Why the FBNAS arm runs ho.py, not train_fbnas_baseline.py

`ho.ho()` runs the search *and* the two-phase final training in one call
(`ho.py:316/318` invoke `nas_phase`, `:343` calls `baseModel.train`), so one
FBNAS job per subject produces a trained, tested model.  `train_search.py` only
searches, so the DARTS arm needs two jobs per subject — nine subjects is 9 FBNAS
jobs against 18 DARTS jobs.

The headline FBNAS numbers come from the authors' own code, which is a stronger
thing to write in a paper than a reimplementation.  `train_fbnas_baseline.py`
re-implements the same two-phase protocol on the tdarts pipeline and is kept as
a **cross-check** instead: run it on one subject and compare against the
official run for that subject.  Agreement is evidence that `train_retrain.py` —
which implements that protocol for the DARTS arm — is faithful; disagreement is
a finding worth chasing before reporting anything.  It is deliberately not wired
into `run/bin/`.

One caveat if you ever do switch: `train_fbnas_baseline.py` imports the official
modules too (`NAS`, `networks`, `eegDataset`), so it carries the same
bytecode-into-`FBNAS/` hazard.  It suppresses it with `sys.dont_write_bytecode`,
but the hazard does not disappear.
