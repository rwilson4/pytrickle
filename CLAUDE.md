# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when
working with code in this repository.

## Commands

Use [uv](https://github.com/astral-sh/uv). Packaging is PEP 621
`pyproject.toml` with a hatchling backend; `uv.lock` is committed.

```bash
uv sync --extra dev          # runtime deps, test/lint tools, sklearn, sphinx
uv run pytest                # 107 tests, ~15s
uv run pytest tests/test_trickle_down.py::TestHierarchy::test_hierarchy
uv run mypy                  # checks trickle_down/ and paper/ (set in pyproject)
uv run ruff check trickle_down tests paper
uv run ruff format trickle_down tests paper   # or --check to verify
```

All tool configuration lives in `pyproject.toml`:
- **Formatting:** `ruff format` replaces black. It also formats the
  Python code blocks inside `trickle_down/USAGE.md`.
- **Linting:** ruff uses the same rule set as `~/Projects/gamdist`.
  Its isort sorts names case-insensitively (`order-by-type = false`),
  matching the existing imports.
- **Type checking:** mypy requires full annotations on our own code. It
  skips missing stubs for scipy (scipy-stubs needs Python 3.12; this
  project supports 3.11) and EoN.
- **Tests:** pytest turns warnings into errors.

## CI and branching

The branching model mirrors `~/Projects/gamdist`:
- Do work on a `claude/<topic>` branch.
- Open a PR against `main`, and merge it with a merge commit.
- Commit subjects start with `[Claude]` when Claude commits.

CI (`.github/workflows/ci.yml`) runs on pushes and PRs to main, on
Python 3.11 and 3.12. It runs `ruff check`, `ruff format --check`,
`mypy`, and `pytest --cov=trickle_down --cov-fail-under=80`. Its install
step uses `uv sync --locked`, so update `uv.lock` (`uv lock`) whenever
you change dependencies.

`.github/workflows/publish.yml` publishes to PyPI by trusted
publishing when a `v*` tag is pushed. The distribution is
`trickle-down` and the import name `trickle_down`; the project was
called pytrickle until November 2026 (`pytrickle` on PyPI is an
unrelated project). Bump `version` in `pyproject.toml` before
tagging.

Licence: Apache 2.0 (`LICENSE.txt`). Add a `changelog.txt` entry for
behaviour changes.

## Docs

The Sphinx site in `docs/` (pydata theme, modeled on `~/Projects/gamdist`
and `~/Projects/PairMatch`) is served at
https://convexanalytics.com/trickle-down/.
- `docs/user_guides/usage.md` includes `trickle_down/USAGE.md` through
  MyST, so the usage guide has one source.
- `docs/user_guides/end_to_end.rst` runs the whole workflow on
  simulated data as doctests: a scikit-learn T-learner, a summary
  tree grown as deep as `min_samples_leaf` allows, trickle down on
  the evaluation split, and pruning of the report (not a retest) to
  the certified claims. Its outputs are pinned by `uv.lock`; a
  scikit-learn upgrade can change them.
- `docs/user_guides/reproducing_the_paper.rst` recomputes the paper's
  exact results (Table 1) as doctests, and shows the simulation study
  beside the paper's numbers.
- `docs/api.rst` renders the docstrings via autodoc.

```bash
cd docs && uv run make html SPHINXOPTS="-W --keep-going" && uv run make doctest
```

`.github/workflows/deploy_docs.yml` builds with `-W` and runs the
doctests on PRs and on pushes to main that touch `docs/`,
`trickle_down/`, `paper/`, `pyproject.toml` or `uv.lock`. On pushes it
rsyncs `docs/_build/html/` to the droplet with the `DOCS_DEPLOY_KEY`,
`DOCS_USER` and `DOCS_HOST` secrets; the key's `rrsync` forced command
on the server fixes the destination, `/srv/docs/convexanalytics/trickle-down`.
`docs/_build/` is not committed.

## Paper reproduction

`paper/` reconstructs the simulation study of §7. The paper's own
script is not public; see "Paper" below.
- `paper/simulation.py` holds the data-generating process, the fixed
  and fitted partitions, the five procedures, and the scoring of true
  claims. It uses only the public `trickle_down` API.
- `ASSUMPTIONS` lists every gap in the paper's description and how it
  is filled, and the docs render that list. Two assumptions were
  calibrated against the paper's tables, and say so: `epsilon_1 ~ N(0,
  9)`, and `+0.30` for the helped quarter.
- `paper/published.py` transcribes Tables 2-4 and Figure 2 from the
  arXiv source.
- `uv run python -m paper.reproduce` runs everything (about 5 minutes
  on 10 cores) and writes `paper/results.json` plus the RST fragments
  in `docs/user_guides/_generated/`. Commit both;
  `--render-only` rebuilds the fragments from the JSON.

Leaves of the interweaved hierarchy stake one. The paper's Figure 3
and the power tables confirm that its simulations did the same, so
don't "fix" this to match the paper's remark that trickle down and
Hierarchical Bonferroni agree exactly under the global null.

## Paper

This library accompanies Wilson (2026), *Interpretable and Validatable
Uplift Modeling*, arXiv:2610.37966. Code comments cite its sections:

- §2 data splitting (fit on training, test on evaluation)
- §3 interweaved effect/heterogeneity hypotheses
- §4 adaptive direction selection
- §5.1 Hierarchical Bonferroni
- §5.2 Fixed Hierarchy
- §5.3 Trickle Down
- §6 confidence bounds for retained hypotheses
- §7 simulation study

The code uses the paper's names for the two hypothesis kinds, *effect*
and *heterogeneity*: `Claim.kind` is `"effect"` or `"heterogeneity"`,
hypotheses are named `E[node]` and `H[node]`, and the stakes are
`effect_stake` and `heterogeneity_stake`. Don't reintroduce the old
names (*magnitude*, *difference*); "difference" survives only for the
arithmetic, as in `ContrastTest.difference_in_differences`.

## Architecture

**Purpose:** Controlling the familywise error rate (FWER) when testing
statistical hypotheses arranged in a tree. The intended application is
uplift modeling: a shallow decision tree summarizes individual
treatment-effect predictions, and its nodes become the hypothesis
hierarchy. Trickle down is one of three procedures in the paper,
alongside Hierarchical Bonferroni and Fixed Hierarchy. It is the most
flexible: testing continues at lower levels even when a parent is
retained.

**`hypothesis.py`:** `Hypothesis` is the abstract base class, with
`weight`, `stake` and `available_level`.
- The *stake* (0–1) is the fraction of the available level wagered on
  this node's test. The node is tested at `stake * available_level`,
  and that rule lives in the base class's `test_hypothesis()`.
- Subclasses implement only `pvalue()`. They can also override
  `confidence_interval(level)`, which inverts the test.
- Concrete classes:
  - `TTest`: Welch, optionally one-sided against a threshold `delta`.
  - `ContrastTest`: a linear contrast across groups, built for an
    uplift split by `ContrastTest.difference_in_differences`.
  - `FTest`: one-way ANOVA.
  - `FixedPValue`: a p-value computed elsewhere.
- Helpers: `contrast_moments`, `t_pvalue`, `invert_test` and
  `select_alternative` (picks the one-sided direction from the
  training split).

**`hierarchy.py`:** `Hierarchy(fwer, root)` runs BFS-order testing in
`test_hypotheses()`. It normalizes child weights recursively. A
rejected node passes `available_level * child.weight` to each child; a
retained node passes `(1 - stake) * available_level * child.weight`.
`hypotheses()` returns the nodes in BFS order. `visualize_hierarchy()`
imports matplotlib, networkx and EoN lazily, so the package imports
without them.

**`uplift.py`:** The paper's interweaved architecture.
- `ArmSummary` holds per-arm moments; `from_samples` uses ddof=1 and
  rejects non-finite values.
- `UpliftNode` is a binary tree node with evaluation-split arms and a
  training-split `tau_train`.
- `interweaved_hierarchy()` builds the tree E[S] → H[S] → {E[L],
  E[R]}. It returns `UpliftClaims`, a list of `Claim`s around a
  `Hierarchy`.
- `Claim` knows its level, its simultaneous confidence `interval`, its
  `reportable_bound` (truncated at `delta`, per §6), and how to state
  itself in a sentence. `print(UpliftClaims)` gives an orgtbl summary.
- Comparators: `bonferroni_rejections` and
  `hierarchical_bonferroni_rejections`. The latter uses
  hypothesis-tree depth, which is not `Claim.depth`.

**`graphical.py`:** `GraphicalProcedure` is a reference implementation
of Bretz et al. (2009). `trickle_down_graph()` encodes a trickle-down
tree as an equivalent allocation and transition matrix.
`tests/test_graphical.py` uses them to check the equivalence
empirically.

**Public API (`__init__.py`):** Exports everything listed in
`__all__`, including `usage()`, which returns the packaged
`trickle_down/USAGE.md`. Keep USAGE.md in sync with the API; a test
checks that it ships and mentions `interweaved_hierarchy`.

**Imports:** Modules import one another as `from trickle_down.x import
...`, never as bare `from x import ...`.
