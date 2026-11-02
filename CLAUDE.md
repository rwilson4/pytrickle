# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when
working with code in this repository.

## Commands

This project uses `poetry` for dependency management (note: the global
style guide prefers `uv`, but this repo uses `poetry`).

```bash
# Install dependencies
poetry install

# Run all tests (includes coverage report)
poetry run pytest

# Run a single test
poetry run pytest test/test_pytrickle.py::TestHierarchy::test_hierarchy

# Type checking
poetry run mypy pytrickle/

# Linting and formatting
poetry run ruff check pytrickle/ test/ simulation.py
poetry run black pytrickle/ test/ simulation.py

# Worked example (needs `brew install libomp` for xgboost on macOS)
poetry run python simulation.py
```

Tests automatically run with coverage (≥80% required) via `tox.ini`.

Tool configuration lives in `pyproject.toml`. Ruff's isort sorts names
case-insensitively (`order-by-type = false`), matching the existing
imports. Mypy skips missing stubs for scipy (scipy-stubs needs Python
3.12; this project supports 3.11), EoN, sklearn and xgboost.

Run `simulation.py` from a scratch directory if you don't want to
overwrite `simulation_results.png`; `poetry -C` runs it from the repo
root.

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

The paper calls the two hypothesis kinds *effect* and *heterogeneity*;
the code calls them *magnitude* and *difference*.

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
- `interweaved_hierarchy()` builds the tree M[S] → D[S] → {M[L],
  M[R]}. It returns `UpliftClaims`, a list of `Claim`s around a
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
`test/test_graphical.py` uses them to check the equivalence
empirically.

**Public API (`__init__.py`):** Exports everything listed in
`__all__`, including `usage()`, which returns the packaged
`pytrickle/USAGE.md`. Keep USAGE.md in sync with the API; a test
checks that it ships and mentions `interweaved_hierarchy`.

**Imports:** Modules import one another as `from pytrickle.x import
...`, never as bare `from x import ...`.

**`simulation.py`:** A worked example. It fits a T-learner with
XGBoost and a depth-2 tree, then tests only the heterogeneity
(`ContrastTest`) hypotheses with trickle down. It does not yet use
`interweaved_hierarchy`, and its data-generating process differs from
the paper's §7.1.
