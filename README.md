# Testing Statistical Hypotheses in a Tree-Like Hierarchy
![Hypothesis Hierarchy](hierarchy.png?raw=true)

This library controls the familywise error rate (FWER) over a family
of statistical hypotheses arranged in a tree. It accompanies

> Bob Wilson (2026). *Interpretable and Validatable Uplift Modeling.*
> [arXiv:2610.37966](https://arxiv.org/abs/2610.37966).

The paper's motivating problem is uplift modeling. We summarize an
uplift model with a shallow decision tree fit on a training split,
then use an independent evaluation split to certify claims about the
tree's subgroups: that the treatment helps one subgroup, harms
another, and that the difference between them is real. Each claim is a
hypothesis test, and the tree tells us how to spend the error budget
across them.

The paper's main procedure, *trickle down*, gives each node a weight
and a *stake*. A node is tested at its stake times its available
level. A rejection passes the whole available level to the node's
children; a retention still passes on the part it did not wager.
Unlike a fixed hierarchy, testing continues below a retention, so a
null effect at the root does not hide opposite effects in its
subgroups. The paper proves the procedure controls the FWER strongly,
and shows it is a special case of the graphical approach of Bretz et
al. (2009).

The library implements:

- the trickle down procedure (`Hierarchy`), over any tree of tests:
  Welch t-tests, linear contrasts, ANOVA, or p-values computed
  elsewhere;
- the paper's *interweaved* hierarchy of effect and heterogeneity
  hypotheses over a fitted uplift tree (`interweaved_hierarchy`), with
  test directions chosen on the training split;
- confidence bounds for retained hypotheses, valid simultaneously with
  the procedure's rejections;
- the Bonferroni, Hierarchical Bonferroni and Fixed Hierarchy
  comparators, and a reference implementation of the graphical
  approach for checking the equivalence.

## Installation

The distribution is named `pytrickle-py`; the package imports as
`pytrickle`.

```bash
pip install pytrickle-py
```

Until the first release reaches PyPI, install from GitHub:

```bash
pip install git+https://github.com/rwilson4/pytrickle
```

## Example

```python
from pytrickle import ArmSummary, UpliftNode, interweaved_hierarchy

men = UpliftNode(
    name="men",
    treated=ArmSummary(mean=0.42, variance=0.24, size=5_000),
    control=ArmSummary(mean=0.35, variance=0.23, size=5_000),
    tau_train=0.065,
)
women = UpliftNode(
    name="women",
    treated=ArmSummary(mean=0.29, variance=0.21, size=5_000),
    control=ArmSummary(mean=0.33, variance=0.22, size=5_000),
    tau_train=-0.038,
)
everyone = UpliftNode(
    name="everyone",
    treated=ArmSummary(mean=0.355, variance=0.229, size=10_000),
    control=ArmSummary(mean=0.340, variance=0.224, size=10_000),
    tau_train=0.014,
    children=[women, men],
)

claims = interweaved_hierarchy(everyone, alpha=0.05).test()
print(claims)
```

The arm summaries come from the evaluation split; `tau_train`, from
the training split, only picks the direction of each one-sided test.
At FWER 0.05 this certifies that the effect is positive overall,
positive for men, negative for women, and larger for men than for
women.

`pytrickle.usage()` returns the full usage guide, with recipes for
building hierarchies by hand, reporting confidence bounds, and
comparing against the other procedures.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check pytrickle tests
uv run mypy
```

## Citation

```bibtex
@misc{wilson2026uplift,
  author        = {Wilson, Bob},
  title         = {Interpretable and Validatable Uplift Modeling},
  year          = {2026},
  eprint        = {2610.37966},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ME},
}
```

## License

Apache License 2.0. See [LICENSE.txt](LICENSE.txt).
