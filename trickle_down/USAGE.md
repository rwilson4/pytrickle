# trickle-down Usage Guide

This guide teaches an AI agent (or human user) how to use `trickle_down`
to test a family of hypotheses arranged in a tree while controlling
the familywise error rate. The library's two entry points are
`interweaved_hierarchy`, which builds the whole hypothesis
architecture for validating an uplift model, and `Hierarchy`, which
runs the procedure over a tree you assemble yourself.

The library accompanies Wilson (2026), *Interpretable and Validatable
Uplift Modeling*, arXiv:2610.37966, which proves the procedure
controls the familywise error rate and studies its power by
simulation. The paper calls the two kinds of uplift hypothesis
*effect* and *heterogeneity*; the code calls them *magnitude* and
*difference*.

**For AI agents:** This guide gives recipes and conceptual
orientation, but every public function and class carries a
NumPy-style docstring documenting its parameters, return values and
edge cases. Read the docstring before calling:

```python
from trickle_down import (
    Hierarchy,
    interweaved_hierarchy,
    select_alternative,
)

help(interweaved_hierarchy)
help(Hierarchy.test_hypotheses)
help(select_alternative)
```

The docstrings are the authoritative reference for parameter names,
types and defaults.

## Conceptual Foundation

Suppose we have fitted an uplift model and distilled it into a shallow
decision tree. We would like to tell a decision-maker several things
at once: that the treatment helps men, that it harms women, and that
the difference between the two is real. Each of those sentences is a
hypothesis test, and reporting whichever ones came out significant
without adjustment inflates the error rate.

The ordinary remedy is Bonferroni: test each of the $m$ hypotheses
at $\alpha / m$. This is valid but wasteful. It charges the same
price for the root of the tree, which we have ample power to test, as
for a leaf describing two percent of the population, and it cannot
reuse the level spent on a hypothesis we went on to reject.

The Trickle Down Procedure exploits the tree. Each node carries two
numbers:

- its **weight** $w$, the share of its parent's level it receives,
  normalized so that siblings' weights sum to one; and
- its **stake** $v \in [0, 1]$, the fraction of its own available
  level it wagers on its own test.

Write $a$ for the level available at a node. The node tests its
hypothesis at $v a$. What it passes to its children depends on the
verdict:

- **Rejected.** The node passes its whole available level $a$ down,
  split among the children in proportion to their weights. If the
  hypothesis was true, rejecting it is already a familywise error, and
  nothing that happens below can make that worse; if it was false, no
  error was made. Either way the level has done its job and can be
  reused.
- **Retained.** The node passes only the unwagered part,
  $(1 - v) a$, again split by weight.

A leaf should stake everything, $v = 1$: there is nothing below it
to hold level in reserve for, and a procedure that leaves level
unspent is uniformly improvable.

The consequence is that evidence trickles downward. A strong effect at
the root buys the full level for the subgroup analysis beneath it, and
the deep, narrow claims that Bonferroni could never afford become
testable. Nothing trickles sideways: a node's level reaches only its
own descendants, never a sibling's subtree, so the split between
siblings is zero-sum.

### What this controls

The procedure controls the familywise error rate strongly at
$\alpha$: the probability of rejecting **any** true hypothesis is at
most $\alpha$, whatever the configuration of true and false
hypotheses. This holds for any non-negative weights and any stakes in
$[0, 1]$, so those are tuning choices and not correctness
conditions. The paper requires $v > 0$; the library also accepts
$v = 0$, which never rejects and passes the node's whole available
level to its children, as if the node were not there.

Strong control is the guarantee a reader needs when the findings are
consumed selectively. False discovery rate control permits a
proportion of false positives, and that guarantee does not survive an
audience applying its own filters to the results.

### Relationship to the graphical approach

The Trickle Down Procedure is a special case of the graphical approach
of Bretz, Maurer, Brannath and Posch (2009), under the encoding in
`trickle_down_graph`. The `graphical` module implements their
Algorithm 1 directly so the equivalence can be checked empirically
rather than only argued on paper; `test_graphical.py` does exactly
that over random trees. For ordinary use you do not need the module,
but it is the thing to reach for when you want to confirm that a
hierarchy behaves as the theory says.

## Imports

```python
from trickle_down import (
    ArmSummary,
    Claim,
    ContrastTest,
    FixedPValue,
    FTest,
    GraphicalProcedure,
    Hierarchy,
    Hypothesis,
    TTest,
    UpliftClaims,
    UpliftNode,
    bonferroni_rejections,
    hierarchical_bonferroni_rejections,
    interweaved_hierarchy,
    select_alternative,
    trickle_down_graph,
)
```

## Recipe: validate an uplift model

This is the case the library was written for. Fit the uplift model on
a training split, evaluate it on a held-out split, and certify what
the evaluation split supports.

```python
from trickle_down import (
    ArmSummary,
    UpliftNode,
    interweaved_hierarchy,
)

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

for claim in claims.certified():
    print(claim.sentence())
```

Three things about the inputs matter.

**Moments come from the evaluation split.** The `treated` and
`control` summaries are what the hypotheses are tested against. Pass
the unbiased sample variance (`ddof=1`), not the squared standard
error; the tests divide by the sample size themselves.

If you are starting from the observations rather than from moments,
use `ArmSummary.from_samples`, which gets the `ddof` right:

```python
# y: outcomes; z: treatment indicator; subgroup: boolean mask
treated = ArmSummary.from_samples(y[subgroup & (z == 1)])
control = ArmSummary.from_samples(y[subgroup & (z == 0)])
```

`np.var` defaults to `ddof=0`, which understates the variance by a
factor of $(n-1)/n$. Feeding that to a Welch test makes it
anticonservative, and nothing downstream will complain.

`from_samples` also refuses a slice containing NaN or infinity.
Drop or impute missing outcomes before you get here; a single NaN
gives a NaN variance, a NaN statistic and a NaN p-value, and
`nan <= level` is false, so the node retains and the report looks
ordinary. Deciding what a missing outcome means is yours, not the
library's.

**`tau_train` comes from the training split, and only picks a
direction.** It never enters a p-value. Choosing the side of a
one-sided test from an independent split is legitimate because,
conditional on that split, the family of hypotheses is fixed (Cox,
1975). A wrong guess costs power, not validity.

**Node names must be unique and children come in pairs.** Duplicate
names raise, because claims are joined back to nodes by name and a
collision would silently report one node's estimate under another
node's hypothesis. A node has either zero children or two.

### The hypothesis architecture

`interweaved_hierarchy` attaches two kinds of hypothesis to the
decision tree and interweaves them. Writing $M_S$ for the magnitude
hypothesis at node $S$ and $D_S$ for the difference hypothesis at
its split,

```text
M_root -> D_root -> { M_left , M_right } -> ...
```

A magnitude hypothesis asks whether the average treatment effect in
the node's subpopulation clears a threshold `delta`. A difference
hypothesis asks whether the effect differs between the node's two
children, which is what it means for the split to have found real
heterogeneity. Each magnitude hypothesis has exactly one child, the
difference hypothesis at the same node; each difference hypothesis has
the two child magnitude hypotheses as its children. Leaves carry a
magnitude hypothesis and nothing below.

The ordering matters: a split is only interesting once we believe
something about the node it splits, so the magnitude test's verdict
sets the level available to the difference test, whose verdict in turn
sets the level available to the children. With a stake below one a
retention reduces that level rather than ending testing.

### The knobs

| Parameter          | Default  | What it does                                                                                                                                                                        |
|--------------------|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `alpha`            | required | Familywise error rate.                                                                                                                                                              |
| `delta`            | `0.0`    | Threshold for the magnitude hypotheses. Zero tests for benefit or harm; positive sets a break-even point an expensive treatment must clear; negative sets a non-inferiority margin. |
| `magnitude_stake`  | `0.5`    | Stake for magnitude hypotheses at internal nodes. Leaves always stake one.                                                                                                          |
| `difference_stake` | `0.5`    | Stake for difference hypotheses.                                                                                                                                                    |
| `adaptive`         | `True`   | Pick each test's direction from the training split. `False` makes every test two-sided.                                                                                             |
| `size_weighted`    | `False`  | Split a node's level between children in proportion to subpopulation size rather than equally.                                                                                      |

A stake below one keeps part of the level in reserve, so testing
continues past a retention at reduced power. A stake of one is
all-or-nothing: retain at a node and everything below it is tested at
level zero.

Equal weights treat a small subpopulation as worth as much attention
as a large one, which is what finding a carve-out requires. Size
weighting buys power where the most people are. Neither is more
correct.

## Recipe: one-sided tests and what they buy

With `adaptive=True` the procedure commits to a direction before
seeing the evaluation split, which buys two things.

The first is a factor of two in level. The second, and the more
important, is the strength of the claim. A two-sided rejection of a
difference hypothesis licenses "the effect differs across this split",
which leaves the reader to infer the direction from an uncertified
point estimate. A one-sided rejection licenses "the effect is larger
for men than for women", which is a sentence someone can act on.
Going from the first to the second is a Type III error argument this
library does not make.

To choose a direction outside the uplift machinery:

```python
from trickle_down import (
    TTest,
    select_alternative,
)

alternative = select_alternative(tau_train=0.065, delta=0.02)  # "greater"

test = TTest(
    mean1=0.42,
    mean2=0.35,
    var1=0.24,
    var2=0.23,
    n1=5_000,
    n2=5_000,
    delta=0.02,
    alternative=alternative,
    name="M[men]",
)
```

`alternative="greater"` tests $H_0: \mu_1 - \mu_2 \leq \delta$, so
rejecting proves the effect exceeds `delta`. `"less"` tests the
reverse inequality. The naming follows scipy's.

## Recipe: build a hierarchy by hand

`Hierarchy` runs the procedure over any tree of `Hypothesis` objects,
so the uplift architecture is one use and not the only one. A common
pattern is an omnibus F-test gating the pairwise comparisons beneath
it: there is often power to show the groups differ without power to
say which is best, and the omnibus test spends nothing on the pairwise
tests until that much is established.

```python
import numpy as np
from trickle_down import (
    FTest,
    Hierarchy,
    TTest,
)

rng = np.random.default_rng(0)
a, b, c = (rng.normal(size=400) for _ in range(3))

omnibus = FTest(a, b, c, stake=0.5, name="any difference")
for label, (x, y) in {"a vs b": (a, b), "a vs c": (a, c)}.items():
    omnibus.add_child(
        TTest(
            mean1=x.mean(),
            mean2=y.mean(),
            var1=x.var(ddof=1),
            var2=y.var(ddof=1),
            n1=len(x),
            n2=len(y),
            weight=1.0,
            stake=1.0,
            name=label,
        )
    )

hierarchy = Hierarchy(fwer=0.05, root=omnibus)
hierarchy.test_hypotheses()

for hypothesis in hierarchy.hypotheses():
    print(hypothesis.name, hypothesis.rejected, hypothesis.available_level)
```

Weights are relative; siblings are normalized to sum to one when the
procedure runs, so passing `1.0` everywhere gives an even split. The
root's own weight is ignored, since it takes no share from a parent.

When the test you want is not one this library implements, a
randomization test or a bootstrap or a model-based Wald test, wrap its
p-value:

```python
from trickle_down import FixedPValue

node = FixedPValue(p=0.013, stake=0.5, name="bootstrap test")
```

`FixedPValue` is also how the simulations drive a hierarchy with
prescribed p-values.

## Recipe: bounds that survive a retention

A retention is not the end of what the data support. The level a
hypothesis was tested at is the level its inversion is valid at, so
inverting the test gives a confidence bound simultaneous with every
other claim the procedure makes.

In the running example every claim is rejected at $\alpha = 0.05$.
At $\alpha = 0.001$ the overall magnitude claim is retained:

```python
strict = interweaved_hierarchy(everyone, alpha=0.001).test()
for claim in strict.claims:
    if not claim.rejected:
        print(claim.bound_sentence())
# the effect for everyone is above -0.00715
```

This turns "we cannot show the treatment helps everyone" into "we
cannot show the treatment helps everyone, and it costs at most 0.007
per person", which is a far more useful sentence.

Three properties of these bounds are worth keeping straight.

**They are simultaneous, not marginal.** They are not the intervals
one would report for a single node in isolation, and for a deep node
they are correspondingly wide. `claim.interval` gives the raw
inversion at `claim.level`.

**A rejecting node may claim only its threshold.** `reportable_bound`
truncates at `delta`. A node that rejected has already passed its
level down to its children, so a threshold past the one it cleared is
a fresh claim it has nothing left to pay for.

**A two-sided claim gets no free bound.** Its enlarged family is not
nested, so the node holds a true member whatever the effect is and
rejecting never releases the level. `bound_sentence` says so rather
than reporting a number.

`FixedPValue` cannot be inverted: a p-value computed elsewhere carries
no estimate or standard error, so `confidence_interval` raises
`NotImplementedError`. Calling `claim.interval` on such a claim will
surface that.

## Recipe: compare against the comparators

```python
plain = bonferroni_rejections(claims.claims, alpha=0.05)
hier = hierarchical_bonferroni_rejections(claims.claims, alpha=0.05)
trickle = [claim.rejected for claim in claims.claims]
```

`bonferroni_rejections` tests everything at $\alpha / m$, ignoring
the tree. `hierarchical_bonferroni_rejections` tests depth $k$ at
$2^{-k} \alpha / n_k$, where $n_k$ counts the hypotheses at that
depth; the levels sum to at most $\alpha$, so it is valid, but power
falls off exponentially with depth. Both are useful as the baseline in
a simulation.

The $k$ there is depth in the **hypothesis** tree, counted from one,
which interleaves the two kinds: a decision-tree node at depth $d$
contributes its magnitude hypothesis at $2d + 1$ and its difference
hypothesis at $2d + 2$. `Claim.depth` is the decision-tree depth,
counted from zero. They are different numbers and are not
interchangeable — swapping them leaves a procedure that still looks
valid and spends quite different levels.

## Recipe: report results by depth

```python
from collections import defaultdict

by_depth = defaultdict(list)
for claim in claims.claims:
    by_depth[claim.depth].append(claim)

for depth in sorted(by_depth):
    at_depth = by_depth[depth]
    certified = sum(claim.rejected for claim in at_depth)
    print(f"depth {depth}: {certified} / {len(at_depth)}")
```

`Claim.depth` is the depth of the decision-tree node the claim is
about, with the root at zero; both claims at a node share it. Read it
off the field rather than parsing the node name — a name is a
human-readable path like `x2>0.5, x9<=-0.4`, and counting its commas
breaks the moment a name contains one for another reason.

Depth is worth separating out. A hierarchy concentrates level near the
top and starves the bottom, so two procedures that look similar on a
total claim count can differ sharply at every individual depth.

## Recipe: check a hierarchy against the graphical approach

```python
import numpy as np
from trickle_down import (
    GraphicalProcedure,
    trickle_down_graph,
)

root = claims.hierarchy.root  # already tested, from the first recipe
ordered, allocation, transitions = trickle_down_graph(root, alpha=0.05)
procedure = GraphicalProcedure(allocation, transitions)
rejected = procedure.reject([h.pvalue() for h in ordered])

assert list(rejected) == [h.rejected for h in ordered]
```

`procedure.alpha` reports the total level allocated, which is less
than the nominal $\alpha$ when some root-to-leaf path ends at a node
with stake below one. That shortfall is real: such a procedure can be
uniformly improved.

The `selection` argument to `reject` exposes which rejectable
hypothesis is taken first ("argmin", "first" or "random"). Bretz et
al. note the final rejection set does not depend on that choice; the
argument exists so the invariance can be tested.

## Pitfalls

**Re-running re-tests everything.** `Hierarchy.test_hypotheses`
re-tests every node on each call, so changing `fwer` and calling it
again gives verdicts consistent with the new levels. The previous
verdicts are overwritten, so record any you want to keep first.

**A `Hypothesis` is single-use.** Each instance caches its p-value
on first call. Do not reuse one across datasets.

**Weights are normalized in place.** Both `test_hypotheses` and
`trickle_down_graph` rewrite `child.weight` so that siblings sum to
one. Read the weights you passed in from your own records, not off the
objects afterwards.

**Level can reach zero.** A node below a chain of retentions with
stake one is tested at level zero, and `pvalue() <= 0` is false for
any positive p-value. This is the procedure working as designed, not a
bug, but it means a printed level of `0.0000` deserves a glance at
what happened above.

**Scoring a selective procedure needs the realized directions.** When
`select_alternative` picks each test's direction from a training
split, which nulls are true becomes data-dependent. If a node's effect
is positive and the split pointed its test at harm, that null is true,
and rejecting it is a Type I error rather than a claim earned. A
simulation that scores against a fixed two-sided null set --- where a
null holds only when the effect sits at the threshold exactly ---
credits the procedure with power for a wrong guess and hides what
guessing wrong costs. The selection cannot be precomputed: recompute
the true-null set per replication, against the alternatives the
procedure actually chose.

**`Hierarchy.visualize_hierarchy` imports its drawing libraries
lazily.** `matplotlib`, `networkx` and `EoN` are installed with
`trickle-down`, but they are imported only when you draw, so importing
the package stays fast and works in environments that lack them.

## References

Wilson, B. (2026). Interpretable and validatable uplift modeling.
arXiv:2610.37966.

Bretz, F., Maurer, W., Brannath, W., and Posch, M. (2009). A graphical
approach to sequentially rejective multiple test procedures.
*Statistics in Medicine*, 28(4), 586-604.

Burman, C.-F., Sonesson, C., and Guilbaud, O. (2009). A recycling
framework for the construction of Bonferroni-based multiple tests.
*Statistics in Medicine*, 28(5), 739-761.

Cox, D. R. (1975). A note on data-splitting for the evaluation of
significance levels. *Biometrika*, 62(2), 441-444.
