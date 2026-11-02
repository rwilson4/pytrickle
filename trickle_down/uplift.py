r"""The hypothesis architecture for validating an uplift model.

An uplift model is distilled into a shallow decision tree, and each node
of that tree carries two kinds of claim we might want to certify:

- a *magnitude* claim about the average treatment effect in the node's
  subpopulation, tau_S, against a threshold delta_S chosen for its
  business meaning; and
- a *difference* claim that tau differs between the node's two
  children, which is what it means for the split to identify real
  heterogeneity.

These interweave into a single tree of hypotheses. Writing M_S for the
magnitude hypothesis at node S and D_S for the difference hypothesis at
its split,

    M_root -> D_root -> { M_L , M_R } -> ...

Each magnitude hypothesis has exactly one child, the difference
hypothesis at the same node; each difference hypothesis has the
magnitude hypotheses of the two child nodes as its children. Leaves of
the decision tree carry a magnitude hypothesis and nothing below it.

The payoff is a set of claims a decision-maker can act on. If the first
split is gender, the procedure can certify simultaneously that the
effect is larger for men than for women, that it is positive for men,
and that it is negative for women --- three sentences, all at the
stated familywise error rate.

"""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from trickle_down.hierarchy import Hierarchy
from trickle_down.hypothesis import (
    Alternative,
    ContrastTest,
    Hypothesis,
    select_alternative,
    TTest,
)


@dataclass(frozen=True)
class ArmSummary:
    """Sample moments for one treatment arm within one subpopulation."""

    mean: float
    variance: float
    size: int

    def as_cell(self) -> tuple[float, float, int]:
        """Return the (mean, variance, size) triple the tests expect."""
        return self.mean, self.variance, self.size

    @classmethod
    def from_samples(cls, y: npt.ArrayLike) -> "ArmSummary":
        """Summarize one arm's observations.

        The variance is the unbiased sample variance, ``ddof=1``.
        ``np.var`` defaults to ``ddof=0``, which understates it by a
        factor of (n-1)/n; a Welch test fed that understated variance
        is anticonservative, and the error is invisible at a glance.
        That is the whole reason this exists rather than leaving every
        caller to write the three lines itself.

        Non-finite values are refused for the same reason. A single
        NaN turns the variance into NaN, which propagates through the
        Welch statistic into a NaN p-value that compares false against
        every level, so the node silently retains. Dropping the
        offending observations instead would be worse: the caller
        would never learn that the slice it handed over was not the
        slice it was summarizing.
        """
        values = np.asarray(y, dtype=float).ravel()
        if values.size < 2:
            raise ValueError(
                "An arm needs at least two observations to have a sample "
                f"variance; got {values.size}."
            )

        if not np.all(np.isfinite(values)):
            bad = int(np.count_nonzero(~np.isfinite(values)))
            raise ValueError(
                f"An arm's observations must all be finite; {bad} of "
                f"{values.size} are NaN or infinite."
            )

        return cls(
            mean=float(np.mean(values)),
            variance=float(np.var(values, ddof=1)),
            size=int(values.size),
        )


@dataclass
class UpliftNode:
    r"""A node of the fitted uplift tree, with what is needed to test it.

    Parameters
    ----------
    name : str
        Human-readable label, e.g. "men" or "x0 < 0.5".
    treated, control : ArmSummary
        Moments computed on the **evaluation** split. These are what the
        hypotheses are tested against.
    tau_train : float
        The effect estimated on the **training** split. Used only to
        choose the direction of the one-sided magnitude test, never to
        compute a p-value.
    children : list[UpliftNode]
        Either empty, or exactly two.

    """

    name: str
    treated: ArmSummary
    control: ArmSummary
    tau_train: float
    children: list["UpliftNode"] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.children) not in (0, 2):
            raise ValueError(
                f"Node {self.name!r} has {len(self.children)} children; "
                "a binary tree needs zero or two."
            )

    @property
    def tau_hat(self) -> float:
        """Evaluation-split estimate of the average treatment effect."""
        return self.treated.mean - self.control.mean

    @property
    def size(self) -> int:
        """Total evaluation-split sample size in this subpopulation."""
        return self.treated.size + self.control.size

    def walk(self) -> list["UpliftNode"]:
        """Every node at or below this one, parents before children."""
        ordered: list[UpliftNode] = []
        queue: list[UpliftNode] = [self]
        while queue:
            node = queue.pop(0)
            ordered.append(node)
            queue.extend(node.children)

        return ordered


@dataclass(frozen=True)
class Claim:
    """One hypothesis in the architecture, and what rejecting it proves."""

    node: str
    kind: str  # "magnitude" or "difference"
    hypothesis: Hypothesis
    delta: float
    alternative: Alternative
    # Depth of the *decision tree* node this claim is about: the root is
    # zero, its children one. Both claims at a node share its depth.
    #
    # This is not the claim's depth in the hypothesis tree, which
    # interleaves the two kinds and so runs 2*depth + 1 for a magnitude
    # and 2*depth + 2 for a difference. `hierarchical_bonferroni_rejections`
    # spends level by the latter; do not substitute this field there.
    #
    # Recorded here because depth is what an aggregate claim count hides
    # --- a hierarchy concentrates level near the top and starves the
    # bottom, so procedures differ at every depth and one total averages
    # the differences away. Callers used to recover it by counting commas
    # in the node name, which breaks on any name containing one.
    depth: int
    # The two children compared, for a difference claim. The contrast
    # estimated is tau_right - tau_left, so a one-sided rejection names
    # one of them, and the sentence cannot be written without them.
    left: str | None = None
    right: str | None = None

    @property
    def rejected(self) -> bool:
        """Whether the procedure rejected this hypothesis."""
        return self.hypothesis.rejected

    @property
    def pvalue(self) -> float:
        """The p-value, computed on the evaluation split."""
        return self.hypothesis.pvalue()

    @property
    def level(self) -> float:
        """The level this hypothesis was actually tested at."""
        available = self.hypothesis.available_level
        if available is None:
            return float("nan")

        return self.hypothesis.stake * available

    @property
    def interval(self) -> tuple[float, float]:
        """Confidence set for this claim's estimand, at its own level.

        The level a hypothesis is tested at is the level its inversion
        is valid at, so these bounds are simultaneous with every other
        claim the procedure makes, and with each other. They are not
        the marginal intervals one would report for a single node, and
        for a deep node they are correspondingly wide.

        A claim tested at level zero has the whole line for its
        confidence set. That is not a degenerate case to guard
        against but the honest answer: a test that can never reject
        excludes nothing, so its inversion excludes nothing either.
        """
        level = self.level
        if level == 0.0:
            return float("-inf"), float("inf")

        return self.hypothesis.confidence_interval(level)

    @property
    def reportable_bound(self) -> tuple[float, float]:
        """The part of ``interval`` the procedure is entitled to claim.

        Inverting the test gives an endpoint that may lie on either
        side of the node's threshold, but only the weaker side is
        free. A node that rejected has passed its level down to its
        children, so a threshold past the one it cleared is a fresh
        claim it has nothing left to pay for; truncating at ``delta``
        is what keeps the whole collection of endpoints
        simultaneously valid. A node that retained kept its level,
        and its endpoint already lies on the weaker side.

        Nothing is free for a two-sided test: its enlarged family is
        not nested, so the node holds a true member whatever the
        effect is, and rejecting never releases the level. Such a
        claim reports no bound, and so does a claim tested at level
        zero, whose interval is already the whole line.
        """
        low, high = self.interval
        if self.alternative == "greater":
            return min(low, self.delta), high
        if self.alternative == "less":
            return low, max(high, self.delta)
        return float("-inf"), float("inf")

    def bound_sentence(self) -> str:
        """The quantitative claim, which survives a retention.

        ``sentence`` says what rejecting proves and is silent when the
        hypothesis is retained. This says what the data support
        either way, within what the procedure can pay for: at a
        retention it bounds how far the other way the truth can lie,
        and at a rejection it restates the threshold already cleared.

        A node handed no level has nothing to say. It is reported as
        such rather than omitted, so that the absence is visible in
        the output instead of being mistaken for a node the procedure
        never reached.
        """
        if self.level == 0.0:
            return f"no bound is free for the level-zero test at {self.node}"

        low, high = self.reportable_bound
        if self.kind == "difference":
            pair = f"the effect for {self.right} relative to {self.left}"
            if self.alternative == "greater":
                return f"{pair} is above {low:.3g}"
            if self.alternative == "less":
                return f"{pair} is below {high:.3g}"
            return f"no bound is free for the two-sided test at {self.node}"

        if self.alternative == "greater":
            return f"the effect for {self.node} is above {low:.3g}"
        if self.alternative == "less":
            return f"the effect for {self.node} is below {high:.3g}"
        return f"no bound is free for the two-sided test at {self.node}"

    def sentence(self) -> str:
        """What rejecting this hypothesis certifies, in words."""
        if self.kind == "difference":
            if self.alternative == "two-sided" or not (self.left and self.right):
                return f"the effect differs across the split at {self.node}"
            larger, smaller = (
                (self.right, self.left)
                if self.alternative == "greater"
                else (self.left, self.right)
            )
            return f"the effect is larger for {larger} than for {smaller}"
        elif self.alternative == "greater":
            if self.delta == 0.0:
                return f"the effect is positive for {self.node}"
            return f"the effect for {self.node} exceeds {self.delta:g}"
        elif self.alternative == "less":
            if self.delta == 0.0:
                return f"the effect is negative for {self.node}"
            return f"the effect for {self.node} falls short of {self.delta:g}"
        else:
            return f"the effect for {self.node} differs from {self.delta:g}"


@dataclass
class UpliftClaims:
    """The hypotheses attached to an uplift tree, and their verdicts.

    Holds the hierarchy so it can be tested, and knows how to report
    what was certified once it has been.
    """

    hierarchy: Hierarchy
    claims: list[Claim]

    def test(self) -> "UpliftClaims":
        """Run the procedure. Returns self, so calls can be chained."""
        self.hierarchy.test_hypotheses()
        return self

    def certified(self) -> list[Claim]:
        """The claims the procedure was able to certify."""
        return [claim for claim in self.claims if claim.rejected]

    def __str__(self) -> str:
        """A table of every hypothesis, with its level and verdict."""
        from tabulate import tabulate

        rows = [
            [
                claim.node,
                claim.kind,
                claim.alternative,
                f"{claim.pvalue:.4f}",
                f"{claim.level:.4f}",
                "rejected" if claim.rejected else "retained",
            ]
            for claim in self.claims
        ]
        table = tabulate(
            rows,
            headers=["node", "kind", "alternative", "p", "level", "decision"],
            tablefmt="orgtbl",
        )
        certified = self.certified()
        retained = [claim for claim in self.claims if not claim.rejected]

        parts = [table]
        if certified:
            lines = "\n".join(f"  - {claim.sentence()}" for claim in certified)
            parts.append(f"Certified at FWER {self.hierarchy.fwer:g}:\n{lines}")
        else:
            parts.append("Nothing certified.")

        # A retention is not the end of what the data support. The
        # inverted test still bounds the estimand, at the same level
        # and simultaneously with everything above.
        if retained:
            lines = "\n".join(f"  - {claim.bound_sentence()}" for claim in retained)
            parts.append(f"Bounded, though not certified:\n{lines}")

        return "\n\n".join(parts)


def interweaved_hierarchy(
    root: UpliftNode,
    alpha: float,
    delta: float = 0.0,
    magnitude_stake: float = 0.5,
    difference_stake: float = 0.5,
    adaptive: bool = True,
    size_weighted: bool = False,
) -> UpliftClaims:
    r"""Build the interweaved hierarchy over a fitted uplift tree.

    Parameters
    ----------
    root : UpliftNode
        Root of the fitted tree.
    alpha : float
        Familywise error rate.
    delta : float
        Threshold for the magnitude hypotheses. Zero tests for benefit
        or harm; a positive value a break-even point an expensive
        treatment must clear; a negative value a non-inferiority margin.
    magnitude_stake, difference_stake : float
        Stakes for the two kinds of hypothesis. A stake below one keeps
        part of the level in reserve, so testing continues past a
        retention at reduced power.
    adaptive : bool
        Choose the direction of every test from the training split. For
        a magnitude hypothesis that is the sign of the node's effect;
        for a difference hypothesis it is the sign of
        ``tau_right - tau_left``, so rejecting says which child
        responds more rather than merely that they differ. When False,
        every hypothesis of both kinds is two-sided --- the natural
        comparator. Two things separate them: the factor of two in
        level that committing to a side in advance buys, and the
        strength of the claim earned. A two-sided rejection certifies
        that the effect differs from the threshold, not which way it
        differs; going on to name the direction is a Type III error
        argument this library does not make.
    size_weighted : bool
        Split a node's level between its children in proportion to
        their subpopulation sizes rather than equally. The procedure
        controls the familywise error rate under any weights, so this
        is a preference and not a correctness matter: equal weights
        treat a small subpopulation as worth as much attention as a
        large one, which is what finding a carve-out requires, while
        size weighting buys power where the most people are. A node's
        level is recycled only to its own descendants, never across to
        a sibling, so the split between siblings is zero-sum and the
        default is the neutral one.

    Returns
    -------
    UpliftClaims
        The hierarchy and the claims attached to it, untested.

    Notes
    -----
    Hypotheses with no children are given a stake of one. There is
    nothing below them to hold level in reserve for, and spending it
    keeps the procedure closer to exhausting alpha --- a procedure that
    leaves level unspent is uniformly improvable.

    Choosing the direction on the training split is legitimate because
    the splits are independent: conditional on the training split the
    family of hypotheses is fixed. A wrong guess costs power, not
    validity.

    A one-sided difference test is not merely more powerful than the
    two-sided one; it answers a different and more useful question.
    Rejecting a two-sided test licenses "the effect differs across this
    split", which leaves the reader to work out which way from an
    uncertified point estimate. Rejecting the one-sided test licenses
    "the effect is larger in this child than in that one", which is a
    sentence someone can act on.

    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")

    # Node names are the key claims are joined back to nodes by, both
    # here and by every consumer of UpliftClaims. Duplicates do not
    # raise anywhere downstream; they silently report one node's
    # estimate under another node's hypothesis.
    names = [node.name for node in root.walk()]
    if len(names) != len(set(names)):
        duplicated = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(
            f"Node names must be unique; these repeat: {duplicated}. "
            "Qualify each name by its path from the root."
        )

    claims: list[Claim] = []

    def magnitude_for(node: UpliftNode, depth: int) -> Hypothesis:
        alternative: Alternative = (
            select_alternative(node.tau_train, delta) if adaptive else "two-sided"
        )
        # A leaf has nothing below it to save level for.
        stake = magnitude_stake if node.children else 1.0
        hypothesis = TTest(
            mean1=node.treated.mean,
            mean2=node.control.mean,
            var1=node.treated.variance,
            var2=node.control.variance,
            n1=node.treated.size,
            n2=node.control.size,
            stake=stake,
            name=f"M[{node.name}]",
            delta=delta,
            alternative=alternative,
        )
        claims.append(
            Claim(
                node=node.name,
                kind="magnitude",
                hypothesis=hypothesis,
                delta=delta,
                alternative=alternative,
                depth=depth,
            )
        )
        return hypothesis

    def difference_for(node: UpliftNode, depth: int) -> Hypothesis:
        left, right = node.children
        # The contrast estimated is tau_right - tau_left, so the
        # training-split sign of that difference picks the side. The
        # question "do the children differ?" is symmetric, but the
        # answer a practitioner wants is not: "the effect is larger on
        # the right than on the left" is both the stronger claim and
        # the one a one-sided test buys at twice the level.
        alternative: Alternative = (
            select_alternative(right.tau_train - left.tau_train)
            if adaptive
            else "two-sided"
        )
        hypothesis = ContrastTest.difference_in_differences(
            treated_right=right.treated.as_cell(),
            control_right=right.control.as_cell(),
            treated_left=left.treated.as_cell(),
            control_left=left.control.as_cell(),
            stake=difference_stake,
            name=f"D[{node.name}]",
            alternative=alternative,
        )
        claims.append(
            Claim(
                node=node.name,
                kind="difference",
                hypothesis=hypothesis,
                delta=0.0,
                alternative=alternative,
                depth=depth,
                left=left.name,
                right=right.name,
            )
        )
        return hypothesis

    def build(node: UpliftNode, depth: int = 0) -> Hypothesis:
        magnitude = magnitude_for(node, depth)
        if not node.children:
            return magnitude

        difference = difference_for(node, depth)
        magnitude.add_child(difference)
        for child in node.children:
            child_magnitude = build(child, depth + 1)
            child_magnitude.weight = float(child.size) if size_weighted else 1.0
            difference.add_child(child_magnitude)

        return magnitude

    return UpliftClaims(
        hierarchy=Hierarchy(fwer=alpha, root=build(root)), claims=claims
    )


def bonferroni_rejections(claims: list[Claim], alpha: float) -> list[bool]:
    """Test every hypothesis at alpha / m, ignoring the tree.

    The comparator the paper argues against: valid, but it pays the full
    multiplicity price at every node however deep, and cannot borrow
    level from a rejection higher up.
    """
    if not claims:
        return []

    level = alpha / len(claims)
    return [claim.pvalue <= level for claim in claims]


def hierarchical_bonferroni_rejections(claims: list[Claim], alpha: float) -> list[bool]:
    """The Hierarchical Bonferroni Procedure of the paper's section 5.1.

    Hypotheses at depth k of the hypothesis tree are tested at level
    2^{-k} alpha / n_k, where n_k counts the hypotheses at that depth.
    The levels sum to at most alpha, so the FWER is controlled, but
    power falls off exponentially with depth. The depth is depth in the
    *hypothesis* tree, counted from one, which interleaves the two kinds
    of claim: a node at decision-tree depth d contributes its magnitude
    hypothesis at 2d + 1 and its difference hypothesis at 2d + 2. That
    is not `Claim.depth`, which is the decision-tree depth. Substituting
    one for the other leaves a procedure that still looks valid and
    spends quite different levels.
    """
    if not claims:
        return []

    by_id = {id(claim.hypothesis): i for i, claim in enumerate(claims)}

    # Find the root rather than assume it is claims[0]. The walk below
    # is what assigns every claim its level, so starting from the wrong
    # node leaves depth_of missing entries and the comprehension raises
    # KeyError -- and on a different claim order it would mis-level the
    # whole family rather than fail.
    has_parent = {id(child) for claim in claims for child in claim.hypothesis.children}
    roots = [claim for claim in claims if id(claim.hypothesis) not in has_parent]
    if len(roots) != 1:
        raise ValueError(
            f"The claims must form a single tree; found {len(roots)} hypotheses "
            "with no parent among them."
        )

    depth_of: dict[int, int] = {}
    queue: list[tuple[Hypothesis, int]] = [(roots[0].hypothesis, 1)]
    counts: Counter[int] = Counter()
    while queue:
        hypothesis, depth = queue.pop(0)
        depth_of[by_id[id(hypothesis)]] = depth
        counts[depth] += 1
        for child in hypothesis.children:
            queue.append((child, depth + 1))

    return [
        claim.pvalue <= 2.0 ** (-depth_of[i]) * alpha / counts[depth_of[i]]
        for i, claim in enumerate(claims)
    ]
