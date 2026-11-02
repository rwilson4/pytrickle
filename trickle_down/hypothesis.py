"""Abstract base class for representing a hypothesis."""

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import stats

# Which departures from the null count as evidence against it. Follows
# scipy's naming. "greater" tests H_0: tau <= delta, so rejecting proves
# the effect exceeds delta; "less" tests H_0: tau >= delta.
Alternative = Literal["two-sided", "greater", "less"]


def select_alternative(tau_train: float, delta: float = 0.0) -> Alternative:
    r"""Choose a one-sided alternative from an independent training split.

    Parameters
    ----------
    tau_train : float
        Estimated effect in this node, computed on the training split.
    delta : float
        Threshold defining the hypothesis. Zero gives an ordinary test of
        benefit or harm; a positive value a superiority test against a
        break-even point; a negative value a non-inferiority test.

    Returns
    -------
    Alternative
        "greater" when the training estimate exceeds delta, so that
        rejecting certifies a benefit beyond delta; "less" otherwise, so
        that rejecting certifies harm.

    Notes
    -----
    The choice is legitimate because the training and evaluation splits
    are independent. Conditional on the training split the selected
    hypotheses are fixed, so a procedure controlling the FWER over a
    fixed family controls it here too. Choosing the wrong direction
    costs power; it does not inflate the error rate.

    References
    ----------
    Cox, D. R. (1975). A note on data-splitting for the evaluation of
    significance levels. Biometrika, 62(2), 441-444.

    """
    if tau_train > delta:
        return "greater"

    return "less"


def contrast_moments(
    means: Sequence[float],
    variances: Sequence[float],
    sizes: Sequence[int],
    coefficients: Sequence[float],
) -> tuple[float, float, float]:
    r"""Estimate, standard error and degrees of freedom for a contrast.

    For independent groups with means mu_i, variances sigma_i^2 and sizes
    n_i, and contrast coefficients c_i, this estimates

       theta = \sum_i c_i mu_i

    with standard error \sqrt{ \sum_i c_i^2 sigma_i^2 / n_i } and the
    Welch-Satterthwaite degrees of freedom

       nu = ( \sum_i u_i )^2 / \sum_i u_i^2 / (n_i - 1),

    where u_i = c_i^2 sigma_i^2 / n_i.

    Parameters
    ----------
    means, variances, sizes : Sequence
        Per-group sample moments. Variances are the usual unbiased
        estimates (ddof=1).
    coefficients : Sequence[float]
        Contrast coefficients. Use (1, -1) for a difference of means and
        (1, -1, -1, 1) for a difference in differences.

    Returns
    -------
    tuple[float, float, float]
        Estimate, standard error, and degrees of freedom.

    Notes
    -----
    Groups with a zero coefficient drop out of every term, including the
    degrees of freedom, so they may be passed harmlessly.

    """
    if not len(means) == len(variances) == len(sizes) == len(coefficients):
        raise ValueError("means, variances, sizes and coefficients must align.")

    if any(n < 2 for n in sizes):
        raise ValueError("Every group needs at least two observations.")

    estimate = sum(c * mu for c, mu in zip(coefficients, means, strict=True))

    # u_i is group i's contribution to the variance of the contrast.
    u = [c * c * v / n for c, v, n in zip(coefficients, variances, sizes, strict=True)]
    variance = sum(u)

    if variance <= 0.0:
        raise ValueError("Contrast has zero variance; cannot form a t statistic.")

    denominator = sum(
        u_i * u_i / (n - 1) for u_i, n in zip(u, sizes, strict=True) if u_i > 0.0
    )
    df = variance * variance / denominator

    return estimate, math.sqrt(variance), df


def t_pvalue(t: float, df: float, alternative: Alternative) -> float:
    """Convert a t statistic to a p-value under the given alternative."""
    if alternative == "two-sided":
        return min(1.0, 2.0 * min(stats.t.cdf(t, df), stats.t.sf(t, df)))
    elif alternative == "greater":
        return float(stats.t.sf(t, df))
    elif alternative == "less":
        return float(stats.t.cdf(t, df))
    else:
        raise ValueError(f"Unrecognized alternative: {alternative!r}")


def invert_test(
    estimate: float,
    standard_error: float,
    df: float,
    level: float,
    alternative: Alternative,
) -> tuple[float, float]:
    r"""Invert a t-test to get the confidence set it implies.

    A test of ``H_0: tau <= delta`` at level ``level`` rejects exactly
    those thresholds ``delta`` lying below

       L = tau_hat - t_{1 - level, df} * SE,

    so the set of thresholds the data can rule out is an interval, and
    reporting its endpoint says strictly more than reporting the
    verdict at one threshold. When the test rejects at ``delta``, the
    endpoint is a sharper claim than ``delta``. When it retains, the
    endpoint still bounds how far the other way the truth can be ---
    which is the difference between "we cannot show the treatment
    helps" and "we cannot show the treatment helps, and it costs at
    most this much".

    Returns
    -------
    tuple[float, float]
        Lower and upper endpoints, with an infinite endpoint on the
        side a one-sided test says nothing about.

    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie in (0, 1); got {level}")

    if alternative == "greater":
        critical = stats.t.ppf(1.0 - level, df)
        return estimate - critical * standard_error, math.inf

    if alternative == "less":
        critical = stats.t.ppf(1.0 - level, df)
        return -math.inf, estimate + critical * standard_error

    critical = stats.t.ppf(1.0 - level / 2.0, df)
    return estimate - critical * standard_error, estimate + critical * standard_error


class Hypothesis(ABC):
    """Abstract base class for representing a hypothesis."""

    def __init__(
        self,
        weight: float = 1.0,
        stake: float = 1.0,
        name: str | None = None,
    ) -> None:
        if weight < 0.0:
            raise ValueError("Weight must be non-negative.")

        if stake < 0.0 or stake > 1.0:
            raise ValueError("Stake must be between 0 and 1.")

        self.weight = weight
        self.stake = stake
        self.name = name
        self.available_level: float | None = None
        self.children: list[Hypothesis] = []
        self.tested = False
        self.rejected = False
        self._weights_normalized = False
        # Per-instance, not functools.cache: caching on a method keys the
        # global cache on self and keeps every hypothesis alive for the
        # life of the process (flake8 B019).
        self._cached_pvalue: float | None = None

    def add_child(self, child: "Hypothesis") -> None:
        """Add child hypothesis."""
        if child.weight < 0.0:
            raise ValueError("Weights must be non-negative.")

        self.children.append(child)
        self._weights_normalized = False

    def _normalize_children_weights_recursively(self) -> None:
        for child in self.children:
            child._normalize_children_weights_recursively()

        if self._weights_normalized:
            return

        if len(self.children) == 0:
            self._weights_normalized = True
            return

        sum_weights = 0.0
        for child in self.children:
            if child.weight < 0.0:
                raise ValueError("Weight must be non-negative.")

            sum_weights += child.weight

        if sum_weights <= 0.0:
            raise ValueError("Must have at least one positive weight.")

        for child in self.children:
            child.weight = child.weight / sum_weights

        self._weights_normalized = True

        return

    def set_available_level(self, available_level: float) -> None:
        """Set available level."""
        if available_level < 0.0:
            raise ValueError("Available level must be non-negative")

        self.available_level = available_level

    def test_hypothesis(self) -> None:
        """Reject when the p-value clears this node's share of the level.

        Subclasses supply the p-value, not the decision rule: the
        comparison against ``stake * available_level`` is what makes the
        node part of the procedure, and is the same for every test.
        """
        if self.available_level is None:
            raise ValueError("`available_level` must be set before calling.")

        if self.pvalue() <= self.stake * self.available_level:
            self.rejected = True
        else:
            self.rejected = False

        self.tested = True

    @abstractmethod
    def pvalue(self) -> float:
        """Calculate p-value."""

    def confidence_interval(self, level: float) -> tuple[float, float]:
        """Confidence set obtained by inverting this test at ``level``.

        Not every hypothesis admits one. A p-value computed elsewhere
        carries no estimate or standard error, so there is nothing to
        invert.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot be inverted into a confidence set."
        )

    def __repr__(self) -> str:
        """Represent hypothesis as a string."""
        if self.name is not None:
            s = self.name + "\n"
        else:
            s = ""

        if self.tested:
            s += f"p={self.pvalue():.03g}"

        return s


class FixedPValue(Hypothesis):
    """A hypothesis whose p-value was computed elsewhere.

    Useful when the test is one this library does not implement --- a
    randomization test, a bootstrap, a model-based Wald test --- and for
    exercising a hierarchy against prescribed p-values in simulation.
    """

    def __init__(
        self,
        p: float,
        weight: float = 1.0,
        stake: float = 1.0,
        name: str | None = None,
    ) -> None:
        super().__init__(weight=weight, stake=stake, name=name)
        if p < 0.0 or p > 1.0:
            raise ValueError("p-value must be between 0 and 1.")

        self.p = p

    def pvalue(self) -> float:
        """Return the supplied p-value."""
        return self.p


class TTest(Hypothesis):
    r"""Test a hypothesis about a difference of means using Welch's t-test.

    With ``delta`` and ``alternative`` left at their defaults this is the
    familiar two-sided test of equal means. Setting them tests

       H_0: mu_1 - mu_2 <= delta     (alternative="greater")

    or the reverse inequality (alternative="less"), which is what the
    effect hypotheses attached to the nodes of an uplift tree need:
    rejecting proves the treatment effect clears, or falls short of, a
    threshold chosen for its business meaning.

    Parameters
    ----------
    mean1, mean2 : float
        Sample means. The contrast estimated is ``mean1 - mean2``, so for
        a treatment effect pass the treated mean first.
    var1, var2 : float
        Unbiased sample variances (ddof=1).
    n1, n2 : int
        Sample sizes.
    delta : float
        Threshold the difference is compared against. Zero recovers a
        test of no effect.
    alternative : Alternative
        Which departures count as evidence. See ``select_alternative``
        for choosing this from an independent training split.

    """

    def __init__(
        self,
        mean1: float,
        mean2: float,
        var1: float,
        var2: float,
        n1: int,
        n2: int,
        weight: float = 1.0,
        stake: float = 1.0,
        name: str | None = None,
        delta: float = 0.0,
        alternative: Alternative = "two-sided",
    ) -> None:
        super().__init__(weight=weight, stake=stake, name=name)
        if alternative not in ("two-sided", "greater", "less"):
            raise ValueError(f"Unrecognized alternative: {alternative!r}")

        self.mean1 = mean1
        self.mean2 = mean2
        self.var1 = var1
        self.var2 = var2
        self.n1 = n1
        self.n2 = n2
        self.delta = delta
        self.alternative: Alternative = alternative

    def pvalue(self) -> float:
        """Calculate p-value.

        Uses Welch's t-test. The two-sided p-value is computed by taking
        both one-sided p-values, doubling the smaller, and capping at 1.

        """
        if self._cached_pvalue is None:
            self._cached_pvalue = t_pvalue(
                self._test_statistic(), self._welch_satterthwaite(), self.alternative
            )

        return self._cached_pvalue

    def confidence_interval(self, level: float) -> tuple[float, float]:
        """The confidence set this test's inversion yields at ``level``."""
        estimate, standard_error, df = self._moments()
        return invert_test(estimate, standard_error, df, level, self.alternative)

    def _moments(self) -> tuple[float, float, float]:
        return contrast_moments(
            means=(self.mean1, self.mean2),
            variances=(self.var1, self.var2),
            sizes=(self.n1, self.n2),
            coefficients=(1.0, -1.0),
        )

    def _test_statistic(self) -> float:
        estimate, standard_error, _ = self._moments()
        return (estimate - self.delta) / standard_error

    def _standard_error(self) -> float:
        _, standard_error, _ = self._moments()
        return standard_error

    def _welch_satterthwaite(self) -> float:
        _, _, df = self._moments()
        return df


class ContrastTest(Hypothesis):
    r"""Test a linear contrast across independent groups.

    Generalizes ``TTest`` to any number of groups and any contrast. The
    case this library was written for is the difference in treatment
    effects across the two children of a tree node,

       (mu_{R1} - mu_{R0}) - (mu_{L1} - mu_{L0}),

    obtained with coefficients (1, -1, -1, 1) over the four
    arm-by-subgroup cells ordered R1, R0, L1, L0. Rejecting the
    two-sided null proves the treatment effect is heterogeneous across
    that split.

    Parameters
    ----------
    means, variances, sizes : Sequence
        Per-cell sample moments. Variances are unbiased (ddof=1).
    coefficients : Sequence[float]
        Contrast coefficients, aligned with the moments.
    delta : float
        Threshold the contrast is compared against.
    alternative : Alternative
        Which departures count as evidence.

    """

    def __init__(
        self,
        means: Sequence[float],
        variances: Sequence[float],
        sizes: Sequence[int],
        coefficients: Sequence[float],
        weight: float = 1.0,
        stake: float = 1.0,
        name: str | None = None,
        delta: float = 0.0,
        alternative: Alternative = "two-sided",
    ) -> None:
        super().__init__(weight=weight, stake=stake, name=name)
        if alternative not in ("two-sided", "greater", "less"):
            raise ValueError(f"Unrecognized alternative: {alternative!r}")

        self.means = tuple(means)
        self.variances = tuple(variances)
        self.sizes = tuple(sizes)
        self.coefficients = tuple(coefficients)
        self.delta = delta
        self.alternative: Alternative = alternative

    @classmethod
    def difference_in_differences(
        cls,
        treated_right: tuple[float, float, int],
        control_right: tuple[float, float, int],
        treated_left: tuple[float, float, int],
        control_left: tuple[float, float, int],
        weight: float = 1.0,
        stake: float = 1.0,
        name: str | None = None,
        delta: float = 0.0,
        alternative: Alternative = "two-sided",
    ) -> "ContrastTest":
        """Build the heterogeneity test for one split of an uplift tree.

        Each argument is a ``(mean, variance, size)`` triple for one
        arm-by-subgroup cell. The contrast estimated is
        ``tau_right - tau_left``.

        """
        cells = (treated_right, control_right, treated_left, control_left)
        return cls(
            means=[cell[0] for cell in cells],
            variances=[cell[1] for cell in cells],
            sizes=[int(cell[2]) for cell in cells],
            coefficients=(1.0, -1.0, -1.0, 1.0),
            weight=weight,
            stake=stake,
            name=name,
            delta=delta,
            alternative=alternative,
        )

    def pvalue(self) -> float:
        """Calculate p-value."""
        if self._cached_pvalue is None:
            self._cached_pvalue = t_pvalue(
                self._test_statistic(), self._welch_satterthwaite(), self.alternative
            )

        return self._cached_pvalue

    def _moments(self) -> tuple[float, float, float]:
        return contrast_moments(
            means=self.means,
            variances=self.variances,
            sizes=self.sizes,
            coefficients=self.coefficients,
        )

    def confidence_interval(self, level: float) -> tuple[float, float]:
        """The confidence set this test's inversion yields at ``level``."""
        estimate, standard_error, df = self._moments()
        return invert_test(estimate, standard_error, df, level, self.alternative)

    def _test_statistic(self) -> float:
        estimate, standard_error, _ = self._moments()
        return (estimate - self.delta) / standard_error

    def _standard_error(self) -> float:
        _, standard_error, _ = self._moments()
        return standard_error

    def _welch_satterthwaite(self) -> float:
        _, _, df = self._moments()
        return df


class FTest(Hypothesis):
    """One-way ANOVA for equality of means across two or more groups.

    Useful as the parent of the pairwise comparisons in a hierarchy:
    there is often power to show that the groups differ without power to
    say which is best, and testing the omnibus hypothesis first spends
    nothing on the pairwise tests until that much is established.

    """

    def __init__(
        self,
        *samples: npt.NDArray[np.float64],
        weight: float = 1.0,
        stake: float = 1.0,
        name: str | None = None,
        **kwargs: int,
    ) -> None:
        super().__init__(weight=weight, stake=stake, name=name)
        self.samples = samples
        self.kwargs = kwargs

    def pvalue(self) -> float:
        """Calculate p-value."""
        if self._cached_pvalue is None:
            _, p = stats.f_oneway(*self.samples, **self.kwargs)
            self._cached_pvalue = float(p)

        return self._cached_pvalue
