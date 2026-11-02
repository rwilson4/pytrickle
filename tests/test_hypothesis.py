"""Tests for one-sided tests, thresholds, contrasts and direction choice."""

import math

import numpy as np
import pytest
from scipy import stats

from trickle_down import (
    contrast_moments,
    ContrastTest,
    select_alternative,
    TTest,
)


def two_samples(seed=4):
    """A pair of normal samples with unequal variances and sizes."""
    rng = np.random.default_rng(seed=seed)
    a = stats.norm.rvs(loc=5.0, scale=10.0, size=400, random_state=rng)
    b = stats.norm.rvs(loc=3.0, scale=4.0, size=250, random_state=rng)
    return a, b


def as_ttest(
    a: np.ndarray,
    b: np.ndarray,
    **kwargs,
) -> TTest:
    """Build a TTest from two raw samples."""
    return TTest(
        mean1=float(np.mean(a)),
        mean2=float(np.mean(b)),
        var1=float(np.var(a, ddof=1)),
        var2=float(np.var(b, ddof=1)),
        n1=len(a),
        n2=len(b),
        **kwargs,
    )


class TestOneSided:
    """One-sided p-values and thresholds."""

    @staticmethod
    @pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
    def test_matches_scipy(alternative: str) -> None:
        """Each alternative reproduces scipy's Welch test."""
        a, b = two_samples()
        expected = stats.ttest_ind(a, b, equal_var=False, alternative=alternative)

        h = as_ttest(a, b, alternative=alternative)

        assert h._test_statistic() == pytest.approx(expected.statistic)
        assert h.pvalue() == pytest.approx(expected.pvalue)

    @staticmethod
    def test_one_sided_pvalues_partition_the_two_sided_one() -> None:
        """The two one-sided p-values sum to one; the smaller doubles."""
        a, b = two_samples()
        greater = as_ttest(a, b, alternative="greater").pvalue()
        less = as_ttest(a, b, alternative="less").pvalue()
        two_sided = as_ttest(a, b, alternative="two-sided").pvalue()

        assert greater + less == pytest.approx(1.0)
        assert two_sided == pytest.approx(min(1.0, 2.0 * min(greater, less)))

    @staticmethod
    def test_delta_shifts_the_null() -> None:
        """Testing against delta equals testing a shifted mean against 0."""
        a, b = two_samples()
        delta = 1.25

        with_delta = as_ttest(a, b, delta=delta, alternative="greater")
        shifted = TTest(
            mean1=float(np.mean(a)) - delta,
            mean2=float(np.mean(b)),
            var1=float(np.var(a, ddof=1)),
            var2=float(np.var(b, ddof=1)),
            n1=len(a),
            n2=len(b),
            alternative="greater",
        )

        assert with_delta.pvalue() == pytest.approx(shifted.pvalue())
        # And the standard error does not move with delta.
        assert with_delta._standard_error() == pytest.approx(shifted._standard_error())

    @staticmethod
    def test_raising_delta_makes_a_benefit_claim_harder() -> None:
        """A superiority test against a higher bar has a larger p-value."""
        a, b = two_samples()
        p_low = as_ttest(a, b, delta=0.0, alternative="greater").pvalue()
        p_high = as_ttest(a, b, delta=2.0, alternative="greater").pvalue()

        assert p_high > p_low

    @staticmethod
    def test_unrecognized_alternative_is_rejected() -> None:
        a, b = two_samples()
        with pytest.raises(ValueError, match="Unrecognized alternative"):
            as_ttest(a, b, alternative="sideways")


class TestSelectAlternative:
    """Choosing the direction of the test on the training split."""

    @staticmethod
    def test_above_threshold_certifies_benefit() -> None:
        assert select_alternative(1.5, delta=0.0) == "greater"
        assert select_alternative(2.5, delta=2.0) == "greater"

    @staticmethod
    def test_below_threshold_certifies_harm() -> None:
        assert select_alternative(-0.5, delta=0.0) == "less"
        assert select_alternative(1.5, delta=2.0) == "less"

    @staticmethod
    def test_ties_go_to_harm() -> None:
        """At exactly delta both nulls hold, so either is defensible.

        The convention matters only for reproducibility; a training
        estimate landing exactly on delta has measure zero.
        """
        assert select_alternative(0.0, delta=0.0) == "less"

    @staticmethod
    def test_direction_follows_the_training_estimate() -> None:
        """The selected test is the one the training data points at."""
        a, b = two_samples()
        tau_train = 4.0  # Pretend the training split saw a large benefit.
        alternative = select_alternative(tau_train, delta=0.0)
        h = as_ttest(a, b, alternative=alternative)

        assert alternative == "greater"
        assert h.pvalue() < 0.05


class TestContrastMoments:
    """The shared contrast machinery."""

    @staticmethod
    def test_difference_of_means_matches_the_closed_form() -> None:
        estimate, se, df = contrast_moments(
            means=(5.0, 3.0),
            variances=(100.0, 16.0),
            sizes=(400, 250),
            coefficients=(1.0, -1.0),
        )

        assert estimate == pytest.approx(2.0)
        assert se == pytest.approx(math.sqrt(100.0 / 400 + 16.0 / 250))

        u1, u2 = 100.0 / 400, 16.0 / 250
        expected_df = (u1 + u2) ** 2 / (u1 * u1 / 399 + u2 * u2 / 249)
        assert df == pytest.approx(expected_df)

    @staticmethod
    def test_zero_coefficients_drop_out() -> None:
        """A group with coefficient zero must not affect the answer."""
        with_extra = contrast_moments(
            means=(5.0, 3.0, 99.0),
            variances=(100.0, 16.0, 1e6),
            sizes=(400, 250, 3),
            coefficients=(1.0, -1.0, 0.0),
        )
        without = contrast_moments(
            means=(5.0, 3.0),
            variances=(100.0, 16.0),
            sizes=(400, 250),
            coefficients=(1.0, -1.0),
        )

        assert with_extra == pytest.approx(without)

    @staticmethod
    def test_misaligned_inputs_are_rejected() -> None:
        with pytest.raises(ValueError, match="must align"):
            contrast_moments((1.0, 2.0), (1.0,), (10, 10), (1.0, -1.0))

    @staticmethod
    def test_singleton_groups_are_rejected() -> None:
        with pytest.raises(ValueError, match="at least two observations"):
            contrast_moments((1.0, 2.0), (1.0, 1.0), (1, 10), (1.0, -1.0))


class TestContrastTest:
    """Linear contrasts across independent groups."""

    @staticmethod
    def test_reduces_to_the_two_sample_test() -> None:
        """Coefficients (1, -1, 0, 0) must reproduce TTest exactly."""
        a, b = two_samples()
        expected = as_ttest(a, b, alternative="greater")

        actual = ContrastTest(
            means=(float(np.mean(a)), float(np.mean(b)), 0.0, 0.0),
            variances=(float(np.var(a, ddof=1)), float(np.var(b, ddof=1)), 1.0, 1.0),
            sizes=(len(a), len(b), 50, 50),
            coefficients=(1.0, -1.0, 0.0, 0.0),
            alternative="greater",
        )

        assert actual._test_statistic() == pytest.approx(expected._test_statistic())
        assert actual.pvalue() == pytest.approx(expected.pvalue())

    @staticmethod
    def test_difference_in_differences_matches_the_closed_form() -> None:
        """The heterogeneity test at one split of an uplift tree."""
        # (mean, variance, size) per arm-by-subgroup cell.
        tr, cr = (1.80, 9.0, 600), (0.40, 9.0, 620)
        tl, cl = (0.09, 9.0, 580), (0.00, 9.0, 590)

        h = ContrastTest.difference_in_differences(tr, cr, tl, cl)

        tau_right = tr[0] - cr[0]
        tau_left = tl[0] - cl[0]
        assert h._test_statistic() * h._standard_error() == pytest.approx(
            tau_right - tau_left
        )

        u = [cell[1] / cell[2] for cell in (tr, cr, tl, cl)]
        assert h._standard_error() == pytest.approx(math.sqrt(sum(u)))

        expected_df = sum(u) ** 2 / sum(
            u_i * u_i / (cell[2] - 1)
            for u_i, cell in zip(u, (tr, cr, tl, cl), strict=True)
        )
        assert h._welch_satterthwaite() == pytest.approx(expected_df)

    @staticmethod
    def test_no_heterogeneity_is_not_rejected() -> None:
        """Equal effects in both children give a large p-value."""
        cell = (0.0, 9.0, 500)
        h = ContrastTest.difference_in_differences(
            (1.0, 9.0, 500), cell, (1.0, 9.0, 500), cell
        )

        assert h.pvalue() == pytest.approx(1.0)

    @staticmethod
    def test_sign_of_the_contrast_follows_right_minus_left() -> None:
        """A larger effect on the right gives a positive statistic."""
        h = ContrastTest.difference_in_differences(
            treated_right=(2.0, 9.0, 500),
            control_right=(0.0, 9.0, 500),
            treated_left=(0.5, 9.0, 500),
            control_left=(0.0, 9.0, 500),
        )

        assert h._test_statistic() > 0.0
