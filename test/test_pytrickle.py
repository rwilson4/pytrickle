"""Test pytrickle."""

import itertools

import numpy as np
import numpy.typing as npt
import pytest
from scipy import stats

import pytrickle
from pytrickle import FixedPValue, FTest, Hierarchy, TTest


class TestTTest:
    """Test cases for TTest."""

    @staticmethod
    def test_ttest() -> None:
        """Examples from scipy.ttest_ind documentation."""
        rng = np.random.default_rng(seed=1)
        rvs1 = rng.normal(loc=5, scale=10, size=500)
        rvs2 = rng.normal(loc=5, scale=10, size=500)

        res = stats.ttest_ind(rvs1, rvs2, equal_var=False)

        h = TTest(
            mean1=np.mean(rvs1),
            mean2=np.mean(rvs2),
            var1=np.var(rvs1, ddof=1),
            var2=np.var(rvs2, ddof=1),
            n1=len(rvs1),
            n2=len(rvs2),
        )

        t = h._test_statistic()
        assert t == pytest.approx(res.statistic)

        df = h._welch_satterthwaite()
        # The scipy stubs still describe the old Ttest_indResult
        # namedtuple; the runtime returns a TtestResult, which carries
        # the Welch-Satterthwaite df this assertion is the whole point
        # of checking against.
        # pyrefly: ignore[missing-attribute]
        assert df == pytest.approx(res.df)

        p = h.pvalue()
        assert p == pytest.approx(res.pvalue)

        h.set_available_level(0.05)
        h.test_hypothesis()
        assert not h.rejected

        rvs5 = rng.normal(loc=8, scale=20, size=1000)

        res = stats.ttest_ind(rvs1, rvs5, equal_var=False)

        h = TTest(
            mean1=np.mean(rvs1),
            mean2=np.mean(rvs5),
            var1=np.var(rvs1, ddof=1),
            var2=np.var(rvs5, ddof=1),
            n1=len(rvs1),
            n2=len(rvs5),
        )

        t = h._test_statistic()
        assert t == pytest.approx(res.statistic)

        df = h._welch_satterthwaite()
        # The scipy stubs still describe the old Ttest_indResult
        # namedtuple; the runtime returns a TtestResult, which carries
        # the Welch-Satterthwaite df this assertion is the whole point
        # of checking against.
        # pyrefly: ignore[missing-attribute]
        assert df == pytest.approx(res.df)

        p = h.pvalue()
        assert p == pytest.approx(res.pvalue)

        h.set_available_level(0.05)
        h.test_hypothesis()
        assert h.rejected


class TestFTest:
    """Tests for FTest."""

    @staticmethod
    def dataset() -> dict[str, npt.NDArray[np.float64]]:
        """Data on shell measurement.

        Source:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html

        From source:
        Here are some data [3] on a shell measurement (the length of the anterior
        adductor muscle scar, standardized by dividing by length) in the mussel Mytilus
        trossulus from five locations: Tillamook, Oregon; Newport, Oregon; Petersburg,
        Alaska; Magadan, Russia; and Tvarminne, Finland, taken from a much larger data
        set used in McDonald et al. (1991).

        [3]: G.H. McDonald, “Handbook of Biological Statistics”, One-way ANOVA.
        http://www.biostathandbook.com/onewayanova.html

        """
        tillamook = [
            0.0571,
            0.0813,
            0.0831,
            0.0976,
            0.0817,
            0.0859,
            0.0735,
            0.0659,
            0.0923,
            0.0836,
        ]
        newport = [0.0873, 0.0662, 0.0672, 0.0819, 0.0749, 0.0649, 0.0835, 0.0725]
        petersburg = [0.0974, 0.1352, 0.0817, 0.1016, 0.0968, 0.1064, 0.105]
        magadan = [0.1033, 0.0915, 0.0781, 0.0685, 0.0677, 0.0697, 0.0764, 0.0689]
        tvarminne = [0.0703, 0.1026, 0.0956, 0.0973, 0.1039, 0.1045]

        samples = dict(**locals())
        # FTest takes arrays; returning lists makes every caller
        # convert, and the lists are the only reason it did not.
        return {
            name: np.asarray(values, dtype=float) for name, values in samples.items()
        }

    @staticmethod
    def test_ftest() -> None:
        """Data from scipy documentation."""

        data = TestFTest.dataset()

        res = stats.f_oneway(*data.values())
        expected = res.pvalue

        h = FTest(*data.values(), axis=0)

        actual = h.pvalue()

        assert actual == pytest.approx(expected)

        h.set_available_level(0.05)
        h.test_hypothesis()
        assert h.rejected


class TestHierarchy:
    """Tests for Hierarchy."""

    @staticmethod
    def test_hierarchy() -> None:
        """Test Hierarchy."""
        data = TestFTest.dataset()
        root = FTest(*data.values(), name="Global Equality")

        for c1, c2 in itertools.combinations(data.keys(), 2):
            pair = TTest(
                mean1=np.mean(data[c1]),
                mean2=np.mean(data[c2]),
                var1=np.var(data[c1], ddof=1),
                var2=np.var(data[c2], ddof=1),
                n1=len(data[c1]),
                n2=len(data[c2]),
                name=f"{c1}\n{c2}",
            )
            root.add_child(pair)

        h = Hierarchy(
            fwer=0.05,
            root=root,
        )

        h.test_hypotheses()

        # Drawing needs matplotlib, networkx and EoN. EoN in particular is
        # absent from some environments this runs in, so the visualization
        # is exercised only where it can be.
        pytest.importorskip("EoN")
        h.visualize_hierarchy()

    @staticmethod
    def test_retesting_at_new_fwer_updates_verdicts() -> None:
        """A second run at a smaller fwer re-tests every node.

        The root rejects at 0.05 and retains at 0.01. Were its first
        verdict kept, it would report a rejection at a level its
        p-value exceeds, and pass the whole level to its children.
        """
        root = FixedPValue(p=0.02, stake=0.5, name="root")
        child = FixedPValue(p=0.004, name="child")
        root.add_child(child)
        h = Hierarchy(fwer=0.05, root=root)

        h.test_hypotheses()
        assert root.rejected
        assert child.available_level == pytest.approx(0.05)

        h.fwer = 0.01
        h.test_hypotheses()
        assert not root.rejected
        assert child.available_level == pytest.approx(0.005)
        assert child.rejected

        for hypothesis in h.hypotheses():
            assert hypothesis.available_level is not None
            level = hypothesis.stake * hypothesis.available_level
            assert hypothesis.rejected == (hypothesis.pvalue() <= level)


class TestUsage:
    """Tests for the packaged usage guide."""

    @staticmethod
    def test_usage_guide_is_packaged() -> None:
        """USAGE.md is reachable at runtime, not only on disk.

        The guide ships as package data. Excluding it from the build
        leaves the source tree looking correct and breaks `usage()` for
        every caller, so the check belongs here.
        """
        guide = pytrickle.usage()
        assert guide.startswith("# pytrickle Usage Guide")
        assert "interweaved_hierarchy" in guide


class TestPublicAPI:
    """What `from pytrickle import ...` is allowed to reach."""

    @staticmethod
    def test_comparators_are_exported() -> None:
        """The baselines are part of the API, not an internal detail.

        Anyone running the procedure wants something to compare it
        against, and reaching into the `uplift` submodule for that is
        the sort of import that quietly becomes load-bearing.
        """
        assert pytrickle.bonferroni_rejections([], 0.05) == []
        assert pytrickle.hierarchical_bonferroni_rejections([], 0.05) == []

    @staticmethod
    def test_dir_matches_all() -> None:
        """`__dir__` returns `__all__`, so every name in it must resolve."""
        for name in pytrickle.__all__:
            assert hasattr(pytrickle, name), f"{name} is in __all__ but missing"
