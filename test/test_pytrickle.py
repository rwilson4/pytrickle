"""Test pytrickle."""

import numpy as np
import pytest
from scipy import stats

from pytrickle import TTest


def test_ttest() -> None:
    """Examples from scipy.ttest_ind documentation."""
    rng = np.random.default_rng(seed=1)
    rvs1 = stats.norm.rvs(loc=5, scale=10, size=500, random_state=rng)
    rvs2 = stats.norm.rvs(loc=5, scale=10, size=500, random_state=rng)

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
    assert df == pytest.approx(res.df)

    p = h.p_value()
    assert p == pytest.approx(res.pvalue)

    h.set_available_level(0.05)
    h.test_hypothesis()
    assert not h.rejected

    rvs5 = stats.norm.rvs(loc=8, scale=20, size=1000, random_state=rng)

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
    assert df == pytest.approx(res.df)

    p = h.p_value()
    assert p == pytest.approx(res.pvalue)

    h.set_available_level(0.05)
    h.test_hypothesis()
    assert h.rejected
