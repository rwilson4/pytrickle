"""Abstract base class for representing a hypothesis."""

import math
from abc import ABC, abstractmethod
from typing import List, Optional

from scipy import stats


class Hypothesis(ABC):
    """Abstract base class for representing a hypothesis."""

    def __init__(self, weight: float = 1.0, stake: float = 1.0):
        if weight < 0.0:
            raise ValueError("Weight must be non-negative.")

        if stake < 0.0 or stake > 1.0:
            raise ValueError("Stake must be between 0 and 1.")

        self.weight = weight
        self.stake = stake
        self.available_level: Optional[float] = None
        self.children: List[Hypothesis] = []
        self.tested = False
        self.rejected = False
        self._weights_normalized = False

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
        """Test hypothesis.

        Abstract method to test the hypothesis. This method should be implemented by
        subclasses to define the actual testing procedure.
        """
        if self.available_level is None:
            raise ValueError("`available_level` must be set before calling.")

        if self.p_value() <= self.stake * self.available_level:
            self.rejected = True
        else:
            self.rejected = False

        self.tested = True

    @abstractmethod
    def p_value(self) -> float:
        """Calculate p-value."""
        pass


class TTest(Hypothesis):
    """Test a hypothesis using Welch's t-test."""

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
    ):
        super().__init__(weight=weight, stake=stake)
        self.mean1 = mean1
        self.mean2 = mean2
        self.var1 = var1
        self.var2 = var2
        self.n1 = n1
        self.n2 = n2

    def p_value(self) -> float:
        """Calculate p-value.

        Uses Welch's t-test to compute a two-sided p-value against the null hypothesis
        of equal means. To compute a two-sided p-value, we compute both one-sided
        hypotheses, double the smaller one, and cap at 1.0

        """
        t = self._test_statistic()
        df = self._welch_satterthwaite()
        p_value = min(1.0, 2.0 * min(stats.t.cdf(t, df=df), stats.t.sf(t, df=df)))

        return p_value

    def _test_statistic(self) -> float:
        return (self.mean1 - self.mean2) / self._standard_error()

    def _standard_error(self) -> float:
        return math.sqrt(self.var1 / self.n1 + self.var2 / self.n2)

    def _welch_satterthwaite(self) -> float:
        df = (self.var1 / self.n1 + self.var2 / self.n2) ** 2
        df /= self.var1**2 / (self.n1 * self.n1 * (self.n1 - 1)) + self.var2**2 / (
            self.n2 * self.n2 * (self.n2 - 1)
        )

        return df
