"""Abstract base class for representing a hypothesis."""

from abc import ABC, abstractmethod
from typing import List, Optional


class Hypothesis(ABC):
    """Abstract base class for representing a hypothesis."""

    def __init__(self, weight: float, stake: float = 1.0):
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

    @abstractmethod
    def test_hypothesis(self) -> None:
        """Test hypothesis.

        Abstract method to test the hypothesis. This method should be implemented by
        subclasses to define the actual testing procedure.
        """
        if self.available_level is None:
            raise ValueError("`available_level` must be set before calling.")