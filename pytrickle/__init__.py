"""Testing Statistical Hypotheses in a Tree-Like Hierarchy."""

__version__ = "0.1.0"

from typing import List

from pytrickle.hierarchy import Hierarchy
from pytrickle.hypothesis import FTest, Hypothesis, TTest

__all__: List[str] = [
    "Hypothesis",
    "Hierarchy",
    "TTest",
    "FTest",
]


def __dir__() -> List[str]:
    return __all__
