"""Testing Statistical Hypotheses in a Tree-Like Hierarchy."""

__version__ = "0.1.0"

from typing import List

from pytrickle.hierarchy import Hierarchy
from pytrickle.hypothesis import Hypothesis

__all__: List[str] = [
    "Hypothesis",
    "Hierarchy",
]


def __dir__() -> List[str]:
    return __all__
