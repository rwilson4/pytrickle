"""Testing Statistical Hypotheses in a Tree-Like Hierarchy."""

import importlib.resources

__version__ = "0.1.0"

from trickle_down.graphical import (
    GraphicalProcedure,
    trickle_down_graph,
)
from trickle_down.hierarchy import Hierarchy
from trickle_down.hypothesis import (
    Alternative,
    contrast_moments,
    ContrastTest,
    FixedPValue,
    FTest,
    Hypothesis,
    select_alternative,
    TTest,
)
from trickle_down.uplift import (
    ArmSummary,
    bonferroni_rejections,
    Claim,
    hierarchical_bonferroni_rejections,
    interweaved_hierarchy,
    UpliftClaims,
    UpliftNode,
)

__all__: list[str] = [
    "Alternative",
    "ArmSummary",
    "Claim",
    "ContrastTest",
    "FTest",
    "FixedPValue",
    "GraphicalProcedure",
    "Hierarchy",
    "Hypothesis",
    "TTest",
    "UpliftClaims",
    "UpliftNode",
    "bonferroni_rejections",
    "contrast_moments",
    "hierarchical_bonferroni_rejections",
    "interweaved_hierarchy",
    "select_alternative",
    "trickle_down_graph",
    "usage",
]


def usage() -> str:
    """Return the contents of USAGE.md."""
    return importlib.resources.files(__name__).joinpath("USAGE.md").read_text()


def __dir__() -> list[str]:
    """Restrict tab completion to the public API."""
    return __all__
