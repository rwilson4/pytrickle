"""Numbers published in Wilson (2026), transcribed from the arXiv source.

Each table below was copied from the generated ``.tex`` file the paper
inputs (``trickle_down_fwer.tex``, ``trickle_down_power.tex``,
``trickle_down_depth.tex`` and ``trickle_down_tree.tex`` in the arXiv
source of 2610.37966v1). The paper gives standard errors for Tables 2
and 3 but not Table 4.
"""

from __future__ import annotations

REPLICATIONS = 4_000
SCALES = (0.10, 0.15, 0.20, 0.30, 0.45, 1.00)
DEPTH_TABLE_SCALE = 0.45

# Table 2: familywise error rate under the global null, (FWER, s.e.).
FWER: dict[str, dict[str, tuple[float, float]]] = {
    "Fixed partition": {
        "Bonferroni": (0.0447, 0.0033),
        "Hierarchical Bonferroni": (0.0447, 0.0033),
        "Fixed hierarchy": (0.0508, 0.0035),
        "Trickle down": (0.0447, 0.0033),
        "Non-adaptive": (0.0460, 0.0033),
    },
    "Fitted tree": {
        "Bonferroni": (0.0345, 0.0029),
        "Hierarchical Bonferroni": (0.0457, 0.0033),
        "Fixed hierarchy": (0.0508, 0.0035),
        "Trickle down": (0.0457, 0.0033),
        "Non-adaptive": (0.0435, 0.0032),
    },
}
# Section 7.2: plain Bonferroni's paired difference, fitted minus fixed.
BONFERRONI_PAIRED_DIFFERENCE = (-0.0103, 0.0042)

# Table 3: mean true claims certified (of 10), (mean, s.e.), by scale.
POWER: dict[str, dict[str, tuple[tuple[float, float], ...]]] = {
    "The running example: one quarter harmed": {
        "Bonferroni": (
            (0.13, 0.01),
            (0.24, 0.01),
            (0.43, 0.01),
            (1.11, 0.02),
            (2.86, 0.03),
            (7.45, 0.01),
        ),
        "Hierarchical Bonferroni": (
            (0.12, 0.01),
            (0.23, 0.01),
            (0.39, 0.01),
            (0.96, 0.02),
            (2.36, 0.02),
            (6.68, 0.01),
        ),
        "Fixed hierarchy": (
            (0.11, 0.01),
            (0.20, 0.01),
            (0.35, 0.01),
            (0.84, 0.02),
            (1.90, 0.02),
            (3.60, 0.02),
        ),
        "Trickle down": (
            (0.15, 0.01),
            (0.28, 0.01),
            (0.49, 0.02),
            (1.25, 0.02),
            (3.15, 0.03),
            (7.58, 0.01),
        ),
        "Non-adaptive": (
            (0.12, 0.01),
            (0.22, 0.01),
            (0.37, 0.01),
            (0.93, 0.02),
            (2.53, 0.03),
            (7.36, 0.01),
        ),
    },
    "Every subgroup helped": {
        "Bonferroni": (
            (0.13, 0.01),
            (0.24, 0.01),
            (0.44, 0.01),
            (1.13, 0.02),
            (2.69, 0.02),
            (6.42, 0.01),
        ),
        "Hierarchical Bonferroni": (
            (0.13, 0.01),
            (0.26, 0.01),
            (0.46, 0.01),
            (1.06, 0.02),
            (2.19, 0.02),
            (5.91, 0.02),
        ),
        "Fixed hierarchy": (
            (0.15, 0.01),
            (0.29, 0.01),
            (0.48, 0.01),
            (0.96, 0.01),
            (1.65, 0.02),
            (4.65, 0.03),
        ),
        "Trickle down": (
            (0.16, 0.01),
            (0.32, 0.01),
            (0.57, 0.02),
            (1.34, 0.02),
            (2.86, 0.02),
            (6.84, 0.02),
        ),
        "Non-adaptive": (
            (0.13, 0.01),
            (0.25, 0.01),
            (0.42, 0.01),
            (1.05, 0.02),
            (2.42, 0.02),
            (6.45, 0.02),
        ),
    },
}

# Table 4: true claims by decision-tree depth at scale 0.45
# (root of 2, depth one of 4, depth two of 4). No standard errors.
DEPTH: dict[str, dict[str, tuple[float, float, float]]] = {
    "The running example: one quarter harmed": {
        "Bonferroni": (0.77, 1.06, 1.02),
        "Hierarchical Bonferroni": (1.13, 0.88, 0.36),
        "Fixed hierarchy": (1.30, 0.56, 0.03),
        "Trickle down": (1.20, 1.04, 0.90),
        "Non-adaptive": (0.99, 0.86, 0.68),
    },
    "Every subgroup helped": {
        "Bonferroni": (0.86, 0.90, 0.93),
        "Hierarchical Bonferroni": (1.07, 0.78, 0.34),
        "Fixed hierarchy": (1.28, 0.36, 0.02),
        "Trickle down": (1.14, 0.94, 0.79),
        "Non-adaptive": (1.02, 0.79, 0.61),
    },
}

# Figure 2: the paper's fitted tree, parents before children, as
# (node, depth, tau_train, tau_eval, certified). "certified" is whether
# the figure draws the node solid: its effect hypothesis was rejected.
TREE: tuple[tuple[str, int, float, float, bool], ...] = (
    ("everyone", 0, 1.09, 1.17, True),
    ("x2 = 0", 1, 0.31, 0.38, True),
    ("x2 = 1", 1, 1.83, 1.91, True),
    ("x2 = 0, x3 = 0", 2, -0.94, -0.60, True),
    ("x2 = 0, x3 = 1", 2, 1.55, 1.37, True),
    ("x2 = 1, x9 <= -0.425", 2, 1.12, 1.92, True),
    ("x2 = 1, x9 > -0.425", 2, 2.18, 1.91, True),
    ("x2 = 0, x3 = 0, x7 <= 1.17", 3, -0.76, -0.47, False),
    ("x2 = 0, x3 = 0, x7 > 1.17", 3, -2.13, -1.40, False),
    ("x2 = 0, x3 = 1, x6 <= 1.38", 3, 1.33, 1.33, True),
    ("x2 = 0, x3 = 1, x6 > 1.38", 3, 3.78, 1.76, False),
    ("x2 = 1, x9 <= -0.425, x5 <= 0.758", 3, 0.68, 1.84, True),
    ("x2 = 1, x9 <= -0.425, x5 > 0.758", 3, 2.53, 2.16, True),
    ("x2 = 1, x9 > -0.425, x9 <= -0.151", 3, 3.22, 1.91, True),
    ("x2 = 1, x9 > -0.425, x9 > -0.151", 3, 1.98, 1.91, True),
)
