# pyre-unsafe
"""Simulation study for the trickle-down method."""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn.tree import DecisionTreeRegressor, export_text
from xgboost import XGBRegressor

from pytrickle import ArmSummary, ContrastTest, Hierarchy
from pytrickle.hypothesis import Hypothesis

_FEATURE_NAMES = [f"x{i}" for i in range(10)]


def generate_data(
    rng: np.random.Generator,
    n: int = 10_000,
    sigma: float = 3.0,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.intp],
    npt.NDArray[np.float64],
]:
    r"""Generate simulation data with heterogeneous treatment effects.

    Features: x0 binary moderator, x1 continuous outcome predictor,
    x2 continuous noise, x3 categorical (3 levels) noise, x4 binary
    moderator, x5-x9 continuous noise.

    True treatment effects by subgroup:

    +------+------+------+
    |  x0  |  x4  |  tau |
    +------+------+------+
    |   0  |   0  | 0.30 |
    |   0  |   1  | 1.20 |
    |   1  |   0  | 2.00 |
    |   1  |   1  | 1.50 |
    +------+------+------+

    Parameters
    ----------
    rng : np.random.Generator
    n : int
    sigma : float
        Standard deviation of outcome noise.

    Returns
    -------
    X : ndarray, shape (n, 10)
    y_obs : ndarray, shape (n,)
    z : ndarray, shape (n,)
        Treatment indicator (0 or 1).
    tau_true : ndarray, shape (n,)
        True individual treatment effects.
    """
    X = np.column_stack(
        [
            rng.integers(0, 2, size=n).astype(float),  # x0: binary moderator
            rng.standard_normal(n),  # x1: outcome predictor
            rng.standard_normal(n),  # x2: noise
            rng.integers(0, 3, size=n).astype(float),  # x3: 3-level noise
            rng.integers(0, 2, size=n).astype(float),  # x4: binary moderator
            rng.standard_normal(n),  # x5-x9: noise
            rng.standard_normal(n),
            rng.standard_normal(n),
            rng.standard_normal(n),
            rng.standard_normal(n),
        ]
    )

    x0, x1, x3, x4 = X[:, 0], X[:, 1], X[:, 3], X[:, 4]

    tau_true = np.where(
        x0 == 0,
        np.where(x4 == 0, 0.3, 1.2),
        np.where(x4 == 0, 2.0, 1.5),
    )

    mu = x1 + 0.3 * x3
    Y0 = mu + rng.normal(0.0, sigma, n)
    Y1 = mu + tau_true + rng.normal(0.0, sigma, n)

    z = rng.integers(0, 2, size=n)
    y_obs = np.where(z == 1, Y1, Y0)

    return X, y_obs, z, tau_true


def compute_pseudo_outcomes(
    X_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.float64],
    z_train: npt.NDArray[np.intp],
) -> npt.NDArray[np.float64]:
    """Fit T-learner via XGBoost and return pseudo-outcomes for the training set.

    Parameters
    ----------
    X_train : ndarray, shape (n, p)
    y_train : ndarray, shape (n,)
    z_train : ndarray, shape (n,)

    Returns
    -------
    tau_pseudo : ndarray, shape (n,)
        For control (z=0): mu_hat_1(x) - y_obs.
        For treated (z=1): y_obs - mu_hat_0(x).
    """
    kw = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.1,
        "random_state": 0,
        "verbosity": 0,
    }
    mu0 = XGBRegressor(**kw)
    mu1 = XGBRegressor(**kw)

    mu0.fit(X_train[z_train == 0], y_train[z_train == 0])
    mu1.fit(X_train[z_train == 1], y_train[z_train == 1])

    return np.where(
        z_train == 1,
        y_train - mu0.predict(X_train),
        mu1.predict(X_train) - y_train,
    )


def _node_mask(
    dt: DecisionTreeRegressor,
    node_id: int,
    X: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """Return boolean mask of samples that pass through node_id."""
    indicator = dt.decision_path(X)
    return indicator[:, node_id].toarray().ravel().astype(bool)


def _cell(y: npt.NDArray[np.float64]) -> tuple[float, float, int]:
    """Return the (mean, variance, size) triple for one arm in one subgroup."""
    return ArmSummary.from_samples(y).as_cell()


def _build_hypothesis(
    dt: DecisionTreeRegressor,
    node_id: int,
    X_eval: npt.NDArray[np.float64],
    y_eval: npt.NDArray[np.float64],
    z_eval: npt.NDArray[np.intp],
    context: str,
    weight: float,
    stake: float,
) -> ContrastTest:
    """Build a ContrastTest for the split at internal node `node_id`.

    Tests H0: tau_right - tau_left = 0 (two-sided), where tau_S is the ATE in child
    subpopulation S. Children in the pytrickle hierarchy are added for
    any child nodes that are themselves internal (i.e., have further splits).
    """
    tree_ = dt.tree_
    left_id = int(tree_.children_left[node_id])
    right_id = int(tree_.children_right[node_id])

    feat = _FEATURE_NAMES[int(tree_.feature[node_id])]
    thresh = float(tree_.threshold[node_id])
    left_cond = f"{feat}<{thresh:.1f}"
    right_cond = f"{feat}≥{thresh:.1f}"

    left_mask = _node_mask(dt, left_id, X_eval)
    right_mask = _node_mask(dt, right_id, X_eval)

    y_t_left = y_eval[left_mask & (z_eval == 1)]
    y_c_left = y_eval[left_mask & (z_eval == 0)]
    y_t_right = y_eval[right_mask & (z_eval == 1)]
    y_c_right = y_eval[right_mask & (z_eval == 0)]

    ate_left = float(np.mean(y_t_left) - np.mean(y_c_left))
    ate_right = float(np.mean(y_t_right) - np.mean(y_c_right))

    title = f"{feat} split | {context}" if context else f"{feat} split"
    name = f"{title}\n{left_cond}: τ̂={ate_left:.2f},  {right_cond}: τ̂={ate_right:.2f}"

    hyp = ContrastTest.difference_in_differences(
        treated_right=_cell(y_t_right),
        control_right=_cell(y_c_right),
        treated_left=_cell(y_t_left),
        control_left=_cell(y_c_left),
        weight=weight,
        stake=stake,
        name=name,
    )

    # Recurse into internal children only; assign equal weights.
    internal_children = [
        (child_id, child_cond)
        for child_id, child_cond in [(left_id, left_cond), (right_id, right_cond)]
        if int(tree_.children_left[child_id]) != -1
    ]
    child_weight = 1.0 / len(internal_children) if internal_children else 0.0
    for child_id, child_cond in internal_children:
        child_hyp = _build_hypothesis(
            dt,
            child_id,
            X_eval,
            y_eval,
            z_eval,
            context=child_cond,
            weight=child_weight,
            stake=0.5,
        )
        hyp.add_child(child_hyp)

    return hyp


def _print_results(hyp: Hypothesis, indent: int = 0) -> None:
    """Print hypothesis testing results as an indented tree."""
    pad = "  " * indent
    status = "REJECTED" if hyp.rejected else "retained"
    name_line = (hyp.name or "").split("\n")[0]
    print(
        f"{pad}{name_line:40s}"
        f"  p={hyp.pvalue():.4f}"
        f"  α={hyp.stake * (hyp.available_level or 0):.5f}"
        f"  [{status}]"
    )
    for child in hyp.children:
        _print_results(child, indent + 1)


def main() -> None:
    """Run the simulation study."""
    rng = np.random.default_rng(seed=42)

    # Steps 1-4: Generate data and treatment assignment.
    print("Generating data...")
    X, y_obs, z, tau_true = generate_data(rng, n=10_000, sigma=3.0)

    # Step 5 (implied): Delete potential outcomes.

    # Step 6: 50/50 train/evaluation split.
    n = len(X)
    idx = rng.permutation(n)
    train_idx = idx[: n // 2]
    eval_idx = idx[n // 2 :]

    X_train, y_train, z_train = X[train_idx], y_obs[train_idx], z[train_idx]
    X_eval, y_eval, z_eval = X[eval_idx], y_obs[eval_idx], z[eval_idx]

    # Step 7: T-learner pseudo-outcomes on the training set.
    print("Fitting T-learner (XGBoost)...")
    tau_pseudo = compute_pseudo_outcomes(X_train, y_train, z_train)

    # Step 8: Decision tree on pseudo-outcomes, depth 2.
    print("Fitting uplift tree...")
    dt = DecisionTreeRegressor(max_depth=2, min_samples_leaf=50, random_state=0)
    dt.fit(X_train, tau_pseudo)

    print("\nDecision tree (fit on training pseudo-outcomes):")
    print(export_text(dt, feature_names=_FEATURE_NAMES))

    # Reference: true and naive estimated ATEs in the evaluation set.
    tau_eval = tau_true[eval_idx]
    print("True and naively-estimated ATEs in evaluation set:")
    for x0v in [0, 1]:
        for x4v in [0, 1]:
            m = (X_eval[:, 0] == x0v) & (X_eval[:, 4] == x4v)
            naive = np.mean(y_eval[m & (z_eval == 1)]) - np.mean(
                y_eval[m & (z_eval == 0)]
            )
            print(
                f"  x0={x0v}, x4={x4v}:"
                f"  true τ = {np.mean(tau_eval[m]):.3f}"
                f"  naive Δ = {naive:.3f}"
                f"  n = {int(m.sum())}"
            )

    # Build hierarchy: one ContrastTest per internal tree node.
    root = _build_hypothesis(
        dt,
        0,
        X_eval,
        y_eval,
        z_eval,
        context="",
        weight=1.0,
        stake=0.5,
    )
    hierarchy = Hierarchy(fwer=0.05, root=root)

    print("\nRunning trickle-down procedure (FWER = 0.05, stake = 0.5 everywhere)...")
    hierarchy.test_hypotheses()

    print("\nHypothesis testing results (H0: ATE_left = ATE_right at each split):")
    print(f"  {'Split':<40}  {'p-value':>8}  {'test α':>8}  decision")
    print(f"  {'-' * 40}  {'-' * 8}  {'-' * 8}  --------")
    _print_results(root)

    # Visualize the hierarchy.
    fig = hierarchy.visualize_hierarchy()
    fig.suptitle(
        "Trickle-Down Procedure  |  FWER = 0.05, stake = 0.5\n"
        "H₀: ATEₗ = ATEᵣ at each split  ·  Blue = rejected  ·  Orange = retained",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig("simulation_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nHierarchy visualization saved to simulation_results.png")


if __name__ == "__main__":
    main()
