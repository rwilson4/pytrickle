r"""Reference implementation of the graphical approach to multiple testing.

The trickle down procedure is a special case of the graphical approach
of Bretz, Maurer, Brannath and Posch (2009). This module implements that
approach directly, and provides the encoding that turns a trickle down
hierarchy into an equivalent graph, so the claim can be checked
empirically rather than only argued on paper.

References
----------
Bretz, F., Maurer, W., Brannath, W., and Posch, M. (2009). A graphical
approach to sequentially rejective multiple test procedures. Statistics
in Medicine, 28(4), 586-604.

"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from pytrickle.hypothesis import Hypothesis


class GraphicalProcedure:
    r"""A sequentially rejective procedure defined by weights and a graph.

    A procedure is specified by an initial allocation of the overall
    level, a = (a_1, ..., a_m) with \sum_i a_i <= alpha, and a transition
    matrix G = (g_ij) satisfying g_ii = 0 and \sum_j g_ij <= 1. Testing
    H_i at level a_i and, on rejecting H_j, passing a fraction g_jl of
    a_j to each surviving H_l controls the FWER strongly at alpha.

    Parameters
    ----------
    allocation : npt.ArrayLike
        Initial levels a_i, in absolute terms rather than as fractions of
        alpha. Must sum to at most the intended alpha. Array-like rather
        than a sequence because `trickle_down_graph`, the main producer
        of this argument, hands back an ndarray.
    transitions : npt.ArrayLike
        The m-by-m matrix G.

    Notes
    -----
    Algorithm 1 of Bretz et al. (2009): repeatedly find a hypothesis
    whose p-value is at or below its current level, reject it, add
    a_j * g_jl to every surviving level, and update the graph by

       g_lk <- (g_lk + g_lj g_jk) / (1 - g_lj g_jl).

    Their remark (iii) notes the final rejection set does not depend on
    which rejectable hypothesis is taken first. ``selection`` exposes
    that choice so the invariance can be tested.

    """

    def __init__(
        self,
        allocation: npt.ArrayLike,
        transitions: npt.ArrayLike,
    ) -> None:
        a = np.asarray(allocation, dtype=np.float64)
        G = np.asarray(transitions, dtype=np.float64)

        m = a.shape[0]
        if G.shape != (m, m):
            raise ValueError(f"transitions must be {m}-by-{m}, got {G.shape}.")

        if np.any(a < 0.0):
            raise ValueError("Allocations must be non-negative.")

        if np.any(G < 0.0) or np.any(G > 1.0):
            raise ValueError("Transition weights must lie in [0, 1].")

        if np.any(np.abs(np.diag(G)) > 0.0):
            raise ValueError("Transition matrix must have a zero diagonal.")

        # Tolerance absorbs the floating-point error of the products
        # accumulated when a tree is encoded as a graph.
        if np.any(G.sum(axis=1) > 1.0 + 1e-9):
            raise ValueError("Transition matrix rows must sum to at most 1.")

        self.allocation = a
        self.transitions = G

    @property
    def alpha(self) -> float:
        """Total level allocated, which may be less than the nominal alpha."""
        return float(self.allocation.sum())

    def reject(
        self,
        pvalues: Sequence[float],
        selection: str = "argmin",
        rng: np.random.Generator | None = None,
    ) -> npt.NDArray[np.bool_]:
        """Return the rejection indicator for each hypothesis.

        Parameters
        ----------
        pvalues : Sequence[float]
            One p-value per hypothesis, in the same order as the
            allocation.
        selection : str
            Which rejectable hypothesis to take at each step: "argmin"
            for the smallest p_i / a_i as in Bretz et al., "first" for
            the lowest index, or "random". All three give the same
            answer; the choice exists to test that.
        rng : np.random.Generator | None
            Source of randomness when ``selection`` is "random".

        """
        p = np.asarray(pvalues, dtype=np.float64)
        m = self.allocation.shape[0]
        if p.shape != (m,):
            raise ValueError(f"Expected {m} p-values, got {p.shape}.")

        levels = self.allocation.copy()
        G = self.transitions.copy()
        active = np.ones(m, dtype=bool)
        rejected = np.zeros(m, dtype=bool)

        while True:
            candidates = np.flatnonzero(active & (p <= levels))
            if candidates.size == 0:
                return rejected

            j = self._select(candidates, p, levels, selection, rng)

            rejected[j] = True
            active[j] = False

            # Redistribute H_j's level along its outgoing edges.
            levels = levels + levels[j] * G[j, :]
            levels[~active] = 0.0

            # g_lk <- (g_lk + g_lj g_jk) / (1 - g_lj g_jl). Note the
            # denominator depends on l alone, not on k, so it is a
            # column vector broadcast across k rather than an outer
            # product. The graphs this module is used with are acyclic,
            # making g_lj g_jl zero and the denominator one, but the
            # general form is kept: this is a reference implementation.
            numerator = G + np.outer(G[:, j], G[j, :])  # shape (m, m)
            denominator = 1.0 - G[:, j] * G[j, :]  # shape (m,), indexed by l
            safe = denominator > 0.0
            updated = np.where(
                safe[:, None],
                numerator / np.where(safe, denominator, 1.0)[:, None],
                0.0,
            )
            np.fill_diagonal(updated, 0.0)
            updated[~active, :] = 0.0
            updated[:, ~active] = 0.0
            G = updated

    @staticmethod
    def _select(
        candidates: npt.NDArray[np.int64],
        p: npt.NDArray[np.float64],
        levels: npt.NDArray[np.float64],
        selection: str,
        rng: np.random.Generator | None,
    ) -> int:
        if selection == "first":
            return int(candidates[0])
        elif selection == "random":
            if rng is None:
                raise ValueError("selection='random' requires an rng.")
            return int(rng.choice(candidates))
        elif selection == "argmin":
            # A candidate with a zero level can only be a candidate when
            # its p-value is also zero; rank those first.
            ratios = np.where(
                levels[candidates] > 0.0,
                p[candidates]
                / np.where(levels[candidates] > 0.0, levels[candidates], 1.0),
                -np.inf,
            )
            return int(candidates[int(np.argmin(ratios))])
        else:
            raise ValueError(f"Unrecognized selection rule: {selection!r}")


def trickle_down_graph(
    root: Hypothesis, alpha: float
) -> tuple[list[Hypothesis], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Encode a trickle down hierarchy as an equivalent graph.

    Let w_i be node i's share of its parent's level, v_i its stake, W_i
    the product of weights along the path from the root to i, and anc(i)
    the strict ancestors of i. The initial allocation is

       a_i = alpha * v_i * W_i * \prod_{k \in anc(i)} (1 - v_k),

    and there is an edge from every node k to every strict descendant i
    with weight

       g_{ki} = v_i * W_{k \to i} * \prod_u (1 - v_u),

    the last product running over the nodes strictly between k and i.

    Parameters
    ----------
    root : Hypothesis
        Root of the hierarchy. Child weights are normalized in place, as
        ``Hierarchy.test_hypotheses`` also does.
    alpha : float
        Familywise error rate.

    Returns
    -------
    tuple
        The hypotheses in breadth-first order, the allocation, and the
        transition matrix, aligned with one another.

    Notes
    -----
    Edges run to every strict descendant, not merely to the children.
    A node's level under the trickle down procedure is a product over
    all of its ancestors' outcomes, and with parent-only edges just one
    of those factors would ever fire.

    The allocation sums to alpha only when every root-to-leaf path ends
    at a node with stake 1. Otherwise the procedure leaves some of the
    level unspent, and can be uniformly improved by adding edges out of
    the leaves.

    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")

    root._normalize_children_weights_recursively()

    ordered: list[Hypothesis] = []
    parent: dict[int, int] = {}
    queue: list[tuple[Hypothesis, int | None]] = [(root, None)]
    while queue:
        node, parent_index = queue.pop(0)
        index = len(ordered)
        ordered.append(node)
        if parent_index is not None:
            parent[index] = parent_index

        for child in node.children:
            queue.append((child, index))

    m = len(ordered)
    # The root takes no share from a parent, so its path weight is 1
    # whatever its own weight attribute happens to be.
    path_weight = np.ones(m, dtype=np.float64)
    ancestor_stakes = np.ones(m, dtype=np.float64)
    for i in range(1, m):
        k = parent[i]
        path_weight[i] = path_weight[k] * ordered[i].weight
        ancestor_stakes[i] = ancestor_stakes[k] * (1.0 - ordered[k].stake)

    allocation = np.array(
        [
            alpha * ordered[i].stake * path_weight[i] * ancestor_stakes[i]
            for i in range(m)
        ],
        dtype=np.float64,
    )

    transitions = np.zeros((m, m), dtype=np.float64)
    for i in range(1, m):
        # w_run is the product of weights from the ancestor's child down
        # to i; v_run the product of (1 - v_u) strictly between them.
        w_run = ordered[i].weight
        v_run = 1.0
        ancestor: int | None = parent[i]
        while ancestor is not None:
            transitions[ancestor, i] = ordered[i].stake * w_run * v_run
            v_run *= 1.0 - ordered[ancestor].stake
            w_run *= ordered[ancestor].weight
            ancestor = parent.get(ancestor)

    return ordered, allocation, transitions
