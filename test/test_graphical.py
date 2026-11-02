"""Check the trickle down procedure against the graphical approach."""

import itertools
from collections.abc import Sequence

import numpy as np
import pytest

from pytrickle import (
    FixedPValue,
    GraphicalProcedure,
    Hierarchy,
    Hypothesis,
    trickle_down_graph,
)

ALPHA = 0.05


def chain(
    stakes: Sequence[float],
    pvalues: Sequence[float],
    weights: Sequence[float] | None = None,
) -> list[FixedPValue]:
    """Build a path graph of hypotheses, root first."""
    if weights is None:
        weights = [1.0] * len(stakes)

    nodes = [
        FixedPValue(p=p, weight=w, stake=v, name=str(i))
        for i, (p, w, v) in enumerate(zip(pvalues, weights, stakes))
    ]
    for parent, child in itertools.pairwise(nodes):
        parent.add_child(child)

    return nodes


class TestTrickleDownGraph:
    """The encoding of a hierarchy as an initial allocation and a graph."""

    @staticmethod
    def test_three_chain_matches_hand_calculation() -> None:
        """Root, child, grandchild with unit weights and stakes of 1/2.

        Worked by hand: the allocation is (alpha/2, alpha/4, alpha/8) and
        the edges are g_RC = 1/2, g_RG = 1/4, g_CG = 1/2. The edge to the
        grandchild carries the factor (1 - v_C), which is exactly the
        term a parent-only encoding would miss.
        """
        nodes = chain(stakes=[0.5, 0.5, 0.5], pvalues=[0.5, 0.5, 0.5])
        ordered, allocation, transitions = trickle_down_graph(nodes[0], ALPHA)

        assert ordered == nodes
        np.testing.assert_allclose(
            allocation, [ALPHA / 2, ALPHA / 4, ALPHA / 8], rtol=1e-12
        )
        np.testing.assert_allclose(
            transitions,
            [[0.0, 0.5, 0.25], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]],
            rtol=1e-12,
        )

    @staticmethod
    def test_unit_stakes_give_the_fixed_hierarchy_procedure() -> None:
        """With every stake 1, all the level sits on the root."""
        nodes = chain(stakes=[1.0, 1.0, 1.0], pvalues=[0.5, 0.5, 0.5])
        _, allocation, transitions = trickle_down_graph(nodes[0], ALPHA)

        np.testing.assert_allclose(allocation, [ALPHA, 0.0, 0.0], rtol=1e-12)
        # A retained node forfeits everything, so no edge skips a level.
        np.testing.assert_allclose(
            transitions,
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            rtol=1e-12,
        )

    @staticmethod
    def test_alpha_is_not_exhausted() -> None:
        """A documented defect: the procedure leaves level unspent.

        Goeman and Solari (2010) call a sequential rejection procedure
        that does not exhaust alpha inadmissible --- it can be uniformly
        improved, here by adding edges out of the leaves.
        """
        nodes = chain(stakes=[0.5, 0.5, 0.5], pvalues=[0.5, 0.5, 0.5])
        _, allocation, transitions = trickle_down_graph(nodes[0], ALPHA)

        assert allocation.sum() == pytest.approx(7.0 * ALPHA / 8.0)
        assert allocation.sum() < ALPHA
        # Every row sum is short of one for the same reason.
        assert np.all(transitions.sum(axis=1) <= 1.0)
        assert transitions[0, :].sum() == pytest.approx(0.75)

    @staticmethod
    def test_regularity_conditions_hold_on_random_trees() -> None:
        """Bretz et al. require sum(a) <= alpha and row sums <= 1."""
        rng = np.random.default_rng(seed=20260930)
        for _ in range(200):
            root, _ = random_hierarchy(rng)
            _, allocation, transitions = trickle_down_graph(root, ALPHA)

            assert allocation.sum() <= ALPHA + 1e-12
            assert np.all(allocation >= 0.0)
            assert np.all(transitions.sum(axis=1) <= 1.0 + 1e-12)
            np.testing.assert_allclose(np.diag(transitions), 0.0, atol=0.0)


def random_hierarchy(
    rng: np.random.Generator, max_nodes: int = 9
) -> tuple[FixedPValue, list[FixedPValue]]:
    """A random rooted tree of hypotheses with random weights and stakes.

    P-values are drawn from a mixture so that rejections actually
    happen: half uniform on the unit interval, half concentrated near
    zero.
    """
    n = int(rng.integers(1, max_nodes + 1))

    def draw_pvalue() -> float:
        if rng.random() < 0.5:
            return float(rng.random())
        return float(rng.uniform(0.0, 0.02))

    root = FixedPValue(
        p=draw_pvalue(), weight=1.0, stake=float(rng.uniform(0.05, 1.0)), name="0"
    )
    nodes = [root]
    for i in range(1, n):
        parent = nodes[int(rng.integers(len(nodes)))]
        child = FixedPValue(
            p=draw_pvalue(),
            weight=float(rng.uniform(0.1, 2.0)),
            stake=float(rng.uniform(0.05, 1.0)),
            name=str(i),
        )
        parent.add_child(child)
        nodes.append(child)

    return root, nodes


class TestEquivalence:
    """The trickle down procedure is a graphical procedure."""

    @staticmethod
    @pytest.mark.parametrize("selection", ["argmin", "first", "random"])
    def test_same_rejections_on_random_trees(selection: str) -> None:
        """Both procedures reject exactly the same hypotheses.

        This is the empirical form of the claim that the trickle down
        procedure is a special case of Bretz et al. (2009). It doubles as
        a correctness test for Hierarchy.test_hypotheses.
        """
        rng = np.random.default_rng(seed=17)
        n_with_rejections = 0

        for _ in range(500):
            root, _ = random_hierarchy(rng)

            # Encode first: both entry points normalize weights in
            # place, and we need the two runs to see the same tree.
            ordered, allocation, transitions = trickle_down_graph(root, ALPHA)
            pvalues = [node.pvalue() for node in ordered]

            graphical = GraphicalProcedure(allocation, transitions)
            expected = graphical.reject(pvalues, selection=selection, rng=rng)

            Hierarchy(fwer=ALPHA, root=root).test_hypotheses()
            actual = np.array([node.rejected for node in ordered], dtype=bool)

            np.testing.assert_array_equal(actual, expected)
            n_with_rejections += int(expected.any())

        # Guard against the test passing because nothing ever rejects.
        assert n_with_rejections > 100

    @staticmethod
    def test_selection_rule_does_not_change_the_outcome() -> None:
        """Remark (iii) of Bretz et al.: the rejection set is invariant.

        The graph is a transitive closure, so a rejection deep in the
        tree augments an ancestor's edges via
        g_lk <- g_lk + g_lj g_jk. That update is what makes the order
        not matter, and this is the test that exercises it.
        """
        rng = np.random.default_rng(seed=99)
        for _ in range(300):
            root, _ = random_hierarchy(rng)
            _, allocation, transitions = trickle_down_graph(root, ALPHA)
            pvalues = [node.pvalue() for node in ordered_pvalues(root)]

            graphical = GraphicalProcedure(allocation, transitions)
            by_argmin = graphical.reject(pvalues, selection="argmin")
            by_first = graphical.reject(pvalues, selection="first")
            by_random = graphical.reject(pvalues, selection="random", rng=rng)

            np.testing.assert_array_equal(by_argmin, by_first)
            np.testing.assert_array_equal(by_argmin, by_random)


def ordered_pvalues(root: Hypothesis) -> list[Hypothesis]:
    """Breadth-first traversal, matching trickle_down_graph's ordering."""
    ordered = []
    queue = [root]
    while queue:
        node = queue.pop(0)
        ordered.append(node)
        queue.extend(node.children)

    return ordered


class TestGraphicalProcedureValidation:
    """Input validation on the reference implementation."""

    @staticmethod
    def test_rejects_nonzero_diagonal() -> None:
        with pytest.raises(ValueError, match="zero diagonal"):
            GraphicalProcedure([0.05], np.array([[0.5]]))

    @staticmethod
    def test_rejects_row_sums_above_one() -> None:
        transitions = np.array([[0.0, 0.7, 0.7], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="rows must sum"):
            GraphicalProcedure([0.05, 0.0, 0.0], transitions)

    @staticmethod
    def test_rejects_mismatched_pvalues() -> None:
        procedure = GraphicalProcedure([0.05, 0.0], np.zeros((2, 2)))
        with pytest.raises(ValueError, match="Expected 2 p-values"):
            procedure.reject([0.01])


class TestGraphIsAdmissible:
    """The conditions Bretz et al. require of a graph.

    A weighting strategy is a pair (a, G) with g_ii = 0,
    0 <= g_ki <= 1, row sums of G at most one, and sum(a) <= alpha
    (Bretz et al. 2011, Section 2). Strong control follows from
    their result only if the encoding satisfies all four, so the
    paper's proof rests on these holding for every hierarchy, not
    just the ones we happened to draw.
    """

    @staticmethod
    def test_conditions_hold_on_random_hierarchies() -> None:
        alpha = 0.05
        rng = np.random.default_rng(20261001)
        for _ in range(300):
            root, _ = random_hierarchy(rng)
            _, allocation, transitions = trickle_down_graph(root, alpha)

            assert np.all(np.diag(transitions) == 0.0)
            assert np.all(transitions >= 0.0)
            assert np.all(transitions <= 1.0 + 1e-12)
            assert np.all(transitions.sum(axis=1) <= 1.0 + 1e-12)
            assert allocation.sum() <= alpha + 1e-12

    @staticmethod
    def test_alpha_is_exhausted_exactly_when_every_leaf_stakes_everything() -> None:
        """The condition is on the leaves alone, not on every node.

        An interior node with stake below one loses nothing: the
        unwagered part is still handed down and eventually staked.
        Only a leaf can strand level, because there is nowhere below
        it to hand the remainder to. Writing the condition as
        "every stake is one" is sufficient but not necessary, and
        the second block here is what distinguishes them.
        """
        alpha = 0.05
        rng = np.random.default_rng(7)
        for _ in range(60):
            root, nodes = random_hierarchy(rng)
            leaves = [node for node in nodes if not node.children]

            for node in nodes:
                node.stake = 1.0
            _, allocation, _ = trickle_down_graph(root, alpha)
            assert allocation.sum() == pytest.approx(alpha)

            # Interior stakes well below one, leaves still at one.
            for node in nodes:
                node.stake = 1.0 if not node.children else 0.3
            _, allocation, _ = trickle_down_graph(root, alpha)
            assert allocation.sum() == pytest.approx(alpha)

            # Drop a single leaf below one and the level is stranded.
            leaves[0].stake = 0.9
            _, allocation, _ = trickle_down_graph(root, alpha)
            assert allocation.sum() < alpha - 1e-12

    @staticmethod
    def test_row_sum_recursion_matches_the_closed_form() -> None:
        r"""S(k) = sum_c w_c [v_c + (1 - v_c) S(c)], the paper's claim.

        The row sums are not an accident of the parameterization;
        they satisfy this recursion, which is what makes the
        induction up from the leaves work.
        """
        alpha = 0.05
        rng = np.random.default_rng(99)
        for _ in range(200):
            root, _ = random_hierarchy(rng)
            ordered, allocation, transitions = trickle_down_graph(root, alpha)
            index = {id(node): i for i, node in enumerate(ordered)}

            def recursion(node: Hypothesis) -> float:
                return sum(
                    child.weight
                    * (child.stake + (1.0 - child.stake) * recursion(child))
                    for child in node.children
                )

            for node in ordered:
                assert transitions[index[id(node)], :].sum() == pytest.approx(
                    recursion(node)
                )

            assert allocation.sum() / alpha == pytest.approx(
                root.stake + (1.0 - root.stake) * recursion(root)
            )

    @staticmethod
    def test_three_chain_with_half_stakes_allocates_seven_eighths() -> None:
        """The figure the paper quotes."""
        alpha = 0.05
        nodes = chain([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
        _, allocation, _ = trickle_down_graph(nodes[0], alpha)
        assert allocation.sum() == pytest.approx(7.0 * alpha / 8.0)

    @staticmethod
    def test_the_graph_is_acyclic_so_the_update_denominator_is_one() -> None:
        """Why Bretz et al.'s cycle correction never fires here.

        Edges run only from a node to its strict descendants, so
        g_lj g_jl = 0 for every pair and the denominator in their
        Algorithm 2 is identically one.
        """
        rng = np.random.default_rng(1234)
        for _ in range(200):
            root, _ = random_hierarchy(rng)
            _, _, g = trickle_down_graph(root, 0.05)
            assert np.all(g * g.T == 0.0)
