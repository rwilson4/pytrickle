"""Classes for representing hypothesis hierarchies."""

from typing import TYPE_CHECKING

from trickle_down.hypothesis import Hypothesis

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# matplotlib, networkx and EoN are imported inside visualize_hierarchy
# rather than here. They are needed only for drawing, and EoN in
# particular is not available in every environment this runs in; a
# top-level import would make the whole package unimportable there.


class Hierarchy:
    """Statistical hypothesis hierarchy."""

    def __init__(self, fwer: float, root: Hypothesis) -> None:
        if not 0.0 < fwer < 1.0:
            raise ValueError(f"fwer must lie in (0, 1); got {fwer}")

        self.fwer = fwer
        self.root = root

    def test_hypotheses(self) -> None:
        """Test every hypothesis at the level the procedure assigns it.

        Safe to call again, for instance after changing ``fwer``: every
        node is re-tested, so verdicts always agree with the levels
        recorded alongside them. P-values are cached on each hypothesis,
        so a re-run costs only the comparisons.
        """
        queue = [self.root]
        self.root.set_available_level(self.fwer)
        self.root._normalize_children_weights_recursively()

        while queue:
            hypothesis = queue.pop(0)
            # Re-test even a node marked tested. Skipping it would keep a
            # verdict reached at an earlier level while the level below
            # is overwritten, so the two would disagree.
            hypothesis.test_hypothesis()

            # Every node reaches the queue only after its parent has set
            # its level, and the root's was set above.
            available = hypothesis.available_level
            if available is None:
                raise ValueError(f"{hypothesis!r} was queued with no available level.")

            # A rejection passes the whole available level down; a
            # retention passes only the unwagered part.
            passed_down = 1.0 if hypothesis.rejected else 1.0 - hypothesis.stake
            for child in hypothesis.children:
                child.set_available_level(passed_down * available * child.weight)
                queue.append(child)

    def hypotheses(self) -> list[Hypothesis]:
        """Return every hypothesis in the tree, parents before children."""
        ordered: list[Hypothesis] = []
        queue = [self.root]
        while queue:
            hypothesis = queue.pop(0)
            ordered.append(hypothesis)
            queue.extend(hypothesis.children)

        return ordered

    def visualize_hierarchy(self) -> "Figure":
        """Draw the tree, coloring rejected hypotheses blue."""
        import matplotlib.pyplot as plt
        import networkx as nx
        from EoN.auxiliary import hierarchy_pos

        G: nx.DiGraph[Hypothesis] = nx.DiGraph()
        labels: dict[Hypothesis, str] = {}
        colors: dict[Hypothesis, str] = {}
        for hypothesis in self.hypotheses():
            G.add_node(hypothesis)
            labels[hypothesis] = repr(hypothesis)
            colors[hypothesis] = "skyblue" if hypothesis.rejected else "orange"
            for child in hypothesis.children:
                G.add_edge(hypothesis, child)

        pos = hierarchy_pos(G)
        fig = plt.figure(figsize=(16, 8))
        ax = plt.gca()
        nx.draw(
            G,
            pos,
            ax=ax,
            with_labels=False,
            node_size=4_000,
            node_color=[colors[node] for node in G.nodes()],
            node_shape="o",
            alpha=0.8,
        )
        nx.draw_networkx_labels(G, pos, labels, font_size=12, ax=ax)
        plt.axis("off")

        return fig
