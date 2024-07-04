"""Classes for representing hypothesis hierarchies."""

import matplotlib.pyplot as plt
import networkx as nx
from EoN.auxiliary import hierarchy_pos

from pytrickle.hypothesis import Hypothesis


class Hierarchy:
    """Statistical hypothesis hierarchy."""

    def __init__(self, fwer: float, root: Hypothesis):
        self.fwer = fwer
        self.root = root

    def test_hypotheses(self):
        """Tests hypotheses."""
        queue = [self.root]
        self.root.set_available_level(self.fwer)
        self.root._normalize_children_weights_recursively()

        while queue:
            hypothesis = queue.pop(0)
            if not hypothesis.tested:
                hypothesis.test_hypothesis()

            if hypothesis.rejected:
                for child in hypothesis.children:
                    child.set_available_level(hypothesis.available_level * child.weight)
                    queue.append(child)
            else:
                for child in hypothesis.children:
                    child.set_available_level(
                        (1 - hypothesis.stake)
                        * hypothesis.available_level
                        * child.weight
                    )
                    queue.append(child)

    def visualize_hierarchy(self):
        """Visualize testing results."""
        G = nx.DiGraph()
        labels = {}
        colors = {}

        def add_nodes_edges(hypothesis, parent=None, level=0, pos=None, horizontal_idx=0):
            G.add_node(hypothesis)
            labels[hypothesis] = repr(hypothesis)
            colors[hypothesis] = "skyblue" if hypothesis.rejected else "orange"
            if parent:
                G.add_edge(parent, hypothesis)

            if parent is not None:
                pos[hypothesis] = (horizontal_idx, -level)

            for idx, child in enumerate(hypothesis.children):
                horizontal_idx = idx - len(hypothesis.children) // 2
                add_nodes_edges(child, hypothesis, level + 1, pos, horizontal_idx)

        pos = {self.root: (0, 0)}
        add_nodes_edges(self.root, pos=pos)

        # pos = nx.spring_layout(G)
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
