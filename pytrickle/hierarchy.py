"""Classes for representing hypothesis hierarchies."""

import matplotlib.pyplot as plt
import networkx as nx

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

        def add_nodes_edges(hypothesis, parent=None):
            G.add_node(hypothesis)
            labels[hypothesis] = "R" if hypothesis.rejected else "NR"
            if parent:
                G.add_edge(parent, hypothesis)
            for child in hypothesis.children:
                add_nodes_edges(child, hypothesis)

        add_nodes_edges(self.root)

        pos = nx.spring_layout(G)
        nx.draw(
            G,
            pos,
            with_labels=False,
            node_size=2000,
            node_color="skyblue",
            node_shape="o",
            alpha=0.8,
        )
        nx.draw_networkx_labels(G, pos, labels, font_size=12)
        plt.show()
