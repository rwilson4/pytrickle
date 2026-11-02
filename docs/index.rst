trickle-down
============

trickle-down controls the familywise error rate (FWER) over a family
of hypotheses arranged in a tree. It accompanies [Wil26]_, which uses
it to validate an uplift model: we summarize the model with a shallow
decision tree fit on a training split, then use an independent
evaluation split to certify claims about the tree's subgroups. The
claims can be that the treatment helps one subgroup, that it harms
another, and that the difference between them is real.

The paper's *trickle down* procedure gives each hypothesis a weight
and a *stake*. A hypothesis is tested at its stake times its
*available level*. A rejection passes the whole available level to its
children; a retention still passes on the part it did not wager. So
unlike a fixed hierarchy, testing continues below a retention, and a
null effect at the root cannot hide opposite effects in its subgroups.

The library provides:

- **The procedure**: :class:`~trickle_down.Hierarchy` runs trickle
  down over any tree of :class:`~trickle_down.Hypothesis` objects:
  Welch t-tests, linear contrasts, ANOVA, or p-values computed
  elsewhere.
- **The uplift architecture**:
  :func:`~trickle_down.interweaved_hierarchy` interweaves effect and
  heterogeneity hypotheses over a fitted tree, choosing each test's
  direction on the training split.
- **Confidence bounds**: every retained hypothesis still reports a
  bound, valid simultaneously with the procedure's rejections.
- **Comparators**: Bonferroni, Hierarchical Bonferroni, and the
  graphical approach of [BMBP09]_, of which trickle down is a special
  case.

Installation
------------

.. code-block:: bash

    pip install trickle-down

The package imports as ``trickle_down``.

Quickstart
----------

Summarize each node's arms on the evaluation split, record the effect
estimated on the training split, and build the hierarchy:

.. doctest::

    >>> from trickle_down import ArmSummary, UpliftNode, interweaved_hierarchy
    >>> men = UpliftNode(
    ...     name="men",
    ...     treated=ArmSummary(mean=0.42, variance=0.24, size=5_000),
    ...     control=ArmSummary(mean=0.35, variance=0.23, size=5_000),
    ...     tau_train=0.065,
    ... )
    >>> women = UpliftNode(
    ...     name="women",
    ...     treated=ArmSummary(mean=0.29, variance=0.21, size=5_000),
    ...     control=ArmSummary(mean=0.33, variance=0.22, size=5_000),
    ...     tau_train=-0.038,
    ... )
    >>> everyone = UpliftNode(
    ...     name="everyone",
    ...     treated=ArmSummary(mean=0.355, variance=0.229, size=10_000),
    ...     control=ArmSummary(mean=0.340, variance=0.224, size=10_000),
    ...     tau_train=0.014,
    ...     children=[women, men],
    ... )
    >>> claims = interweaved_hierarchy(everyone, alpha=0.05).test()
    >>> for claim in claims.certified():
    ...     print(claim.sentence())
    the effect is positive for everyone
    the effect is larger for men than for women
    the effect is negative for women
    the effect is positive for men

``print(claims)`` tabulates every hypothesis with its level, p-value
and verdict. Full citations are on the :doc:`references` page.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   user_guides/index
   api.rst
   references.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
