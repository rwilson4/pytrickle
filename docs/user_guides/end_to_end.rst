End-to-End: From Uplift Model to Certified Claims
=================================================

The usage guide starts from the arm summaries of a tree someone has
already fitted. This one starts from raw data. We simulate a
randomized experiment, fit an uplift model on one half of it, distill
the model into a decision tree, and then use trickle down on the other
half to decide where the treatment pays for itself. This is the
workflow of [Wil26]_, with a scikit-learn model standing in for
whatever learner you prefer.

Every code block below is executed when the documentation is built,
and the printed output is checked, so the guide cannot drift from the
library. Because the data are simulated we know the true effects, and
we finish by comparing the conclusions against them.

One rule organizes everything that follows:

    Anything that shapes the hypotheses (the model, the tree, the
    threshold, the direction of each test) may be chosen using only
    the training split, or nothing at all. The evaluation split is
    used once, to compute p-values.

The familywise guarantee is conditional on the training split, so it
survives any amount of exploration there [Cox75]_. It does not survive
a single peek at the evaluation split: change the tree or the
threshold after seeing which claims came out, and the guarantee is
gone.

The code needs scikit-learn, which is in the ``dev`` extra but is not
a dependency of trickle-down itself.

Simulating an Experiment
------------------------

A subscription service is considering mailing customers a gift box.
Each box costs $3 to make and ship. The outcome is each customer's
gross margin over the following quarter, in dollars, which does not
include the box, so the gift pays for itself in a group of customers
exactly when it raises their margin by more than $3. To find out
where it does, the service mails boxes to a random half of its
customers.

The gift never hurts, but it does not help everyone equally. Customers
in their first year respond strongly; established customers respond
weakly, premium ones a little more than basic ones. The average effect
clears the cost, so a blanket rollout would pay, but most customers
would cost more than they return.

There are eight features. Tenure and plan moderate the effect; tenure,
plan and weekly sessions predict the baseline outcome; age and four
noise columns do neither. Outcomes are noisy, with a standard
deviation of 20 against effects of a few dollars.

.. testcode::

    import math

    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.tree import DecisionTreeRegressor, export_text

    from trickle_down import (
        ArmSummary,
        bonferroni_rejections,
        hierarchical_bonferroni_rejections,
        interweaved_hierarchy,
        UpliftNode,
    )

    COST = 3.0  # dollars per gift box
    FEATURES = [
        "tenure", "premium", "age", "sessions",
        "noise1", "noise2", "noise3", "noise4",
    ]


    def true_effect(X):
        """The effect of the gift on each customer's quarterly margin."""
        tenure, premium = X[:, 0], X[:, 1] == 1
        return np.where(tenure < 12, 7.0, np.where(premium, 2.0, 0.5))


    def simulate(rng, n):
        tenure = np.floor(rng.exponential(24.0, size=n))  # months
        premium = (rng.random(n) < 0.4).astype(float)
        age = np.round(rng.uniform(18, 75, size=n))
        sessions = rng.poisson(3.0, size=n).astype(float)
        noise = rng.standard_normal((n, 4))
        X = np.column_stack([tenure, premium, age, sessions, noise])

        baseline = 40 + 0.5 * np.minimum(tenure, 36) + 2 * sessions + 10 * premium
        y0 = baseline + rng.normal(0, 20, size=n)
        z = rng.random(n) < 0.5  # who is mailed a box
        y = np.where(z, y0 + true_effect(X), y0)
        return X, z, y


    rng = np.random.default_rng(4)
    X, z, y = simulate(rng, 30_000)

    tau = true_effect(X)
    print(f"{'everyone':24} 100%  effect {tau.mean():+.2f}")
    for label, mask in [
        ("tenure < 12", X[:, 0] < 12),
        ("tenure >= 12, premium", (X[:, 0] >= 12) & (X[:, 1] == 1)),
        ("tenure >= 12, basic", (X[:, 0] >= 12) & (X[:, 1] == 0)),
    ]:
        print(f"{label:24} {mask.mean():4.0%}  effect {tau[mask].mean():+.2f}")

.. testoutput::

    everyone                 100%  effect +3.42
    tenure < 12               39%  effect +7.00
    tenure >= 12, premium     24%  effect +2.00
    tenure >= 12, basic       37%  effect +0.50

Only first-year customers, about two fifths of the base, clear the
$3 cost. Mailing everyone would earn :math:`3.42 - 3 = 0.42` per
customer; mailing only first-year customers would earn
:math:`0.39 \times (7 - 3) \approx 1.56` per customer in the base. The
analysis should find that policy, and it should not be fooled by the
fact that the gift helps everyone a little.

Splitting the Data
------------------

Split once, at random, before doing anything else. Half the data
trains; half evaluates.

.. testcode::

    train = rng.random(len(y)) < 0.5
    X_train, z_train, y_train = X[train], z[train], y[train]
    X_eval, z_eval, y_eval = X[~train], z[~train], y[~train]
    print(f"{train.sum()} training, {(~train).sum()} evaluation")

.. testoutput::

    14903 training, 15097 evaluation

An even split is conventional, not optimal. More training data gives
a better tree; more evaluation data gives more power to certify claims
about it. The paper uses halves.

Fitting the Uplift Model
------------------------

The library never sees the uplift model, only the tree that summarizes
it, so any learner of individual treatment effects will do: an S-,
T-, X- or DR-learner, a causal forest, or something bespoke. We use a
T-learner, which fits one outcome model per arm and predicts the
effect as their difference:

.. testcode::

    def fit_t_learner(X, z, y):
        """Return a function predicting each row's treatment effect."""
        models = []
        for arm in (False, True):
            model = HistGradientBoostingRegressor(
                max_depth=3,
                learning_rate=0.05,
                max_iter=200,
                early_stopping=False,
                random_state=0,
            )
            models.append(model.fit(X[z == arm], y[z == arm]))

        control, treated = models
        return lambda X: treated.predict(X) - control.predict(X)

The individual predictions from such a model are too noisy to report
and too numerous to test. What a decision-maker can act on is a
handful of subgroups, so we distill the predictions into a regression
tree. ``min_samples_leaf=0.05`` keeps every leaf at 5% of the data or
more: a smaller leaf would be cheap to grow but too small to certify
anything about. It is the only limit we put on the tree's size; the
section on depth below explains why.

.. testcode::

    def summary_tree(X, cate):
        """A tree approximating the predicted effects."""
        tree = DecisionTreeRegressor(min_samples_leaf=0.05, random_state=0)
        return tree.fit(X, cate)

From Tree to Hypotheses
-----------------------

:func:`~trickle_down.interweaved_hierarchy` takes a tree of
:class:`~trickle_down.UpliftNode` objects. Each node needs two things,
from two different splits:

- the treated and control outcome moments in the node, from the
  **evaluation** split, which the tests are computed against; and
- ``tau_train``, the effect estimated in the node from the
  **training** split, which only chooses the direction of each
  one-sided test (§4).

scikit-learn's ``decision_path`` tells us which rows reach each node,
so the conversion is a short recursion. We also give each node a name a
reader can understand, since the names end up in the claims.

.. testcode::

    def describe(feature, threshold, below):
        if feature == "premium":
            return "basic plan" if below else "premium plan"
        if feature == "tenure":
            months = math.ceil(threshold)  # tenure is a whole number of months
            return f"tenure < {months} months" if below else f"tenure >= {months} months"
        return f"{feature} {'<=' if below else '>'} {threshold:.3g}"


    def node_names(tree):
        """A readable name for every node, indexed like ``tree.tree_``."""
        structure = tree.tree_
        names = ["everyone"] * structure.node_count
        for node in range(structure.node_count):  # parents precede children
            left, right = structure.children_left[node], structure.children_right[node]
            if left == -1:
                continue
            feature = FEATURES[structure.feature[node]]
            threshold = structure.threshold[node]
            prefix = "" if node == 0 else names[node] + ", "
            names[left] = prefix + describe(feature, threshold, below=True)
            names[right] = prefix + describe(feature, threshold, below=False)
        return names


    def to_uplift_tree(tree, train, evaluation):
        """Summarize every node of ``tree`` for interweaved_hierarchy."""
        (X_t, z_t, y_t), (X_e, z_e, y_e) = train, evaluation
        in_train = tree.decision_path(X_t).toarray().astype(bool)
        in_eval = tree.decision_path(X_e).toarray().astype(bool)
        names = node_names(tree)
        structure = tree.tree_

        def build(node):
            left, right = structure.children_left[node], structure.children_right[node]
            t, e = in_train[:, node], in_eval[:, node]
            return UpliftNode(
                name=names[node],
                treated=ArmSummary.from_samples(y_e[e & z_e]),
                control=ArmSummary.from_samples(y_e[e & ~z_e]),
                tau_train=float(y_t[t & z_t].mean() - y_t[t & ~z_t].mean()),
                children=[] if left == -1 else [build(left), build(right)],
            )

        return build(0)

Two details are deliberate. Membership comes from ``decision_path``
rather than from re-applying the thresholds, because scikit-learn
compares features in single precision, and a comparison in double
precision can send a row within rounding of a threshold the other
way. And
``tau_train`` is the difference of arm means on the training split,
as in the paper, rather than the tree's own prediction (the node's
mean predicted effect). Either is legitimate, because both use only
the training split; a poor choice costs power, never validity.

Setting the Threshold
---------------------

Every effect hypothesis compares a node's average effect with a
threshold ``delta`` (§3). The default, zero, asks whether the
treatment helps at all. That is not our question. The gift costs $3,
so the question for each group is whether the effect exceeds $3, and
we set ``delta=COST``. As the paper puts it, "when applying the
treatment requires some resource, setting a suitable positive
:math:`\delta_i` can determine whether the benefit justifies the
resource" (§3).

With a positive threshold both directions of a claim are worth
having. Rejecting :math:`E^+_S` certifies that the gift *pays* in
:math:`S`; rejecting :math:`E^-_S` certifies that it *does not*.
Which one a node tests is chosen on the training split by comparing
its training estimate with ``delta`` (§4), so a group whose training
effect is $2 is tested for falling short of $3, not for being
positive.

The threshold must come from the business case, before the
evaluation split is touched. It is part of the family of hypotheses,
and choosing it after seeing the results, or trying several and
reporting the most favorable, is a peek like any other. Nor can a
threshold be recovered from an analysis run at another one: a rejected
hypothesis may claim only the threshold it was tested against (§6).

How Deep? Grow, Then Prune
--------------------------

How deep should the tree be? The paper's answer is that the question
needs no tuning: "there is no reason to control the tree depth when
constructing the tree on the training set" (§5.2), and "the method
works with arbitrarily deep trees and offers a disciplined pruning
strategy" (§7.1). Grow the tree as deep as ``min_samples_leaf``
allows, let the procedure find where the evidence runs out, and prune
what lies beyond. Two facts make this work.

**Depth does not dilute the tests above it.** The level at which
trickle down tests a node depends only on the verdicts and weights of
its ancestors, never on how many descendants the node has. Growing the
tree adds hypotheses at the bottom and leaves the level of every
existing one unchanged, with one exception: a former leaf, which
staked its whole level because it had nothing to pass on, now stakes
``effect_stake``, so its effect test runs at that fraction of the
level it had (half, by default). This is the paper's "no risk of
sacrificing reliability or power (except at the leaves)" (§7.3).
Bonferroni is different: every hypothesis added anywhere dilutes the
level of all the others, which makes it "useless in deep trees whose
hypothesis count doubles with each new level" (§7.3).

**The shallow tree is the top of the deep one.** CART is greedy, so
for the same training data and ``min_samples_leaf`` the first :math:`k`
layers of a deep tree are exactly the tree we would have got by
stopping at depth :math:`k`. A deep tree loses none of a shallow
tree's splits; it only adds splits beneath them.

The tree's size is still limited by ``min_samples_leaf``, which "ensures
the sample size is large enough to run the associated tests" (§7.1).
The paper uses a hundredth of the sample; we use a twentieth, which
keeps this guide's tree small enough to read.

.. testcode::

    cate = fit_t_learner(X_train, z_train, y_train)(X_train)
    tree = summary_tree(X_train, cate)
    print(f"depth {tree.get_depth()}, {tree.get_n_leaves()} leaves")
    print(export_text(tree, feature_names=FEATURES, decimals=1, max_depth=1))

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    depth 5, 15 leaves
    |--- tenure <= 11.5
    |   |--- sessions <= 2.5
    |   |   |--- truncated branch of depth 2
    |   |--- sessions >  2.5
    |   |   |--- truncated branch of depth 3
    |--- tenure >  11.5
    |   |--- tenure <= 30.5
    |   |   |--- truncated branch of depth 4
    |   |--- tenure >  30.5
    |   |   |--- truncated branch of depth 4

``export_text`` prints only the first two layers here. The first split
recovers the true one, at twelve months of tenure. Below it, the tree
carries on splitting wherever that reduces the error in the predicted
effects, on noise features as readily as on real ones. Nothing on the
training split can tell us which of these splits matter. The
evaluation split can.

Testing on the Evaluation Split
-------------------------------

Now, and only now, touch the evaluation split:

.. testcode::

    root = to_uplift_tree(
        tree, (X_train, z_train, y_train), (X_eval, z_eval, y_eval)
    )
    claims = interweaved_hierarchy(root, alpha=0.05, delta=COST).test()
    print(f"{len(claims.claims)} hypotheses, {len(claims.certified())} certified")

.. testoutput::

    43 hypotheses, 12 certified

``print(claims)`` would tabulate all of them. Here are the six at the
top of the tree, at its root and its first split:

.. testcode::

    for claim in claims.claims:
        if claim.depth <= 1:
            verdict = "rejected" if claim.rejected else "retained"
            print(
                f"{claim.node:20} {claim.kind:14} {claim.alternative:8} "
                f"p = {claim.pvalue:.4f}  level = {claim.level:.4f}  {verdict}"
            )

.. testoutput::

    everyone             effect         greater  p = 0.0060  level = 0.0250  rejected
    everyone             heterogeneity  less     p = 0.0000  level = 0.0250  rejected
    tenure < 12 months   effect         greater  p = 0.0000  level = 0.0125  rejected
    tenure < 12 months   heterogeneity  less     p = 0.4308  level = 0.0125  retained
    tenure >= 12 months  effect         less     p = 0.0040  level = 0.0125  rejected
    tenure >= 12 months  heterogeneity  less     p = 0.5314  level = 0.0125  retained

Read them top to bottom, as the procedure does.

- **Mailing everyone pays, on average.** The training estimate,
  :math:`+3.04`, is above the $3 cost, so the root tests whether the
  effect *exceeds* $3. It does (:math:`p = 0.006` at level 0.025). The
  rejection passes the root's whole level on, so the heterogeneity
  test at the root runs at 0.025 rather than 0.0125.
- **The first split is certified**: the gift does more for first-year
  customers than for established ones. That rejection passes its whole
  level to the two halves, 0.0125 each.
- **The gift pays among first-year customers** (:math:`p < 0.0001`).
- **It does not pay among established customers.** Their training
  estimate, :math:`+0.99`, is below the cost, so this node tests
  whether the effect *falls short of* $3, and it does (:math:`p =
  0.004`). This is the claim the decision turns on: it says the
  customers a blanket rollout would add to the targeted one cost more
  than they return.
- **Neither half's own split is certified** (:math:`p = 0.43` and
  :math:`0.53`), but testing continues below both, and eight more
  claims are certified further down.

Pruning
-------

Most of the remaining claims sit deeper in the tree, and most of them
repeat a decision already made higher up: a subgroup of first-year
customers, split on a noise feature, also clears the cost. That is
true, but it changes nothing. The power available deep in the tree
falls exponentially, so "at some point, we will retain *all*
hypotheses below a certain depth. We may prune these nodes, leaving a
shallower tree emphasizing the certified claims" (§5.2).

We prune branch by branch, keeping a split only where it matters for
the decision. A node's certified effect claim, if it has one, says
whether to mail its customers. A split is worth reporting if something
certified below it calls for a *different* action: a part of a
profitable group certified not to pay, a part of an unprofitable one
certified to pay, or a decided part of a node that is itself
undecided. Otherwise the split is cut, with everything beneath it.

.. testcode::

    import dataclasses


    def action(claim):
        return "mail" if claim.alternative == "greater" else "do not mail"


    def prune(node, claims):
        """Cut every split below which nothing certified changes the action."""
        decided = {
            claim.node: action(claim)
            for claim in claims.certified()
            if claim.kind == "effect"
        }

        def actions_below(node):
            found = set()
            for child in node.children:
                if child.name in decided:
                    found.add(decided[child.name])
                found |= actions_below(child)
            return found

        def cut(node):
            keep = bool(actions_below(node) - {decided.get(node.name)})
            children = [cut(child) for child in node.children] if keep else []
            return dataclasses.replace(node, children=children)

        return cut(node)


    pruned = prune(root, claims)

The pruned tree is the analysis's result. To present it, we list under
each node what the evaluation split certified about it, and, for each
hypothesis it retained, the confidence bound of §6. Those bounds hold
simultaneously with the certified claims. Indentation shows the path,
so each node is labeled by its last split alone; ``[A] - [B]`` is the
effect in node A minus the effect in node B. A heterogeneity
hypothesis is listed only if its split survived the pruning.

.. testcode::

    def label(name):
        return name.split(", ")[-1]


    def verdict(claim):
        low, high = claim.reportable_bound
        if claim.kind == "effect":
            quantity = "effect"
        else:
            quantity = f"[{label(claim.right)}] - [{label(claim.left)}]"
        if claim.alternative == "greater":
            return f"{quantity} > {low:.3g}"
        return f"{quantity} < {high:.3g}"


    def show(node, claims, indent=0):
        pad = "    " * indent
        print(f"{pad}{label(node.name)}: n = {node.size}, estimated effect {node.tau_hat:+.2f}")
        for claim in claims.claims:
            if claim.node != node.name:
                continue
            if claim.kind == "heterogeneity" and not node.children:
                continue  # its split was pruned away
            status = "certified" if claim.rejected else "bounded"
            print(f"{pad}  {status:9}  {verdict(claim)}")
        for child in node.children:
            show(child, claims, indent + 1)


    show(pruned, claims)

.. testoutput::

    everyone: n = 15097, estimated effect +3.89
      certified  effect > 3
      certified  [tenure >= 12 months] - [tenure < 12 months] < 0
        tenure < 12 months: n = 6004, estimated effect +6.98
          certified  effect > 3
        tenure >= 12 months: n = 9093, estimated effect +1.80
          certified  effect < 3

The estimated effects are the evaluation split's differences in arm
means. Because the tree was chosen on the other split, they are
unbiased for each node, but they are point estimates and carry no
simultaneous guarantee; the certified and bounded sentences do.

Pruning is legitimate because it only chooses which claims to report,
and the familywise guarantee covers every subset of them. What is
*not* legitimate is to test the pruned tree again. Its new leaves would
stake their whole level instead of half of it, and choosing which
nodes get that boost after seeing the evaluation split is exactly the
kind of peek the guarantee does not survive. Prune the report, not the
hierarchy.

Findings
--------

Here is what the analysis lets us tell the decision-maker, with the
familywise error rate held at 5% across every statement together.

**Mail the gift to customers in their first year.** Among customers
with under twelve months' tenure the gift raises quarterly margin by
more than its $3 cost.

**Do not mail it to established customers.** Among customers with
twelve months or more, the gift raises margin by less than $3. It may
well help them (the estimate is :math:`+1.80`), but not by enough to
pay for itself.

**Targeting beats a blanket rollout.** Mailing everyone would pay on
average, and that is certified too, so an analysis of the average
effect alone would have approved the rollout. But the difference
between mailing everyone and mailing only first-year customers is
exactly the gift's net return on established customers, and that is
certified to be negative. Every box sent to an established customer
loses money.

What the analysis does *not* support matters as much:

- **How much the gift pays.** A certified effect claim establishes
  which side of the threshold a group lies on, not how far. A
  rejected hypothesis may claim only its threshold, because its level
  has already passed to its children (§6).
- **Finer targeting.** No split below the first is certified to change
  the decision. Eight deeper claims are certified, but all of them say
  that some subgroup of first-year customers also clears the cost,
  which the policy already assumes. Nothing identifies established
  customers who would pay for a box, nor first-year customers who
  would not.
- **Differences that do not change the decision.** The tree split
  established customers by tenure, and the procedure certified
  nothing about that split: it is retained, and its bound says only
  that the difference across it is below 2.08.

One caveat applies to all of it. These conclusions are conditional on
one random split of the data into training and evaluation halves,
which "may lead two analysts to draw different conclusions about
treatment effects" (§8). The guarantee covers the error rate of the
procedure, not the variability across splits.

Why the Threshold Matters
-------------------------

Suppose we had left ``delta`` at its default of zero. That is a
different question, whether the gift helps at all, and in a real
analysis it would have to be chosen in advance instead of ours, not
run alongside it. Running it here, purely for contrast:

.. testcode::

    helps = interweaved_hierarchy(root, alpha=0.05).test()
    print(f"{len(helps.certified())} certified, for example:")
    for claim in helps.certified()[:4]:
        print(f"  - {claim.sentence()}")

.. testoutput::

    16 certified, for example:
      - the effect is positive for everyone
      - the effect is larger for tenure < 12 months than for tenure >= 12 months
      - the effect is positive for tenure < 12 months
      - the effect is positive for tenure < 12 months, sessions <= 2.5

Sixteen claims, every one of them true: the gift helps everyone,
established customers included ("the effect is positive for tenure >=
12 months" is among them). An analyst who asked this question would
report that the gift works everywhere, and pruning by the same rule
would collapse the tree to its root, since no part of it calls for a
different action from "the gift helps". The heterogeneity claim at the
root is still certified, but it reads as a curiosity: the gift helps
new customers more.

The threshold is what turns that curiosity into a policy. Testing
against the cost asks of each group the question the decision needs
answered, and the same data that certified "positive" for established
customers certify "below break-even" for them. Certifying that a
treatment helps is not the same as certifying that it pays.

Checking Against the Truth
--------------------------

Because we simulated the data, we can see what each node's effect
really is. ``node_names`` lines the tree's nodes up with
``decision_path``, so the true effect in each node is a mask away:

.. testcode::

    in_eval = tree.decision_path(X_eval).toarray().astype(bool)
    column = {name: i for i, name in enumerate(node_names(tree))}
    truth = true_effect(X_eval)
    print(f"{'node':20} {'train':>6} {'eval':>6} {'true':>6}")
    for node in pruned.walk():
        actual = truth[in_eval[:, column[node.name]]].mean()
        print(f"{node.name:20} {node.tau_train:+6.2f} {node.tau_hat:+6.2f} {actual:+6.2f}")

.. testoutput::

    node                  train   eval   true
    everyone              +3.04  +3.89  +3.44
    tenure < 12 months    +6.22  +6.98  +7.00
    tenure >= 12 months   +0.99  +1.80  +1.09

Every certified claim is true: the gift pays in the base as a whole
(:math:`+3.44`), pays among first-year customers (:math:`+7.00`), and
does not pay among established ones (:math:`+1.09`), and the eight
deeper claims are about subgroups of first-year customers, whose
effect is :math:`+7` throughout. So are the bounds, since there is no
real heterogeneity within either half. That had to hold with
probability at least 95%: trickle down bounds the chance that *any*
certified claim or reported bound is false by the familywise error
rate.

The policy is the right one. Mailing first-year customers earns about
:math:`0.39 \times (7 - 3) \approx 1.56` per customer in the base,
against :math:`3.42 - 3 = 0.42` for mailing everyone. And it is the
best policy available: established premium customers respond more
than established basic ones (:math:`+2.00` against :math:`+0.50`), a
difference the analysis never found, but both fall short of the cost,
so finding it would not have changed what to do.

Comparing Procedures
--------------------

The same p-values give each comparator of §5 on the full tree. Fixed
Hierarchy is trickle down with every stake set to one:

.. testcode::

    fixed = interweaved_hierarchy(
        root, alpha=0.05, delta=COST, effect_stake=1.0, heterogeneity_stake=1.0
    ).test()
    counts = {
        "Bonferroni": sum(bonferroni_rejections(claims.claims, 0.05)),
        "Hierarchical Bonferroni": sum(
            hierarchical_bonferroni_rejections(claims.claims, 0.05)
        ),
        "Fixed hierarchy": len(fixed.certified()),
        "Trickle down": len(claims.certified()),
    }
    for procedure, count in counts.items():
        print(f"{procedure:24} {count} claims")

.. testoutput::

    Bonferroni               10 claims
    Hierarchical Bonferroni  7 claims
    Fixed hierarchy          4 claims
    Trickle down             12 claims

The counts mislead, for the same reason pruning was needed: most
certified claims repeat a decision. What matters is whether each
procedure certifies the four claims the decision rests on (that
mailing everyone pays, that the first split is real, that mailing
first-year customers pays, and that mailing established ones does
not). Trickle down and Fixed Hierarchy certify all four.

Bonferroni tests each of the 43 hypotheses at
:math:`0.05 / 43 \approx 0.0012` and misses two of them: the effect
for everyone (:math:`p = 0.006`) and, crucially, the effect for
established customers (:math:`p = 0.004`). Without the second, it
cannot say that targeting beats a blanket rollout. Eight of its ten
claims are the deep, redundant subgroups of first-year customers,
whose effects are large enough to clear even its level, which is the
paper's observation that Bonferroni gives "more weight to the leaf
hypotheses" (§7.3). Hierarchical Bonferroni also misses the
established customers' claim, which it tests at 0.0031 against trickle
down's 0.0125.

Fixed Hierarchy matches trickle down on what matters here because the
root's effect hypothesis was rejected. Had it been retained, Fixed
Hierarchy would have tested everything
below the root at level zero and certified nothing, while trickle down
would still have tested the subgroups with the half of the root's level
it had kept in reserve. The simulation study in
:doc:`reproducing_the_paper` measures these trade-offs over thousands
of replications.

Checklist
---------

- Split before you look. Fit the model and the tree, and compute
  ``tau_train``, on the training split only.
- Set ``delta`` from the business case, in advance: the uplift at
  which the treatment pays for itself, in the units of the outcome.
  Make sure the outcome does not already net out the cost, or it will
  be counted twice.
- Grow deep, then prune the report branch by branch, keeping the
  splits below which something certified changes the decision. Under
  trickle down, depth costs only a factor of ``effect_stake`` at what
  would have been the leaves; ``min_samples_leaf`` keeps the tree from
  splitting into pieces too small to test.
- Test once. Retesting a pruned tree, or a different tree or
  threshold, after seeing the evaluation results voids the guarantee;
  if you must iterate, the evaluation split has become training data
  and you need a new one.
- Name nodes so the claims read as sentences, and keep the names
  unique; ``interweaved_hierarchy`` raises on duplicates.
- Report the bounds as well as the rejections. A retained hypothesis
  with a tight bound is often the more useful sentence.
