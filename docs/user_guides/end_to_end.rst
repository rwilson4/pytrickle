End-to-End: From Uplift Model to Certified Claims
=================================================

The usage guide starts from the arm summaries of a tree someone has
already fitted. This one starts from raw data. We simulate a randomized experiment, fit an uplift model on one
half of it, distill the model into a decision tree, and then use trickle down on
the other half to decide
which statements about the tree's subgroups we may make to a
decision-maker. This is the workflow of [Wil26]_, with a
scikit-learn model standing in for whatever learner you prefer.

Every code block below is executed when the documentation is built,
and the printed output is checked, so the guide cannot drift from the
library. Because the data are simulated we know the true effects, and
we finish by comparing the claims against them.

One rule organizes everything that follows:

    Anything that shapes the hypotheses (the model, the tree, the
    direction of each test) may use only the training split. The evaluation split is used once, to compute p-values.

The familywise guarantee is conditional on the training split, so it
survives any amount of exploration there [Cox75]_. It does not survive
a single peek at the evaluation split: refit the tree after seeing
which claims came out, and the guarantee is gone.

The code needs scikit-learn, which is in the ``dev`` extra but is not
a dependency of trickle-down itself.

Simulating an Experiment
------------------------

A subscription service offers a discount to a random half of its
customers and measures spend over the following quarter. The discount
wins over new customers on the basic plan, does little for established
ones, and costs money on premium customers, who would have paid full
price anyway. Averaged over everyone the effect is close to zero, which
is the situation in which a model of the *heterogeneity* matters most:
a single A/B readout would conclude that the discount does nothing.

There are eight features. Tenure and plan moderate the effect; tenure,
plan and weekly sessions predict the baseline outcome; age and four
noise columns do neither. Outcomes are noisy, with a standard
deviation of 25 against effects of a few dollars.

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

    FEATURES = [
        "tenure", "premium", "age", "sessions",
        "noise1", "noise2", "noise3", "noise4",
    ]


    def true_effect(X):
        """The effect of the discount on each customer's spend."""
        tenure, premium = X[:, 0], X[:, 1] == 1
        return np.where(premium, -3.0, np.where(tenure < 12, 5.0, 1.0))


    def simulate(rng, n):
        tenure = np.floor(rng.exponential(24.0, size=n))  # months
        premium = (rng.random(n) < 0.4).astype(float)
        age = np.round(rng.uniform(18, 75, size=n))
        sessions = rng.poisson(3.0, size=n).astype(float)
        noise = rng.standard_normal((n, 4))
        X = np.column_stack([tenure, premium, age, sessions, noise])

        baseline = 40 + 0.5 * np.minimum(tenure, 36) + 2 * sessions + 10 * premium
        y0 = baseline + rng.normal(0, 25, size=n)
        z = rng.random(n) < 0.5  # the randomized offer
        y = np.where(z, y0 + true_effect(X), y0)
        return X, z, y


    rng = np.random.default_rng(4)
    X, z, y = simulate(rng, 20_000)

    tau = true_effect(X)
    print(f"average effect: {tau.mean():+.2f}")
    for label, mask in [
        ("basic, tenure < 12", (X[:, 1] == 0) & (X[:, 0] < 12)),
        ("basic, tenure >= 12", (X[:, 1] == 0) & (X[:, 0] >= 12)),
        ("premium", X[:, 1] == 1),
    ]:
        print(f"{label:20} {mask.mean():4.0%} of customers, effect {tau[mask].mean():+.1f}")

.. testoutput::

    average effect: +0.34
    basic, tenure < 12    24% of customers, effect +5.0
    basic, tenure >= 12   36% of customers, effect +1.0
    premium               40% of customers, effect -3.0

So there are three true statements of the sort we hope to certify: the
discount helps basic customers, harms premium ones, and helps new basic
customers more than established ones. Whether the data can certify
them is another matter; with this much noise, some may be out of reach.

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

    10076 training, 9924 evaluation

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

    depth 6, 16 leaves
    |--- premium <= 0.5
    |   |--- tenure <= 9.5
    |   |   |--- truncated branch of depth 3
    |   |--- tenure >  9.5
    |   |   |--- truncated branch of depth 5
    |--- premium >  0.5
    |   |--- tenure <= 11.5
    |   |   |--- truncated branch of depth 3
    |   |--- tenure >  11.5
    |   |   |--- truncated branch of depth 4

``export_text`` prints only the first two layers here. The top of the
tree recovers the plan split and, on the basic side, a tenure split close to the true
one at twelve months. On the premium side it has also split on tenure,
though the true effect is :math:`-3` throughout; the gap between the
predictions is noise the model mistook for structure. Below that, the
tree carries on splitting (on sessions and tenure, in the next layer)
wherever that reduces the error in the predicted effects, whether or
not the effect really varies there. Nothing on the training
split can tell us which of these splits are real. The evaluation split
can.

Testing on the Evaluation Split
-------------------------------

Now, and only now, touch the evaluation split:

.. testcode::

    root = to_uplift_tree(
        tree, (X_train, z_train, y_train), (X_eval, z_eval, y_eval)
    )
    claims = interweaved_hierarchy(root, alpha=0.05).test()
    print(f"{len(claims.claims)} hypotheses, {len(claims.certified())} certified:")
    for claim in claims.certified():
        print(f"  - {claim.sentence()}")

.. testoutput::

    46 hypotheses, 4 certified:
      - the effect is larger for basic plan than for premium plan
      - the effect is positive for basic plan
      - the effect is negative for premium plan
      - the effect is negative for premium plan, tenure < 12 months

``print(claims)`` would tabulate all of the hypotheses. Here are the
six at the top of the tree, at its root and its first split:

.. testcode::

    for claim in claims.claims:
        if claim.depth <= 1:
            verdict = "rejected" if claim.rejected else "retained"
            print(
                f"{claim.node:14} {claim.kind:14} {claim.alternative:8} "
                f"p = {claim.pvalue:.4f}  level = {claim.level:.4f}  {verdict}"
            )

.. testoutput::

    everyone       effect         greater  p = 0.5985  level = 0.0250  retained
    everyone       heterogeneity  less     p = 0.0000  level = 0.0125  rejected
    basic plan     effect         greater  p = 0.0051  level = 0.0063  rejected
    basic plan     heterogeneity  less     p = 0.1343  level = 0.0063  retained
    premium plan   effect         less     p = 0.0001  level = 0.0063  rejected
    premium plan   heterogeneity  less     p = 0.8879  level = 0.0063  retained

Read them top to bottom, as the procedure does.

- **The overall effect is retained.** With :math:`p = 0.60` there is
  no evidence the discount helps on average, which is what an
  ordinary A/B readout would report. Under trickle down that is not
  the end: the root staked half its level and passes the other half
  down.
- **The plan split is certified**, in the direction the training split
  chose: the effect is larger for basic than for premium customers.
  That rejection passes its whole level to the two plans, and both
  effects are then certified: the discount helps basic customers and
  harms premium ones.
- **Neither plan's tenure split is certified.** On the premium side
  that is the right answer, since the split is spurious (:math:`p =
  0.89`). On the basic side it is a miss, a Type II error: the split
  is real, but :math:`p = 0.13` falls well short of its level.
- **Testing continues below both retentions**, at ever smaller levels,
  and one claim further down is certified: the discount harms premium
  customers of under a year's tenure. That test ran at 0.0016. Had we
  stopped the tree at depth two, the same node would have been a leaf
  staking its whole level, tested at 0.0031. That factor of two, at
  each of the depth-two tree's four leaves, is the whole price of
  growing the tree, and here it cost nothing.

No claim is certified about any node more than two splits down, which
is how the procedure tells us that the tree's deeper splits are not
supported by the evaluation split.

Pruning
-------

Pruning turns the 46 verdicts into a tree a decision-maker can read.
The power available deep in the tree falls exponentially, so "at some
point, we will retain *all* hypotheses below a certain depth. We may
prune these nodes, leaving a shallower tree emphasizing the certified
claims" (§5.2).

Prune branch by branch rather than layer by layer. Evidence runs out
at different depths in different branches: here nothing is certified
below the basic plan's first split, while the premium side has a
certified claim one split further down. So we keep a node's split if
anything below it was certified, whether the split itself (its
heterogeneity hypothesis) or a claim about any of its descendants, and
cut it otherwise:

.. testcode::

    import dataclasses


    def prune(node, claims):
        """Cut every split below which nothing was certified."""
        certified = {(claim.node, claim.kind) for claim in claims.certified()}

        def anything_certified(node):
            return (node.name, "effect") in certified or any(
                anything_certified(child) for child in node.children
            )

        def cut(node):
            keep = (node.name, "heterogeneity") in certified or any(
                anything_certified(child) for child in node.children
            )
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

    everyone: n = 9924, estimated effect -0.13
      bounded    effect > -1.15
      certified  [premium plan] - [basic plan] < 0
        basic plan: n = 5953, estimated effect +1.70
          certified  effect > 0
        premium plan: n = 3971, estimated effect -3.19
          certified  effect < 0
          bounded    [tenure >= 12 months] - [tenure < 12 months] < 6.11
            tenure < 12 months: n = 1577, estimated effect -4.52
              certified  effect < 0
            tenure >= 12 months: n = 2394, estimated effect -2.52
              bounded    effect < 0.567

The estimated effects are the evaluation split's differences in arm
means. Because the tree was chosen on the other split, they are
unbiased for each node, but they are point estimates and carry no
simultaneous guarantee; the certified and bounded sentences do.

Pruning is legitimate because it only chooses which claims to report,
and the familywise guarantee covers every subset of them.
What is *not* legitimate is to test the pruned tree again. Its new
leaves would stake their whole level instead of half of it, and
choosing which nodes get that boost after seeing the evaluation split
is exactly the kind of peek the guarantee does not survive. Prune the
report, not the hierarchy.

Findings
--------

Here is what the analysis lets us tell the decision-maker, with the
familywise error rate held at 5% across every statement together.

**A blanket rollout is not supported.** We cannot show that the
discount raises spend on average (:math:`p = 0.60`). We can bound the
downside: the average effect is above :math:`-1.15` per customer.

**The discount helps basic-plan customers and hurts premium ones.**
Both signs are certified, and so is the difference between the two
plans. This is the finding that matters, and it is invisible in the
average: the two plans' effects nearly cancel.

**Premium customers in their first year are harmed.** The harm among
premium customers under twelve months' tenure is certified on its
own. For longer-tenured premium customers we cannot certify harm, but
the effect is below :math:`+0.57`, so there is no meaningful benefit
to be had there either. Nothing in the data argues for discounting any
premium customer.

**The recommendation follows**: offer the discount to basic-plan
customers and withhold it from premium ones.

What the analysis does *not* support matters as much:

- **Magnitudes.** A certified effect claim establishes a sign, not a
  size. A rejected hypothesis may claim only its threshold, because
  its level has already passed to its children (§6), so "the effect
  is positive for basic plan" is all we may say there; the bounds of
  §6 come only with retained hypotheses. If the discount
  costs money, set ``delta`` to the break-even uplift and the claims
  become "the effect exceeds the break-even point".
- **Tenure within the basic plan.** The model suggested new basic
  customers respond more strongly, and the tree split on tenure, but
  the evaluation split did not certify the split (:math:`p = 0.13`),
  and pruning removed it. We should not tell anyone to target new
  basic customers in particular.
- **Tenure within the premium plan.** The premium split survives the
  pruning only because one of its halves carries a certified claim.
  Its heterogeneity hypothesis was retained (:math:`p = 0.89`), and
  its bound, that the longer-tenured half's effect exceeds the newer
  half's by less than 6.11, rules out almost nothing. We cannot say
  whether newer and longer-tenured premium customers respond
  differently, only that the newer ones are harmed.
- **Other splits.** The full tree carried 36 more hypotheses, about
  nodes the pruning removed, and none was certified. The model's finer
  structure, splits on sessions and on tenure further down, is
  unsupported.

One caveat applies to all of it. These conclusions are conditional on
one random split of the data into training and evaluation halves,
which "may lead two analysts to draw different conclusions about
treatment effects" (§8). The guarantee covers the error rate of the procedure,
not the variability across splits.

Checking Against the Truth
--------------------------

Because we simulated the data, we can see what each node's effect
really is. ``node_names`` lines the tree's nodes up with
``decision_path``, so the true effect in each node is a mask away:

.. testcode::

    in_eval = tree.decision_path(X_eval).toarray().astype(bool)
    column = {name: i for i, name in enumerate(node_names(tree))}
    truth = true_effect(X_eval)
    print(f"{'node':34} {'train':>6} {'eval':>6} {'true':>6}")
    for node in pruned.walk():
        actual = truth[in_eval[:, column[node.name]]].mean()
        print(f"{node.name:34} {node.tau_train:+6.2f} {node.tau_hat:+6.2f} {actual:+6.2f}")

.. testoutput::

    node                                train   eval   true
    everyone                            +0.84  -0.13  +0.35
    basic plan                          +3.81  +1.70  +2.59
    premium plan                        -3.44  -3.19  -3.00
    premium plan, tenure < 12 months    -1.51  -4.52  -3.00
    premium plan, tenure >= 12 months   -4.82  -2.52  -3.00

All four certified claims are true, and so are the bounds: the overall
effect (:math:`+0.35`) is above :math:`-1.15`, and that of
longer-tenured premium customers (:math:`-3.00`) is below
:math:`+0.57`. They had to be, with probability at least 95%: trickle
down bounds the chance that *any* certified claim or reported bound is
false by the familywise error rate. The recommendation is right too.
The true effects are :math:`+2.59` on the basic plan and :math:`-3.00`
on premium, so targeting basic customers gains about :math:`2.59
\times 0.60 \approx 1.55` per customer in the population, against
:math:`0.35` for a blanket rollout.

The miss is the basic plan's tenure split. New basic customers really
do respond more (:math:`+5.00` against :math:`+1.35` beyond ten
months), but the evaluation split estimated a contrast of :math:`1.54`
where the truth is :math:`3.65`, and the test had a level of 0.006.
That is a Type II error, the price of controlling the familywise rate
at a sample size this noisy.

The premium leaves show the winner's curse. The training split put
them at :math:`-1.51` and :math:`-4.82`, because the tree chose the
split that made them look most different; on the evaluation split the
order reverses. Estimates from the split that chose the tree are
biased by the search that produced it, which is why the tree is fit on
one split and tested on another [AI16]_. Here the bias was harmless,
because ``tau_train`` only chose directions, and the evaluation split
refused the spurious split.

Comparing Procedures
--------------------

The same p-values give each comparator of §5 on the full tree. Fixed
Hierarchy is trickle down with every stake set to one:

.. testcode::

    fixed = interweaved_hierarchy(
        root, alpha=0.05, effect_stake=1.0, heterogeneity_stake=1.0
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

    Bonferroni               3 claims
    Hierarchical Bonferroni  3 claims
    Fixed hierarchy          0 claims
    Trickle down             4 claims

Fixed Hierarchy certifies nothing: it stakes the root's whole level on
the overall effect, and once that is retained everything below is
tested at level zero. A near-null average effect concealing opposite
effects in subgroups is exactly the case it cannot handle, and the one
trickle down was designed for. Bonferroni tests each of the 46
hypotheses at :math:`0.05 / 46 \approx 0.0011`. It certifies the three
claims with the smallest p-values and misses the basic-plan effect
(:math:`p = 0.0051`), which trickle down certifies at 0.0063 because
the plan split's rejection passed its whole level down. Unlike trickle
down, Bonferroni would gain from a shallower tree: at depth two it
tests ten hypotheses at 0.005, and still just misses that claim.

This is one draw from one data-generating process. Trickle down does
not dominate Bonferroni on every dataset, since it can spend a lot of
level on a retained root; the simulation study in
:doc:`reproducing_the_paper` measures the procedures' power over
thousands of replications.

Checklist
---------

- Split before you look. Fit the model and the tree, and compute
  ``tau_train``, on the training split only.
- Grow deep, then prune the report branch by branch. Under trickle
  down, depth costs only a factor of ``effect_stake`` at what would
  have been the leaves; ``min_samples_leaf`` keeps the tree from
  splitting into pieces too small to test.
- Test once. Retesting a pruned tree, or a different tree, after seeing
  the evaluation results voids the guarantee; if you must iterate, the
  evaluation split has become training data and you need a new one.
- Name nodes so the claims read as sentences, and keep the names
  unique; ``interweaved_hierarchy`` raises on duplicates.
- Report the bounds as well as the rejections. A retained hypothesis
  with a tight bound is often the more useful sentence.
