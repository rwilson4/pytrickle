"""Tests for the interweaved hypothesis architecture."""

import pytest

from pytrickle.uplift import (
    ArmSummary,
    bonferroni_rejections,
    ContrastTest,
    hierarchical_bonferroni_rejections,
    interweaved_hierarchy,
    TTest,
    UpliftClaims,
    UpliftNode,
)

ALPHA = 0.05
VARIANCE = 9.0


def arm(mean: float, size: int = 1000) -> ArmSummary:
    return ArmSummary(mean=mean, variance=VARIANCE, size=size)


def opposite_effects_tree() -> UpliftNode:
    """Treatment helps the left subgroup and harms the right one.

    The overall effect is exactly zero because the two cancel. This is
    the case the paper is built around: a procedure that stops at the
    first retention learns nothing, while one that keeps testing at
    reduced level certifies both subgroup claims and the difference.
    """
    left = UpliftNode(name="left", treated=arm(2.0), control=arm(0.0), tau_train=2.0)
    right = UpliftNode(
        name="right", treated=arm(-2.0), control=arm(0.0), tau_train=-2.0
    )
    return UpliftNode(
        name="root",
        treated=arm(0.0, size=2000),
        control=arm(0.0, size=2000),
        tau_train=0.0,
        children=[left, right],
    )


class TestAdaptiveDifferenceDirection:
    """Difference hypotheses are oriented from the training split too."""

    @staticmethod
    def test_direction_follows_the_training_contrast() -> None:
        """The contrast is tau_right - tau_left, so its training sign picks
        the side.

        In the fixture the left subgroup is helped (+2) and the right
        one harmed (-2), so tau_right - tau_left is negative on the
        training split and the one-sided test should be "less".
        Rejecting then licenses "the effect is smaller on the right
        than on the left" rather than merely "they differ".
        """
        claims = interweaved_hierarchy(opposite_effects_tree(), ALPHA)
        difference = next(c for c in claims.claims if c.kind == "difference")

        assert difference.alternative == "less"
        # The claim records the direction; narrow to read it back off
        # the test object, which is where it actually takes effect.
        assert isinstance(difference.hypothesis, ContrastTest)
        assert difference.hypothesis.alternative == "less"

    @staticmethod
    def test_direction_flips_with_the_training_split() -> None:
        """Same evaluation data, mirrored training estimates.

        Only tau_train differs between the two trees, and tau_train is
        not used by any test statistic. If the selected side did not
        flip, the direction would be coming from the evaluation split
        --- which would invalidate the error rate, not merely cost
        power.
        """
        tree = opposite_effects_tree()
        for child in tree.children:
            child.tau_train = -child.tau_train

        claims = interweaved_hierarchy(tree, ALPHA)
        difference = next(c for c in claims.claims if c.kind == "difference")

        assert difference.alternative == "greater"

    @staticmethod
    def test_non_adaptive_keeps_the_two_sided_test() -> None:
        """The comparator certifies direction nowhere."""
        claims = interweaved_hierarchy(opposite_effects_tree(), ALPHA, adaptive=False)
        difference = next(c for c in claims.claims if c.kind == "difference")

        assert difference.alternative == "two-sided"

    @staticmethod
    def test_one_sided_p_value_is_half_the_two_sided_one() -> None:
        """Where the guess is right, the test is exactly twice as cheap.

        This is the whole gain, and it is exact rather than asymptotic:
        for a symmetric reference distribution the one-sided tail is
        half the two-sided one whenever the observed contrast falls on
        the chosen side.
        """
        adaptive = interweaved_hierarchy(opposite_effects_tree(), ALPHA)
        two_sided = interweaved_hierarchy(
            opposite_effects_tree(), ALPHA, adaptive=False
        )

        one = next(
            c for c in adaptive.claims if c.kind == "difference"
        ).hypothesis.pvalue()
        both = next(
            c for c in two_sided.claims if c.kind == "difference"
        ).hypothesis.pvalue()

        assert one == pytest.approx(both / 2.0)


class TestNodeNames:
    """Names are the join key, so they have to be unique."""

    @staticmethod
    def test_duplicate_node_names_are_rejected() -> None:
        """A repeated name silently reports the wrong node's estimate.

        This is not hypothetical: the fitted tree named its children
        for the split that produced them, so a tree that chose the
        same split in both branches had two nodes called
        ``x4<=0.5``, and the worked example's table reported one
        leaf's effect under the other leaf's hypothesis.
        """
        root = opposite_effects_tree()
        root.children[1].name = root.children[0].name

        with pytest.raises(ValueError, match="unique"):
            interweaved_hierarchy(root, ALPHA)


class TestStructure:
    """The shape of the hierarchy."""

    @staticmethod
    def test_magnitude_and_difference_alternate() -> None:
        """M_root -> D_root -> {M_left, M_right}."""
        claims = interweaved_hierarchy(opposite_effects_tree(), ALPHA)

        root_magnitude = claims.hierarchy.root
        assert root_magnitude.name == "M[root]"

        assert len(root_magnitude.children) == 1
        difference = root_magnitude.children[0]
        assert difference.name == "D[root]"

        assert [child.name for child in difference.children] == [
            "M[left]",
            "M[right]",
        ]
        assert difference.children[0].children == []

    @staticmethod
    def test_leaves_stake_everything() -> None:
        """Nothing below a leaf, so nothing to hold in reserve."""
        claims = interweaved_hierarchy(
            opposite_effects_tree(), ALPHA, magnitude_stake=0.5
        )
        by_name = {c.hypothesis.name: c.hypothesis for c in claims.claims}

        assert by_name["M[root]"].stake == 0.5
        assert by_name["M[left]"].stake == 1.0
        assert by_name["M[right]"].stake == 1.0

    @staticmethod
    def test_size_weighting() -> None:
        """A larger child takes a larger share of the level."""
        root = opposite_effects_tree()
        root.children[0].treated = arm(2.0, size=3000)

        claims = interweaved_hierarchy(root, ALPHA, size_weighted=True)
        claims.test()
        by_name = {c.hypothesis.name: c.hypothesis for c in claims.claims}

        assert by_name["M[left]"].weight > by_name["M[right]"].weight

    @staticmethod
    def test_odd_child_count_is_rejected() -> None:
        leaf = UpliftNode("a", arm(0.0), arm(0.0), 0.0)
        with pytest.raises(ValueError, match="zero or two"):
            UpliftNode("root", arm(0.0), arm(0.0), 0.0, children=[leaf])


class TestDirectionSelection:
    """Choosing which one-sided hypothesis to test."""

    @staticmethod
    def test_adaptive_follows_the_training_estimate() -> None:
        claims = interweaved_hierarchy(opposite_effects_tree(), ALPHA, adaptive=True)
        by_node = {(c.node, c.kind): c for c in claims.claims}

        assert by_node[("left", "magnitude")].alternative == "greater"
        assert by_node[("right", "magnitude")].alternative == "less"

    @staticmethod
    def test_non_adaptive_tests_everything_two_sided() -> None:
        """The comparator forgoes the side, in both families.

        Not "always test for benefit": that is a fixed guess rather
        than the absence of one, and against a population containing
        harm it loses for a reason that has nothing to do with
        adaptivity.
        """
        claims = interweaved_hierarchy(opposite_effects_tree(), ALPHA, adaptive=False)

        assert all(c.alternative == "two-sided" for c in claims.claims)


class TestShowcase:
    """The claim set the paper is arguing for."""

    @staticmethod
    def test_certifies_opposite_signed_effects_under_a_null_average() -> None:
        """Three claims, simultaneously, at FWER 0.05.

        The root magnitude hypothesis is retained --- averaged over
        everyone the treatment does nothing --- and the procedure
        nonetheless goes on to certify that the effect is larger on
        the left than on the right, positive on the left, and
        negative on the right.
        """
        claims = interweaved_hierarchy(opposite_effects_tree(), ALPHA).test()
        verdicts = {(c.node, c.kind): c.rejected for c in claims.claims}

        assert verdicts[("root", "magnitude")] is False
        assert verdicts[("root", "difference")] is True
        assert verdicts[("left", "magnitude")] is True
        assert verdicts[("right", "magnitude")] is True

        sentences = {c.sentence() for c in claims.certified()}
        assert sentences == {
            # Not "differs across the split at root": the one-sided
            # difference test names which child responds more.
            "the effect is larger for left than for right",
            "the effect is positive for left",
            "the effect is negative for right",
        }

        # Pin the trickle-down arithmetic, not just the verdicts. The
        # effects here are large enough that the verdicts alone would
        # survive a broken level computation.
        #
        #   M[root]  0.05 available, staked 0.5      -> 0.0250
        #   retained, so D[root] inherits (1 - 0.5) * 0.05 = 0.025,
        #   staked 0.5                               -> 0.0125
        #   rejected, so each leaf inherits 0.5 * 0.025 = 0.0125,
        #   staked 1.0                               -> 0.0125
        levels = {(c.node, c.kind): c.level for c in claims.claims}
        assert levels[("root", "magnitude")] == pytest.approx(0.0250)
        assert levels[("root", "difference")] == pytest.approx(0.0125)
        assert levels[("left", "magnitude")] == pytest.approx(0.0125)
        assert levels[("right", "magnitude")] == pytest.approx(0.0125)

    @staticmethod
    def test_fixed_hierarchy_would_have_learned_nothing() -> None:
        """Stakes of one stop dead at the retained root.

        This is the comparison that motivates the stake: the Fixed
        Hierarchy Procedure is the special case v = 1, and here it
        certifies nothing at all.
        """
        claims = interweaved_hierarchy(
            opposite_effects_tree(),
            ALPHA,
            magnitude_stake=1.0,
            difference_stake=1.0,
        ).test()

        assert claims.certified() == []

    @staticmethod
    def test_non_adaptive_finds_the_harmed_subgroup_but_cannot_name_it() -> None:
        """Two-sided reaches the harmed node; it just says less there.

        This is the whole of what direction selection buys on the
        magnitude side: the same node is reached, but the sentence is
        "differs from 0" rather than "is negative". Naming the side
        off a two-sided rejection is a Type III error argument, and
        this library does not make it.
        """
        claims = interweaved_hierarchy(
            opposite_effects_tree(), ALPHA, adaptive=False
        ).test()
        sentences = {c.sentence() for c in claims.certified()}

        assert "the effect for right differs from 0" in sentences
        assert not any("negative" in sentence for sentence in sentences)
        assert not any("positive" in sentence for sentence in sentences)

    @staticmethod
    def test_threshold_withholds_a_claim_that_does_not_clear_it() -> None:
        """An effect of 2.0 does not clear a break-even point of 5.0."""
        claims = interweaved_hierarchy(opposite_effects_tree(), ALPHA, delta=5.0).test()
        by_node = {(c.node, c.kind): c for c in claims.claims}

        assert by_node[("left", "magnitude")].alternative == "less"
        assert by_node[("left", "magnitude")].rejected is True
        assert by_node[("left", "magnitude")].sentence() == (
            "the effect for left falls short of 5"
        )

    @staticmethod
    def test_str_reports_the_certified_claims() -> None:
        claims = interweaved_hierarchy(opposite_effects_tree(), ALPHA).test()
        report = str(claims)

        assert "Certified at FWER 0.05" in report
        assert "the effect is negative for right" in report
        assert "magnitude" in report and "difference" in report


class TestNoEffect:
    """Nothing to find means nothing certified."""

    @staticmethod
    def test_global_null_certifies_nothing() -> None:
        left = UpliftNode("left", arm(0.0), arm(0.0), 0.01)
        right = UpliftNode("right", arm(0.0), arm(0.0), -0.01)
        root = UpliftNode(
            "root",
            arm(0.0, size=2000),
            arm(0.0, size=2000),
            0.0,
            children=[left, right],
        )

        claims = interweaved_hierarchy(root, ALPHA).test()

        assert claims.certified() == []


class TestComparators:
    """The procedures the paper measures itself against."""

    @staticmethod
    def test_bonferroni_splits_alpha_evenly() -> None:
        claims = interweaved_hierarchy(opposite_effects_tree(), ALPHA).test()
        rejected = bonferroni_rejections(claims.claims, ALPHA)

        # Four hypotheses, so each is tested at 0.0125.
        assert len(rejected) == 4
        for claim, flag in zip(claims.claims, rejected, strict=True):
            assert flag == (claim.pvalue <= ALPHA / 4)

    @staticmethod
    def test_bonferroni_also_misses_the_root() -> None:
        """A null root is retained by every valid procedure."""
        claims = interweaved_hierarchy(opposite_effects_tree(), ALPHA).test()
        rejected = bonferroni_rejections(claims.claims, ALPHA)
        by_index = {
            c.hypothesis.name: r for c, r in zip(claims.claims, rejected, strict=True)
        }

        assert by_index["M[root]"] is False

    @staticmethod
    def test_hierarchical_bonferroni_levels_decay_with_depth() -> None:
        """Depth k is tested at 2^-k alpha / n_k; the root at alpha / 2."""
        root = opposite_effects_tree()
        claims = interweaved_hierarchy(root, ALPHA).test()
        rejected = hierarchical_bonferroni_rejections(claims.claims, ALPHA)
        by_name = {
            c.hypothesis.name: r for c, r in zip(claims.claims, rejected, strict=True)
        }

        # M[root] at depth 1 sees alpha/2; the leaf magnitudes sit at
        # depth 3 with two hypotheses there, so alpha/16 each.
        assert by_name["M[left]"] == (
            by_name["M[left]"] and claims.claims[2].pvalue <= ALPHA / 16
        )
        assert len(rejected) == 4

    @staticmethod
    def test_empty_claim_list_is_handled() -> None:
        assert bonferroni_rejections([], ALPHA) == []
        assert hierarchical_bonferroni_rejections([], ALPHA) == []

    @staticmethod
    def test_hierarchical_bonferroni_does_not_depend_on_claim_order() -> None:
        """The root is found, not assumed to be the first claim.

        The walk is what assigns every claim its level, so a procedure
        that starts from whichever claim happens to be first mis-levels
        the whole family when handed the same claims in another order
        --- or, as here, fails outright on the missing key.
        """
        claims = interweaved_hierarchy(opposite_effects_tree(), ALPHA).test()
        in_order = dict(
            zip(
                (c.hypothesis.name for c in claims.claims),
                hierarchical_bonferroni_rejections(claims.claims, ALPHA),
                strict=True,
            )
        )

        shuffled = list(reversed(claims.claims))
        reordered = dict(
            zip(
                (c.hypothesis.name for c in shuffled),
                hierarchical_bonferroni_rejections(shuffled, ALPHA),
                strict=True,
            )
        )

        assert reordered == in_order

    @staticmethod
    def test_hierarchical_bonferroni_rejects_two_unrelated_families() -> None:
        """Claims from two hierarchies are not one tree, so n_k is wrong."""
        first = interweaved_hierarchy(opposite_effects_tree(), ALPHA).test()
        second = interweaved_hierarchy(opposite_effects_tree(), ALPHA).test()

        with pytest.raises(ValueError, match="single tree"):
            hierarchical_bonferroni_rejections(first.claims + second.claims, ALPHA)


class TestConfidenceBounds:
    """Inverting the test, so a retention still says something."""

    @staticmethod
    def test_bound_is_exactly_the_thresholds_the_test_rejects() -> None:
        """The defining property of inversion, checked either side.

        A test of H_0: tau <= delta at a given level rejects exactly
        those delta below the lower endpoint. If the two ever disagree
        the bound is not the test's inversion and means nothing.
        """
        level = 0.0123
        tree = opposite_effects_tree()
        claims = interweaved_hierarchy(tree, ALPHA)
        hypothesis = next(
            c for c in claims.claims if c.node == "left" and c.kind == "magnitude"
        ).hypothesis
        assert isinstance(hypothesis, TTest)
        low, high = hypothesis.confidence_interval(level)

        assert high == float("inf")  # "greater" leaves the top open

        # Probe immediately either side of the endpoint. A loose
        # offset here lets a wrong critical value slip through: the
        # one- and two-sided values differ by only ~0.03 on this
        # fixture, so anything coarser steps clean over the error.
        for delta, rejects in ((low - 1e-9, True), (low + 1e-9, False)):
            probe = TTest(
                mean1=hypothesis.mean1,
                mean2=hypothesis.mean2,
                var1=hypothesis.var1,
                var2=hypothesis.var2,
                n1=hypothesis.n1,
                n2=hypothesis.n2,
                delta=delta,
                alternative="greater",
            )
            assert (probe.pvalue() <= level) is rejects

    @staticmethod
    def test_a_retained_claim_still_bounds_the_effect() -> None:
        """The whole point: a retention is not silence.

        The two subgroups cancel exactly, so the root effect is zero
        and the root magnitude hypothesis is retained. The inverted
        test still bounds the root effect, and truncating at the
        threshold does not touch it: a retention never recycled its
        level, so the whole endpoint is paid for.
        """
        claims = interweaved_hierarchy(opposite_effects_tree(), ALPHA).test()

        root = next(
            c for c in claims.claims if c.node == "root" and c.kind == "magnitude"
        )
        assert not root.rejected

        # tau_train is 0.0 at the root, so select_alternative orients
        # this one toward harm and the endpoint is an upper bound.
        assert root.alternative == "less"
        low, high = root.interval
        assert low == -float("inf")
        assert high > root.delta
        assert root.reportable_bound == (low, high)
        assert "is below" in root.bound_sentence()

    @staticmethod
    def test_a_tighter_level_gives_a_weaker_bound() -> None:
        """Less level, less to say. Monotone, and the deep nodes pay."""
        hypothesis = next(
            c
            for c in interweaved_hierarchy(opposite_effects_tree(), ALPHA).claims
            if c.node == "left" and c.kind == "magnitude"
        ).hypothesis

        generous, _ = hypothesis.confidence_interval(0.05)
        stingy, _ = hypothesis.confidence_interval(0.001)
        assert stingy < generous

    @staticmethod
    def test_two_sided_inversion_is_the_ordinary_interval() -> None:
        """A sanity anchor against the textbook formula."""
        claim = next(
            c
            for c in interweaved_hierarchy(
                opposite_effects_tree(), ALPHA, adaptive=False
            ).claims
            if c.kind == "difference"
        )
        assert claim.alternative == "two-sided"
        hypothesis = claim.hypothesis
        assert isinstance(hypothesis, ContrastTest)
        low, high = hypothesis.confidence_interval(0.05)

        estimate, _, _ = hypothesis._moments()
        assert (low + high) / 2.0 == pytest.approx(estimate)
        assert low < estimate < high


class TestReportableBounds:
    """Only the weaker side of the threshold comes free.

    A node that rejects hands its level to its children. Any
    threshold past the one it cleared is a fresh claim it has nothing
    left to pay for, so the endpoint it may report is the threshold
    itself. A node that retained kept its level and reports the whole
    endpoint. Without the truncation the collection of bounds is not
    simultaneously valid, which is the one thing the procedure
    promises.
    """

    @staticmethod
    def test_a_rejecting_node_may_claim_only_its_threshold() -> None:
        """Both directions, since the truncation is a min and a max."""
        claims = interweaved_hierarchy(opposite_effects_tree(), ALPHA).test()

        benefit = next(
            c for c in claims.claims if c.node == "left" and c.kind == "magnitude"
        )
        assert benefit.rejected and benefit.alternative == "greater"
        assert benefit.interval[0] > benefit.delta  # the raw inversion
        assert benefit.reportable_bound == (benefit.delta, float("inf"))

        harm = next(
            c for c in claims.claims if c.node == "right" and c.kind == "magnitude"
        )
        assert harm.rejected and harm.alternative == "less"
        assert harm.interval[1] < harm.delta
        assert harm.reportable_bound == (-float("inf"), harm.delta)

    @staticmethod
    def test_truncation_is_at_the_threshold_not_at_zero() -> None:
        """A nonzero delta is the whole point of delta.

        Truncating at 0.0 passes every other test in this class,
        because the paper's own example runs delta = 0.
        """
        delta = 0.5
        claims = interweaved_hierarchy(
            opposite_effects_tree(), ALPHA, delta=delta
        ).test()

        benefit = next(
            c for c in claims.claims if c.node == "left" and c.kind == "magnitude"
        )
        assert benefit.rejected and benefit.delta == delta
        assert benefit.reportable_bound == (delta, float("inf"))
        assert f"{delta:.3g}" in benefit.bound_sentence()

    @staticmethod
    def test_a_two_sided_claim_reports_no_bound() -> None:
        """Its enlarged family is not nested, so nothing is free.

        The node holds a true threshold whatever the effect is, so
        rejecting never releases the level and no truncation rescues
        it. The claim says so rather than quoting an interval it
        cannot pay for.
        """
        claim = next(
            c
            for c in interweaved_hierarchy(
                opposite_effects_tree(), ALPHA, adaptive=False
            )
            .test()
            .claims
            if c.kind == "difference"
        )
        assert claim.alternative == "two-sided"
        assert claim.reportable_bound == (-float("inf"), float("inf"))
        assert "no bound is free" in claim.bound_sentence()


class TestZeroLevelClaims:
    """A node can be handed no level at all and must still report."""

    @staticmethod
    def _global_null_at_full_stake() -> UpliftClaims:
        """A null tree where the root keeps the whole level for itself.

        With ``magnitude_stake`` one, a retention at the root passes
        ``(1 - stake) * level``, which is exactly zero, to everything
        below it. Those nodes are then tested at level zero. The
        default stakes of a half never reach this; the paper's fixed
        hierarchy does.
        """
        left = UpliftNode("left", arm(0.0), arm(0.0), 0.01)
        right = UpliftNode("right", arm(0.0), arm(0.0), -0.01)
        root = UpliftNode(
            "root",
            arm(0.0, size=2000),
            arm(0.0, size=2000),
            0.0,
            children=[left, right],
        )
        return interweaved_hierarchy(
            root, ALPHA, magnitude_stake=1.0, difference_stake=1.0
        ).test()

    def test_a_zero_level_claim_bounds_nothing(self) -> None:
        """Inverting a test that cannot reject excludes nothing.

        The whole line is the honest confidence set, not an error
        condition: the node spent nothing, so it bought nothing.
        """
        claims = self._global_null_at_full_stake()

        zero = [claim for claim in claims.claims if claim.level == 0.0]

        assert zero, "expected the full-stake retention to strand some node"
        for claim in zero:
            assert not claim.rejected
            assert claim.interval == (-float("inf"), float("inf"))
            assert claim.reportable_bound == (-float("inf"), float("inf"))
            assert "no bound is free" in claim.bound_sentence()

    def test_the_report_renders_with_a_zero_level_claim(self) -> None:
        """``__str__`` asks every retained claim for its bound.

        A zero-level claim is always retained, so a report that does
        not handle it raises from deep inside the test inversion
        instead of printing.
        """
        report = str(self._global_null_at_full_stake())

        assert "no bound is free for the level-zero test" in report


class TestArmSummaryFromSamples:
    """Building an arm from observations instead of from moments."""

    @staticmethod
    def test_uses_the_unbiased_variance() -> None:
        summary = ArmSummary.from_samples([1.0, 2.0, 3.0, 4.0])

        assert summary.mean == pytest.approx(2.5)
        assert summary.size == 4
        # Sum of squared deviations is 5.0, so ddof=1 gives 5/3 and the
        # ddof=0 default would give 5/4. The distinction is the point.
        assert summary.variance == pytest.approx(5.0 / 3.0)
        assert summary.variance != pytest.approx(5.0 / 4.0)

    @staticmethod
    def test_matches_a_hand_built_summary() -> None:
        built = ArmSummary.from_samples([0.0, 0.0, 6.0])

        assert built == ArmSummary(mean=2.0, variance=12.0, size=3)

    @staticmethod
    def test_one_observation_has_no_sample_variance() -> None:
        with pytest.raises(ValueError, match="at least two observations"):
            ArmSummary.from_samples([1.0])

    @staticmethod
    def test_a_nan_is_refused_rather_than_propagated() -> None:
        """Counting elements is not counting observations.

        A size check passes a slice that is long enough but not
        finite. ``np.var`` then returns NaN, the Welch statistic
        returns NaN, and ``NaN <= level`` is false, so the node
        retains without anyone being told why.
        """
        with pytest.raises(ValueError, match="must all be finite"):
            ArmSummary.from_samples([1.0, 2.0, float("nan")])

    @staticmethod
    def test_an_infinity_is_refused_too() -> None:
        with pytest.raises(ValueError, match="must all be finite"):
            ArmSummary.from_samples([1.0, float("inf")])


class TestClaimDepth:
    """Claims know the decision-tree depth of the node they describe."""

    @staticmethod
    def test_root_is_zero_and_children_are_one() -> None:
        claims = interweaved_hierarchy(opposite_effects_tree(), ALPHA).test()
        by_name = {c.hypothesis.name: c.depth for c in claims.claims}

        assert by_name["M[root]"] == 0
        assert by_name["D[root]"] == 0
        assert by_name["M[left]"] == 1
        assert by_name["M[right]"] == 1

    @staticmethod
    def test_is_not_the_hypothesis_tree_depth() -> None:
        """The two depths are different numbers, deliberately.

        The hypothesis tree interleaves the kinds, so D[root] sits one
        level below M[root] there while both describe the root node.
        `hierarchical_bonferroni_rejections` spends level by the
        hypothesis-tree depth; `Claim.depth` is the decision-tree one.
        """
        claims = interweaved_hierarchy(opposite_effects_tree(), ALPHA).test()
        by_name = {c.hypothesis.name: c for c in claims.claims}

        assert by_name["M[root]"].depth == by_name["D[root]"].depth
        assert by_name["D[root]"].hypothesis in by_name["M[root]"].hypothesis.children
