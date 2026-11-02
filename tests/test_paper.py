"""Tests for the reconstruction of the paper's simulation study."""

import math

import numpy as np
import pytest

from paper import published
from paper.simulation import (
    ALL_CELLS,
    apply_procedures,
    ASSUMPTIONS,
    build_hierarchy,
    claim_estimand,
    Condition,
    EVERY_SUBGROUP_HELPED,
    fixed_partition,
    fwer_replicate,
    generate_population,
    GLOBAL_NULL,
    null_is_true,
    power_replicate,
    PROCEDURES,
    Region,
    RUNNING_EXAMPLE,
    split_halves,
    to_uplift_tree,
    Verdicts,
    worked_example,
)


def halves(seed: int, n: int, effects=RUNNING_EXAMPLE):
    """A population split into training and evaluation halves."""
    rng = np.random.default_rng(seed)
    return split_halves(rng, generate_population(rng, n, effects))


class TestEffects:
    """The effect configurations of sections 7.1 and 7.3."""

    @staticmethod
    def test_running_example_matches_the_paper() -> None:
        """Section 7.1: an ATE of +0.925, and +0.1 among x2 = 0."""
        assert RUNNING_EXAMPLE.average(ALL_CELLS) == pytest.approx(0.925)
        x2_zero = frozenset(c for c in ALL_CELLS if c[0] == 0)
        assert RUNNING_EXAMPLE.average(x2_zero) == pytest.approx(0.1)

    @staticmethod
    def test_every_subgroup_helped_differs_only_in_the_harmed_cell() -> None:
        """The second configuration helps the quarter the first harms."""
        for cell in ALL_CELLS:
            if cell == (0, 0):
                assert RUNNING_EXAMPLE.tau[cell] < 0.0 < EVERY_SUBGROUP_HELPED.tau[cell]
            else:
                assert RUNNING_EXAMPLE.tau[cell] == EVERY_SUBGROUP_HELPED.tau[cell]

    @staticmethod
    def test_scaling_multiplies_every_cell() -> None:
        """An effect scale of 0.45 multiplies each subgroup's tau."""
        scaled = RUNNING_EXAMPLE.scaled(0.45)
        for cell in ALL_CELLS:
            assert scaled.tau[cell] == pytest.approx(0.45 * RUNNING_EXAMPLE.tau[cell])

    @staticmethod
    def test_population_moments() -> None:
        """Arms are balanced and the observed contrast estimates the ATE."""
        rng = np.random.default_rng(0)
        sample = generate_population(rng, 200_000, RUNNING_EXAMPLE)
        assert sample.X.shape == (200_000, 10)
        assert sample.z.mean() == pytest.approx(0.5, abs=0.01)
        contrast = sample.y[sample.z].mean() - sample.y[~sample.z].mean()
        assert contrast == pytest.approx(0.925, abs=0.06)


class TestRegions:
    """Conditions, regions and partitions."""

    @staticmethod
    def test_moderator_conditions_restrict_cells() -> None:
        """A split on x2 or x3 admits only matching cells."""
        x2_one = Condition(feature=2, threshold=0.5, above=True)
        assert x2_one.label() == "x2 = 1"
        assert {c for c in ALL_CELLS if x2_one.admits(c)} == {(1, 0), (1, 1)}

    @staticmethod
    def test_other_conditions_admit_every_cell() -> None:
        """A split on a noise feature says nothing about the moderators."""
        noise = Condition(feature=7, threshold=1.17, above=False)
        assert noise.label() == "x7 <= 1.17"
        assert all(noise.admits(c) for c in ALL_CELLS)

    @staticmethod
    def test_fixed_partition_has_ten_hypotheses() -> None:
        """Seven nodes; 2, 4 and 4 claims by depth, as in Table 4."""
        train, evaluation = halves(1, 4_000)
        region = fixed_partition()
        tree = to_uplift_tree(region, train, evaluation)
        assert tree is not None
        claims = build_hierarchy(tree, "Trickle down").claims
        assert len(region.walk()) == 7
        depths = [claim.depth for claim in claims]
        assert [depths.count(d) for d in range(3)] == [2, 4, 4]

    @staticmethod
    def test_every_claim_is_true_in_both_configurations() -> None:
        """'Ten true claims are available in each' (Table 3 caption)."""
        train, evaluation = halves(2, 4_000)
        region = fixed_partition()
        regions = {r.name: r for r in region.walk()}
        tree = to_uplift_tree(region, train, evaluation)
        assert tree is not None
        for effects in (RUNNING_EXAMPLE, EVERY_SUBGROUP_HELPED):
            for claim in build_hierarchy(tree, "Trickle down").claims:
                assert claim_estimand(claim, regions, effects) != pytest.approx(0.0)

    @staticmethod
    def test_untestable_split_is_dropped() -> None:
        """A child with no evaluation data turns its parent into a leaf."""
        train, evaluation = halves(3, 2_000)
        region = Region().split(4, 100.0)  # Nobody has x4 > 100.
        tree = to_uplift_tree(region, train, evaluation)
        assert tree is not None
        assert tree.children == []


class TestScoring:
    """Which rejections are errors and which are claims."""

    @staticmethod
    @pytest.mark.parametrize(
        ("estimand", "alternative", "expected"),
        [
            (0.5, "greater", False),
            (0.5, "less", True),
            (-0.5, "greater", True),
            (-0.5, "less", False),
            (0.0, "greater", True),
            (0.0, "less", True),
            (0.0, "two-sided", True),
            (0.5, "two-sided", False),
        ],
    )
    def test_null_is_true(estimand: float, alternative: str, expected: bool) -> None:
        """A null holds on its own side of the threshold, boundary included."""
        assert null_is_true(estimand, 0.0, alternative) is expected

    @staticmethod
    def test_wrong_direction_is_an_error_not_a_claim() -> None:
        """Rejecting a true null in the wrong direction costs, not counts."""
        verdicts = Verdicts(
            rejected=(True, True, False),
            null_true=(True, False, False),
            depths=(0, 0, 1),
        )
        assert verdicts.familywise_error
        assert verdicts.true_claims == 1
        assert verdicts.true_claims_by_depth(2) == (1, 0, 0)

    @staticmethod
    def test_global_null_makes_every_null_true() -> None:
        """Under tau = 0, nothing a procedure rejects can be a true claim."""
        train, evaluation = halves(4, 4_000, GLOBAL_NULL)
        region = fixed_partition()
        regions = {r.name: r for r in region.walk()}
        tree = to_uplift_tree(region, train, evaluation)
        assert tree is not None
        for verdicts in apply_procedures(tree, regions, GLOBAL_NULL).values():
            assert all(verdicts.null_true)
            assert verdicts.true_claims == 0


class TestReplicates:
    """Whole replicates of sections 7.2 and 7.3."""

    @staticmethod
    def test_fwer_replicate_reports_every_cell() -> None:
        """Both partitions, all five procedures."""
        outcome = fwer_replicate(np.random.SeedSequence(5), n=2_000)
        assert set(outcome) == {"Fixed partition", "Fitted tree"}
        for partition in outcome.values():
            assert set(partition) == set(PROCEDURES)

    @staticmethod
    def test_fixed_hierarchy_errs_only_through_the_root() -> None:
        """Under the global null fixed hierarchy rejects only after the root.

        So its familywise error is the root's rejection, the same on both
        partitions. This is why Table 2's fixed-hierarchy row repeats.
        """
        for seed in np.random.SeedSequence(6).spawn(20):
            outcome = fwer_replicate(seed, n=2_000)
            assert (
                outcome["Fixed partition"]["Fixed hierarchy"]
                == outcome["Fitted tree"]["Fixed hierarchy"]
            )

    @staticmethod
    def test_power_replicate_is_reproducible() -> None:
        """One seed, one answer: results do not depend on worker count."""
        seed = np.random.SeedSequence(7)
        first = power_replicate(seed, RUNNING_EXAMPLE)
        again = power_replicate(np.random.SeedSequence(7), RUNNING_EXAMPLE)
        assert first == again
        assert all(len(depths) == 3 for depths in first.values())
        assert all(sum(depths) <= 10 for depths in first.values())

    @staticmethod
    def test_worked_example_finds_the_main_split() -> None:
        """Section 7.1: the fitted tree's first split is on x2."""
        example = worked_example(seed=71)
        assert example.region.children[0].name == "x2 = 0"
        rows = example.node_rows()
        assert rows[0][0] == "everyone"
        assert rows[0][4] == pytest.approx(0.925)


class TestPublished:
    """The transcribed tables are internally consistent."""

    @staticmethod
    def test_table_four_rows_sum_to_table_three() -> None:
        """Table 4 is Table 3's 0.45 column split by depth (within rounding)."""
        column = published.SCALES.index(published.DEPTH_TABLE_SCALE)
        for name, rows in published.DEPTH.items():
            for procedure, by_depth in rows.items():
                total, _ = published.POWER[name][procedure][column]
                assert math.isclose(sum(by_depth), total, abs_tol=0.021)

    @staticmethod
    def test_fwer_standard_errors_are_binomial() -> None:
        """Table 2's s.e. is sqrt(p (1 - p) / 4000)."""
        for column in published.FWER.values():
            for p, se in column.values():
                assert math.sqrt(p * (1 - p) / published.REPLICATIONS) == pytest.approx(
                    se, abs=5e-5
                )

    @staticmethod
    def test_every_assumption_is_explained() -> None:
        """Each assumption has a title and a sentence of justification."""
        assert len(ASSUMPTIONS) >= 10
        for title, text in ASSUMPTIONS:
            assert title
            assert text.endswith(".")
