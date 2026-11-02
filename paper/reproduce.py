r"""Run the reconstructed simulation study and render it for the docs.

Usage::

    uv run python -m paper.reproduce               # everything, 4,000 reps
    uv run python -m paper.reproduce --replications 400 --workers 4
    uv run python -m paper.reproduce --render-only # rebuild the tables

Raw results go to ``paper/results.json``. The RST fragments the
*Reproducing the Paper* guide includes go to
``docs/user_guides/_generated/``. Both are committed, so building the
docs never reruns the simulation.

Every replicate draws from its own child of one ``SeedSequence``, so the
results do not depend on how many workers share the work.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from paper import published
from paper.simulation import (
    ASSUMPTIONS,
    EffectConfiguration,
    EVERY_SUBGROUP_HELPED,
    fwer_replicate,
    power_replicate,
    PROCEDURES,
    RUNNING_EXAMPLE,
    worked_example,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "paper" / "results.json"
GENERATED = ROOT / "docs" / "user_guides" / "_generated"

# arXiv 2610.37966, so the seed is easy to find and hard to tune.
ROOT_SEED = 261_037_966
# Seed for the worked example of section 7.1.
EXAMPLE_SEED = 71
CONFIGURATIONS = (RUNNING_EXAMPLE, EVERY_SUBGROUP_HELPED)
PARTITIONS = ("Fixed partition", "Fitted tree")


def child_seeds(spawn_key: tuple[int, ...], count: int) -> list[np.random.SeedSequence]:
    """Independent seeds for one study, reproducible from ROOT_SEED."""
    return np.random.SeedSequence(ROOT_SEED, spawn_key=spawn_key).spawn(count)


def run_parallel(
    function: Callable[[np.random.SeedSequence], Any],
    seeds: Sequence[np.random.SeedSequence],
    workers: int,
) -> list[Any]:
    """Map ``function`` over ``seeds``, in order."""
    if workers <= 1:
        return [function(seed) for seed in seeds]

    chunksize = max(1, len(seeds) // (workers * 8))
    # Spawn rather than fork: the main process has already run OpenMP
    # (the worked example's boosting fits), and forking after that can
    # deadlock the workers on Linux.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        return list(pool.map(function, seeds, chunksize=chunksize))


def mean_and_se(values: Sequence[float]) -> tuple[float, float]:
    """Sample mean and its standard error."""
    x = np.asarray(values, dtype=np.float64)
    if x.size < 2:
        return float(x.mean()), math.nan

    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(x.size))


def study_fwer(replications: int, workers: int) -> dict[str, Any]:
    """Section 7.2: FWER under the global null, paired across partitions."""
    outcomes = run_parallel(fwer_replicate, child_seeds((1,), replications), workers)

    errors = {
        partition: {
            procedure: int(sum(o[partition][procedure] for o in outcomes))
            for procedure in PROCEDURES
        }
        for partition in PARTITIONS
    }
    paired = {
        procedure: mean_and_se(
            [
                float(o["Fitted tree"][procedure])
                - float(o["Fixed partition"][procedure])
                for o in outcomes
            ]
        )
        for procedure in PROCEDURES
    }
    # The paper says trickle down and Hierarchical Bonferroni agree
    # exactly under the global null. They do at every internal node, but
    # leaves stake one under trickle down, so count where they part.
    disagreements = {
        partition: int(
            sum(
                o[partition]["Trickle down"] != o[partition]["Hierarchical Bonferroni"]
                for o in outcomes
            )
        )
        for partition in PARTITIONS
    }
    return {
        "replications": replications,
        "errors": errors,
        "paired_difference": paired,
        "trickle_down_vs_hierarchical_bonferroni": disagreements,
    }


def study_power(replications: int, workers: int) -> dict[str, Any]:
    """Section 7.3: true claims certified on the fixed partition."""
    results: dict[str, Any] = {}
    for c, configuration in enumerate(CONFIGURATIONS):
        by_scale: dict[str, Any] = {}
        for s, scale in enumerate(published.SCALES):
            function = partial(_power, effects=configuration.scaled(scale))
            outcomes = run_parallel(
                function, child_seeds((2, c, s), replications), workers
            )
            by_scale[f"{scale:.2f}"] = {
                procedure: {
                    "total": mean_and_se([sum(o[procedure]) for o in outcomes]),
                    "by_depth": [
                        mean_and_se([o[procedure][d] for o in outcomes])
                        for d in range(3)
                    ],
                }
                for procedure in PROCEDURES
            }
            print(f"  power: {configuration.name}, scale {scale:.2f}", file=sys.stderr)

        results[configuration.name] = by_scale

    return {"replications": replications, "configurations": results}


def _power(
    seed: np.random.SeedSequence, effects: EffectConfiguration
) -> dict[str, tuple[int, ...]]:
    """Module-level so worker processes can unpickle it."""
    return power_replicate(seed, effects)


def study_example() -> dict[str, Any]:
    """Section 7.1: one fitted tree and its tested hierarchy."""
    example = worked_example(EXAMPLE_SEED)
    nodes = [
        {
            "node": name,
            "depth": depth,
            "tau_train": tau_train,
            "tau_eval": tau_eval,
            "tau_true": tau_true,
        }
        for name, depth, tau_train, tau_eval, tau_true in example.node_rows()
    ]
    claims = []
    for claim in example.claims.claims:
        low, high = claim.reportable_bound
        claims.append(
            {
                "node": claim.node,
                "kind": claim.kind,
                "alternative": claim.alternative,
                "level": claim.level,
                "pvalue": claim.pvalue,
                "rejected": claim.rejected,
                "statement": claim.sentence()
                if claim.rejected
                else claim.bound_sentence(),
                "bound": [low, high],
            }
        )

    return {
        "seed": EXAMPLE_SEED,
        "n": example.train.size + example.evaluation.size,
        "nodes": nodes,
        "claims": claims,
    }


def metadata(replications: int) -> dict[str, Any]:
    """What produced these results."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"

    return {
        "date": date.today().isoformat(),
        "git_revision": revision,
        "root_seed": ROOT_SEED,
        "replications": replications,
        "python": sys.version.split()[0],
        "versions": {
            package: version(package)
            for package in ("trickle-down", "numpy", "scipy", "scikit-learn")
        },
    }


# --- Rendering -------------------------------------------------------------


def list_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    caption: str | None = None,
    widths: Sequence[int] | None = None,
) -> str:
    """An RST list-table, which tolerates any cell content."""
    lines = [f".. list-table::{' ' + caption if caption else ''}", "   :header-rows: 1"]
    if widths is not None:
        lines.append(f"   :widths: {' '.join(str(w) for w in widths)}")

    lines.append("")
    for row in [list(headers), *[list(r) for r in rows]]:
        lines.append(f"   * - {row[0]}")
        lines.extend(f"     - {cell}" for cell in row[1:])

    return "\n".join(lines) + "\n"


def binomial_se(p: float, n: int) -> float:
    """Standard error of a proportion, as the paper's Table 2 reports."""
    return math.sqrt(p * (1.0 - p) / n)


def gap(paper: float, paper_se: float, ours: float, ours_se: float) -> str:
    """Difference in units of the standard error of the difference."""
    se = math.hypot(paper_se, ours_se)
    if se == 0.0 or math.isnan(se):
        return "--"

    return f"{(ours - paper) / se:+.1f}"


def render_fwer(fwer: dict[str, Any]) -> str:
    """Table 2 beside the reconstruction."""
    n = fwer["replications"]
    rows = []
    for procedure in PROCEDURES:
        row = [procedure]
        for partition in PARTITIONS:
            paper, paper_se = published.FWER[partition][procedure]
            ours = fwer["errors"][partition][procedure] / n
            ours_se = binomial_se(ours, n)
            row += [
                f"{paper:.4f} ({paper_se:.4f})",
                f"{ours:.4f} ({ours_se:.4f})",
                gap(paper, paper_se, ours, ours_se),
            ]
        rows.append(row)

    table = list_table(
        [
            "Procedure",
            "Fixed: paper",
            "Fixed: ours",
            "z",
            "Fitted: paper",
            "Fitted: ours",
            "z",
        ],
        rows,
        widths=[24, 13, 13, 5, 13, 13, 5],
    )
    diff, diff_se = fwer["paired_difference"]["Bonferroni"]
    paper_diff, paper_diff_se = published.BONFERRONI_PAIRED_DIFFERENCE
    note = (
        f"\nPlain Bonferroni's paired difference, fitted tree minus fixed "
        f"partition: paper {paper_diff:+.4f} (s.e. {paper_diff_se:.4f}); "
        f"reconstruction {diff:+.4f} (s.e. {diff_se:.4f}).\n"
    )
    parted = fwer["trickle_down_vs_hierarchical_bonferroni"]
    note += (
        "\nTrickle down and Hierarchical Bonferroni disagree on whether a "
        f"familywise error occurred in {parted['Fixed partition']} of {n:,} "
        f"replications on the fixed partition and {parted['Fitted tree']} on "
        "the fitted tree.\n"
    )
    return table + note


def render_power(power: dict[str, Any]) -> str:
    """Table 3 beside the reconstruction, one table per configuration."""
    parts = []
    for configuration in CONFIGURATIONS:
        results = power["configurations"][configuration.name]
        rows = []
        for procedure in PROCEDURES:
            row = [procedure]
            for s, scale in enumerate(published.SCALES):
                paper, _ = published.POWER[configuration.name][procedure][s]
                ours, _ = results[f"{scale:.2f}"][procedure]["total"]
                row.append(f"{paper:.2f} / {ours:.2f}")
            rows.append(row)

        parts.append(
            list_table(
                ["Procedure", *(f"{scale:.2f}" for scale in published.SCALES)],
                rows,
                caption=f"{configuration.name} (paper / reconstruction, of 10)",
                widths=[24, *([10] * len(published.SCALES))],
            )
        )

    return "\n".join(parts)


def render_power_gaps(power: dict[str, Any]) -> str:
    """The largest disagreements with Table 3, in standard errors."""
    rows = []
    for configuration in CONFIGURATIONS:
        results = power["configurations"][configuration.name]
        for procedure in PROCEDURES:
            for s, scale in enumerate(published.SCALES):
                paper, paper_se = published.POWER[configuration.name][procedure][s]
                ours, ours_se = results[f"{scale:.2f}"][procedure]["total"]
                se = math.hypot(paper_se, ours_se)
                rows.append(
                    (
                        abs(ours - paper) / se,
                        configuration.name,
                        procedure,
                        scale,
                        paper,
                        ours,
                    )
                )

    rows.sort(reverse=True)
    within_two = sum(1 for r in rows if r[0] <= 2.0)
    body = list_table(
        ["Configuration", "Procedure", "Scale", "Paper", "Ours", "|z|"],
        [
            [name, procedure, f"{scale:.2f}", f"{paper:.2f}", f"{ours:.2f}", f"{z:.1f}"]
            for z, name, procedure, scale, paper, ours in rows[:8]
        ],
        caption="The eight largest differences from Table 3",
        widths=[30, 22, 8, 8, 8, 6],
    )
    return (
        f"{within_two} of {len(rows)} cells of Table 3 lie within two standard "
        "errors of the paper's value. The paper rounds its standard errors to "
        "0.01, so these z-scores are approximate.\n\n" + body
    )


def render_depth(power: dict[str, Any]) -> str:
    """Table 4 beside the reconstruction."""
    scale = f"{published.DEPTH_TABLE_SCALE:.2f}"
    parts = []
    for configuration in CONFIGURATIONS:
        results = power["configurations"][configuration.name][scale]
        rows = []
        for procedure in PROCEDURES:
            paper = published.DEPTH[configuration.name][procedure]
            ours = [m for m, _ in results[procedure]["by_depth"]]
            rows.append(
                [
                    procedure,
                    *(f"{p:.2f} / {o:.2f}" for p, o in zip(paper, ours, strict=True)),
                ]
            )

        parts.append(
            list_table(
                ["Procedure", "Root (of 2)", "Depth 1 (of 4)", "Depth 2 (of 4)"],
                rows,
                caption=f"{configuration.name}, scale {scale} (paper / reconstruction)",
                widths=[24, 14, 14, 14],
            )
        )

    return "\n".join(parts)


def render_example(example: dict[str, Any]) -> tuple[str, str, str]:
    """The paper's tree, our tree, and our tested hierarchy."""
    paper_tree = list_table(
        ["Node", "τ̂ train", "τ̂ eval", "Effect certified"],
        [
            [
                name,
                f"{tau_train:+.2f}",
                f"{tau_eval:+.2f}",
                "yes" if certified else "no",
            ]
            for name, depth, tau_train, tau_eval, certified in published.TREE
        ],
        caption="The paper's fitted tree (Figure 2)",
        widths=[44, 10, 10, 12],
    )

    certified = {
        c["node"]
        for c in example["claims"]
        if c["kind"] == "magnitude" and c["rejected"]
    }
    our_tree = list_table(
        ["Node", "τ̂ train", "τ̂ eval", "True τ", "Effect certified"],
        [
            [
                node["node"],
                f"{node['tau_train']:+.2f}",
                f"{node['tau_eval']:+.2f}",
                f"{node['tau_true']:+.2f}",
                "yes" if node["node"] in certified else "no",
            ]
            for node in example["nodes"]
        ],
        caption=f"The reconstruction's fitted tree (seed {example['seed']})",
        widths=[44, 10, 10, 10, 12],
    )

    hierarchy = list_table(
        ["Node", "Hypothesis", "Level", "p", "Decision", "What we may say"],
        [
            [
                c["node"],
                ("effect" if c["kind"] == "magnitude" else "heterogeneity")
                + f" ({c['alternative']})",
                f"{c['level']:.4f}",
                f"{c['pvalue']:.2g}",
                "rejected" if c["rejected"] else "retained",
                c["statement"],
            ]
            for c in example["claims"]
        ],
        caption="The reconstruction's interweaved hierarchy, as tested",
        widths=[26, 18, 8, 8, 9, 40],
    )
    return paper_tree, our_tree, hierarchy


def render_assumptions() -> str:
    """Every assumption, as a definition list."""
    return "\n".join(f"**{title}**\n   {text}\n" for title, text in ASSUMPTIONS)


def render_meta(meta: dict[str, Any]) -> str:
    """Provenance line for the guide."""
    versions = ", ".join(f"{k} {v}" for k, v in meta["versions"].items())
    return (
        f"Generated on {meta['date']} at revision ``{meta['git_revision']}`` "
        f"with {meta['replications']:,} replications per cell, root seed "
        f"``{meta['root_seed']}``, Python {meta['python']}, {versions}.\n"
    )


def render(results: dict[str, Any]) -> None:
    """Write every RST fragment the guide includes."""
    GENERATED.mkdir(parents=True, exist_ok=True)
    paper_tree, our_tree, hierarchy = render_example(results["example"])
    fragments = {
        "meta.rst": render_meta(results["meta"]),
        "assumptions.rst": render_assumptions(),
        "fwer.rst": render_fwer(results["fwer"]),
        "power.rst": render_power(results["power"]),
        "power_gaps.rst": render_power_gaps(results["power"]),
        "depth.rst": render_depth(results["power"]),
        "paper_tree.rst": paper_tree,
        "example_tree.rst": our_tree,
        "example_hierarchy.rst": hierarchy,
    }
    header = ".. Generated by paper/reproduce.py. Do not edit.\n\n"
    for name, text in fragments.items():
        (GENERATED / name).write_text(header + text)


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--replications", type=int, default=published.REPLICATIONS)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Rebuild the RST tables from paper/results.json without simulating.",
    )
    args = parser.parse_args(argv)

    if args.render_only:
        render(json.loads(RESULTS.read_text()))
        return

    # One thread per worker process: the boosting fits otherwise each
    # grab every core and the workers fight over them.
    os.environ["OMP_NUM_THREADS"] = "1"

    print("Worked example...", file=sys.stderr)
    example = study_example()
    print(f"FWER, {args.replications} replications...", file=sys.stderr)
    fwer = study_fwer(args.replications, args.workers)
    print(f"Power, {args.replications} replications per cell...", file=sys.stderr)
    power = study_power(args.replications, args.workers)

    results = {
        "meta": metadata(args.replications),
        "example": example,
        "fwer": fwer,
        "power": power,
    }
    RESULTS.write_text(json.dumps(results, indent=1) + "\n")
    render(results)


if __name__ == "__main__":
    main()
