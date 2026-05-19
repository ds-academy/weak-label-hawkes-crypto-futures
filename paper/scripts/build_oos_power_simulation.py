"""Build the K5 / Tier-1 E1 OOS power simulation table.

E1 closes the OOS-power complaint from the third reviewer round: with
n_test = 2 held-out exogenous events per cohort, the 5.0-nat materiality
threshold reported in Section 6.3 cannot in principle reject any
practically interesting effect size, and the reviewer asked for a stated
power calculation rather than a deferred limitation paragraph.

The simulation does not estimate a new effect. It simulates, under a
range of true M1-vs-M0 effect sizes Delta_LL_true, what fraction of
n_test = 2 OOS draws would yield an observed delta below the
pre-registered 5.0-nat materiality threshold.

We deliberately do NOT use the asymptotic chi-square approximation; with
two test events the asymptotic argument does not apply, and using it
would reproduce the same inferential issue the audit table already has.
We instead simulate per-event log-likelihood contributions as Gaussian
draws around a per-event mean, with a per-event noise scale derived
conservatively from the in-sample LL_M1 - LL_M0 magnitude divided by
sqrt(n_train); this yields an OOS noise scale that is, if anything,
larger than what a real per-event likelihood export would deliver, and
is therefore a conservative power floor for the audit threshold.

Inputs:
    table_5_hawkes_comparison.csv (j0_expansion) for n_train and the
    fitted in-sample LL_M1 - LL_M0.

Outputs:
    table_11_oos_power_simulation.csv with columns
        cohort_name, n_test, delta_ll_true_nats, threshold_nats,
        detection_probability, monte_carlo_se, sigma_total
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = REPO_ROOT / "research/2026-weak-label-exo-hawkes"
DEFAULT_HAWKES_TABLE = (
    PROJECT_ROOT / "004_paper/j0_expansion/tables/table_5_hawkes_comparison.csv"
)
DEFAULT_OOS_AUDIT_TABLE = (
    PROJECT_ROOT / "004_paper/j0_expansion/tables/table_7_oos_audit.csv"
)
DEFAULT_OUTPUT_TABLE = (
    PROJECT_ROOT
    / "004_paper/j0_expansion/tables/table_11_oos_power_simulation.csv"
)

DEFAULT_N_TEST = 2
DEFAULT_THRESHOLD_NATS = 5.0
DEFAULT_DELTA_GRID = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
DEFAULT_N_REPLICATES = 20_000
DEFAULT_SEED = 20260505

FIELDNAMES = [
    "cohort_name",
    "n_train",
    "n_test",
    "in_sample_delta_ll",
    "sigma_per_event",
    "delta_ll_true_nats",
    "threshold_nats",
    "detection_probability",
    "monte_carlo_se",
    "sigma_total",
    "n_replicates",
]


@dataclass(frozen=True)
class CohortInputs:
    cohort_name: str
    n_train: int
    in_sample_delta_ll: float


def read_inputs(hawkes_path: Path, oos_audit_path: Path) -> list[CohortInputs]:
    """Combine n_train from the OOS audit table with in_sample_delta_ll from the
    Hawkes comparison table.

    The Hawkes comparison artifact carries the in-sample LL delta but not
    the train/test split sizes; the OOS audit artifact carries the split
    sizes but not the LL delta. We join them on cohort_name.
    """
    train_counts: dict[str, int] = {}
    for row in _read_csv(oos_audit_path):
        if row["model"] != "M1":
            continue
        cohort = row["cohort_name"]
        n_train = int(row["n_events_in_sample"]) if row.get("n_events_in_sample") else 0
        if n_train > 0:
            train_counts[cohort] = n_train

    cohorts: list[CohortInputs] = []
    seen = set()
    for row in _read_csv(hawkes_path):
        if row["model"] != "M1":
            continue
        cohort = row["cohort_name"]
        if cohort in seen:
            continue
        n_train = train_counts.get(cohort, 0)
        if n_train <= 0:
            continue
        in_sample_delta_ll = float(row["ll_delta_vs_m0"])
        cohorts.append(CohortInputs(cohort, n_train, in_sample_delta_ll))
        seen.add(cohort)
    if not cohorts:
        raise ValueError(
            f"no usable M1 rows after joining {hawkes_path} with {oos_audit_path}"
        )
    return cohorts


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def conservative_sigma_per_event(in_sample_delta_ll: float, n_train: int) -> float:
    """Conservative per-event noise scale for the OOS LL contribution.

    We assume the per-event Gaussian contribution to the LL difference has
    a standard deviation comparable to the absolute in-sample delta divided
    by sqrt(n_train). This treats the entire observed in-sample improvement
    as if it were dispersion noise, which is conservative; a real
    per-event likelihood export would in general yield a smaller per-event
    sigma.
    """
    if n_train <= 0:
        raise ValueError("n_train must be positive")
    return abs(in_sample_delta_ll) / math.sqrt(n_train) if in_sample_delta_ll != 0 else 1.0


def simulate_detection_probability(
    *,
    delta_ll_true_nats: float,
    sigma_per_event: float,
    n_test: int,
    threshold_nats: float,
    n_replicates: int,
    rng: random.Random,
) -> tuple[float, float]:
    """Monte Carlo detection probability for a Gaussian OOS LL delta.

    Returns the fraction of replicates whose simulated total OOS delta
    exceeds the materiality threshold, plus the binomial Monte Carlo
    standard error of that fraction.
    """
    sigma_total = sigma_per_event * math.sqrt(n_test)
    mean_total = delta_ll_true_nats
    hits = 0
    for _ in range(n_replicates):
        sample = rng.gauss(mean_total, sigma_total)
        if sample > threshold_nats:
            hits += 1
    p = hits / n_replicates
    se = math.sqrt(max(p * (1.0 - p), 0.0) / n_replicates)
    return p, se


def build_power_simulation(
    *,
    hawkes_table: Path = DEFAULT_HAWKES_TABLE,
    oos_audit_table: Path = DEFAULT_OOS_AUDIT_TABLE,
    output_table: Path = DEFAULT_OUTPUT_TABLE,
    n_test: int = DEFAULT_N_TEST,
    threshold_nats: float = DEFAULT_THRESHOLD_NATS,
    delta_grid: tuple[float, ...] = DEFAULT_DELTA_GRID,
    n_replicates: int = DEFAULT_N_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> Path:
    cohorts = read_inputs(hawkes_table, oos_audit_table)
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for cohort in cohorts:
        sigma_per_event = conservative_sigma_per_event(
            cohort.in_sample_delta_ll, cohort.n_train
        )
        sigma_total = sigma_per_event * math.sqrt(n_test)
        for delta_true in delta_grid:
            p, se = simulate_detection_probability(
                delta_ll_true_nats=delta_true,
                sigma_per_event=sigma_per_event,
                n_test=n_test,
                threshold_nats=threshold_nats,
                n_replicates=n_replicates,
                rng=rng,
            )
            rows.append(
                {
                    "cohort_name": cohort.cohort_name,
                    "n_train": str(cohort.n_train),
                    "n_test": str(n_test),
                    "in_sample_delta_ll": f"{cohort.in_sample_delta_ll:.6f}",
                    "sigma_per_event": f"{sigma_per_event:.6f}",
                    "delta_ll_true_nats": f"{delta_true:.6f}",
                    "threshold_nats": f"{threshold_nats:.6f}",
                    "detection_probability": f"{p:.6f}",
                    "monte_carlo_se": f"{se:.6f}",
                    "sigma_total": f"{sigma_total:.6f}",
                    "n_replicates": str(n_replicates),
                }
            )

    output_table.parent.mkdir(parents=True, exist_ok=True)
    with output_table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return output_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hawkes-table", type=Path, default=DEFAULT_HAWKES_TABLE)
    parser.add_argument("--oos-audit-table", type=Path, default=DEFAULT_OOS_AUDIT_TABLE)
    parser.add_argument("--output-table", type=Path, default=DEFAULT_OUTPUT_TABLE)
    parser.add_argument("--n-test", type=int, default=DEFAULT_N_TEST)
    parser.add_argument("--threshold-nats", type=float, default=DEFAULT_THRESHOLD_NATS)
    parser.add_argument(
        "--delta-grid",
        type=str,
        default=",".join(str(d) for d in DEFAULT_DELTA_GRID),
        help="Comma-separated list of true Delta_LL values (nats)",
    )
    parser.add_argument("--n-replicates", type=int, default=DEFAULT_N_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    delta_grid = tuple(float(x) for x in args.delta_grid.split(","))
    out = build_power_simulation(
        hawkes_table=args.hawkes_table,
        oos_audit_table=args.oos_audit_table,
        output_table=args.output_table,
        n_test=args.n_test,
        threshold_nats=args.threshold_nats,
        delta_grid=delta_grid,
        n_replicates=args.n_replicates,
        seed=args.seed,
    )
    print(out)


if __name__ == "__main__":
    main()
