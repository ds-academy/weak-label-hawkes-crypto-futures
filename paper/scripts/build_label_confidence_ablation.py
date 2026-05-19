"""Build the K5 / Tier-1 E4 label-confidence ablation table.

E4 closes the third reviewer's "Information label semantics" complaint
descriptively rather than by re-running the Hawkes fitter. The reviewer
asked whether the directional finding survives if we strip out the 30
Low-confidence Information rows in which the PPI verdict and the RR
verdict disagree. A full refit on the High-confidence-only subset
requires either an event-level likelihood export from the Hawkes fitter
or a fresh H3 run; both are deliberately deferred to J2 (see the K5-0
plan lock in REVIEW_INTEGRATED_RESPONSE).

What this script does provide is the descriptive composition of three
label-policy variants on the existing 46 computed labels:

* baseline: PPI-primary / RR-validator, 34 Information + 12 Noise.
* strict: PPI/RR-agreement Information + Noise, 4 Information + 12 Noise.
* low_conf_excluded: same as strict (the only Low-confidence rows are
  in the Information class). Reported separately so the policy intent is
  explicit even though the resulting subset coincides.

For each variant the script reports the implied class share, the number
of release-batch deduplicated events implied by Section 6.4 grouping,
and a one-line caveat that the full Hawkes refit is in J2 scope.

Inputs (defaults):
    research/2026-weak-label-exo-hawkes/004_paper/tables/table_4_ppi_rr_matrix.csv
    research/2026-weak-label-exo-hawkes/004_paper/j0_expansion/tables/table_8_dependence_audit.csv

Outputs:
    research/2026-weak-label-exo-hawkes/004_paper/j0_expansion/tables/table_13_label_confidence_ablation.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = REPO_ROOT / "research/2026-weak-label-exo-hawkes"
DEFAULT_PPI_RR_TABLE = (
    PROJECT_ROOT / "004_paper/tables/table_4_ppi_rr_matrix.csv"
)
DEFAULT_DEPENDENCE_TABLE = (
    PROJECT_ROOT / "004_paper/j0_expansion/tables/table_8_dependence_audit.csv"
)
DEFAULT_OUTPUT_TABLE = (
    PROJECT_ROOT
    / "004_paper/j0_expansion/tables/table_13_label_confidence_ablation.csv"
)

FIELDNAMES = [
    "policy_variant",
    "information_count",
    "noise_count",
    "computed_total",
    "information_share",
    "deduplicated_event_estimate",
    "hawkes_refit_status",
    "note",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def composition_from_ppi_rr(rows: list[dict[str, str]]) -> tuple[int, int, int]:
    high_info = 0
    low_info = 0
    noise = 0
    for row in rows:
        if row.get("label") == "Information" and row.get("label_confidence") == "High":
            high_info += int(row["count"])
        elif row.get("label") == "Information" and row.get("label_confidence") == "Low":
            low_info += int(row["count"])
        elif row.get("label") == "Noise":
            noise += int(row["count"])
    return high_info, low_info, noise


def deduplicated_event_estimate(dependence_rows: list[dict[str, str]]) -> int:
    """Sum unique deduplicated computed events across cohorts.

    Section 6.4 reports BTC = 12 deduplicated events and ETH = 11; we
    report the simple sum 23 as the dedup estimate that an
    Information-and-Noise refit would face on the full computed
    surface. Subset variants scale this estimate proportionally to the
    Information count, which is conservative for the strict variant.
    """
    return sum(int(row["deduplicated_event_count"]) for row in dependence_rows)


def build_label_confidence_ablation(
    *,
    ppi_rr_table: Path = DEFAULT_PPI_RR_TABLE,
    dependence_table: Path = DEFAULT_DEPENDENCE_TABLE,
    output_table: Path = DEFAULT_OUTPUT_TABLE,
) -> Path:
    ppi_rr_rows = read_csv_rows(ppi_rr_table)
    dependence_rows = read_csv_rows(dependence_table)

    high_info, low_info, noise = composition_from_ppi_rr(ppi_rr_rows)
    base_dedup = deduplicated_event_estimate(dependence_rows)

    base_info = high_info + low_info
    base_total = base_info + noise

    rows_out: list[dict[str, str]] = []

    rows_out.append(
        {
            "policy_variant": "baseline_ppi_primary",
            "information_count": str(base_info),
            "noise_count": str(noise),
            "computed_total": str(base_total),
            "information_share": f"{base_info / base_total:.6f}",
            "deduplicated_event_estimate": str(base_dedup),
            "hawkes_refit_status": "fitted_in_main_table",
            "note": "PPI primary, RR validator; full computed surface used for the Hawkes comparison.",
        }
    )

    strict_total = high_info + noise
    strict_share = high_info / strict_total if strict_total else 0.0
    strict_dedup = max(0, round(base_dedup * strict_total / base_total)) if base_total else 0
    rows_out.append(
        {
            "policy_variant": "strict_agreement_only",
            "information_count": str(high_info),
            "noise_count": str(noise),
            "computed_total": str(strict_total),
            "information_share": f"{strict_share:.6f}",
            "deduplicated_event_estimate": str(strict_dedup),
            "hawkes_refit_status": "deferred_to_J2",
            "note": "PPI=Information AND RR=Information for Information; Noise unchanged. A Hawkes refit on this 16-event subset is in J2 scope.",
        }
    )

    rows_out.append(
        {
            "policy_variant": "low_confidence_excluded",
            "information_count": str(high_info),
            "noise_count": str(noise),
            "computed_total": str(strict_total),
            "information_share": f"{strict_share:.6f}",
            "deduplicated_event_estimate": str(strict_dedup),
            "hawkes_refit_status": "deferred_to_J2",
            "note": "Drop the 30 Low-confidence Information rows where PPI=Information and RR=Noise. Coincides with strict_agreement_only on this surface; reported separately to make the policy intent explicit.",
        }
    )

    output_table.parent.mkdir(parents=True, exist_ok=True)
    with output_table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows_out)
    return output_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppi-rr-table", type=Path, default=DEFAULT_PPI_RR_TABLE)
    parser.add_argument("--dependence-table", type=Path, default=DEFAULT_DEPENDENCE_TABLE)
    parser.add_argument("--output-table", type=Path, default=DEFAULT_OUTPUT_TABLE)
    args = parser.parse_args()
    out = build_label_confidence_ablation(
        ppi_rr_table=args.ppi_rr_table,
        dependence_table=args.dependence_table,
        output_table=args.output_table,
    )
    print(out)


if __name__ == "__main__":
    main()
