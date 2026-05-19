"""Build the J1-1 OOS audit table from pre-existing J0 H4 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = REPO_ROOT / "research/2026-weak-label-exo-hawkes"
DEFAULT_INPUT_TABLE = PROJECT_ROOT / "004_paper/j0_expansion/tables/table_5_hawkes_comparison.csv"
DEFAULT_H3_MANIFEST = (
    REPO_ROOT
    / "data/derived/weak_label_exo_hawkes/J0_v0"
    / "h3_hawkes_j0_ppi_primary_gap_p98_v0_20260503/manifest.json"
)
DEFAULT_OUTPUT_TABLE = PROJECT_ROOT / "004_paper/j0_expansion/tables/table_7_oos_audit.csv"
DEFAULT_EPSILON_NATS = 5.0

FIELDNAMES = [
    "cohort_name",
    "model",
    "log_likelihood_in_sample",
    "log_likelihood_oos",
    "ll_oos_delta_vs_m0",
    "oos_rank",
    "n_events_in_sample",
    "n_events_oos",
    "lrt_p_value_vs_m0",
    "scenario_indicator",
    "epsilon_nats",
]


@dataclass(frozen=True)
class EventSplitCounts:
    """Train/test exogenous event counts used by the H3 chronological split."""

    train: int | None
    test: int | None


def build_oos_audit(
    *,
    input_table: Path = DEFAULT_INPUT_TABLE,
    output_table: Path = DEFAULT_OUTPUT_TABLE,
    h3_manifest: Path | None = DEFAULT_H3_MANIFEST,
    epsilon_nats: float = DEFAULT_EPSILON_NATS,
) -> Path:
    """Build a deterministic J1-1 OOS audit CSV."""
    rows = read_csv(input_table)
    event_counts = read_event_counts(h3_manifest) if h3_manifest is not None and h3_manifest.exists() else {}
    audit_rows = oos_audit_rows(rows, event_counts=event_counts, epsilon_nats=epsilon_nats)
    output_table.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_table, FIELDNAMES, audit_rows)
    return output_table


def oos_audit_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    event_counts: Mapping[str, EventSplitCounts] | None = None,
    epsilon_nats: float = DEFAULT_EPSILON_NATS,
) -> list[dict[str, str]]:
    """Return OOS audit rows with pre-registered scenario indicators."""
    event_counts = event_counts or {}
    cohorts = sorted({row["cohort_name"] for row in rows})
    out: list[dict[str, str]] = []
    for cohort in cohorts:
        cohort_rows = [row for row in rows if row["cohort_name"] == cohort]
        m0 = next((row for row in cohort_rows if row["model"] == "M0"), None)
        if m0 is None:
            raise ValueError(f"Missing M0 row for cohort {cohort!r}")
        m0_oos = parse_float(m0["log_likelihood_oos"])
        ranks = oos_ranks(cohort_rows)
        counts = event_counts.get(cohort, EventSplitCounts(train=None, test=None))
        for row in sorted(cohort_rows, key=lambda item: str(item["model"])):
            model = row["model"]
            ll_oos = parse_float(row["log_likelihood_oos"])
            delta = ll_oos - m0_oos
            out.append(
                {
                    "cohort_name": cohort,
                    "model": model,
                    "log_likelihood_in_sample": fmt(row.get("log_likelihood_in_sample")),
                    "log_likelihood_oos": fmt(row.get("log_likelihood_oos")),
                    "ll_oos_delta_vs_m0": fmt_number(delta),
                    "oos_rank": str(ranks[model]),
                    "n_events_in_sample": fmt_count(counts.train),
                    "n_events_oos": fmt_count(counts.test),
                    "lrt_p_value_vs_m0": fmt(row.get("lrt_p_value_vs_m0")),
                    "scenario_indicator": scenario_indicator(model, delta, epsilon_nats),
                    "epsilon_nats": fmt_number(epsilon_nats),
                }
            )
    return out


def scenario_indicator(model: str, delta_vs_m0: float, epsilon_nats: float) -> str:
    """Classify the OOS result using the pre-registered J1 epsilon."""
    if model == "M0":
        return "baseline_m0"
    if model == "M2":
        return "ablation_check_m2"
    if model != "M1":
        return "unsupported_model"
    if delta_vs_m0 > epsilon_nats:
        return "A_m1_materially_better_oos"
    if delta_vs_m0 < -epsilon_nats:
        return "C_m1_materially_worse_oos"
    return "B_m1_similar_oos"


def oos_ranks(rows: Sequence[Mapping[str, str]]) -> dict[str, int]:
    """Rank models by OOS log-likelihood within one cohort; higher is better."""
    ranked = sorted(
        ((row["model"], parse_float(row["log_likelihood_oos"])) for row in rows),
        key=lambda item: (-item[1], item[0]),
    )
    return {model: index + 1 for index, (model, _) in enumerate(ranked)}


def read_event_counts(path: Path) -> dict[str, EventSplitCounts]:
    """Extract train/test event counts from the J0 H3 manifest."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    summaries = manifest.get("observation_summaries", {})
    counts: dict[str, EventSplitCounts] = {}
    for key, summary in summaries.items():
        symbol = str(key).split(":", maxsplit=1)[0]
        cohort = cohort_from_symbol(symbol)
        if cohort is None:
            continue
        counts[cohort] = EventSplitCounts(
            train=int(summary["train_exogenous_count"]),
            test=int(summary["test_exogenous_count"]),
        )
    return counts


def cohort_from_symbol(symbol: str) -> str | None:
    if symbol == "BTCUSDT":
        return "main"
    if symbol == "ETHUSDT":
        return "robustness_symbol"
    return None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_float(value: str | None) -> float:
    if value is None or value == "" or value == "NULL":
        raise ValueError(f"Expected numeric value, got {value!r}")
    return float(value)


def fmt(value: str | None) -> str:
    if value is None or value == "":
        return "NULL"
    return value


def fmt_number(value: float) -> str:
    return f"{value:.6f}"


def fmt_count(value: int | None) -> str:
    return "NULL" if value is None else str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-table", type=Path, default=DEFAULT_INPUT_TABLE)
    parser.add_argument("--output-table", type=Path, default=DEFAULT_OUTPUT_TABLE)
    parser.add_argument("--h3-manifest", type=Path, default=DEFAULT_H3_MANIFEST)
    parser.add_argument("--epsilon-nats", type=float, default=DEFAULT_EPSILON_NATS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = build_oos_audit(
        input_table=args.input_table,
        output_table=args.output_table,
        h3_manifest=args.h3_manifest,
        epsilon_nats=args.epsilon_nats,
    )
    print(path)


if __name__ == "__main__":
    main()
