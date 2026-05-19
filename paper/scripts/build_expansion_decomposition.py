"""Build the K5 / Tier-1 E6 expanded-universe decomposition table.

E6 closes the third reviewer's "where did the 41 added candidates go?"
complaint about Section 6 of the manuscript: the expanded run grew the
primary TE candidate set from 54 to 95 yet kept the activation-positive
count at 60. Without a decomposition, a reader cannot tell whether the
expansion (a) lay outside the market-data window, (b) hit the gap-p99
activation gate, or (c) was simply weaker candidates than the original
54.

This script joins three artifacts:

1. The baseline H_v0 event-universe activation input (54 rows).
2. The expanded J0_v0 event-universe activation input (95 rows).
3. The expanded J0_v0 event_activation outcomes (380 rows = 95 events
   x 2 symbols x 2 activation rules).

For each of the 41 added candidates (J0_v0 minus H_v0 by event_id) we
report the macro family from the canonical body_or_url path, the
country, whether the event timestamp falls within the J0 market-data
window (2026-02-26 20:10 UTC to 2026-04-27 14:32 UTC), and the
worst-case activation outcome across its four attempts. We also report
aggregate counts so a reviewer can read the breakdown without scanning
the per-event rows.

Inputs (defaults):
    /Users/jhjeon/workspace/future_polios/data/derived/weak_label_exo_hawkes/H_v0/h1_activation_v0_20260501/event_universe/event_universe_activation_input.jsonl
    /Users/jhjeon/workspace/future_polios/data/derived/weak_label_exo_hawkes/J0_v0/h1_activation_j0_gap_p99_v0_20260503/event_universe/event_universe_activation_input.jsonl
    /Users/jhjeon/workspace/future_polios/data/derived/weak_label_exo_hawkes/J0_v0/h1_activation_j0_gap_p99_v0_20260503/event_activation/event_activation.jsonl

Outputs:
    research/2026-weak-label-exo-hawkes/004_paper/j0_expansion/tables/table_12_expansion_decomposition.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = REPO_ROOT / "research/2026-weak-label-exo-hawkes"
DATA_ROOT = REPO_ROOT / "data/derived/weak_label_exo_hawkes"

DEFAULT_BASELINE_INPUT = (
    DATA_ROOT
    / "H_v0/h1_activation_v0_20260501/event_universe/event_universe_activation_input.jsonl"
)
DEFAULT_EXPANDED_INPUT = (
    DATA_ROOT
    / "J0_v0/h1_activation_j0_gap_p99_v0_20260503/event_universe/event_universe_activation_input.jsonl"
)
DEFAULT_ACTIVATION_OUTPUT = (
    DATA_ROOT
    / "J0_v0/h1_activation_j0_gap_p99_v0_20260503/event_activation/event_activation.jsonl"
)
DEFAULT_OUTPUT_TABLE = (
    PROJECT_ROOT
    / "004_paper/j0_expansion/tables/table_12_expansion_decomposition.csv"
)

MARKET_WINDOW_START = dt.datetime(2026, 2, 26, 20, 10, 16, tzinfo=dt.timezone.utc)
MARKET_WINDOW_END = dt.datetime(2026, 4, 27, 14, 32, 59, tzinfo=dt.timezone.utc)

FAMILY_KEYWORDS = (
    ("CPI/inflation", ("inflation", "cpi-")),
    ("PPI/producer", ("producer-prices", "ppi", "producer-price")),
    ("Labor", ("non-farm", "unemployment", "jobless", "payrolls", "employment", "average-hourly")),
    ("ISM/PMI", ("ism-", "pmi", "manufacturing-pmi", "services-pmi")),
    ("Retail sales", ("retail-sales",)),
    ("GDP/growth", ("gdp-growth", "gdp-deflator")),
    ("Central bank speech", ("speech", "fed-chair", "lagarde", "bailey", "barr", "williams")),
    ("Rate decision", ("interest-rate-decision", "fed-decision", "ecb-decision", "boe-decision")),
)

OUTCOME_PRIORITY = (
    "computed_positive",
    "computed_negative",
    "insufficient_baseline",
    "missing_market_data",
    "rule_not_applicable",
    "pipeline_version_mismatch",
)


FIELDNAMES = [
    "category",
    "subcategory",
    "count",
    "share_of_added",
    "note",
]


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def family_for(row: dict) -> str:
    body = (row.get("body_or_url") or "").lower()
    title = (row.get("title") or "").lower()
    blob = body + " " + title
    for family, keywords in FAMILY_KEYWORDS:
        for keyword in keywords:
            if keyword in blob:
                return family
    return "Other macro"


def in_market_window(row: dict) -> bool:
    ts = row.get("event_ts")
    if not ts:
        return False
    cleaned = ts.replace("Z", "+00:00")
    parsed = dt.datetime.fromisoformat(cleaned)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return MARKET_WINDOW_START <= parsed <= MARKET_WINDOW_END


def worst_outcome_per_event(activations: Iterable[dict]) -> dict[str, str]:
    """Reduce 4 activation attempts per event_id to one worst-case outcome.

    "Worst" here means: if any attempt is computed_positive we keep
    computed_positive (one positive symbol/rule is enough to count the
    event as activated); otherwise we apply OUTCOME_PRIORITY in order.
    """
    by_event: dict[str, list[str]] = {}
    for row in activations:
        by_event.setdefault(row["event_id"], []).append(row["activation_status"])

    summary: dict[str, str] = {}
    for event_id, statuses in by_event.items():
        if "computed_positive" in statuses:
            summary[event_id] = "computed_positive"
            continue
        for outcome in OUTCOME_PRIORITY[1:]:
            if outcome in statuses:
                summary[event_id] = outcome
                break
        else:
            summary[event_id] = statuses[0]
    return summary


def build_decomposition(
    *,
    baseline_input: Path = DEFAULT_BASELINE_INPUT,
    expanded_input: Path = DEFAULT_EXPANDED_INPUT,
    activation_output: Path = DEFAULT_ACTIVATION_OUTPUT,
    output_table: Path = DEFAULT_OUTPUT_TABLE,
) -> Path:
    baseline_rows = read_jsonl(baseline_input)
    expanded_rows = read_jsonl(expanded_input)
    activation_rows = read_jsonl(activation_output)

    baseline_ids = {row["event_id"] for row in baseline_rows}
    expanded_ids = {row["event_id"] for row in expanded_rows}
    added_ids = expanded_ids - baseline_ids

    added_rows = [row for row in expanded_rows if row["event_id"] in added_ids]

    outcome_summary = worst_outcome_per_event(activation_rows)

    n_added = len(added_rows)

    family_counts: Counter[str] = Counter()
    country_counts: Counter[str] = Counter()
    window_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()

    for row in added_rows:
        family_counts[family_for(row)] += 1
        country_counts[row.get("country") or "Unknown"] += 1
        in_window = in_market_window(row)
        window_counts["in_market_window" if in_window else "out_of_market_window"] += 1
        outcome = outcome_summary.get(row["event_id"], "no_attempt_recorded")
        outcome_counts[outcome] += 1

    rows_out: list[dict[str, str]] = []

    rows_out.append(
        {
            "category": "Header",
            "subcategory": "added_candidates",
            "count": str(n_added),
            "share_of_added": "1.000000",
            "note": "Expanded universe minus baseline universe by event_id (95 - 54).",
        }
    )

    for family, count in sorted(family_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        rows_out.append(
            {
                "category": "Family",
                "subcategory": family,
                "count": str(count),
                "share_of_added": f"{count / n_added:.6f}",
                "note": "Macro release family inferred from canonical body_or_url.",
            }
        )

    for country, count in sorted(country_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        rows_out.append(
            {
                "category": "Country",
                "subcategory": country,
                "count": str(count),
                "share_of_added": f"{count / n_added:.6f}",
                "note": "Source country from the canonical event row.",
            }
        )

    window_notes = {
        "in_market_window": "Event timestamp inside [2026-02-26, 2026-04-27] market-data window.",
        "out_of_market_window": "Event timestamp outside the market-data window; activation impossible by construction.",
    }
    for status in ("in_market_window", "out_of_market_window"):
        count = window_counts.get(status, 0)
        rows_out.append(
            {
                "category": "Window",
                "subcategory": status,
                "count": str(count),
                "share_of_added": f"{count / n_added:.6f}",
                "note": window_notes[status],
            }
        )

    outcome_notes = {
        "computed_positive": "At least one of the four (symbol, rule) attempts produced a computed-positive activation row.",
        "computed_negative": "No computed-positive attempt; at least one computed-negative attempt.",
        "insufficient_baseline": "Activation rejected for insufficient pre-event baseline coverage.",
        "missing_market_data": "Activation rejected because the event window had no usable market data.",
        "rule_not_applicable": "Pre-snapshot gate: the rule did not apply to this event.",
        "pipeline_version_mismatch": "Pre-snapshot gate: pipeline-version mismatch.",
        "no_attempt_recorded": "Event id has no row in the activation output (should not happen).",
    }
    for outcome, count in sorted(outcome_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        rows_out.append(
            {
                "category": "Activation outcome",
                "subcategory": outcome,
                "count": str(count),
                "share_of_added": f"{count / n_added:.6f}",
                "note": outcome_notes.get(outcome, "Activation outcome category."),
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
    parser.add_argument("--baseline-input", type=Path, default=DEFAULT_BASELINE_INPUT)
    parser.add_argument("--expanded-input", type=Path, default=DEFAULT_EXPANDED_INPUT)
    parser.add_argument("--activation-output", type=Path, default=DEFAULT_ACTIVATION_OUTPUT)
    parser.add_argument("--output-table", type=Path, default=DEFAULT_OUTPUT_TABLE)
    args = parser.parse_args()
    out = build_decomposition(
        baseline_input=args.baseline_input,
        expanded_input=args.expanded_input,
        activation_output=args.activation_output,
        output_table=args.output_table,
    )
    print(out)


if __name__ == "__main__":
    main()
