"""Build the K5 / Tier-1 E2 family-aware Hawkes input table.

E2 closes the third reviewer's "event-family confounding" complaint
descriptively rather than by re-running the Hawkes fitter on a
family-fixed-effect specification. The reviewer asked whether the
M0/M1 contrast that Section 6 reports is in practice a CPI vs
(Labor + Retail + ISM) contrast across release families with very
different release calendars and surprise distributions, rather than a
within-family Information vs Noise contrast.

A full Hawkes refit with a family fixed effect on the exogenous decay
requires either an event-level likelihood export from H3 or a fresh H3
run; both are deferred to J2 (see the K5-0 plan lock in
REVIEW_INTEGRATED_RESPONSE.md). What this script does provide is the
exact family composition of the events that actually entered the
Hawkes-fit table: per cohort, per family, the count of computed
Information / Noise rows and the implied independent release moments.

The output makes Barrier 2 in Section 6.6 quantitatively visible. It
shows that the BTCUSDT main cohort's Information class is essentially
Labor + Retail + ISM rows while its Noise class is essentially
CPI/inflation rows, and that the M0/M1 likelihood contrast therefore
spans only a handful of distinct release moments per cohort.

Inputs (defaults):
    /Users/jhjeon/workspace/future_polios/data/derived/weak_label_exo_hawkes/J0_v0/h1_activation_j0_gap_p99_v0_20260503/event_universe/event_universe_activation_input.jsonl
    /Users/jhjeon/workspace/future_polios/data/derived/weak_label_exo_hawkes/J0_v0/h3_hawkes_j0_ppi_primary_gap_p98_v0_20260503/hawkes_comparison/hawkes_label_input.jsonl

Outputs:
    research/2026-weak-label-exo-hawkes/004_paper/j0_expansion/tables/table_14_family_aware_hawkes_input.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = REPO_ROOT / "research/2026-weak-label-exo-hawkes"
DATA_ROOT = REPO_ROOT / "data/derived/weak_label_exo_hawkes"

DEFAULT_UNIVERSE_INPUT = (
    DATA_ROOT
    / "J0_v0/h1_activation_j0_gap_p99_v0_20260503/event_universe/event_universe_activation_input.jsonl"
)
DEFAULT_HAWKES_LABEL_INPUT = (
    DATA_ROOT
    / "J0_v0/h3_hawkes_j0_ppi_primary_gap_p98_v0_20260503/hawkes_comparison/hawkes_label_input.jsonl"
)
DEFAULT_OUTPUT_TABLE = (
    PROJECT_ROOT
    / "004_paper/j0_expansion/tables/table_14_family_aware_hawkes_input.csv"
)

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

COHORT_TO_SYMBOL = {
    "main": "BTCUSDT",
    "robustness_symbol": "ETHUSDT",
}

FIELDNAMES = [
    "cohort_name",
    "family",
    "computed_information",
    "computed_noise",
    "abstain_or_other",
    "total_rows",
    "unique_event_timestamps",
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


def build_family_aware_hawkes(
    *,
    universe_input: Path = DEFAULT_UNIVERSE_INPUT,
    hawkes_label_input: Path = DEFAULT_HAWKES_LABEL_INPUT,
    output_table: Path = DEFAULT_OUTPUT_TABLE,
) -> Path:
    universe_rows = read_jsonl(universe_input)
    hawkes_rows = read_jsonl(hawkes_label_input)

    # event_id -> (family, event_ts)
    by_event_id: dict[str, tuple[str, str]] = {}
    for row in universe_rows:
        by_event_id[row["event_id"]] = (family_for(row), row.get("event_ts") or "")

    # cohort -> family -> {Information: int, Noise: int, abstain: int, ts_set}
    grid: dict[str, dict[str, dict[str, set | int]]] = defaultdict(
        lambda: defaultdict(lambda: {"Information": 0, "Noise": 0, "abstain": 0, "ts": set()})
    )

    for row in hawkes_rows:
        symbol = row.get("symbol") or ""
        # Map symbol to cohort label expected by the manuscript.
        if symbol == "BTCUSDT":
            cohort = "main"
        elif symbol == "ETHUSDT":
            cohort = "robustness_symbol"
        else:
            continue
        family, event_ts = by_event_id.get(row["event_id"], ("Other macro", ""))
        cell = grid[cohort][family]
        label = row.get("label")
        if label == "Information":
            cell["Information"] += 1
        elif label == "Noise":
            cell["Noise"] += 1
        else:
            cell["abstain"] += 1
        if event_ts:
            cell["ts"].add(event_ts)

    rows_out: list[dict[str, str]] = []
    for cohort in ("main", "robustness_symbol"):
        cohort_rows = grid.get(cohort, {})
        for family, cell in sorted(
            cohort_rows.items(), key=lambda kv: (-(kv[1]["Information"] + kv[1]["Noise"]), kv[0])
        ):
            total_rows = cell["Information"] + cell["Noise"] + cell["abstain"]
            unique_ts = len(cell["ts"])
            note_parts = []
            if cell["Information"] and cell["Noise"]:
                note_parts.append("mixed family")
            elif cell["Information"]:
                note_parts.append("Information-only family")
            elif cell["Noise"]:
                note_parts.append("Noise-only family")
            else:
                note_parts.append("abstain-only family")
            note_parts.append(
                f"{COHORT_TO_SYMBOL.get(cohort, cohort)} cohort"
            )
            rows_out.append(
                {
                    "cohort_name": cohort,
                    "family": family,
                    "computed_information": str(cell["Information"]),
                    "computed_noise": str(cell["Noise"]),
                    "abstain_or_other": str(cell["abstain"]),
                    "total_rows": str(total_rows),
                    "unique_event_timestamps": str(unique_ts),
                    "note": "; ".join(note_parts),
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
    parser.add_argument("--universe-input", type=Path, default=DEFAULT_UNIVERSE_INPUT)
    parser.add_argument("--hawkes-label-input", type=Path, default=DEFAULT_HAWKES_LABEL_INPUT)
    parser.add_argument("--output-table", type=Path, default=DEFAULT_OUTPUT_TABLE)
    args = parser.parse_args()
    out = build_family_aware_hawkes(
        universe_input=args.universe_input,
        hawkes_label_input=args.hawkes_label_input,
        output_table=args.output_table,
    )
    print(out)


if __name__ == "__main__":
    main()
