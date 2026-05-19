"""Build J1-4 threshold-sensitivity and event-family audit tables.

This script is intentionally safe to import before result generation. J1-4a
pre-registers the grid, materiality thresholds, and classification logic; the
default J0 output tables are generated only when the script is executed.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import product
from math import isfinite
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = REPO_ROOT / "research/2026-weak-label-exo-hawkes"

DEFAULT_LABELS = (
    REPO_ROOT
    / "data/derived/weak_label_exo_hawkes/J0_v0"
    / "h2_labeling_j0_ppi_primary_gap_p98_v0_20260503/event_label/event_label.jsonl"
)
DEFAULT_EVENTS = (
    REPO_ROOT
    / "data/derived/weak_label_exo_hawkes/J0_v0"
    / "h1_activation_j0_gap_p99_v0_20260503/event_universe/event_universe_combined.jsonl"
)
DEFAULT_THRESHOLD_OUTPUT = PROJECT_ROOT / "004_paper/j0_expansion/tables/table_9_threshold_sensitivity.csv"
DEFAULT_FAMILY_OUTPUT = PROJECT_ROOT / "004_paper/j0_expansion/tables/table_10_event_family_heterogeneity.csv"

PPI_INFORMATION_GRID = (1.5, 2.0, 2.5, 3.0)
PPI_NOISE_GRID = (0.50, 0.75, 1.00)
RECOVERY_INFORMATION_GRID = (1.15, 1.25, 1.35)
RECOVERY_NOISE_GRID = (1.02, 1.05, 1.10)

BASELINE_GRID_POINT = (2.0, 0.75, 1.25, 1.05)
MAX_COMPUTED_RELATIVE_CHANGE = 0.25
MAX_INFORMATION_SHARE_SHIFT = 0.15
MAX_BASELINE_COMPUTED_FLIP_RATE = 0.20
STABLE_GRID_POINT_SHARE = 0.80
MIN_EVENT_FAMILY_COUNT = 3

THRESHOLD_FIELDNAMES = [
    "grid_id",
    "ppi_information_threshold_z",
    "ppi_noise_threshold_z",
    "recovery_information_min_ratio",
    "recovery_noise_max_ratio",
    "valid_scored_rows",
    "baseline_computed_count",
    "computed_count",
    "information_count",
    "noise_count",
    "high_confidence_count",
    "low_confidence_count",
    "computed_relative_change",
    "information_share",
    "information_share_shift",
    "baseline_computed_flip_count",
    "baseline_computed_flip_rate",
    "gridpoint_indicator",
    "overall_scenario_indicator",
]

FAMILY_FIELDNAMES = [
    "event_family",
    "scored_rows",
    "computed_count",
    "information_count",
    "noise_count",
    "low_confidence_count",
    "ppi_rr_disagreement_count",
    "abstain_or_insufficient_count",
    "interpretation_status",
]


@dataclass(frozen=True)
class ThresholdGridPoint:
    """One pre-registered J1-4 threshold variant."""

    ppi_information_threshold_z: float
    ppi_noise_threshold_z: float
    recovery_information_min_ratio: float
    recovery_noise_max_ratio: float

    @property
    def grid_id(self) -> str:
        return (
            f"ppi_i{self.ppi_information_threshold_z:g}_"
            f"ppi_n{self.ppi_noise_threshold_z:g}_"
            f"rr_i{self.recovery_information_min_ratio:g}_"
            f"rr_n{self.recovery_noise_max_ratio:g}"
        )


@dataclass(frozen=True)
class ClassifiedLabel:
    """Label state after applying one threshold-grid point."""

    status: str
    label: str | None
    confidence: str | None
    ppi_verdict: str
    recovery_verdict: str


def build_j1_threshold_sensitivity(
    *,
    labels_path: Path = DEFAULT_LABELS,
    events_path: Path = DEFAULT_EVENTS,
    threshold_output: Path = DEFAULT_THRESHOLD_OUTPUT,
    family_output: Path = DEFAULT_FAMILY_OUTPUT,
    grid_points: Sequence[ThresholdGridPoint] | None = None,
) -> tuple[Path, Path]:
    """Build deterministic J1-4 audit tables."""
    labels = read_jsonl(labels_path)
    events = read_jsonl(events_path)
    grid_points = tuple(grid_points or pre_registered_grid())

    scored = scored_label_rows(labels)
    threshold_rows = threshold_sensitivity_rows(scored, grid_points=grid_points)
    family_rows = event_family_rows(scored, events)

    threshold_output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(threshold_output, THRESHOLD_FIELDNAMES, threshold_rows)
    write_csv(family_output, FAMILY_FIELDNAMES, family_rows)
    return threshold_output, family_output


def pre_registered_grid() -> list[ThresholdGridPoint]:
    """Return all valid pre-registered J1-4 threshold cross-products."""
    grid: list[ThresholdGridPoint] = []
    for ppi_info, ppi_noise, rr_info, rr_noise in product(
        PPI_INFORMATION_GRID,
        PPI_NOISE_GRID,
        RECOVERY_INFORMATION_GRID,
        RECOVERY_NOISE_GRID,
    ):
        if ppi_info <= ppi_noise:
            continue
        if rr_info <= rr_noise:
            continue
        grid.append(ThresholdGridPoint(ppi_info, ppi_noise, rr_info, rr_noise))
    return grid


def threshold_sensitivity_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    grid_points: Sequence[ThresholdGridPoint],
) -> list[dict[str, str]]:
    """Return per-grid sensitivity rows and the pre-registered scenario."""
    if not rows:
        return [
            empty_threshold_row(
                grid_point=grid_point,
                index=index,
                scenario="I_threshold_audit_failed",
                gridpoint_indicator="missing_scored_rows",
            )
            for index, grid_point in enumerate(grid_points, start=1)
        ]

    baseline_point = ThresholdGridPoint(*BASELINE_GRID_POINT)
    baseline = [classify_label(row, baseline_point) for row in rows]
    baseline_computed_count = sum(item.status == "computed" for item in baseline)
    baseline_information_share = information_share(baseline)
    if baseline_computed_count == 0:
        scenario = "I_threshold_audit_failed"
    else:
        scenario = "pending"

    output: list[dict[str, str]] = []
    stable_count = 0
    for index, grid_point in enumerate(grid_points, start=1):
        classified = [classify_label(row, grid_point) for row in rows]
        metrics = sensitivity_metrics(
            baseline=baseline,
            classified=classified,
            baseline_computed_count=baseline_computed_count,
            baseline_information_share=baseline_information_share,
        )
        indicator = gridpoint_indicator(metrics)
        if indicator == "stable_grid_point":
            stable_count += 1
        output.append(threshold_row(index, grid_point, len(rows), metrics, indicator))

    if scenario == "pending":
        scenario = overall_threshold_scenario(stable_count=stable_count, total_count=len(output))
    return [{**row, "overall_scenario_indicator": scenario} for row in output]


def sensitivity_metrics(
    *,
    baseline: Sequence[ClassifiedLabel],
    classified: Sequence[ClassifiedLabel],
    baseline_computed_count: int,
    baseline_information_share: float | None,
) -> dict[str, float | int | None]:
    computed_count = sum(item.status == "computed" for item in classified)
    information_count = sum(item.status == "computed" and item.label == "Information" for item in classified)
    noise_count = sum(item.status == "computed" and item.label == "Noise" for item in classified)
    high_confidence_count = sum(
        item.status == "computed" and item.confidence == "High" for item in classified
    )
    low_confidence_count = sum(
        item.status == "computed" and item.confidence == "Low" for item in classified
    )
    computed_relative_change = safe_ratio(
        computed_count - baseline_computed_count,
        baseline_computed_count,
    )
    info_share = safe_ratio(information_count, computed_count)
    info_share_shift = None
    if info_share is not None and baseline_information_share is not None:
        info_share_shift = info_share - baseline_information_share
    flip_count = baseline_computed_flip_count(baseline, classified)
    flip_rate = safe_ratio(flip_count, baseline_computed_count)
    return {
        "baseline_computed_count": baseline_computed_count,
        "computed_count": computed_count,
        "information_count": information_count,
        "noise_count": noise_count,
        "high_confidence_count": high_confidence_count,
        "low_confidence_count": low_confidence_count,
        "computed_relative_change": computed_relative_change,
        "information_share": info_share,
        "information_share_shift": info_share_shift,
        "baseline_computed_flip_count": flip_count,
        "baseline_computed_flip_rate": flip_rate,
    }


def classify_label(row: Mapping[str, object], grid_point: ThresholdGridPoint) -> ClassifiedLabel:
    """Classify one fixed-surface label row under one threshold variant."""
    ppi_score = finite_float(row.get("ppi_score"))
    recovery_score = finite_float(row.get("recovery_score"))
    if ppi_score is None or recovery_score is None:
        return ClassifiedLabel("insufficient_horizon_data", None, None, "missing", "missing")

    ppi_verdict = ppi_threshold_verdict(
        ppi_score,
        information_threshold=grid_point.ppi_information_threshold_z,
        noise_threshold=grid_point.ppi_noise_threshold_z,
    )
    recovery_verdict = recovery_threshold_verdict(
        recovery_score,
        information_min_ratio=grid_point.recovery_information_min_ratio,
        noise_max_ratio=grid_point.recovery_noise_max_ratio,
    )
    if ppi_verdict == "undetermined":
        return ClassifiedLabel("abstain_ambiguous", None, None, ppi_verdict, recovery_verdict)
    label = "Information" if ppi_verdict == "information" else "Noise"
    confidence = "High" if ppi_verdict == recovery_verdict else "Low"
    return ClassifiedLabel("computed", label, confidence, ppi_verdict, recovery_verdict)


def ppi_threshold_verdict(
    score: float,
    *,
    information_threshold: float,
    noise_threshold: float,
) -> str:
    if score >= information_threshold:
        return "information"
    if score <= noise_threshold:
        return "noise"
    return "undetermined"


def recovery_threshold_verdict(
    score: float,
    *,
    information_min_ratio: float,
    noise_max_ratio: float,
) -> str:
    if score >= information_min_ratio:
        return "information"
    if score <= noise_max_ratio:
        return "noise"
    return "undetermined"


def baseline_computed_flip_count(
    baseline: Sequence[ClassifiedLabel],
    classified: Sequence[ClassifiedLabel],
) -> int:
    flips = 0
    for before, after in zip(baseline, classified, strict=True):
        if before.status != "computed":
            continue
        if after.status != "computed" or before.label != after.label:
            flips += 1
    return flips


def gridpoint_indicator(metrics: Mapping[str, float | int | None]) -> str:
    values = (
        abs_float(metrics["computed_relative_change"]),
        abs_float(metrics["information_share_shift"]),
        abs_float(metrics["baseline_computed_flip_rate"]),
    )
    if any(value is None for value in values):
        return "failed_grid_point"
    if (
        values[0] <= MAX_COMPUTED_RELATIVE_CHANGE
        and values[1] <= MAX_INFORMATION_SHARE_SHIFT
        and values[2] <= MAX_BASELINE_COMPUTED_FLIP_RATE
    ):
        return "stable_grid_point"
    return "sensitive_grid_point"


def overall_threshold_scenario(*, stable_count: int, total_count: int) -> str:
    if total_count == 0:
        return "I_threshold_audit_failed"
    stable_share = stable_count / total_count
    if stable_share >= STABLE_GRID_POINT_SHARE:
        return "G_threshold_limited_sensitivity"
    return "H_threshold_material_sensitivity"


def event_family_rows(
    labels: Sequence[Mapping[str, object]],
    events: Sequence[Mapping[str, object]],
) -> list[dict[str, str]]:
    """Return descriptive event-family heterogeneity rows."""
    events_by_id = {str(row["event_id"]): row for row in events}
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    for row in labels:
        event = events_by_id.get(str(row.get("event_id")))
        family = classify_event_family(event or {})
        classified = classify_label(row, ThresholdGridPoint(*BASELINE_GRID_POINT))
        key = family
        counters[key]["scored_rows"] += 1
        if classified.status == "computed":
            counters[key]["computed_count"] += 1
            counters[key][f"{classified.label}_count"] += 1
            if classified.confidence == "Low":
                counters[key]["low_confidence_count"] += 1
                counters[key]["ppi_rr_disagreement_count"] += 1
        else:
            counters[key]["abstain_or_insufficient_count"] += 1

    output: list[dict[str, str]] = []
    for family in sorted(counters):
        counts = counters[family]
        scored_rows = counts["scored_rows"]
        output.append(
            {
                "event_family": family,
                "scored_rows": str(scored_rows),
                "computed_count": str(counts["computed_count"]),
                "information_count": str(counts["Information_count"]),
                "noise_count": str(counts["Noise_count"]),
                "low_confidence_count": str(counts["low_confidence_count"]),
                "ppi_rr_disagreement_count": str(counts["ppi_rr_disagreement_count"]),
                "abstain_or_insufficient_count": str(counts["abstain_or_insufficient_count"]),
                "interpretation_status": (
                    "interpretable_family"
                    if scored_rows >= MIN_EVENT_FAMILY_COUNT
                    else "too_sparse_for_interpretation"
                ),
            }
        )
    return output


def classify_event_family(event: Mapping[str, object]) -> str:
    """Classify TE event text into the fixed J1-4 family map."""
    raw_payload = event.get("raw_payload")
    te_row = raw_payload.get("te_row", {}) if isinstance(raw_payload, Mapping) else {}
    text_parts = [
        str(te_row.get("Category", "")),
        str(te_row.get("Event", "")),
        str(event.get("title", "")),
    ]
    text = " ".join(text_parts).lower()
    if any(token in text for token in ("consumer price", "cpi", "inflation rate")):
        return "cpi_inflation"
    if any(token in text for token in ("producer price", "ppi")):
        return "ppi"
    if "pce" in text or "personal consumption expenditure" in text:
        return "pce"
    if any(
        token in text
        for token in (
            "payroll",
            "unemployment",
            "jobless",
            "employment",
            "average hourly",
            "non farm",
            "nonfarm",
        )
    ):
        return "labor"
    if any(token in text for token in ("fomc", "fed interest", "federal funds", "interest rate")):
        return "fomc_rates"
    if "gdp" in text or "gross domestic product" in text:
        return "gdp"
    if "retail sales" in text:
        return "retail_sales"
    if "ism" in text or "pmi" in text:
        return "ism_pmi"
    return "other_macro"


def scored_label_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Keep rows with enough stored score surface for threshold reclassification."""
    scored: list[dict[str, object]] = []
    for row in rows:
        if finite_float(row.get("ppi_score")) is None:
            continue
        if finite_float(row.get("recovery_score")) is None:
            continue
        scored.append(dict(row))
    return scored


def threshold_row(
    index: int,
    grid_point: ThresholdGridPoint,
    valid_scored_rows: int,
    metrics: Mapping[str, float | int | None],
    indicator: str,
) -> dict[str, str]:
    return {
        "grid_id": f"{index:03d}_{grid_point.grid_id}",
        "ppi_information_threshold_z": fmt_number(grid_point.ppi_information_threshold_z),
        "ppi_noise_threshold_z": fmt_number(grid_point.ppi_noise_threshold_z),
        "recovery_information_min_ratio": fmt_number(grid_point.recovery_information_min_ratio),
        "recovery_noise_max_ratio": fmt_number(grid_point.recovery_noise_max_ratio),
        "valid_scored_rows": str(valid_scored_rows),
        "baseline_computed_count": str(int(metrics.get("baseline_computed_count", 0) or 0)),
        "computed_count": str(int(metrics["computed_count"] or 0)),
        "information_count": str(int(metrics["information_count"] or 0)),
        "noise_count": str(int(metrics["noise_count"] or 0)),
        "high_confidence_count": str(int(metrics["high_confidence_count"] or 0)),
        "low_confidence_count": str(int(metrics["low_confidence_count"] or 0)),
        "computed_relative_change": fmt_optional_number(metrics["computed_relative_change"]),
        "information_share": fmt_optional_number(metrics["information_share"]),
        "information_share_shift": fmt_optional_number(metrics["information_share_shift"]),
        "baseline_computed_flip_count": str(int(metrics["baseline_computed_flip_count"] or 0)),
        "baseline_computed_flip_rate": fmt_optional_number(metrics["baseline_computed_flip_rate"]),
        "gridpoint_indicator": indicator,
        "overall_scenario_indicator": "pending",
    }


def empty_threshold_row(
    *,
    grid_point: ThresholdGridPoint,
    index: int,
    scenario: str,
    gridpoint_indicator: str,
) -> dict[str, str]:
    metrics = {
        "computed_count": 0,
        "information_count": 0,
        "noise_count": 0,
        "high_confidence_count": 0,
        "low_confidence_count": 0,
        "computed_relative_change": None,
        "information_share": None,
        "information_share_shift": None,
        "baseline_computed_flip_count": 0,
        "baseline_computed_flip_rate": None,
    }
    row = threshold_row(index, grid_point, 0, metrics, gridpoint_indicator)
    row["overall_scenario_indicator"] = scenario
    return row


def information_share(classified: Sequence[ClassifiedLabel]) -> float | None:
    computed = sum(item.status == "computed" for item in classified)
    information = sum(item.status == "computed" and item.label == "Information" for item in classified)
    return safe_ratio(information, computed)


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def finite_float(value: object) -> float | None:
    if value in (None, "", "NULL"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if isfinite(parsed) else None


def abs_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    return abs(float(value))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt_number(value: float) -> str:
    return f"{value:.6f}"


def fmt_optional_number(value: float | int | None) -> str:
    if value is None:
        return "NULL"
    return fmt_number(float(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--threshold-output", type=Path, default=DEFAULT_THRESHOLD_OUTPUT)
    parser.add_argument("--family-output", type=Path, default=DEFAULT_FAMILY_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    threshold_path, family_path = build_j1_threshold_sensitivity(
        labels_path=args.labels,
        events_path=args.events,
        threshold_output=args.threshold_output,
        family_output=args.family_output,
    )
    print(threshold_path)
    print(family_path)


if __name__ == "__main__":
    main()
