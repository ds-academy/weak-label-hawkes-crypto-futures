from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_j1_threshold_sensitivity.py"


def _module():
    spec = importlib.util.spec_from_file_location("build_j1_threshold_sensitivity", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pre_registered_grid_size_and_fixed_baseline():
    mod = _module()
    grid = mod.pre_registered_grid()

    assert len(grid) == 108
    assert mod.BASELINE_GRID_POINT == (2.0, 0.75, 1.25, 1.05)
    assert any(
        (
            item.ppi_information_threshold_z,
            item.ppi_noise_threshold_z,
            item.recovery_information_min_ratio,
            item.recovery_noise_max_ratio,
        )
        == mod.BASELINE_GRID_POINT
        for item in grid
    )


def test_threshold_classification_uses_ppi_primary_rr_validator_policy():
    mod = _module()
    point = mod.ThresholdGridPoint(2.0, 0.75, 1.25, 1.05)

    information_low = mod.classify_label(_label("e1", ppi_score=2.5, recovery_score=1.0), point)
    noise_low = mod.classify_label(_label("e2", ppi_score=0.5, recovery_score=1.3), point)
    ambiguous = mod.classify_label(_label("e3", ppi_score=1.2, recovery_score=1.3), point)

    assert information_low.status == "computed"
    assert information_low.label == "Information"
    assert information_low.confidence == "Low"
    assert noise_low.status == "computed"
    assert noise_low.label == "Noise"
    assert noise_low.confidence == "Low"
    assert ambiguous.status == "abstain_ambiguous"
    assert ambiguous.label is None


def test_event_family_classifier_uses_fixed_mapping():
    mod = _module()

    assert mod.classify_event_family(_event("a", category="Consumer Price Index", event="CPI YoY")) == "cpi_inflation"
    assert mod.classify_event_family(_event("b", category="Non Farm Payrolls", event="Payrolls")) == "labor"
    assert mod.classify_event_family(_event("c", category="Interest Rate", event="Fed Interest Rate Decision")) == "fomc_rates"
    assert mod.classify_event_family(_event("d", category="Business Confidence", event="Survey")) == "other_macro"


def test_build_threshold_sensitivity_writes_expected_schema(tmp_path):
    mod = _module()
    labels = tmp_path / "labels.jsonl"
    events = tmp_path / "events.jsonl"
    threshold_output = tmp_path / "table_9_threshold_sensitivity.csv"
    family_output = tmp_path / "table_10_event_family_heterogeneity.csv"
    baseline = mod.ThresholdGridPoint(*mod.BASELINE_GRID_POINT)
    restrictive = mod.ThresholdGridPoint(3.0, 0.50, 1.35, 1.02)

    labels.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                _label("e1", ppi_score=2.4, recovery_score=1.0, symbol="BTCUSDT"),
                _label("e2", ppi_score=0.5, recovery_score=1.3, symbol="BTCUSDT"),
                _label("e3", ppi_score=2.2, recovery_score=1.0, symbol="ETHUSDT"),
                _label("e4", ppi_score=1.0, recovery_score=1.0, symbol="ETHUSDT"),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    events.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                _event("e1", category="Consumer Price Index", event="CPI YoY"),
                _event("e2", category="Consumer Price Index", event="CPI MoM"),
                _event("e3", category="Consumer Price Index", event="Core CPI"),
                _event("e4", category="Non Farm Payrolls", event="Payrolls"),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    threshold_path, family_path = mod.build_j1_threshold_sensitivity(
        labels_path=labels,
        events_path=events,
        threshold_output=threshold_output,
        family_output=family_output,
        grid_points=(baseline, restrictive),
    )

    threshold_rows = list(csv.DictReader(threshold_path.open("r", encoding="utf-8", newline="")))
    family_rows = list(csv.DictReader(family_path.open("r", encoding="utf-8", newline="")))

    assert list(threshold_rows[0]) == mod.THRESHOLD_FIELDNAMES
    assert threshold_rows[0]["baseline_computed_count"] == "3"
    assert threshold_rows[0]["computed_count"] == "3"
    assert threshold_rows[0]["gridpoint_indicator"] == "stable_grid_point"
    assert threshold_rows[1]["gridpoint_indicator"] == "sensitive_grid_point"
    assert threshold_rows[0]["overall_scenario_indicator"] == "H_threshold_material_sensitivity"
    assert list(family_rows[0]) == mod.FAMILY_FIELDNAMES
    cpi = next(row for row in family_rows if row["event_family"] == "cpi_inflation")
    assert cpi["scored_rows"] == "3"
    assert cpi["interpretation_status"] == "interpretable_family"


def test_threshold_sensitivity_reports_scenario_g_when_grid_is_stable():
    mod = _module()
    baseline = mod.ThresholdGridPoint(*mod.BASELINE_GRID_POINT)
    rows = [_label(f"info-{index}", ppi_score=4.0, recovery_score=2.0) for index in range(10)]

    output = mod.threshold_sensitivity_rows(rows, grid_points=(baseline,))

    assert output[0]["gridpoint_indicator"] == "stable_grid_point"
    assert output[0]["overall_scenario_indicator"] == "G_threshold_limited_sensitivity"


def test_threshold_sensitivity_reports_scenario_i_when_scores_are_missing():
    mod = _module()
    baseline = mod.ThresholdGridPoint(*mod.BASELINE_GRID_POINT)

    output = mod.threshold_sensitivity_rows([], grid_points=(baseline,))

    assert output[0]["valid_scored_rows"] == "0"
    assert output[0]["gridpoint_indicator"] == "missing_scored_rows"
    assert output[0]["overall_scenario_indicator"] == "I_threshold_audit_failed"


def _label(
    event_id: str,
    *,
    ppi_score: float,
    recovery_score: float,
    symbol: str = "BTCUSDT",
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "symbol": symbol,
        "ppi_score": ppi_score,
        "recovery_score": recovery_score,
    }


def _event(event_id: str, *, category: str, event: str) -> dict[str, object]:
    return {
        "event_id": event_id,
        "raw_payload": {
            "te_row": {
                "Category": category,
                "Event": event,
            }
        },
        "title": f"United States - {event}",
    }
