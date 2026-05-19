from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_j1_oos_audit.py"


def _module():
    spec = importlib.util.spec_from_file_location("build_j1_oos_audit", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_scenario_classification_uses_pre_registered_epsilon():
    mod = _module()

    assert mod.scenario_indicator("M1", 5.1, 5.0) == "A_m1_materially_better_oos"
    assert mod.scenario_indicator("M1", 5.0, 5.0) == "B_m1_similar_oos"
    assert mod.scenario_indicator("M1", -5.0, 5.0) == "B_m1_similar_oos"
    assert mod.scenario_indicator("M1", -5.1, 5.0) == "C_m1_materially_worse_oos"
    assert mod.scenario_indicator("M0", 0.0, 5.0) == "baseline_m0"
    assert mod.scenario_indicator("M2", 4.0, 5.0) == "ablation_check_m2"


def test_build_oos_audit_writes_expected_schema(tmp_path):
    mod = _module()
    input_table = tmp_path / "table_5_hawkes_comparison.csv"
    output_table = tmp_path / "table_7_oos_audit.csv"
    manifest = tmp_path / "manifest.json"

    _write_csv(
        input_table,
        [
            {
                "cohort_name": "main",
                "model": "M0",
                "log_likelihood_in_sample": "-100.0",
                "log_likelihood_oos": "-20.0",
                "lrt_p_value_vs_m0": "NULL",
            },
            {
                "cohort_name": "main",
                "model": "M1",
                "log_likelihood_in_sample": "-98.0",
                "log_likelihood_oos": "-14.8",
                "lrt_p_value_vs_m0": "0.4",
            },
            {
                "cohort_name": "main",
                "model": "M2",
                "log_likelihood_in_sample": "-99.0",
                "log_likelihood_oos": "-18.0",
                "lrt_p_value_vs_m0": "NULL",
            },
            {
                "cohort_name": "robustness_symbol",
                "model": "M0",
                "log_likelihood_in_sample": "-80.0",
                "log_likelihood_oos": "-30.0",
                "lrt_p_value_vs_m0": "NULL",
            },
            {
                "cohort_name": "robustness_symbol",
                "model": "M1",
                "log_likelihood_in_sample": "-79.0",
                "log_likelihood_oos": "-35.2",
                "lrt_p_value_vs_m0": "0.8",
            },
            {
                "cohort_name": "robustness_symbol",
                "model": "M2",
                "log_likelihood_in_sample": "-79.5",
                "log_likelihood_oos": "-29.0",
                "lrt_p_value_vs_m0": "NULL",
            },
        ],
    )
    manifest.write_text(
        json.dumps(
            {
                "observation_summaries": {
                    "BTCUSDT:12": {"train_exogenous_count": 10, "test_exogenous_count": 2},
                    "ETHUSDT:11": {"train_exogenous_count": 9, "test_exogenous_count": 2},
                }
            }
        ),
        encoding="utf-8",
    )

    result = mod.build_oos_audit(
        input_table=input_table,
        output_table=output_table,
        h3_manifest=manifest,
        epsilon_nats=5.0,
    )

    rows = list(csv.DictReader(result.open("r", encoding="utf-8", newline="")))
    assert list(rows[0]) == mod.FIELDNAMES
    main_m1 = _find(rows, "main", "M1")
    robust_m1 = _find(rows, "robustness_symbol", "M1")
    main_m2 = _find(rows, "main", "M2")

    assert main_m1["ll_oos_delta_vs_m0"] == "5.200000"
    assert main_m1["scenario_indicator"] == "A_m1_materially_better_oos"
    assert main_m1["n_events_in_sample"] == "10"
    assert main_m1["n_events_oos"] == "2"
    assert robust_m1["ll_oos_delta_vs_m0"] == "-5.200000"
    assert robust_m1["scenario_indicator"] == "C_m1_materially_worse_oos"
    assert main_m2["scenario_indicator"] == "ablation_check_m2"
    assert main_m2["oos_rank"] == "2"


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _find(rows: list[dict[str, str]], cohort: str, model: str) -> dict[str, str]:
    return next(row for row in rows if row["cohort_name"] == cohort and row["model"] == model)
