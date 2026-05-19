from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_h4_artifacts.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("build_h4_artifacts", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_artifacts_from_minimal_fixture(tmp_path):
    mod = _module()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "paper"
    h1_manifest = input_dir / "h1_manifest.json"
    h2_manifest = input_dir / "h2_manifest.json"
    h2_labels = input_dir / "labels.jsonl"
    h3_manifest = input_dir / "h3_manifest.json"
    h3_comparison = input_dir / "hawkes.jsonl"

    h1_manifest.write_text(
        json.dumps(
            {
                "activation_input_count": 2,
                "attempt_count": 8,
                "combined_event_count": 5,
                "event_activation_sha256": "h1",
                "row_status_counts": {
                    "computed_positive": 3,
                    "missing_market_data": 5,
                },
            }
        ),
        encoding="utf-8",
    )
    h2_manifest.write_text(
        json.dumps(
            {
                "event_label_count": 3,
                "event_label_sha256": "h2",
                "label_counts": {"Information": 2, "Noise": 1},
                "row_status_counts": {"computed": 3},
            }
        ),
        encoding="utf-8",
    )
    h2_labels.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "label": "Information",
                    "label_confidence": "Low",
                    "label_status": "computed",
                    "ppi_verdict": "information",
                    "recovery_verdict": "noise",
                },
                {
                    "label": "Information",
                    "label_confidence": "High",
                    "label_status": "computed",
                    "ppi_verdict": "information",
                    "recovery_verdict": "information",
                },
                {
                    "label": "Noise",
                    "label_confidence": "High",
                    "label_status": "computed",
                    "ppi_verdict": "noise",
                    "recovery_verdict": "noise",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    h3_manifest.write_text(
        json.dumps(
            {
                "hawkes_comparison_count": 2,
                "hawkes_comparison_sha256": "h3",
                "maxiter": 80,
            }
        ),
        encoding="utf-8",
    )
    h3_comparison.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "aic_delta_vs_m0": 0.0,
                    "beta_information": None,
                    "beta_noise": None,
                    "bic_delta_vs_m0": 0.0,
                    "cohort_name": "main",
                    "fit_success": True,
                    "half_life_information_sec": None,
                    "half_life_noise_sec": None,
                    "ll_delta_vs_m0": 0.0,
                    "log_likelihood_in_sample": -10.0,
                    "log_likelihood_oos": -2.0,
                    "lrt_p_value_vs_m0": None,
                    "model": "M0",
                    "optimizer_iterations": 3,
                },
                {
                    "aic_delta_vs_m0": 4.0,
                    "beta_information": 0.4,
                    "beta_noise": 0.1,
                    "bic_delta_vs_m0": 10.0,
                    "cohort_name": "main",
                    "fit_success": True,
                    "half_life_information_sec": 1.7,
                    "half_life_noise_sec": 6.9,
                    "ll_delta_vs_m0": 0.5,
                    "log_likelihood_in_sample": -9.5,
                    "log_likelihood_oos": -1.9,
                    "lrt_p_value_vs_m0": 0.7,
                    "model": "M1",
                    "optimizer_iterations": 7,
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    outputs = mod.build_artifacts(
        inputs=mod.H4Inputs(
            h1_manifest=h1_manifest,
            h2_manifest=h2_manifest,
            h2_labels=h2_labels,
            h3_manifest=h3_manifest,
            h3_comparison=h3_comparison,
        ),
        output_root=output_dir,
    )

    assert outputs["summary"].exists()
    assert outputs["hawkes"].exists()
    summary = outputs["summary"].read_text(encoding="utf-8")
    assert "No statistically significant" in summary
    assert "class-specific decay is detected" in summary
    hawkes_csv = outputs["hawkes"].read_text(encoding="utf-8")
    assert "true" in hawkes_csv


def test_default_inputs_accepts_explicit_run_family(tmp_path):
    mod = _module()
    inputs = mod.default_inputs(
        tmp_path / "J0_v0",
        h1_run_id="h1_activation_j0",
        h2_run_id="h2_labeling_j0",
        h3_run_id="h3_hawkes_j0",
    )

    assert inputs.h1_manifest == tmp_path / "J0_v0/h1_activation_j0/manifest.json"
    assert inputs.h2_manifest == tmp_path / "J0_v0/h2_labeling_j0/manifest.json"
    assert inputs.h2_labels == tmp_path / "J0_v0/h2_labeling_j0/event_label/event_label.jsonl"
    assert inputs.h3_manifest == tmp_path / "J0_v0/h3_hawkes_j0/manifest.json"
    assert inputs.h3_comparison == tmp_path / "J0_v0/h3_hawkes_j0/hawkes_comparison/hawkes_comparison.jsonl"
