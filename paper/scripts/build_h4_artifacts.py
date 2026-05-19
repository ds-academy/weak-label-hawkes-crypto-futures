"""Build H4 paper-ready tables and lightweight figures from H empirical artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = REPO_ROOT / "research/2026-weak-label-exo-hawkes"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "004_paper"
DEFAULT_H_ROOT = REPO_ROOT / "data/derived/weak_label_exo_hawkes/H_v0"
DEFAULT_H1_RUN_ID = "h1_activation_gap_p99_v0_20260501"
DEFAULT_H2_RUN_ID = "h2_labeling_ppi_primary_gap_p98_v0_20260501"
DEFAULT_H3_RUN_ID = "h3_hawkes_empirical_ppi_primary_p98_smoke_v0_20260501"


@dataclass(frozen=True)
class H4Inputs:
    """Input artifact locations for the H4 paper summary builder."""

    h1_manifest: Path
    h2_manifest: Path
    h2_labels: Path
    h3_manifest: Path
    h3_comparison: Path


def default_inputs(
    h_root: Path = DEFAULT_H_ROOT,
    *,
    h1_run_id: str = DEFAULT_H1_RUN_ID,
    h2_run_id: str = DEFAULT_H2_RUN_ID,
    h3_run_id: str = DEFAULT_H3_RUN_ID,
) -> H4Inputs:
    """Return H4 input paths for a named empirical run family."""
    h1 = h_root / h1_run_id
    h2 = h_root / h2_run_id
    h3 = h_root / h3_run_id
    return H4Inputs(
        h1_manifest=h1 / "manifest.json",
        h2_manifest=h2 / "manifest.json",
        h2_labels=h2 / "event_label/event_label.jsonl",
        h3_manifest=h3 / "manifest.json",
        h3_comparison=h3 / "hawkes_comparison/hawkes_comparison.jsonl",
    )


def build_artifacts(
    *,
    inputs: H4Inputs,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Path]:
    """Build deterministic H4 tables, figures, and summary markdown."""
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    h1_manifest = read_json(inputs.h1_manifest)
    h2_manifest = read_json(inputs.h2_manifest)
    h2_labels = read_jsonl(inputs.h2_labels)
    h3_manifest = read_json(inputs.h3_manifest)
    h3_rows = read_jsonl(inputs.h3_comparison)

    table_paths = {
        "run_funnel": tables_dir / "table_1_run_funnel.csv",
        "label_status": tables_dir / "table_2_label_status_counts.csv",
        "label_confidence": tables_dir / "table_3_label_confidence_by_label.csv",
        "ppi_rr": tables_dir / "table_4_ppi_rr_matrix.csv",
        "hawkes": tables_dir / "table_5_hawkes_comparison.csv",
        "claim": tables_dir / "table_6_claim_summary.csv",
    }
    write_csv(
        table_paths["run_funnel"],
        ["stage", "metric", "value", "note"],
        run_funnel_rows(h1_manifest, h2_manifest, h3_manifest, h3_rows),
    )
    write_csv(
        table_paths["label_status"],
        ["label_status", "count", "share_of_label_input"],
        label_status_rows(h2_labels),
    )
    write_csv(
        table_paths["label_confidence"],
        ["label", "label_confidence", "count"],
        label_confidence_rows(h2_labels),
    )
    write_csv(
        table_paths["ppi_rr"],
        ["ppi_verdict", "recovery_verdict", "label_status", "label", "label_confidence", "count"],
        ppi_rr_rows(h2_labels),
    )
    write_csv(
        table_paths["hawkes"],
        [
            "cohort_name",
            "model",
            "fit_success",
            "optimizer_iterations",
            "log_likelihood_in_sample",
            "log_likelihood_oos",
            "ll_delta_vs_m0",
            "aic_delta_vs_m0",
            "bic_delta_vs_m0",
            "lrt_p_value_vs_m0",
            "beta_information",
            "beta_noise",
            "half_life_information_sec",
            "half_life_noise_sec",
        ],
        hawkes_comparison_rows(h3_rows),
    )
    write_csv(
        table_paths["claim"],
        ["claim", "evidence", "paper_framing"],
        claim_summary_rows(h2_manifest, h3_rows),
    )

    markdown_tables = output_root / "tables.md"
    write_tables_markdown(markdown_tables, table_paths)

    figure_paths = {
        "pipeline_funnel": figures_dir / "figure_1_pipeline_funnel.svg",
        "label_confidence": figures_dir / "figure_2_label_confidence_split.svg",
        "hawkes_pvalues": figures_dir / "figure_3_hawkes_m1_pvalues.svg",
    }
    figure_paths["pipeline_funnel"].write_text(
        pipeline_funnel_svg(h1_manifest, h2_manifest, h3_rows),
        encoding="utf-8",
    )
    figure_paths["label_confidence"].write_text(
        label_confidence_svg(label_confidence_rows(h2_labels)),
        encoding="utf-8",
    )
    figure_paths["hawkes_pvalues"].write_text(
        hawkes_pvalue_svg(h3_rows),
        encoding="utf-8",
    )

    summary_path = output_root / "paper_results_summary.md"
    summary_path.write_text(
        paper_summary(
            h1_manifest=h1_manifest,
            h2_manifest=h2_manifest,
            h3_manifest=h3_manifest,
            h3_rows=h3_rows,
            table_paths=table_paths,
            figure_paths=figure_paths,
        ),
        encoding="utf-8",
    )
    return {**table_paths, **figure_paths, "summary": summary_path, "tables_md": markdown_tables}


def run_funnel_rows(
    h1_manifest: Mapping[str, object],
    h2_manifest: Mapping[str, object],
    h3_manifest: Mapping[str, object],
    h3_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    h1_counts = dict(h1_manifest["row_status_counts"])  # type: ignore[arg-type]
    h2_counts = dict(h2_manifest["row_status_counts"])  # type: ignore[arg-type]
    fit_success_count = sum(1 for row in h3_rows if row.get("fit_success") is True)
    activation_input_note = (
        "primary macro_te events selected for activation, including events after market-data end"
        if h1_manifest.get("include_after_market_end")
        else "market-window primary macro_te events"
    )
    symbol_count = len(h1_manifest.get("symbols", ()))
    rule_count = len(h1_manifest.get("selected_rule_versions", ()))
    return [
        {
            "stage": "event_universe",
            "metric": "combined_event_rows",
            "value": h1_manifest["combined_event_count"],
            "note": "TE plus Binance canonical event universe before activation filtering",
        },
        {
            "stage": "activation",
            "metric": "primary_te_candidates",
            "value": h1_manifest["activation_input_count"],
            "note": activation_input_note,
        },
        {
            "stage": "activation",
            "metric": "activation_attempt_rows",
            "value": h1_manifest["attempt_count"],
            "note": (
                f"{h1_manifest['activation_input_count']} events x "
                f"{symbol_count} symbols x {rule_count} empirical activation rules"
            ),
        },
        {
            "stage": "activation",
            "metric": "computed_positive",
            "value": h1_counts.get("computed_positive", 0),
            "note": "input rows passed to empirical labeling",
        },
        {
            "stage": "activation",
            "metric": "missing_market_data",
            "value": h1_counts.get("missing_market_data", 0),
            "note": "gap-p99 still rejected these activation windows",
        },
        {
            "stage": "labeling",
            "metric": "label_rows",
            "value": h2_manifest["event_label_count"],
            "note": "label rows emitted from computed-positive activation rows",
        },
        {
            "stage": "labeling",
            "metric": "computed_labels",
            "value": h2_counts.get("computed", 0),
            "note": "PPI-primary/RR-validator computed Information or Noise",
        },
        {
            "stage": "labeling",
            "metric": "information_labels",
            "value": dict(h2_manifest["label_counts"]).get("Information", 0),  # type: ignore[arg-type]
            "note": "computed Information labels",
        },
        {
            "stage": "labeling",
            "metric": "noise_labels",
            "value": dict(h2_manifest["label_counts"]).get("Noise", 0),  # type: ignore[arg-type]
            "note": "computed Noise labels",
        },
        {
            "stage": "hawkes",
            "metric": "model_fit_rows",
            "value": h3_manifest["hawkes_comparison_count"],
            "note": "2 cohorts x M0/M1/M2",
        },
        {
            "stage": "hawkes",
            "metric": "fit_success_rows",
            "value": fit_success_count,
            "note": "selected Hawkes optimization convergence",
        },
    ]


def label_status_rows(labels: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    counts = Counter(str(row["label_status"]) for row in labels)
    total = len(labels)
    return [
        {
            "label_status": status,
            "count": count,
            "share_of_label_input": f"{count / total:.4f}",
        }
        for status, count in sorted(counts.items())
    ]


def label_confidence_rows(labels: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    counts = Counter(
        (
            str(row.get("label") or "NULL"),
            str(row.get("label_confidence") or "NULL"),
        )
        for row in labels
        if row.get("label_status") == "computed"
    )
    return [
        {"label": label, "label_confidence": confidence, "count": count}
        for (label, confidence), count in sorted(counts.items())
    ]


def ppi_rr_rows(labels: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    counts = Counter(
        (
            str(row.get("ppi_verdict") or "NULL"),
            str(row.get("recovery_verdict") or "NULL"),
            str(row.get("label_status") or "NULL"),
            str(row.get("label") or "NULL"),
            str(row.get("label_confidence") or "NULL"),
        )
        for row in labels
    )
    return [
        {
            "ppi_verdict": ppi,
            "recovery_verdict": rr,
            "label_status": status,
            "label": label,
            "label_confidence": confidence,
            "count": count,
        }
        for (ppi, rr, status, label, confidence), count in sorted(counts.items())
    ]


def hawkes_comparison_rows(
    h3_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "cohort_name": row["cohort_name"],
            "model": row["model"],
            "fit_success": fmt_number(row["fit_success"]),
            "optimizer_iterations": row["optimizer_iterations"],
            "log_likelihood_in_sample": fmt_number(row.get("log_likelihood_in_sample")),
            "log_likelihood_oos": fmt_number(row.get("log_likelihood_oos")),
            "ll_delta_vs_m0": fmt_number(row.get("ll_delta_vs_m0")),
            "aic_delta_vs_m0": fmt_number(row.get("aic_delta_vs_m0")),
            "bic_delta_vs_m0": fmt_number(row.get("bic_delta_vs_m0")),
            "lrt_p_value_vs_m0": fmt_number(row.get("lrt_p_value_vs_m0")),
            "beta_information": fmt_number(row.get("beta_information")),
            "beta_noise": fmt_number(row.get("beta_noise")),
            "half_life_information_sec": fmt_number(row.get("half_life_information_sec")),
            "half_life_noise_sec": fmt_number(row.get("half_life_noise_sec")),
        }
        for row in sorted(h3_rows, key=lambda item: (str(item["cohort_name"]), str(item["model"])))
    ]


def claim_summary_rows(
    h2_manifest: Mapping[str, object],
    h3_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, str]]:
    m1_rows = [row for row in h3_rows if row.get("model") == "M1"]
    pvalues = {
        str(row["cohort_name"]): fmt_number(row.get("lrt_p_value_vs_m0"))
        for row in m1_rows
    }
    labels = dict(h2_manifest["label_counts"])  # type: ignore[arg-type]
    return [
        {
            "claim": "Pipeline executability and paper-facing reproducibility",
            "evidence": "Versioned artifacts, digest snapshot, paper-render command, and all six Hawkes fits converged.",
            "paper_framing": "The weak-label exogenous Hawkes framework is executable on empirical BTC/ETH data; the rendered tables, figures, and PDF are reproducible from the shipped artifacts.",
        },
        {
            "claim": "Class-specific exogenous-component significance",
            "evidence": f"main M1 p={pvalues.get('main')}; robustness_symbol M1 p={pvalues.get('robustness_symbol')}",
            "paper_framing": "No statistically significant class-specific exogenous component is detected in this first-pass sample.",
        },
        {
            "claim": "Weak-label sample size",
            "evidence": f"computed labels: Information={labels.get('Information', 0)}, Noise={labels.get('Noise', 0)}",
            "paper_framing": "Power is limited; production calibration and larger event coverage are future work.",
        },
    ]


def pipeline_funnel_svg(
    h1_manifest: Mapping[str, object],
    h2_manifest: Mapping[str, object],
    h3_rows: Sequence[Mapping[str, object]],
) -> str:
    values = [
        ("TE primary events", int(h1_manifest["activation_input_count"])),
        ("Activation positive rows", int(dict(h1_manifest["row_status_counts"]).get("computed_positive", 0))),  # type: ignore[arg-type]
        ("Computed labels", int(dict(h2_manifest["row_status_counts"]).get("computed", 0))),  # type: ignore[arg-type]
        ("Information labels", int(dict(h2_manifest["label_counts"]).get("Information", 0))),  # type: ignore[arg-type]
        ("Noise labels", int(dict(h2_manifest["label_counts"]).get("Noise", 0))),  # type: ignore[arg-type]
        ("Successful Hawkes fits", sum(1 for row in h3_rows if row.get("fit_success") is True)),
    ]
    return bar_svg(
        title="Empirical Pipeline Funnel",
        rows=values,
        width=860,
        color="#2c6f7c",
        subtitle="Counts are row-level after activation/labeling rule expansion.",
    )


def label_confidence_svg(rows: Sequence[Mapping[str, object]]) -> str:
    values = [
        (f"{row['label']} / {row['label_confidence']}", int(row["count"]))
        for row in rows
    ]
    return bar_svg(
        title="Computed Label Confidence Split",
        rows=values,
        width=760,
        color="#b2693c",
        subtitle="Low confidence indicates PPI/RR disagreement preserved by the validator policy.",
    )


def hawkes_pvalue_svg(h3_rows: Sequence[Mapping[str, object]]) -> str:
    rows = [
        (f"{row['cohort_name']} M1 p-value", float(row["lrt_p_value_vs_m0"]))
        for row in h3_rows
        if row.get("model") == "M1"
    ]
    max_value = 1.0
    width = 780
    left = 260
    top = 80
    bar_max = width - left - 80
    height = top + len(rows) * 70 + 70
    parts = [
        svg_header(width, height),
        '<text x="30" y="36" class="title">M1-vs-M0 LRT p-values</text>',
        '<text x="30" y="58" class="subtitle">Both cohorts are far above the 0.05 reference line.</text>',
    ]
    threshold_x = left + 0.05 / max_value * bar_max
    parts.append(
        f'<line x1="{threshold_x:.1f}" y1="70" x2="{threshold_x:.1f}" y2="{height - 45}" stroke="#a33" stroke-dasharray="4 4"/>'
    )
    parts.append(f'<text x="{threshold_x + 6:.1f}" y="76" class="small">p=0.05</text>')
    for index, (label, value) in enumerate(rows):
        y = top + index * 70
        bar_width = value / max_value * bar_max
        parts.append(f'<text x="30" y="{y + 23}" class="label">{escape_xml(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="30" rx="4" fill="#5f6f91"/>')
        parts.append(f'<text x="{left + bar_width + 8:.1f}" y="{y + 21}" class="value">{value:.3f}</text>')
    parts.append("</svg>\n")
    return "\n".join(parts)


def bar_svg(
    *,
    title: str,
    rows: Sequence[tuple[str, int]],
    width: int,
    color: str,
    subtitle: str,
) -> str:
    left = 260
    top = 86
    bar_max = width - left - 90
    max_value = max(value for _, value in rows) or 1
    height = top + len(rows) * 58 + 55
    parts = [
        svg_header(width, height),
        f'<text x="30" y="36" class="title">{escape_xml(title)}</text>',
        f'<text x="30" y="58" class="subtitle">{escape_xml(subtitle)}</text>',
    ]
    for index, (label, value) in enumerate(rows):
        y = top + index * 58
        bar_width = value / max_value * bar_max
        parts.append(f'<text x="30" y="{y + 23}" class="label">{escape_xml(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="28" rx="4" fill="{color}"/>')
        parts.append(f'<text x="{left + bar_width + 8:.1f}" y="{y + 20}" class="value">{value}</text>')
    parts.append("</svg>\n")
    return "\n".join(parts)


def paper_summary(
    *,
    h1_manifest: Mapping[str, object],
    h2_manifest: Mapping[str, object],
    h3_manifest: Mapping[str, object],
    h3_rows: Sequence[Mapping[str, object]],
    table_paths: Mapping[str, Path],
    figure_paths: Mapping[str, Path],
) -> str:
    m1 = {str(row["cohort_name"]): row for row in h3_rows if row.get("model") == "M1"}
    main_m1 = m1.get("main", {})
    robustness_m1 = m1.get("robustness_symbol", {})
    labels = dict(h2_manifest["label_counts"])  # type: ignore[arg-type]
    return f"""# Paper Results Summary

Status: empirical framework result. The selected H3 run converges all six
Hawkes fits, but the M1-vs-M0 evidence is weak and should not be framed as a
statistically significant class-specific decay finding.

## Key Numbers

| Quantity | Value |
|---|---:|
| Primary TE activation input rows | {h1_manifest["activation_input_count"]} |
| Activation attempts | {h1_manifest["attempt_count"]} |
| Computed-positive activation rows | {dict(h1_manifest["row_status_counts"]).get("computed_positive", 0)} |
| Label rows | {h2_manifest["event_label_count"]} |
| Computed labels | {dict(h2_manifest["row_status_counts"]).get("computed", 0)} |
| Information labels | {labels.get("Information", 0)} |
| Noise labels | {labels.get("Noise", 0)} |
| Hawkes comparison rows | {h3_manifest["hawkes_comparison_count"]} |
| Hawkes fit_success rows | {sum(1 for row in h3_rows if row.get("fit_success") is True)} |

## Hawkes Interpretation

| Cohort | M1 LL delta vs M0 | M1 p-value | Framing |
|---|---:|---:|---|
| main | {fmt_number(main_m1.get("ll_delta_vs_m0"))} | {fmt_number(main_m1.get("lrt_p_value_vs_m0"))} | converged, weak evidence |
| robustness_symbol | {fmt_number(robustness_m1.get("ll_delta_vs_m0"))} | {fmt_number(robustness_m1.get("lrt_p_value_vs_m0"))} | converged, essentially flat vs M0 |

## Paper Framing

The contribution should be framed as a reproducible empirical framework for
weak-label exogenous Hawkes validation. No statistically significant
class-specific decay is detected in the current sample, so the results do not
support a strong positive claim that Information and Noise labels have
significantly different Hawkes decay dynamics.

## Generated Tables

{artifact_list(table_paths)}

## Generated Figures

{artifact_list(figure_paths)}

## Source Artifact Hashes

| Artifact | SHA-256 |
|---|---|
| H1 activation rows | {h1_manifest["event_activation_sha256"]} |
| H2 label rows | {h2_manifest["event_label_sha256"]} |
| H3 Hawkes comparison | {h3_manifest["hawkes_comparison_sha256"]} |
"""


def write_tables_markdown(path: Path, table_paths: Mapping[str, Path]) -> None:
    sections = ["# H4 Generated Tables", ""]
    for name, table_path in table_paths.items():
        rows = read_csv(table_path)
        sections.append(f"## {name.replace('_', ' ').title()}")
        sections.append("")
        sections.extend(markdown_table(rows))
        sections.append("")
    path.write_text("\n".join(sections), encoding="utf-8")


def artifact_list(paths: Mapping[str, Path]) -> str:
    rows = []
    for path in paths.values():
        try:
            display = path.relative_to(DEFAULT_OUTPUT_ROOT)
        except ValueError:
            display = path
        rows.append(f"- `{display}`")
    return "\n".join(rows)


def markdown_table(rows: Sequence[Mapping[str, str]]) -> list[str]:
    if not rows:
        return ["(empty)"]
    headers = list(rows[0])
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return out


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fmt_number(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def svg_header(width: int, height: int) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
<style>
  .title {{ font: 700 24px Georgia, serif; fill: #223; }}
  .subtitle {{ font: 14px Georgia, serif; fill: #556; }}
  .label {{ font: 14px Georgia, serif; fill: #223; }}
  .value {{ font: 700 14px Georgia, serif; fill: #223; }}
  .small {{ font: 12px Georgia, serif; fill: #733; }}
</style>
<rect width="100%" height="100%" fill="#fbf8f1"/>
'''


def escape_xml(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--h-root", type=Path, default=DEFAULT_H_ROOT)
    parser.add_argument("--h1-run-id", default=DEFAULT_H1_RUN_ID)
    parser.add_argument("--h2-run-id", default=DEFAULT_H2_RUN_ID)
    parser.add_argument("--h3-run-id", default=DEFAULT_H3_RUN_ID)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = build_artifacts(
        inputs=default_inputs(
            args.h_root,
            h1_run_id=args.h1_run_id,
            h2_run_id=args.h2_run_id,
            h3_run_id=args.h3_run_id,
        ),
        output_root=args.output_root,
    )
    print(json.dumps({key: str(path) for key, path in sorted(outputs.items())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
