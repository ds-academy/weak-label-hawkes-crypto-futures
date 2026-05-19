from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent
DEFAULT_SOURCE_TEX = SCRIPT_DIR / "manuscript.tex"
DEFAULT_SOURCE_BIB = SCRIPT_DIR / "references.bib"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "arxiv"
DEFAULT_MAIN_TEX = DEFAULT_OUTPUT_DIR / "main.tex"
DEFAULT_BIB = DEFAULT_OUTPUT_DIR / "references.bib"
DEFAULT_TABLES_DIR = PAPER_DIR / "j0_expansion" / "tables"
DEFAULT_SOURCE_FIGURES_DIR = PAPER_DIR / "j0_expansion" / "figures"
DEFAULT_RENDERED_FIGURES_DIRNAME = "figures"
PDF_METADATA = {
    "Creator": "future_polios build_arxiv_package.py",
    "CreationDate": datetime(2026, 5, 5, tzinfo=timezone.utc),
    "ModDate": datetime(2026, 5, 5, tzinfo=timezone.utc),
}


LATEX_REPLACEMENTS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
}


TABLE_CAPTIONS = {
    1: "Run funnel from candidate events through Hawkes fit rows. Each stage carries its own unit; counts are not directly comparable across stages.",
    2: "Label status counts on the expanded-universe diagnostic surface.",
    3: "Computed label confidence by Information(schema)/Noise(schema) class.",
    4: "PPI verdict by recovery-ratio verdict and label status. All nine cells of the three-by-three finite-verdict grid are shown; cells with count zero indicate PPI/RR verdict combinations the empirical sample never produced. The 12 activation-positive rows with NULL PPI/RR verdicts (4 confounded and 8 insufficient-horizon rows in Table 2) do not appear in this finite-verdict grid by construction; the 2 ambiguous rows have finite PPI/RR verdicts and appear as Undetermined/Noise.",
    5: "Hawkes M0/M1/M2 comparison for the main and robustness cohorts. $\\beta_{I}$ and $\\beta_{N}$ are fitted exogenous decay rates (per second) for Information(schema) and Noise(schema) impulses; $T^{1/2}_{I}$ and $T^{1/2}_{N}$ are the corresponding half-lives in seconds. BIC delta and Fit columns expose the BIC contrast against M0 and the optimizer convergence flag respectively. BIC uses the stitched in-sample point-process event count ($n=1{,}920$ for BTCUSDT main and $n=1{,}760$ for ETHUSDT robustness; see Section 4), not the smaller release-batch cluster count used for inferential calibration. M0 is the pooled-exogenous baseline, M1 is the class-specific exogenous test, and M2 is the Information(schema)-only ablation; see Section 4 for the model definitions and Section 6.2 for the standard-error caveat. Estimates are point values. The LRT p-value column reports a nominal $\\chi^2$ diagnostic with df = 3; it is not cluster-calibrated, since the cohort-level effective sample is 4 release-batch clusters (BTCUSDT main) and 3 (ETHUSDT robustness).",
    6: "Claim summary derived from the empirical evidence and paper framing.",
    7: "Out-of-sample audit on two held-out events per cohort, classified against the pre-specified 5.0-nat materiality threshold. M2 rows are reported alongside M0 and M1 to make the OOS rank visible. Because $n_{\\text{test}} = 2$, the classification should be read as a threshold-crossing small-sample materiality check; see Table 11 for the illustrative threshold-crossing limitation.",
    8: "Release-batch dependence audit and the reason per-cluster likelihood contributions are not exported by the current Hawkes-fit output.",
    9: "Fixed-surface threshold-sensitivity summary over the 108-point grid. The 48 valid scored rows comprise the 46 baseline computed labels plus the 2 finite-score PPI/RR-ambiguous rows from Table 4 that the grid can reclassify.",
    10: "Descriptive event-family heterogeneity on the fixed scored-row surface. The Abstain column reports rows that did not produce a computed label.",
    11: "Illustrative threshold-crossing simulation (not a formal power analysis) for the 5.0-nat OOS materiality threshold. The per-event noise scale is heuristically derived from the in-sample LL delta and $\\sqrt{n_{\\text{train}}}$; the table reports the fraction of replicates that cross the threshold for selected true effect sizes. $\\sigma_{\\text{total}}$ is the simulated OOS noise scale at $n_{\\text{test}} = 2$.",
    12: "Decomposition of the 41 candidates added by the expanded universe (95 minus the baseline 54). Family is inferred from the canonical event path; Window indicates whether the event timestamp falls inside the BTC/ETH market-data interval; Activation outcome is the worst-case status across the four (symbol, rule) attempts.",
    13: "Label-policy ablation on the 46 computed labels. The strict-agreement and low-confidence-excluded variants coincide on this surface; both reduce the Information(schema) class to its four High-confidence rows. A Hawkes refit on the resulting 16-label-row / 8-deduplicated-event subset is part of the Section 7 expansion.",
    14: "Family-aware composition of the cohort label surface used to construct the Hawkes-fit input. Abstain rows are shown for accounting only and are not used as exogenous Hawkes impulses. \"Unique event ts\" reports the number of distinct event timestamps within each (cohort, family) cell. A family-fixed-effect Hawkes refit is part of the Section 7 expansion.",
    "5c": "Endogenous-only baseline diagnostic. $M_{\\mathrm{endog}}$ ($x_q(t)\\equiv 0$, ten parameters) is fit under the same log-space bounds $[-24, 8]$ used by the K17 H3 optimizer. For Table 15 only, $\\Delta\\mathrm{IC} = \\mathrm{IC}(M_0) - \\mathrm{IC}(M_{\\mathrm{endog}})$; positive values favor $M_{\\mathrm{endog}}$ and negative values favor $M_0$. Table 5 reports candidate $-$ $M_0$ deltas.",
    "5d": "Endogenous branching spectral radius for each fit, including the K17-frozen M0/M1/M2 specifications and the K18 endogenous-only diagnostic ($M_{\\mathrm{endog}}$, K17-bounded). All values are below 1.0; every fit is subcritical.",
}


TABLE_COLUMN_LABELS = {
    1: ("Stage", "Unit", "Count", "Note"),
    2: ("Label status", "Count", "Share"),
    3: ("Label", "Confidence", "Count"),
    4: ("PPI verdict", "RR verdict", "Label status", "Label", "Confidence", "Count"),
    5: (
        "Cohort",
        "Model",
        "LL delta",
        "AIC delta",
        "BIC delta",
        "LRT p",
        r"$\beta_{I}$",
        r"$\beta_{N}$",
        r"$T^{1/2}_{I}$",
        r"$T^{1/2}_{N}$",
        "Fit",
    ),
    6: ("Claim", "Evidence", "Paper framing"),
    7: ("Cohort", "Model", "OOS LL delta", "Rank", "OOS n", "Audit result"),
    8: ("Cohort", "Clusters", "Max size", "Original p", "Audit result", "Reason"),
    9: ("Metric", "Value"),
    10: ("Family", "Rows", "Info", "Noise", "Low-conf", "PPI/RR disagree", "Abstain", "Status"),
    11: (
        "Cohort",
        r"$\Delta_{\text{LL}}^{\text{true}}$",
        "Threshold-crossing rate",
        "MC SE",
        r"$\sigma_{\text{total}}$",
    ),
    12: ("Category", "Subcategory", "Count", "Share"),
    13: (
        "Policy variant",
        "Info",
        "Noise",
        "Total",
        "Info share",
        "Dedup events",
        "Hawkes refit",
    ),
    14: (
        "Cohort",
        "Family",
        "Info",
        "Noise",
        "Abstain",
        "Total",
        "Unique event ts",
        "Note",
    ),
    "5c": (
        "Cohort",
        "Model",
        "LL endog",
        "LL M0",
        "LL gap",
        r"$\Delta\mathrm{AIC}(M_0-M_e)$",
        "AIC fav.",
        r"$\Delta\mathrm{BIC}(M_0-M_e)$",
        "BIC fav.",
        r"$\rho$",
    ),
    "5d": (
        "Cohort",
        "Model",
        r"$\rho$",
        "Stationary",
        "Source",
    ),
}

TABLE_ALIGNMENTS = {
    1: "l l r Y",
    2: "l r r",
    3: "l l r",
    4: "l l l l l r",
    5: "l l r r r r r r r r l",
    6: "l Y Y",
    7: "l l r r r l",
    8: "l r r r l Y",
    9: "l r",
    10: "l r r r r r r l",
    11: "l r r r r",
    12: "l l r r",
    13: "l r r r r r l",
    14: "l l r r r r r Y",
    "5c": "l l r r r r l r l r",
    "5d": "l l r l Y",
}

TABLE_FONT_SIZES = {
    1: r"\small",
    2: r"\small",
    3: r"\small",
    4: r"\footnotesize",
    5: r"\footnotesize",
    6: r"\footnotesize",
    7: r"\footnotesize",
    8: r"\footnotesize",
    9: r"\small",
    10: r"\footnotesize",
    11: r"\footnotesize",
    12: r"\footnotesize",
    13: r"\footnotesize",
    14: r"\footnotesize",
    "5c": r"\scriptsize",
    "5d": r"\footnotesize",
}


FIGURE_CAPTIONS = {
    1: "Empirical pipeline funnel from primary macro candidates through successful Hawkes fits.",
    2: "Computed label confidence split under the PPI-primary/RR-validator policy.",
    3: "Nominal M1-vs-M0 p-values for the main and robustness cohorts. Values are shown only as diagnostic summaries of Table 5; no calibrated rejection threshold is plotted because cluster-calibrated thresholds are unavailable at the current cohort-level effective sample (4 release-batch clusters for BTCUSDT main; 3 for ETHUSDT robustness). See Section 6.2 for the diagnostic and Section 6.6 for the three structural barriers behind it.",
}


@dataclass(frozen=True)
class ArxivPackageResult:
    output_dir: Path
    main_tex: Path
    bibliography: Path
    abstract_chars: int
    rendered_tables: int
    rendered_figures: int
    compiled_pdf: Path | None
    compiled_bibliography: Path | None


def extract_abstract_chars(tex: str) -> int:
    start = tex.find(r"\begin{abstract}")
    end = tex.find(r"\end{abstract}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("main manuscript tex does not contain an abstract environment")
    abstract = tex[start + len(r"\begin{abstract}") : end].strip()
    # arXiv counts plain text, but preserving the LaTeX payload is the
    # conservative pre-submission check for this generated package.
    return len(" ".join(abstract.split()))


def build_arxiv_package(
    *,
    source_tex: Path = DEFAULT_SOURCE_TEX,
    source_bib: Path = DEFAULT_SOURCE_BIB,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    tables_dir: Path = DEFAULT_TABLES_DIR,
    source_figures_dir: Path = DEFAULT_SOURCE_FIGURES_DIR,
    render_artifacts: bool = True,
    compile_pdf: bool = False,
) -> ArxivPackageResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    main_tex = output_dir / "main.tex"
    bibliography = output_dir / "references.bib"

    tex = source_tex.read_text(encoding="utf-8")
    if "% GENERATED FILE. DO NOT EDIT." not in tex:
        raise ValueError(f"refusing to package non-generated tex source: {source_tex}")

    tex = tex.replace("% Source: ../manuscript.md", "% Source: ../manuscript.md via latex/manuscript.tex")
    rendered_tables = 0
    rendered_figures = 0
    if render_artifacts:
        tex = ensure_arxiv_packages(tex)
        rendered_tables_tex = render_tables(tables_dir)
        rendered_tables_count = rendered_tables_tex.count(r"\begin{table}")
        rendered_figures = render_figures(tables_dir, source_figures_dir, output_dir / DEFAULT_RENDERED_FIGURES_DIRNAME)
        tex = insert_rendered_artifacts(tex, rendered_tables_tex, render_figure_latex())
        rendered_tables = rendered_tables_count

    main_tex.write_text(tex, encoding="utf-8")
    shutil.copyfile(source_bib, bibliography)

    compiled_pdf: Path | None = None
    compiled_bibliography: Path | None = None
    if compile_pdf:
        compile_arxiv_pdf(output_dir)
        compiled_pdf = output_dir / "main.pdf"
        compiled_bibliography = validate_compiled_bibliography(output_dir)

    return ArxivPackageResult(
        output_dir=output_dir,
        main_tex=main_tex,
        bibliography=bibliography,
        abstract_chars=extract_abstract_chars(tex),
        rendered_tables=rendered_tables,
        rendered_figures=rendered_figures,
        compiled_pdf=compiled_pdf,
        compiled_bibliography=compiled_bibliography,
    )


def ensure_arxiv_packages(tex: str) -> str:
    package_lines = []
    for package in ("graphicx", "booktabs", "tabularx", "array"):
        if rf"\usepackage{{{package}}}" not in tex:
            package_lines.append(rf"\usepackage{{{package}}}")
    if package_lines:
        tex = tex.replace(
            r"\usepackage{amsmath,amssymb}",
            "\\usepackage{amsmath,amssymb}\n" + "\n".join(package_lines),
        )
    column_type = r"\newcolumntype{Y}{>{\raggedright\arraybackslash}X}"
    if column_type not in tex:
        tex = tex.replace(r"\usepackage{hyperref}", "\\usepackage{hyperref}\n" + column_type)
    return tex


def insert_rendered_artifacts(tex: str, rendered_tables_tex: str, rendered_figures_tex: str) -> str:
    marker = "\n\\end{document}"
    if marker not in tex:
        raise ValueError("main manuscript tex does not end with \\\\end{document}")
    rendered = "\n\\clearpage\n\\section{Paper Tables and Figures}\n\n" + rendered_tables_tex
    rendered += "\n\\clearpage\n" + rendered_figures_tex
    return tex.replace(marker, rendered + marker, 1)


def _model_display_label(value: str) -> str:
    """Render M_endog using a plain-text label that survives latex_escape.

    The default latex_escape pass would turn `$M_{\\mathrm{endog}}$` into a
    literal dollar sign and underscores in the rendered cell. Tables 5c and
    5d therefore use the plain-text form `M_endog`; the math-mode label
    `$M_{\\mathrm{endog}}$` stays in the manuscript prose where the markdown
    is already in math context.
    """
    return value


def _format_spectral_radius(value: str) -> str:
    """Format a spectral radius for Table 5c/5d.

    The K17 M0/M1/M2 spectral radii are near 2e-9 and would round to 0 under
    a fixed-precision %.4f format. This helper falls back to scientific
    notation for very small magnitudes so the floor near the lower log-bound
    is visible to the reader. Three significant figures is enough to
    distinguish the K17 floor from the K18 endogenous-only diagnostic.
    """
    if value in {"", "NULL", "None", None}:
        return "--"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(x) < 1e-3:
        return f"{x:.2e}".replace("e-0", "e-").replace("e+0", "e+")
    return f"{x:.4f}"


def _favored_model_from_delta(value: str) -> str:
    """Return the criterion-favored model for a M0-minus-M_endog delta."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "--"
    if x > 0:
        return "M_endog"
    if x < 0:
        return "M0"
    return "tie"


def latex_escape(value: object) -> str:
    text = "" if value is None else str(value)
    if text == "":
        return "--"
    return "".join(LATEX_REPLACEMENTS.get(char, char) for char in text)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def table_path(tables_dir: Path, index: int) -> Path:
    matches = sorted(tables_dir.glob(f"table_{index}_*.csv"))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one table_{index}_*.csv in {tables_dir}, found {matches}")
    return matches[0]


def render_tables(tables_dir: Path) -> str:
    parts: list[str] = []
    for index in range(1, 15):
        if not list(tables_dir.glob(f"table_{index}_*.csv")):
            continue
        path = table_path(tables_dir, index)
        columns, rows = paper_table(index, path)
        parts.append(render_table_environment(index, rows, columns))
    # K18 sensitivity diagnostic tables (5c, 5d). String indices keep them
    # outside the integer 1..14 loop so the legacy table layout is unchanged.
    for sub_index, glob_pattern in (("5c", "table_5c_*.csv"), ("5d", "table_5d_*.csv")):
        matches = sorted(tables_dir.glob(glob_pattern))
        if not matches:
            continue
        path = matches[0]
        columns, rows = paper_table(sub_index, path)
        parts.append(render_table_environment(sub_index, rows, columns))
    return "\n\n".join(parts)


def paper_table(index: int | str, path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    rows = read_csv_rows(path)
    if index == 1:
        return TABLE_COLUMN_LABELS[index], summarize_run_funnel(rows)
    if index == 2:
        return TABLE_COLUMN_LABELS[index], [
            {
                "Label status": friendly_label(row["label_status"]),
                "Count": row["count"],
                "Share": format_decimal(row["share_of_label_input"]),
            }
            for row in rows
        ]
    if index == 3:
        return TABLE_COLUMN_LABELS[index], [
            {"Label": row["label"], "Confidence": row["label_confidence"], "Count": row["count"]}
            for row in rows
        ]
    if index == 4:
        return TABLE_COLUMN_LABELS[index], expand_ppi_rr_matrix(rows)
    if index == 5:
        return TABLE_COLUMN_LABELS[index], summarize_hawkes_table(rows)
    if index == 6:
        return TABLE_COLUMN_LABELS[index], [
            {
                "Claim": row["claim"],
                "Evidence": friendly_evidence(row["evidence"]),
                "Paper framing": friendly_paper_framing(row["paper_framing"]),
            }
            for row in rows
        ]
    if index == 7:
        return TABLE_COLUMN_LABELS[index], [
            {
                "Cohort": friendly_cohort(row["cohort_name"]),
                "Model": row["model"],
                "OOS LL delta": format_decimal(row["ll_oos_delta_vs_m0"]),
                "Rank": row["oos_rank"],
                "OOS n": row["n_events_oos"],
                "Audit result": friendly_scenario(row["scenario_indicator"]),
            }
            for row in rows
        ]
    if index == 8:
        return TABLE_COLUMN_LABELS[index], [
            {
                "Cohort": friendly_cohort(row["cohort_name"]),
                "Clusters": row["cluster_count"],
                "Max size": row["max_cluster_size"],
                "Original p": format_decimal(row["original_lrt_p_value"]),
                "Audit result": friendly_scenario(row["scenario_indicator"]),
                "Reason": friendly_failure(row["failure_reason"]),
            }
            for row in rows
        ]
    if index == 9:
        return TABLE_COLUMN_LABELS[index], summarize_threshold_table(path)
    if index == 10:
        return TABLE_COLUMN_LABELS[index], [
            {
                "Family": friendly_family(row["event_family"]),
                "Rows": row["scored_rows"],
                "Info": row["information_count"],
                "Noise": row["noise_count"],
                "Low-conf": row["low_confidence_count"],
                "PPI/RR disagree": row["ppi_rr_disagreement_count"],
                "Abstain": row.get("abstain_or_insufficient_count", "0"),
                "Status": friendly_label(row["interpretation_status"]),
            }
            for row in rows
        ]
    if index == 11:
        return TABLE_COLUMN_LABELS[index], [
            {
                "Cohort": friendly_cohort(row["cohort_name"]),
                r"$\Delta_{\text{LL}}^{\text{true}}$": format_decimal(
                    row["delta_ll_true_nats"], digits=2
                ),
                "Threshold-crossing rate": format_decimal(row["detection_probability"]),
                "MC SE": format_decimal(row["monte_carlo_se"]),
                r"$\sigma_{\text{total}}$": format_decimal(row["sigma_total"]),
            }
            for row in rows
        ]
    if index == 12:
        return TABLE_COLUMN_LABELS[index], [
            {
                "Category": row["category"],
                "Subcategory": row["subcategory"].replace("_", " "),
                "Count": row["count"],
                "Share": format_decimal(row["share_of_added"]),
            }
            for row in rows
        ]
    if index == 13:
        return TABLE_COLUMN_LABELS[index], [
            {
                "Policy variant": row["policy_variant"].replace("_", " "),
                "Info": row["information_count"],
                "Noise": row["noise_count"],
                "Total": row["computed_total"],
                "Info share": format_decimal(row["information_share"]),
                "Dedup events": row["deduplicated_event_estimate"],
                "Hawkes refit": _refit_status_label(row["hawkes_refit_status"]),
            }
            for row in rows
        ]
    if index == 14:
        return TABLE_COLUMN_LABELS[index], [
            {
                "Cohort": friendly_cohort(row["cohort_name"]),
                "Family": row["family"],
                "Info": row["computed_information"],
                "Noise": row["computed_noise"],
                "Abstain": row["abstain_or_other"],
                "Total": row["total_rows"],
                "Unique event ts": row["unique_event_timestamps"],
                "Note": row["note"],
            }
            for row in rows
        ]
    if index == "5c":
        return TABLE_COLUMN_LABELS[index], [
            {
                "Cohort": friendly_cohort(row["cohort"]),
                "Model": _model_display_label(row["model"]),
                "LL endog": format_decimal(row["ll_in_sample"]),
                "LL M0": format_decimal(row["ll_in_sample_m0"]),
                "LL gap": format_decimal(row["ll_diff_m0_vs_m_endog"]),
                r"$\Delta\mathrm{AIC}(M_0-M_e)$": format_decimal(row["aic_delta_m0_vs_m_endog"]),
                "AIC fav.": _favored_model_from_delta(row["aic_delta_m0_vs_m_endog"]),
                r"$\Delta\mathrm{BIC}(M_0-M_e)$": format_decimal(row["bic_delta_m0_vs_m_endog"]),
                "BIC fav.": _favored_model_from_delta(row["bic_delta_m0_vs_m_endog"]),
                r"$\rho$": _format_spectral_radius(row["branching_spectral_radius"]),
            }
            for row in rows
        ]
    if index == "5d":
        return TABLE_COLUMN_LABELS[index], [
            {
                "Cohort": friendly_cohort(row["cohort"]),
                "Model": _model_display_label(row["model"]),
                r"$\rho$": _format_spectral_radius(row["spectral_radius"]),
                "Stationary": "yes" if row["stationary"] == "true" else "no",
                "Source": row["source"],
            }
            for row in rows
        ]
    raise ValueError(f"unsupported table index: {index}")


def summarize_run_funnel(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_metric = {row["metric"]: row for row in rows}
    wanted = [
        (
            "Universe",
            "combined_event_rows",
            "canonical event rows",
            "TE plus Binance canonical universe before activation filtering.",
        ),
        (
            "Candidates",
            "primary_te_candidates",
            "primary TE macro events",
            "Macro events selected for activation.",
        ),
        (
            "Activation attempts",
            "activation_attempt_rows",
            "symbol-by-rule attempts",
            "Candidates times 2 symbols times 2 activation rules.",
        ),
        (
            "Activation positive",
            "computed_positive",
            "activation rows",
            "Rows that pass the empirical activation gate.",
        ),
        (
            "Computed labels",
            "computed_labels",
            "label rows",
            "PPI-primary/RR-validator computed labels.",
        ),
        (
            "Hawkes fits",
            "fit_success_rows",
            "model fit rows",
            "2 cohorts crossed with M0/M1/M2 successful optimizations.",
        ),
    ]
    return [
        {
            "Stage": stage,
            "Unit": unit,
            "Count": by_metric[metric]["value"],
            "Note": note,
        }
        for stage, metric, unit, note in wanted if metric in by_metric
    ]


def summarize_threshold_table(path: Path) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"threshold table is empty: {path}")
    computed = [int(row["computed_count"]) for row in rows]
    information = [int(row["information_count"]) for row in rows]
    noise = [int(row["noise_count"]) for row in rows]
    flip_rate = max(float(row["baseline_computed_flip_rate"]) for row in rows)
    share_shift = max(float(row["information_share_shift"]) for row in rows)
    scenarios = sorted({row["overall_scenario_indicator"] for row in rows})
    return [
        {"Metric": "Grid points", "Value": str(len(rows))},
        {"Metric": "Audit result", "Value": "; ".join(friendly_scenario(scenario) for scenario in scenarios)},
        {"Metric": "Computed count range", "Value": f"{min(computed)}--{max(computed)}"},
        {"Metric": "Information(schema) count range", "Value": f"{min(information)}--{max(information)}"},
        {"Metric": "Noise(schema) count range", "Value": f"{min(noise)}--{max(noise)}"},
        {"Metric": "Max baseline flip rate", "Value": f"{flip_rate:.6f}"},
        {"Metric": "Max Information(schema)-share shift", "Value": f"{share_shift:.6f}"},
    ]


def render_table_environment(index: int | str, rows: list[dict[str, str]], columns: tuple[str, ...]) -> str:
    align = TABLE_ALIGNMENTS[index]
    # Column labels and captions are authored as LaTeX-aware strings (they may
    # contain math like $\beta_{I}$ or escapes like \times). Do not pass them
    # through latex_escape, which would double-escape and break math mode.
    header = " & ".join(columns) + r" \\ \hline"
    body = "\n".join(
        " & ".join(latex_escape(row.get(column, "")) for column in columns) + r" \\"
        for row in rows
    )
    caption = TABLE_CAPTIONS[index]
    label = f"tab:table-{index}"
    font_size = TABLE_FONT_SIZES[index]
    tabular_begin = rf"\begin{{tabularx}}{{\textwidth}}{{{align}}}" if "Y" in align else rf"\begin{{tabular}}{{{align}}}"
    tabular_end = r"\end{tabularx}" if "Y" in align else r"\end{tabular}"
    return "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\centering",
            font_size,
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            tabular_begin,
            r"\toprule",
            header.replace(r"\hline", r"\midrule"),
            body,
            r"\bottomrule",
            tabular_end,
            r"\end{table}",
        ]
    )


def summarize_hawkes_table(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    result = []
    for row in rows:
        fit_value = row.get("fit_success", "")
        if fit_value == "true":
            fit_friendly = "yes"
        elif fit_value == "false":
            fit_friendly = "no"
        else:
            fit_friendly = "--"
        result.append(
            {
                "Cohort": friendly_cohort(row["cohort_name"]),
                "Model": row["model"],
                "LL delta": format_decimal(row["ll_delta_vs_m0"]),
                "AIC delta": format_decimal(row["aic_delta_vs_m0"]),
                "BIC delta": format_decimal(row.get("bic_delta_vs_m0", "")),
                "LRT p": format_decimal(row["lrt_p_value_vs_m0"]),
                r"$\beta_{I}$": format_decimal(row.get("beta_information", "")),
                r"$\beta_{N}$": format_decimal(row.get("beta_noise", "")),
                r"$T^{1/2}_{I}$": format_decimal(row.get("half_life_information_sec", ""), digits=2),
                r"$T^{1/2}_{N}$": format_decimal(row.get("half_life_noise_sec", ""), digits=2),
                "Fit": fit_friendly,
            }
        )
    return result


def _refit_status_label(value: str) -> str:
    """Map raw artifact tokens for the Hawkes-refit-status column to
    reader-facing labels. Internal phase names like J2 must not appear in
    the rendered table; the regression test in
    `tests/test_build_arxiv_package.py` enforces this contract.
    """
    mapping = {
        "fitted_in_main_table": "fitted in main table",
        "deferred_to_J2": "future validation expansion",
        "deferred_to_j2": "future validation expansion",
    }
    return mapping.get(value, value.replace("_", " "))


def expand_ppi_rr_matrix(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Render Table 4 with all nine PPI x RR cells, including empty cells.

    The artifact records only non-zero (PPI verdict, RR verdict) combinations.
    For the reader-facing table we expand to the full 3x3 grid so the absence
    of evidence in cells like (PPI=Noise, RR=Information) is explicit rather
    than hidden by a missing row.
    """
    verdicts = ("information", "noise", "undetermined")
    by_pair: dict[tuple[str, str], dict[str, str]] = {
        (row["ppi_verdict"] or "undetermined", row["recovery_verdict"] or "undetermined"): row
        for row in rows
    }
    expanded: list[dict[str, str]] = []
    for ppi in verdicts:
        for rr in verdicts:
            row = by_pair.get((ppi, rr))
            if row is None:
                expanded.append(
                    {
                        "PPI verdict": friendly_label(ppi),
                        "RR verdict": friendly_label(rr),
                        "Label status": "--",
                        "Label": "--",
                        "Confidence": "--",
                        "Count": "0",
                    }
                )
            else:
                expanded.append(
                    {
                        "PPI verdict": friendly_label(row["ppi_verdict"] or ppi),
                        "RR verdict": friendly_label(row["recovery_verdict"] or rr),
                        "Label status": friendly_label(row["label_status"]),
                        "Label": friendly_label(row["label"]),
                        "Confidence": friendly_label(row["label_confidence"]),
                        "Count": row["count"],
                    }
                )
    return expanded


def format_decimal(value: str, digits: int = 3) -> str:
    if value in {"", "NULL", "None", None}:
        return "--"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def friendly_label(value: str) -> str:
    if value in {"", "NULL", "None", None}:
        return "--"
    text = str(value)
    replacements = {
        "abstain_ambiguous": "Abstain: ambiguous",
        "abstain_confounded": "Abstain: confounded",
        "computed": "Computed",
        "information": "Information(schema)",
        "insufficient_horizon_data": "Insufficient horizon",
        "interpretable_family": "Interpretable family",
        "noise": "Noise(schema)",
        "undetermined": "Undetermined",
    }
    return replacements.get(text, text.replace("_", " "))


def friendly_evidence(value: str) -> str:
    replacements = {
        "All H3-E model rows converged with fit_success=true": "All Hawkes model rows converged.",
        "main M1 p=0.666442; robustness_symbol M1 p=0.999293": (
            "BTC main M1 p=0.666442; ETH robustness M1 p=0.999293"
        ),
        "computed labels: Information=34, Noise=12": "Computed labels: Information(schema)=34, Noise(schema)=12",
    }
    if value in replacements:
        return replacements[value]
    # Long, sentence-shaped evidence strings (e.g. the K8 Table 6 reword)
    # should pass through unchanged rather than being underscore-replaced as
    # if they were short slug tokens.
    if " " in value or "." in value:
        return value
    return friendly_label(value)


def friendly_paper_framing(value: str) -> str:
    """Demote inferential language in Table 6 paper-framing rows.

    K17-3 demotes "No statistically significant ..." to a nominal-diagnostic
    statement so the Table 6 row matches the cluster-uncalibrated wording
    used in Section 6.2 and Section 6.6. Other paper_framing strings pass
    through unchanged.
    """
    replacements = {
        (
            "No statistically significant class-specific exogenous component "
            "is detected in this first-pass sample."
        ): (
            "Under the nominal chi-square reference distribution, the M1-vs-M0 "
            "likelihood-ratio diagnostic does not reject the pooled-exogenous "
            "M0 specification; cluster-calibrated inference is deferred to "
            "the future-validation expansion (Section 7)."
        ),
    }
    return replacements.get(value, value)


def friendly_cohort(value: str) -> str:
    return {
        "main": "BTC main",
        "robustness_symbol": "ETH robustness",
        "robustness": "ETH robustness",
    }.get(value, friendly_label(value))


def friendly_family(value: str) -> str:
    return {
        "cpi_inflation": "CPI/inflation",
        "ism_pmi": "ISM/PMI",
        "labor": "Labor",
        "retail_sales": "Retail sales",
    }.get(value, friendly_label(value))


def friendly_scenario(value: str) -> str:
    if value in {"", "NULL", "None", None}:
        return "--"
    text = str(value)
    if text.startswith("B_"):
        return "Materially similar"
    if text.startswith("F_"):
        return "Artifact-contract limitation"
    if text.startswith("G_"):
        return "Stable across grid"
    if text.startswith("baseline"):
        return "baseline"
    if text.startswith("ablation"):
        return "ablation"
    return friendly_label(text)


def friendly_failure(value: str) -> str:
    if value == "missing_per_cluster_likelihood_contributions":
        return "missing per-cluster likelihood contributions"
    return friendly_label(value)


def render_figures(tables_dir: Path, source_figures_dir: Path, output_dir: Path) -> int:
    # The SVG source directory is checked to preserve the package contract even
    # though arXiv receives PDF figures generated from the same artifact data.
    for index in range(1, 4):
        matches = sorted(source_figures_dir.glob(f"figure_{index}_*.svg"))
        if len(matches) != 1:
            raise ValueError(f"expected exactly one figure_{index}_*.svg in {source_figures_dir}, found {matches}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )

    render_pipeline_funnel_figure(tables_dir, output_dir, plt)
    render_label_confidence_figure(tables_dir, output_dir, plt)
    render_hawkes_pvalue_figure(tables_dir, output_dir, plt)
    return 3


def _configure_matplotlib_cache() -> None:
    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "future-polios-mpl-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))


def render_pipeline_funnel_figure(tables_dir: Path, output_dir: Path, plt) -> None:
    rows = read_csv_rows(table_path(tables_dir, 1))
    wanted = {
        "primary_te_candidates": "TE primary events",
        "computed_positive": "Activation-positive rows",
        "computed_labels": "Computed labels",
        "fit_success_rows": "Successful Hawkes fits",
    }
    values = [(wanted[row["metric"]], int(row["value"])) for row in rows if row["metric"] in wanted]
    labels, counts = zip(*values)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.barh(labels, counts, color="#2c6f7c")
    ax.invert_yaxis()
    ax.set_xlabel("Rows")
    ax.set_title("Empirical Pipeline Funnel")
    for i, count in enumerate(counts):
        ax.text(count + max(counts) * 0.02, i, str(count), va="center")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_1_pipeline_funnel.pdf", metadata=PDF_METADATA)
    plt.close(fig)


def render_label_confidence_figure(tables_dir: Path, output_dir: Path, plt) -> None:
    rows = read_csv_rows(table_path(tables_dir, 3))
    labels = [f"{row['label']} / {row['label_confidence']}" for row in rows]
    counts = [int(row["count"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(labels, counts, color=["#386fa4", "#f2a65a", "#5b8c5a"])
    ax.set_ylabel("Computed labels")
    ax.set_title("Label Confidence Split")
    ax.tick_params(axis="x", rotation=20)
    for i, count in enumerate(counts):
        ax.text(i, count + 0.5, str(count), ha="center")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_2_label_confidence_split.pdf", metadata=PDF_METADATA)
    plt.close(fig)


def render_hawkes_pvalue_figure(tables_dir: Path, output_dir: Path, plt) -> None:
    rows = [
        row for row in read_csv_rows(table_path(tables_dir, 5))
        if row["model"] == "M1"
    ]
    labels = ["BTCUSDT main" if row["cohort_name"] == "main" else "ETHUSDT robustness" for row in rows]
    pvalues = [float(row["lrt_p_value_vs_m0"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(labels, pvalues, color="#733c6b")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("M1-vs-M0 p-value")
    ax.set_title("Hawkes M1 Non-Rejection Diagnostics")
    for i, pvalue in enumerate(pvalues):
        ax.text(i, min(pvalue + 0.035, 1.02), f"{pvalue:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_3_hawkes_m1_pvalues.pdf", metadata=PDF_METADATA)
    plt.close(fig)


def render_figure_latex() -> str:
    parts: list[str] = []
    for index in range(1, 4):
        stem = {
            1: "figure_1_pipeline_funnel.pdf",
            2: "figure_2_label_confidence_split.pdf",
            3: "figure_3_hawkes_m1_pvalues.pdf",
        }[index]
        parts.append(
            "\n".join(
                [
                    r"\begin{figure}[htbp]",
                    r"\centering",
                    rf"\includegraphics[width=0.92\textwidth]{{{DEFAULT_RENDERED_FIGURES_DIRNAME}/{stem}}}",
                    rf"\caption{{{latex_escape(FIGURE_CAPTIONS[index])}}}",
                    rf"\label{{fig:figure-{index}}}",
                    r"\end{figure}",
                ]
            )
        )
    return "\n\n".join(parts)


def compile_arxiv_pdf(output_dir: Path) -> None:
    commands = [
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["bibtex", "main"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
    ]
    for command in commands:
        subprocess.run(command, cwd=output_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def validate_compiled_bibliography(output_dir: Path) -> Path:
    bbl_path = output_dir / "main.bbl"
    if not bbl_path.exists():
        raise ValueError(f"expected BibTeX to generate {bbl_path}")
    bbl_text = bbl_path.read_text(encoding="utf-8")
    if r"\bibitem" not in bbl_text:
        raise ValueError(f"compiled bibliography has no bibitems: {bbl_path}")
    return bbl_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tex", type=Path, default=DEFAULT_SOURCE_TEX)
    parser.add_argument("--source-bib", type=Path, default=DEFAULT_SOURCE_BIB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--source-figures-dir", type=Path, default=DEFAULT_SOURCE_FIGURES_DIR)
    parser.add_argument("--skip-rendered-artifacts", action="store_true")
    parser.add_argument("--compile-pdf", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_arxiv_package(
        source_tex=args.source_tex,
        source_bib=args.source_bib,
        output_dir=args.output_dir,
        tables_dir=args.tables_dir,
        source_figures_dir=args.source_figures_dir,
        render_artifacts=not args.skip_rendered_artifacts,
        compile_pdf=args.compile_pdf,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "main_tex": str(result.main_tex),
                "bibliography": str(result.bibliography),
                "abstract_chars": result.abstract_chars,
                "rendered_tables": result.rendered_tables,
                "rendered_figures": result.rendered_figures,
                "compiled_pdf": str(result.compiled_pdf) if result.compiled_pdf else None,
                "compiled_bibliography": (
                    str(result.compiled_bibliography) if result.compiled_bibliography else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
