from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "latex" / "build_arxiv_package.py"


def _module():
    spec = importlib.util.spec_from_file_location("build_arxiv_package", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_arxiv_package_copies_generated_tex_and_bib(tmp_path):
    mod = _module()
    source_tex = tmp_path / "manuscript.tex"
    source_bib = tmp_path / "references.bib"
    output_dir = tmp_path / "arxiv"
    source_tex.write_text(
        """
% GENERATED FILE. DO NOT EDIT.
% Source: ../manuscript.md
\\documentclass{article}
\\begin{document}
\\begin{abstract}
Short abstract.
\\end{abstract}
\\bibliography{references}
\\end{document}
""".lstrip(),
        encoding="utf-8",
    )
    source_bib.write_text("@article{Example2026Paper,\n  year = {2026}\n}\n", encoding="utf-8")

    result = mod.build_arxiv_package(
        source_tex=source_tex,
        source_bib=source_bib,
        output_dir=output_dir,
        render_artifacts=False,
    )

    assert result.abstract_chars == len("Short abstract.")
    assert result.rendered_tables == 0
    assert result.rendered_figures == 0
    assert result.compiled_pdf is None
    assert result.compiled_bibliography is None
    assert (output_dir / "main.tex").read_text(encoding="utf-8").startswith("% GENERATED FILE. DO NOT EDIT.")
    assert "% Source: ../manuscript.md via latex/manuscript.tex" in (
        output_dir / "main.tex"
    ).read_text(encoding="utf-8")
    assert (output_dir / "references.bib").read_text(encoding="utf-8") == source_bib.read_text(
        encoding="utf-8"
    )


def test_build_arxiv_package_refuses_non_generated_tex(tmp_path):
    mod = _module()
    source_tex = tmp_path / "manuscript.tex"
    source_bib = tmp_path / "references.bib"
    source_tex.write_text("\\documentclass{article}\n", encoding="utf-8")
    source_bib.write_text("", encoding="utf-8")

    try:
        mod.build_arxiv_package(
            source_tex=source_tex,
            source_bib=source_bib,
            output_dir=tmp_path / "arxiv",
            render_artifacts=False,
        )
    except ValueError as exc:
        assert "non-generated tex source" in str(exc)
    else:
        raise AssertionError("non-generated tex source should fail")


def test_build_arxiv_package_inserts_rendered_tables_and_figures(tmp_path):
    mod = _module()
    source_tex = tmp_path / "manuscript.tex"
    source_bib = tmp_path / "references.bib"
    tables_dir = tmp_path / "tables"
    source_figures_dir = tmp_path / "figures"
    output_dir = tmp_path / "arxiv"
    tables_dir.mkdir()
    source_figures_dir.mkdir()
    source_tex.write_text(
        """
% GENERATED FILE. DO NOT EDIT.
% Source: ../manuscript.md
\\documentclass{article}
\\usepackage{amsmath,amssymb}
\\begin{document}
\\begin{abstract}
Short abstract.
\\end{abstract}
\\bibliography{references}
\\appendix
\\section{Artifact Map}
\\end{document}
""".lstrip(),
        encoding="utf-8",
    )
    source_bib.write_text("@article{Example2026Paper,\n  year = {2026}\n}\n", encoding="utf-8")
    _write_fixture_tables(tables_dir)
    for index in range(1, 4):
        (source_figures_dir / f"figure_{index}_fixture.svg").write_text(
            "<svg xmlns='http://www.w3.org/2000/svg'></svg>\n",
            encoding="utf-8",
        )

    result = mod.build_arxiv_package(
        source_tex=source_tex,
        source_bib=source_bib,
        output_dir=output_dir,
        tables_dir=tables_dir,
        source_figures_dir=source_figures_dir,
    )

    main_tex = (output_dir / "main.tex").read_text(encoding="utf-8")
    assert result.rendered_tables == 14
    assert result.rendered_figures == 3
    assert r"\usepackage{graphicx}" in main_tex
    assert r"\usepackage{booktabs}" in main_tex
    assert r"\section{Paper Tables and Figures}" in main_tex
    assert main_tex.count(r"\begin{table}") == 14
    # K7 regression: the Tier-1 ablation table must not leak the internal
    # phase label J2 into the reader-facing tex; the builder maps the raw
    # artifact token deferred_to_J2 to the semantic label "future validation
    # expansion".
    assert "J2" not in main_tex
    assert "future validation expansion" in main_tex
    assert main_tex.count(r"\includegraphics") == 3
    assert r"\resizebox" not in main_tex
    # K4 reader-facing terminology pass: the rendered threshold-sensitivity
    # table reports a semantic indicator ("Stable across grid") instead of
    # the raw scenario letter. Scenario letters remain only as parenthetical
    # pre-registration tags in manuscript prose, which is not part of this
    # fixture.
    assert "Stable across grid" in main_tex
    assert "G\\_threshold\\_limited\\_sensitivity" not in main_tex
    assert "primary\\_te\\_candidates" not in main_tex
    assert "fit\\_success=true" not in main_tex
    assert "robustness\\_symbol" not in main_tex
    assert "insufficient\\_horizon\\_data" not in main_tex
    assert "Interpretable family" in main_tex
    # K4 follow-up: the audit-classification column header is rendered as the
    # semantic label "Audit result". The protocol-side label "Scenario" must
    # not appear as a column header on Tables 7, 8, or 9. The parenthetical
    # "(pre-registered Scenario B/F/G)" tags are still permitted in prose, so
    # the assertion targets only the LaTeX table-row separators where a column
    # header would land.
    assert "Audit result" in main_tex
    assert "& Scenario &" not in main_tex
    assert "& Scenario \\\\" not in main_tex
    figure_path = output_dir / "figures" / "figure_1_pipeline_funnel.pdf"
    assert figure_path.exists()
    first_render = figure_path.read_bytes()

    mod.build_arxiv_package(
        source_tex=source_tex,
        source_bib=source_bib,
        output_dir=output_dir,
        tables_dir=tables_dir,
        source_figures_dir=source_figures_dir,
    )

    assert figure_path.read_bytes() == first_render


def test_rendered_main_tex_has_no_internal_phase_label_or_pre_registered_token():
    """The reader-facing rendered arXiv tex (committed in latex/arxiv/main.tex)
    must not leak internal phase labels as standalone tokens, and must not
    use the word "pre-registered" anywhere in prose.

    "Internal phase labels" here means the K3/K4/K5/J2 phase identifiers
    used to plan the project. The Appendix C reproducibility snapshot
    legitimately references protocol file paths such as
    `J1_empirical_validation_protocol.md`; that is a file name, not a
    reader-facing phase label, so the regex-based check uses word
    boundaries to allow the file-path mention while still rejecting raw
    phase tokens like J0 / J1 / J2 / J3 / Path A appearing as standalone
    words in prose.
    """
    import re

    main_tex = (
        Path(__file__).resolve().parents[1] / "latex" / "arxiv" / "main.tex"
    ).read_text(encoding="utf-8")
    # Reject standalone J-phase tokens as words. The negative-lookahead
    # `(?![\\_])` excludes file-path forms like
    # `J1_empirical_validation_protocol.md` and the LaTeX-escaped
    # `J1\_empirical...` that build_latex emits in Appendix C.
    for token in ("J0", "J1", "J2", "J3"):
        pattern = rf"\b{token}(?![\\_a-zA-Z])"
        assert not re.search(pattern, main_tex), (
            f"reader-facing main.tex must not contain internal phase label {token!r}"
        )
    assert "Path A" not in main_tex, (
        "reader-facing main.tex must not contain the internal label 'Path A'"
    )
    # The "pre-registered" token must not appear; "pre-specified" is the
    # accepted form because the protocol document is not yet mirrored to a
    # public timestamping service.
    assert "pre-registered" not in main_tex, (
        "reader-facing main.tex must use 'pre-specified', not 'pre-registered'"
    )


def test_refit_status_label_maps_internal_phase_tokens():
    """The Table 13 hawkes_refit_status column must not pass internal phase
    names like J2 through to the rendered table. The mapping function in
    build_arxiv_package translates artifact tokens to reader-facing labels.
    """
    mod = _module()
    assert mod._refit_status_label("deferred_to_J2") == "future validation expansion"
    assert mod._refit_status_label("deferred_to_j2") == "future validation expansion"
    assert mod._refit_status_label("fitted_in_main_table") == "fitted in main table"
    # Unknown tokens fall back to the underscore-replace form, which is
    # safe but logs the absence of an explicit mapping. A future artifact
    # token that uses J2 again would still bypass this fallback only if
    # the new token contains "J2" verbatim; the rendered-tex regression
    # test above is the second line of defence.


def test_rendered_main_tex_does_not_leak_private_repo_url():
    """K12 regression: the private working repository URL
    (github.com/ds-academy/future_polios) and the LaTeX-escaped form
    (future\\_polios) must not appear in the reader-facing main.tex.
    The K12 chain replaced this with the public artifact repository
    URL; this test prevents a future edit from regressing back to the
    private URL through any path.
    """
    main_tex = (
        Path(__file__).resolve().parents[1] / "latex" / "arxiv" / "main.tex"
    ).read_text(encoding="utf-8")
    assert "ds-academy/future_polios" not in main_tex, (
        "reader-facing main.tex must not reference the private working repository "
        "github.com/ds-academy/future_polios"
    )
    assert "future\\_polios" not in main_tex, (
        "reader-facing main.tex must not reference the LaTeX-escaped private repo "
        "name future\\_polios"
    )


def test_rendered_main_tex_references_public_artifact_repo():
    """K12 regression: the reader-facing main.tex must reference the
    public artifact repository created by the K12 chain. Appendix C
    of the manuscript points readers at this repository for
    paper-facing reproduction; if the URL is missing the reproducibility
    contract advertised in the appendix breaks.
    """
    main_tex = (
        Path(__file__).resolve().parents[1] / "latex" / "arxiv" / "main.tex"
    ).read_text(encoding="utf-8")
    assert "weak-label-hawkes-crypto-futures" in main_tex, (
        "reader-facing main.tex must reference the public artifact repository "
        "github.com/ds-academy/weak-label-hawkes-crypto-futures"
    )


def test_validate_compiled_bibliography_requires_bibitems(tmp_path):
    mod = _module()
    (tmp_path / "main.bbl").write_text(
        "\\begin{thebibliography}{1}\n\\bibitem{Example2026Paper} Example.\n\\end{thebibliography}\n",
        encoding="utf-8",
    )

    assert mod.validate_compiled_bibliography(tmp_path) == tmp_path / "main.bbl"

    (tmp_path / "main.bbl").write_text("\\begin{thebibliography}{1}\n\\end{thebibliography}\n", encoding="utf-8")
    try:
        mod.validate_compiled_bibliography(tmp_path)
    except ValueError as exc:
        assert "no bibitems" in str(exc)
    else:
        raise AssertionError("empty compiled bibliography should fail")


def _write_fixture_tables(tables_dir: Path) -> None:
    fixtures = {
        "table_1_run_funnel.csv": """stage,metric,value,note
activation,primary_te_candidates,95,primary macro events
activation,computed_positive,60,activation positive
labeling,computed_labels,46,computed labels
hawkes,fit_success_rows,6,successful fits
""",
        "table_2_label_status_counts.csv": """label_status,count,share_of_label_input
computed,46,0.7667
""",
        "table_3_label_confidence_by_label.csv": """label,label_confidence,count
Information,High,4
Information,Low,30
Noise,High,12
""",
        "table_4_ppi_rr_matrix.csv": """ppi_verdict,recovery_verdict,label_status,label,label_confidence,count
information,noise,computed,Information,Low,30
NULL,NULL,insufficient_horizon_data,NULL,NULL,8
""",
        "table_5_hawkes_comparison.csv": """cohort_name,model,fit_success,optimizer_iterations,log_likelihood_in_sample,log_likelihood_oos,ll_delta_vs_m0,aic_delta_vs_m0,bic_delta_vs_m0,lrt_p_value_vs_m0,beta_information,beta_noise,half_life_information_sec,half_life_noise_sec
main,M0,true,20,-1,-1,0,0,0,NULL,NULL,NULL,NULL,NULL
main,M1,true,42,-1,-1,0.7,4.4,21.1,0.666442,0.4,0.1,1.6,6.9
robustness_symbol,M1,true,54,-1,-1,0.0,5.9,22.3,0.999293,0.05,0.06,12.4,10.6
""",
        "table_6_claim_summary.csv": """claim,evidence,paper_framing
Class-specific decay significance,p=0.666,No detected significance
End-to-end reproducibility,All H3-E model rows converged with fit_success=true,Executable
""",
        "table_7_oos_audit.csv": """cohort_name,model,log_likelihood_in_sample,log_likelihood_oos,ll_oos_delta_vs_m0,oos_rank,n_events_in_sample,n_events_oos,lrt_p_value_vs_m0,scenario_indicator,epsilon_nats
main,M1,-1,-1,0.1,2,10,2,0.666442,B_m1_similar_oos,5
""",
        "table_8_dependence_audit.csv": """cohort_name,symbol,cluster_strategy,n_bootstrap,random_seed,alpha,material_pvalue_shift,raw_computed_label_rows,deduplicated_event_count,cluster_count,max_cluster_size,singleton_cluster_count,original_lrt_p_value,adjusted_lrt_p_value,scenario_indicator,failure_reason
main,BTCUSDT,same_source_same_time,1000,20260504,0.05,0.1,24,12,4,7,2,0.666442,NULL,F_dependence_audit_failed,missing_per_cluster_likelihood_contributions
""",
        "table_9_threshold_sensitivity.csv": """grid_id,ppi_information_threshold_z,ppi_noise_threshold_z,recovery_information_min_ratio,recovery_noise_max_ratio,valid_scored_rows,baseline_computed_count,computed_count,information_count,noise_count,high_confidence_count,low_confidence_count,computed_relative_change,information_share,information_share_shift,baseline_computed_flip_count,baseline_computed_flip_rate,gridpoint_indicator,overall_scenario_indicator
g1,1.5,0.5,1.15,1.02,48,46,48,36,12,16,32,0.04,0.75,0.010870,0,0,stable_grid_point,G_threshold_limited_sensitivity
g2,2.0,0.75,1.25,1.05,48,46,46,34,12,16,30,0,0.73,0,0,0,stable_grid_point,G_threshold_limited_sensitivity
""",
        "table_10_event_family_heterogeneity.csv": """event_family,scored_rows,computed_count,information_count,noise_count,low_confidence_count,ppi_rr_disagreement_count,abstain_or_insufficient_count,interpretation_status
labor,16,16,16,0,16,16,0,interpretable_family
""",
        "table_11_oos_power_simulation.csv": """cohort_name,n_train,n_test,in_sample_delta_ll,sigma_per_event,delta_ll_true_nats,threshold_nats,detection_probability,monte_carlo_se,sigma_total,n_replicates
main,10,2,0.7,0.22,5.0,5.0,0.5,0.0035,0.31,20000
""",
        "table_12_expansion_decomposition.csv": """category,subcategory,count,share_of_added,note
Header,added_candidates,41,1.0,Expansion decomposition fixture row.
Activation outcome,missing_market_data,41,1.0,All added candidates rejected as missing_market_data.
""",
        "table_13_label_confidence_ablation.csv": """policy_variant,information_count,noise_count,computed_total,information_share,deduplicated_event_estimate,hawkes_refit_status,note
baseline_ppi_primary,34,12,46,0.7391,23,fitted_in_main_table,baseline
strict_agreement_only,4,12,16,0.25,8,deferred_to_J2,strict ablation
""",
        "table_14_family_aware_hawkes_input.csv": """cohort_name,family,computed_information,computed_noise,abstain_or_other,total_rows,unique_event_timestamps,note
main,Labor,8,0,2,10,2,Information-only family; BTCUSDT cohort
""",
    }
    for name, text in fixtures.items():
        (tables_dir / name).write_text(text, encoding="utf-8")
