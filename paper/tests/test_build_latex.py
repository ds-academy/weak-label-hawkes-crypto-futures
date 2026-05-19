from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "latex" / "build_latex.py"


def _module():
    spec = importlib.util.spec_from_file_location("build_latex", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _bib(tmp_path: Path) -> Path:
    bib = tmp_path / "references.bib"
    bib.write_text(
        """
@article{MacKinlay1997EventStudies,
  author = {MacKinlay, A. Craig},
  title = {Event Studies in Economics and Finance},
  year = {1997},
  keywords = {core,cited}
}

@article{AndersenEtAl2003MacroAnnouncements,
  author = {Andersen, Torben G.},
  title = {Micro Effects of Macro Announcements},
  year = {2003},
  keywords = {core,cited}
}

@article{Hawkes1971Spectra,
  author = {Hawkes, Alan G.},
  title = {Spectra of Some Self-Exciting and Mutually Exciting Point Processes},
  year = {1971},
  keywords = {candidate,metadata-unverified,not-cited}
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return bib


def test_convert_citation_cluster_without_touching_inline_code(tmp_path):
    mod = _module()
    bib_index = mod.parse_bib_index(_bib(tmp_path))
    markdown = """
# Example

## Abstract

Event studies (`MacKinlay1997EventStudies`; `AndersenEtAl2003MacroAnnouncements`) use windows.
The status `computed_positive` is code.

## References

- `MacKinlay1997EventStudies`

## Appendix A. Artifact Map

- `j0_expansion/tables/table_1_run_funnel.csv`
""".strip()

    tex = mod.convert_markdown_to_latex(markdown, bib_index)

    assert r"\cite{MacKinlay1997EventStudies,AndersenEtAl2003MacroAnnouncements}" in tex
    assert r"\texttt{computed\_positive}" in tex
    assert r"\texttt{j0\_expansion/tables/table\_1\_run\_funnel.csv}" in tex
    assert r"\bibliography{references}" in tex
    assert r"\appendix" in tex


def test_heading_numbers_are_removed_without_truncating_titles(tmp_path):
    mod = _module()
    bib_index = mod.parse_bib_index(_bib(tmp_path))
    markdown = """
# Example

## 2. Related Work

### 2.1 Event Studies and Market Reactions

Text.

## Appendix A. Artifact Map

### A.1 Generated Tables

Text.
""".strip()

    tex = mod.convert_markdown_to_latex(markdown, bib_index)

    assert r"\section{Related Work}" in tex
    assert r"\subsection{Event Studies and Market Reactions}" in tex
    assert r"\subsection{A.1 Generated Tables}" in tex
    assert r"\subsection{1 Event Studies" not in tex


def test_candidate_only_citation_fails(tmp_path):
    mod = _module()
    bib_index = mod.parse_bib_index(_bib(tmp_path))
    markdown = "# Example\n\n## Abstract\n\nDo not cite (`Hawkes1971Spectra`).\n"

    try:
        mod.convert_markdown_to_latex(markdown, bib_index)
    except ValueError as exc:
        assert "candidate-only references" in str(exc)
    else:
        raise AssertionError("candidate-only citation should fail")


def test_raw_latex_table_block_passes_through(tmp_path):
    mod = _module()
    bib_index = mod.parse_bib_index(_bib(tmp_path))
    markdown = r"""
# Example

## Abstract

Text before.

\begin{table}[ht]
\centering
\caption{A raw diagnostic table.}
\begin{tabular}{l r}
\hline
Stage & Count \\
\hline
Computed & 46 \\
\hline
\end{tabular}
\end{table}
""".strip()

    tex = mod.convert_markdown_to_latex(markdown, bib_index)

    assert r"\begin{table}[ht]" in tex
    assert r"\caption{A raw diagnostic table.}" in tex
    assert r"\begin\{table\}" not in tex


def test_raw_latex_center_block_passes_through(tmp_path):
    mod = _module()
    bib_index = mod.parse_bib_index(_bib(tmp_path))
    markdown = r"""
# Example

## Abstract

\begin{center}
\small
\begin{tabular}{l r}
Stage & Count \\
Computed & 46 \\
\end{tabular}
\end{center}
""".strip()

    tex = mod.convert_markdown_to_latex(markdown, bib_index)

    assert r"\begin{center}" in tex
    assert r"\begin{tabular}{l r}" in tex
    assert r"\begin\{center\}" not in tex


def test_build_latex_writes_generated_marker(tmp_path):
    mod = _module()
    manuscript = tmp_path / "manuscript.md"
    output = tmp_path / "manuscript.tex"
    manuscript.write_text(
        "# Example\n\n## Abstract\n\nA citation (`MacKinlay1997EventStudies`).\n",
        encoding="utf-8",
    )

    result = mod.build_latex(
        manuscript=manuscript,
        bibliography=_bib(tmp_path),
        output=output,
    )

    tex = output.read_text(encoding="utf-8")
    assert result.cite_count == 1
    assert tex.startswith("% GENERATED FILE. DO NOT EDIT.")
    assert r"\cite{MacKinlay1997EventStudies}" in tex
