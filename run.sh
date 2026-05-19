#!/usr/bin/env bash
# Reader-facing reproduction command for the rendered paper package.
#
# This script regenerates the LaTeX manuscript from manuscript.md and the
# arXiv source package (main.tex, main.bbl, references.bib, figures, and
# main.pdf). It does NOT rerun the upstream activation, weak-labeling, or
# Hawkes-fitting stages; those produce the input CSVs whose SHA-256
# digests are listed in Appendix C of the manuscript and are part of
# the public artifact release rather than re-derived here.
#
# Usage:
#   In the authoring repository (which uses the long internal layout):
#       bash research/2026-weak-label-exo-hawkes/004_paper/run.sh
#
#   In the public artifact repository at
#   https://github.com/ds-academy/weak-label-hawkes-crypto-futures
#   (which flattens the layout to `paper/`):
#       bash run.sh
#
# Both environments work because the script resolves its own directory
# and walks up to the repository root, then constructs the paper-package
# path relative to that root. Set REPO_LAYOUT=public to override the
# default (authoring) layout when the repository top-level is the
# public-mirror layout.
#
# Optional environment variables:
#     PYTHON          Python interpreter (default: ./.venv/bin/python
#                     from the repository root)
#     REPO_LAYOUT     authoring (default) | public

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_LAYOUT="${REPO_LAYOUT:-authoring}"
case "${REPO_LAYOUT}" in
    authoring)
        REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
        PAPER_DIR="research/2026-weak-label-exo-hawkes/004_paper"
        ;;
    public)
        REPO_ROOT="${SCRIPT_DIR}"
        PAPER_DIR="paper"
        ;;
    *)
        echo "error: REPO_LAYOUT must be 'authoring' or 'public' (got '${REPO_LAYOUT}')" >&2
        exit 2
        ;;
esac

PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "error: Python interpreter not found at ${PYTHON}" >&2
    echo "       Set PYTHON=/path/to/python and rerun." >&2
    exit 1
fi

cd "${REPO_ROOT}"

echo "[1/3] Regenerating ${PAPER_DIR}/latex/manuscript.tex from manuscript.md ..."
"${PYTHON}" "${PAPER_DIR}/latex/build_latex.py"

echo "[2/3] Regenerating arXiv source package and compiling main.pdf ..."
"${PYTHON}" "${PAPER_DIR}/latex/build_arxiv_package.py" --compile-pdf

echo "[3/3] Running paper-side test suite ..."
"${PYTHON}" -m pytest "${PAPER_DIR}/tests" -q

echo
echo "Done. Outputs (relative to ${REPO_ROOT}):"
echo "  - ${PAPER_DIR}/latex/manuscript.tex"
echo "  - ${PAPER_DIR}/latex/arxiv/main.tex"
echo "  - ${PAPER_DIR}/latex/arxiv/main.bbl"
echo "  - ${PAPER_DIR}/latex/arxiv/main.pdf"
