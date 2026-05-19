# weak-label-hawkes-crypto-futures

This repository hosts the paper artifact for

  Joohyoung Jeon (Korea University)
  "Diagnostic Weak-Label Hawkes Validation for Event-Driven Crypto Futures"
  arXiv: [identifier pending]

It is maintained under the `ds-academy` organization for archival
continuity.

## What is in this repository

The artifact reproduces the paper-facing PDF, the 14 tables, and the 3
figures from a fixed set of CSV artifacts. It does not re-run the
upstream activation, weak-labeling, or Hawkes-fitting stages from raw
market data; those stages produce the CSV artifacts and remain
upstream of this release.

The repository contains:

- `paper/manuscript.md` — the paper source in markdown.
- `paper/latex/` — the LaTeX builders and the rendered manuscript.
- `paper/latex/arxiv/main.pdf` — the rendered PDF.
- `paper/latex/arxiv/main.tex`, `main.bbl`, `references.bib`,
  `figures/` — the arXiv submission package.
- `paper/j0_expansion/tables/table_1_*.csv` ... `table_14_*.csv` —
  the 14 paper-facing CSV table artifacts whose SHA-256 digests are
  recorded in Appendix C of the manuscript.
- `paper/figures/` — SVG sources for the 3 figures.
- `paper/scripts/` — the build scripts that materialize the J1 audit
  tables (Tables 7, 8, 9) and the K5 Tier-1 expansion tables
  (Tables 11, 12, 13, 14).
- `paper/tests/` — the 21 paper tests that verify the rendering
  pipeline.
- `run.sh` — the single reader-facing reproduction command.
- `CITATION.cff` — machine-readable citation metadata.

## How to reproduce

```
git clone https://github.com/ds-academy/weak-label-hawkes-crypto-futures.git
cd weak-label-hawkes-crypto-futures
bash run.sh
```

`run.sh` regenerates `paper/latex/manuscript.tex`,
`paper/latex/arxiv/main.tex`, `paper/latex/arxiv/main.bbl`, and
`paper/latex/arxiv/main.pdf`, then runs the 21 paper tests.

The script requires Python 3.12+ with `pandas`, `matplotlib`, and
`pytest` available, plus a TeX Live installation that can run
`pdflatex` and `bibtex`.

## License

MIT (see `LICENSE`).

## Citation

See `CITATION.cff` for the machine-readable citation record. GitHub
will render a `Cite this repository` button on the landing page that
exposes the same metadata as BibTeX and APA.
