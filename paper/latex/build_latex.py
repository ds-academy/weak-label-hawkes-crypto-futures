from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent
DEFAULT_MANUSCRIPT = PAPER_DIR / "manuscript.md"
DEFAULT_BIB = SCRIPT_DIR / "references.bib"
DEFAULT_OUTPUT = SCRIPT_DIR / "manuscript.tex"

CITATION_KEY_PATTERN = r"[A-Z][A-Za-z0-9]*[0-9]{4}[A-Za-z0-9]*"
CITATION_KEY_RE = re.compile(rf"^{CITATION_KEY_PATTERN}$")
BACKTICK_CITATION_RE = re.compile(rf"`({CITATION_KEY_PATTERN})`")
CITATION_CLUSTER_RE = re.compile(
    rf"\((`{CITATION_KEY_PATTERN}`(?:\s*;\s*`{CITATION_KEY_PATTERN}`)*)\)"
)
INLINE_MATH_RE = re.compile(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$")
BIB_ENTRY_RE = re.compile(r"@(?P<kind>\w+)\{(?P<key>[^,]+),(?P<body>.*?)(?=\n@\w+\{|\Z)", re.S)


@dataclass(frozen=True)
class BibIndex:
    keys: frozenset[str]
    candidate_keys: frozenset[str]


@dataclass(frozen=True)
class BuildResult:
    manuscript: Path
    bibliography: Path
    output: Path
    cite_count: int
    cited_keys: tuple[str, ...]


def parse_bib_index(path: Path) -> BibIndex:
    text = path.read_text(encoding="utf-8")
    keys: set[str] = set()
    candidate_keys: set[str] = set()
    for match in BIB_ENTRY_RE.finditer(text):
        key = match.group("key").strip()
        keys.add(key)
        body = match.group("body")
        keywords_match = re.search(r"keywords\s*=\s*\{([^}]*)\}", body)
        if keywords_match:
            keywords = {part.strip() for part in keywords_match.group(1).split(",")}
            if "candidate" in keywords:
                candidate_keys.add(key)
    return BibIndex(keys=frozenset(keys), candidate_keys=frozenset(candidate_keys))


def extract_manuscript_citation_keys(markdown: str) -> set[str]:
    body = markdown.split("\n## References\n", maxsplit=1)[0]
    return set(BACKTICK_CITATION_RE.findall(body))


def validate_citations(markdown: str, bib_index: BibIndex) -> tuple[str, ...]:
    cited_keys = extract_manuscript_citation_keys(markdown)
    missing = sorted(cited_keys - bib_index.keys)
    if missing:
        raise ValueError(f"citation keys missing from references.bib: {missing}")
    candidate_cited = sorted(cited_keys & bib_index.candidate_keys)
    if candidate_cited:
        raise ValueError(f"candidate-only references are cited by manuscript: {candidate_cited}")
    return tuple(sorted(cited_keys))


def escape_latex_text(text: str) -> str:
    replacements = {
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
    return "".join(replacements.get(char, char) for char in text)


def _restore_placeholders(text: str, placeholders: dict[str, str]) -> str:
    for token, value in placeholders.items():
        text = text.replace(token, value)
    return text


def convert_inline(text: str, bib_index: BibIndex) -> str:
    placeholders: dict[str, str] = {}

    def protect(value: str) -> str:
        token = f"ZZZLATEXPLACEHOLDER{len(placeholders)}ZZZ"
        placeholders[token] = value
        return token

    def math_repl(match: re.Match[str]) -> str:
        return protect(f"${match.group(1)}$")

    def citation_cluster_repl(match: re.Match[str]) -> str:
        keys = BACKTICK_CITATION_RE.findall(match.group(1))
        _validate_inline_keys(keys, bib_index)
        return protect(r"\cite{" + ",".join(keys) + "}")

    def backtick_repl(match: re.Match[str]) -> str:
        content = match.group(1)
        if CITATION_KEY_RE.match(content):
            _validate_inline_keys([content], bib_index)
            return protect(r"\cite{" + content + "}")
        return protect(r"\texttt{" + escape_latex_text(content) + "}")

    text = INLINE_MATH_RE.sub(math_repl, text)
    text = CITATION_CLUSTER_RE.sub(citation_cluster_repl, text)
    text = re.sub(r"`([^`]+)`", backtick_repl, text)
    return _restore_placeholders(escape_latex_text(text), placeholders)


def _validate_inline_keys(keys: list[str], bib_index: BibIndex) -> None:
    missing = sorted(set(keys) - bib_index.keys)
    if missing:
        raise ValueError(f"citation keys missing from references.bib: {missing}")
    candidate = sorted(set(keys) & bib_index.candidate_keys)
    if candidate:
        raise ValueError(f"candidate-only references are cited by manuscript: {candidate}")


def strip_html_comments(markdown: str) -> list[str]:
    lines = markdown.splitlines()
    output: list[str] = []
    in_comment = False
    for line in lines:
        stripped = line.strip()
        if in_comment:
            if "-->" in stripped:
                in_comment = False
            continue
        if stripped.startswith("<!--"):
            if "-->" not in stripped:
                in_comment = True
            continue
        output.append(line)
    return output


def heading_title(line: str, prefix: str) -> str:
    title = line.removeprefix(prefix).strip()
    title = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", title)
    title = re.sub(r"^Appendix\s+[A-Z]\.\s*", "", title)
    return title


def convert_markdown_to_latex(markdown: str, bib_index: BibIndex) -> str:
    cited_keys = validate_citations(markdown, bib_index)
    lines = strip_html_comments(markdown)
    title = "Untitled"
    body: list[str] = []
    paragraph: list[str] = []
    in_abstract = False
    appendix_started = False
    skip_references = False
    i = 0

    def flush_paragraph() -> None:
        if not paragraph:
            return
        text = " ".join(part.strip() for part in paragraph if part.strip())
        body.append(convert_inline(text, bib_index))
        body.append("")
        paragraph.clear()

    def close_abstract() -> None:
        nonlocal in_abstract
        if in_abstract:
            flush_paragraph()
            body.append(r"\end{abstract}")
            body.append("")
            in_abstract = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if skip_references and not line.startswith("## Appendix"):
            i += 1
            continue
        if skip_references and line.startswith("## Appendix"):
            skip_references = False

        if line.startswith("# "):
            title = heading_title(line, "# ")
            i += 1
            continue

        if not stripped:
            flush_paragraph()
            i += 1
            continue

        if stripped == "```bash" or stripped == "```":
            flush_paragraph()
            fence = stripped
            code: list[str] = []
            i += 1
            while i < len(lines) and lines[i].strip() != "```":
                code.append(lines[i])
                i += 1
            if i < len(lines) and lines[i].strip() == "```":
                i += 1
            body.append(r"\begin{verbatim}")
            body.extend(code)
            body.append(r"\end{verbatim}")
            body.append("")
            continue

        if stripped == "$$":
            flush_paragraph()
            math_lines: list[str] = []
            i += 1
            while i < len(lines) and lines[i].strip() != "$$":
                math_lines.append(lines[i])
                i += 1
            if i < len(lines) and lines[i].strip() == "$$":
                i += 1
            body.append(r"\[")
            body.extend(math_lines)
            body.append(r"\]")
            body.append("")
            continue

        if stripped.startswith((r"\begin{center}", r"\begin{table}")):
            flush_paragraph()
            end_token = r"\end{center}" if stripped.startswith(r"\begin{center}") else r"\end{table}"
            raw_latex: list[str] = [line]
            i += 1
            while i < len(lines):
                raw_latex.append(lines[i])
                if lines[i].strip() == end_token:
                    i += 1
                    break
                i += 1
            body.extend(raw_latex)
            body.append("")
            continue

        if line.startswith("## "):
            flush_paragraph()
            close_abstract()
            section = heading_title(line, "## ")
            if section == "Abstract":
                body.append(r"\begin{abstract}")
                in_abstract = True
            elif section == "References":
                body.append(r"\bibliographystyle{plain}")
                body.append(r"\bibliography{references}")
                body.append("")
                skip_references = True
            elif line.startswith("## Appendix"):
                if not appendix_started:
                    body.append(r"\appendix")
                    body.append("")
                    appendix_started = True
                body.append(r"\section{" + escape_latex_text(section) + "}")
                body.append("")
            else:
                body.append(r"\section{" + escape_latex_text(section) + "}")
                body.append("")
            i += 1
            continue

        if line.startswith("### "):
            flush_paragraph()
            section = heading_title(line, "### ")
            body.append(r"\subsection{" + escape_latex_text(section) + "}")
            body.append("")
            i += 1
            continue

        if re.match(r"^\d+\.\s+", stripped) or stripped.startswith("- "):
            flush_paragraph()
            ordered = bool(re.match(r"^\d+\.\s+", stripped))
            environment = "enumerate" if ordered else "itemize"
            items: list[list[str]] = []
            current: list[str] = []
            while i < len(lines):
                item_line = lines[i]
                item_stripped = item_line.strip()
                numbered = re.match(r"^\d+\.\s+(.*)", item_stripped)
                bulleted = re.match(r"^-\s+(.*)", item_stripped)
                if ordered and numbered:
                    if current:
                        items.append(current)
                    current = [numbered.group(1)]
                    i += 1
                    continue
                if not ordered and bulleted:
                    if current:
                        items.append(current)
                    current = [bulleted.group(1)]
                    i += 1
                    continue
                if item_line.startswith("  ") and current:
                    current.append(item_stripped)
                    i += 1
                    continue
                break
            if current:
                items.append(current)
            body.append(r"\begin{" + environment + "}")
            for item_lines in items:
                head = convert_inline(item_lines[0], bib_index)
                continuations = item_lines[1:]
                if continuations and all(
                    re.fullmatch(r"`[^`]+`", line) for line in continuations
                ):
                    rendered = r"\item " + head
                    for cont in continuations:
                        rendered += r" \\" + "\n" + r"\hspace*{2em}" + convert_inline(cont, bib_index)
                    body.append(rendered)
                else:
                    body.append(r"\item " + convert_inline(" ".join(item_lines), bib_index))
            body.append(r"\end{" + environment + "}")
            body.append("")
            continue

        paragraph.append(line)
        i += 1

    flush_paragraph()
    close_abstract()

    preamble = [
        "% GENERATED FILE. DO NOT EDIT.",
        "% Source: ../manuscript.md",
        "% Build: python build_latex.py",
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{hyperref}",
        r"\title{" + escape_latex_text(title) + "}",
        r"\author{Joohyoung Jeon \\ Korea University \\ Seoul, Republic of Korea}",
        r"\date{}",
        "",
        r"\begin{document}",
        r"\maketitle",
        "",
    ]
    ending = ["", r"\end{document}", ""]
    return "\n".join(preamble + body + ending)


def build_latex(
    manuscript: Path = DEFAULT_MANUSCRIPT,
    bibliography: Path = DEFAULT_BIB,
    output: Path = DEFAULT_OUTPUT,
) -> BuildResult:
    bib_index = parse_bib_index(bibliography)
    markdown = manuscript.read_text(encoding="utf-8")
    cited_keys = validate_citations(markdown, bib_index)
    tex = convert_markdown_to_latex(markdown, bib_index)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(tex, encoding="utf-8")
    return BuildResult(
        manuscript=manuscript,
        bibliography=bibliography,
        output=output,
        cite_count=tex.count(r"\cite{"),
        cited_keys=cited_keys,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX manuscript from Markdown source.")
    parser.add_argument("--manuscript", type=Path, default=DEFAULT_MANUSCRIPT)
    parser.add_argument("--bibliography", type=Path, default=DEFAULT_BIB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = build_latex(
        manuscript=args.manuscript,
        bibliography=args.bibliography,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "manuscript": str(result.manuscript),
                "bibliography": str(result.bibliography),
                "output": str(result.output),
                "cite_count": result.cite_count,
                "cited_keys": list(result.cited_keys),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
