"""
Find numbers in prose that disagree with, or cannot be traced to, the tables.

Every numerical error in the April 2026 draft was a transcription: a standard
error of 4.8 where the code produced 7.45, an out-of-sample table matching no
estimator in the repository, Friday counts from the wrong sample period. Later,
three cells of a ten-row table drifted by 0.1 between the generated table and a
hand-typed copy, purely from rounding at the boundary.

Rather than generate prose from templates, which makes writing awkward, this
checks it. Every decimal in prose must either appear in a generated table under
tabs/, be a named value in tabs/_numbers.json, or be listed in ALLOWED with a
reason. A number that drifts stops matching and gets reported.

Run: make check-prose
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
TABS = ROOT / "tabs"
NUMBERS = TABS / "_numbers.json"

PROSE = [ROOT / "README.md", ROOT / "ms" / "ms.tex"]

# Literals that are not ours to generate: figures reported by Patel et al.,
# years, and quantities fixed by the design rather than estimated.
ALLOWED = {
    "18.2",
    "15.1",
    "5.5",
    "120.9",
    "139.1",
    "123.3",
    "86.1",
    "34866",
    "0.05",
    "0.5",
    "1.96",
    "0.08",
    "2.0",
    "1.0",
    "0.72",
    "11.0",
    "1.1",
    "0.001",  # quoted as a threshold, not as an estimate
    "1.10",  # odds ratio reported by Patel et al., not estimated here
}
YEAR = re.compile(r"^(19|20)\d\d$")
NUM = re.compile(r"(?<![\w.\-])\d+\.\d+(?![\w])")


def strip_noise(text, is_tex):
    if is_tex:
        text = text.split(r"\begin{document}", 1)[-1]
        text = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", "", text, flags=re.S)
        text = re.sub(
            r"\\(input|includegraphics|usepackage)(\[[^\]]*\])?\{[^}]*\}", "", text
        )
        text = re.sub(r"%.*", "", text)
    else:
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        text = re.sub(r"\]\([^)]*\)", "]()", text)
    return text


def parse_md_tables(text):
    """Every markdown table in a document, as a list of rows of cells."""
    tables, current = [], []
    for line in text.splitlines():
        if line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not all(set(c) <= set("-: ") for c in cells):
                current.append(cells)
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return tables


def generated_tables():
    out = {}
    for path in sorted(TABS.glob("t*.md")):
        rows = parse_md_tables(path.read_text())
        if rows:
            out[path.stem] = rows[0]
    return out


def referenced_tables(text, is_tex):
    """Table stems this document actually points at.

    Scoping matters. Accepting a value because it appears in *any* generated
    table validates existence rather than correspondence: 122.1 sat in
    ms.tex for months describing the release-day mean, which is 144.9, and
    passed because 122.1 happens to be a cell in t02, a table the manuscript
    never mentions.
    """
    if is_tex:
        return set(re.findall(r"\\input\{[^}]*?/(t[0-9a-z_]+)\}", text))
    return set(re.findall(r"\((?:\./)?tabs/(t[0-9a-z_]+)\.md\)", text))


def generated_values(gen, stems=None):
    """Numeric cells in the generated tables this document references.

    A number quoted from a table the document shows is traceable even if it
    has no macro. What is left over is the genuinely orphan set: numbers that
    appear in prose and nowhere the reader can check them.
    """
    vals = set()
    for stem, rows in gen.items():
        if stems is not None and stem not in stems:
            continue
        for row in rows:
            for cell in row:
                token = cell.strip().lstrip("+")
                if NUM.fullmatch(token):
                    vals.add(token)
                    vals.add(token.rstrip("0").rstrip("."))
                    try:
                        vals.add(f"{float(token):.1f}")
                        vals.add(f"{float(token):.2f}")
                    except ValueError:
                        pass
    return vals


def check_numbers(name, text, is_tex, defined):
    body = strip_noise(text, is_tex)
    bad = []
    for m in NUM.finditer(body):
        token = m.group(0)
        if token in ALLOWED or YEAR.match(token) or token in defined:
            continue
        bad.append((token, body[: m.start()].count("\n") + 1))
    if bad:
        print(f"\n{name}: {len(bad)} number(s) not traceable to tabs/_numbers.json")
        for token, line in bad[:30]:
            print(f"  {name}:{line}: {token}")
    return len(bad)


def main():
    if not NUMBERS.exists():
        print("tabs/_numbers.json missing; run the pipeline first")
        return 1
    # the extractor reads bare decimals out of prose, so a recorded value
    # formatted with a sign ("-0.0074") never matched the "0.0074" it quotes
    named = {
        v["formatted"].lstrip("+-\u2212")
        for v in json.loads(NUMBERS.read_text()).values()
    }
    gen = generated_tables()

    total = 0
    for path in PROSE:
        if not path.exists():
            continue
        name = path.relative_to(ROOT).as_posix()
        text = path.read_text()
        is_tex = path.suffix == ".tex"
        stems = referenced_tables(text, is_tex)
        defined = named | generated_values(gen, stems)
        print(f"{name}: checking against {len(stems)} referenced table(s)")
        total += check_numbers(name, text, is_tex, defined)

    if total == 0:
        print("prose check: clean")
        return 0
    print(f"\nprose check: {total} issue(s). Either record() the number in the")
    print("pipeline and quote it, or add it to ALLOWED with a reason.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
