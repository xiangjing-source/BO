#!/usr/bin/env python3
"""Render project Markdown to self-contained HTML for LibreOffice conversion.

This is a documentation-build helper, not a production-model dependency.
Python-Markdown is required only when regenerating PDF/DOCX files.
"""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
import re

import markdown


STYLE = """
@page { size: A4; margin: 18mm 17mm 18mm 17mm; }
body {
  font-family: "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
  color: #172033; font-size: 10.5pt; line-height: 1.55;
  max-width: 180mm; margin: 0 auto;
}
h1 { color: #123d67; font-size: 22pt; border-bottom: 2px solid #2b78b8; }
h2 { color: #17598c; font-size: 16pt; margin-top: 18pt; }
h3 { color: #236d91; font-size: 13pt; margin-top: 13pt; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0 12pt; font-size: 9pt; }
th { background: #dceaf5; color: #123d67; }
th, td { border: 0.6pt solid #8ca6ba; padding: 4pt 5pt; vertical-align: top; }
blockquote {
  background: #eef6fb; border-left: 3pt solid #2b78b8;
  margin: 8pt 0; padding: 6pt 10pt;
}
code { font-family: "Noto Sans Mono CJK SC", "Consolas", monospace; font-size: 9pt; }
pre { background: #f3f6f8; border: 0.5pt solid #c8d2da; padding: 8pt; white-space: pre-wrap; }
img { display: block; max-width: 100%; height: auto; margin: 10pt auto; }
li { margin-bottom: 3pt; }
p { orphans: 3; widows: 3; }
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    body = markdown.markdown(
        source.read_text(encoding="utf-8"),
        extensions=("tables", "fenced_code", "sane_lists"),
    )
    # LibreOffice Writer/Web uses a legacy scale for the numeric HTML width
    # attribute (about 39 units per inch), not CSS's 96 px/in. A value of 240
    # therefore renders near 156 mm on A4 while the embedded source remains
    # 1600 px wide for zooming and print.
    body = re.sub(
        r"<img\s+([^>]*?)(?:\s*/?)>",
        r'<img \1 width="240" style="max-width:100%;height:auto;" />',
        body,
    )
    document = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{escape(source.stem)}</title><style>{STYLE}</style></head>"
        f"<body>{body}</body></html>\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")


if __name__ == "__main__":
    main()
