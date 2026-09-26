#!/bin/bash
# Build the preprint, and report the things that actually matter.
#
#   bash preprint/build.sh
#
# latexmk, not a hand-rolled pass sequence. Running pdflatex/bibtex by hand raced with itself
# more than once and produced an empty .bbl with "I found no \bibdata command", which surfaces
# as several dozen undefined citations that have nothing to do with the source.
set -u
cd "$(dirname "$0")"
# latexmk -C alone has left a stale aux behind after an interrupted run, which then throws
# "Missing \begin{document}" from inside hyperref's .out file. Remove them outright.
rm -f neurips_2023.aux neurips_2023.out neurips_2023.toc neurips_2023.bbl \
      neurips_2023.blg neurips_2023.fls neurips_2023.fdb_latexmk
latexmk -C > /dev/null 2>&1
latexmk -pdf -interaction=nonstopmode -halt-on-error neurips_2023.tex > /tmp/latexmk.log 2>&1
python3 - <<'PY'
import re
log = open("neurips_2023.log", errors="ignore").read()
pages = re.findall(r"Output written on .*?\((\d+) pages", log)
errs = re.findall(r"(?m)^! .*", log)
print(f"pages {pages[0] if pages else '?'} | errors {len(errs)} | "
      f"undefined {len(re.findall('undefined', log))} | "
      f"overfull {len(re.findall('Overfull', log))} | "
      f"em dashes {open('neurips_2023.tex').read().count(chr(8212))}")
for e in errs[:5]:
    print("  ", e.strip())
PY
# do not pipe this script through head: SIGPIPE kills it mid-build
echo "a clean log is not a clean page: rasterise one and look at it"
echo "  gs -dNOPAUSE -dBATCH -sDEVICE=png16m -r62 -dFirstPage=N -dLastPage=N -sOutputFile=/tmp/p.png neurips_2023.pdf"
