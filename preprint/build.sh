#!/bin/bash
# Build the preprint, and report the things that actually matter.
#
#   bash preprint/build.sh
#
# Always from a clean aux. An aux left over from a run whose labels have since changed
# produces "File ended while scanning use of \@writefile" and forty undefined references that
# have nothing to do with the source.
set -u
cd "$(dirname "$0")"
rm -f neurips_2023.aux neurips_2023.bbl neurips_2023.blg neurips_2023.out neurips_2023.toc
pdflatex -interaction=nonstopmode neurips_2023.tex > /dev/null 2>&1
bibtex neurips_2023 > /dev/null 2>&1
pdflatex -interaction=nonstopmode neurips_2023.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode neurips_2023.tex > /dev/null 2>&1
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
echo "a clean log is not a clean page: rasterise one and look at it"
echo "  gs -dNOPAUSE -dBATCH -sDEVICE=png16m -r62 -dFirstPage=N -dLastPage=N -sOutputFile=/tmp/p.png neurips_2023.pdf"
