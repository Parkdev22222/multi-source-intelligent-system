#!/usr/bin/env bash
# Rebuild the paper and refuse anything that is not submission-ready.
#
# The layout has no slack: a single added sentence has repeatedly cost a
# seventh page, and it has been pushed that way more than once because the
# page count was not checked after an edit. This turns that check into
# something a session cannot forget.
#
#     scripts/check_paper.sh          # build and check
#     PAGES_MAX=6 scripts/check_paper.sh
#
# Exits non-zero on: wrong page count, any overfull box, any undefined
# reference or citation.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAGES_MAX="${PAGES_MAX:-6}"

cd "$ROOT/paper" || exit 2
latexmk -pdf -interaction=nonstopmode main.tex >/dev/null 2>&1

python3 - "$PAGES_MAX" <<'PY'
import re, sys, pathlib
limit = int(sys.argv[1])
log = pathlib.Path("main.log").read_text(errors="ignore")

m = re.findall(r"Output written on main\.pdf \((\d+) page", log)
if not m:
    print("FAIL  no PDF was written -- read paper/main.log")
    sys.exit(1)
pages = int(m[-1])
overfull = len(re.findall(r"Overfull", log))
undef = len(re.findall(r"undefined", log, re.I))

ok = pages <= limit and not overfull and not undef
print(f"pages {pages}/{limit} | overfull {overfull} | undefined {undef}"
      f"  ->  {'OK' if ok else 'FAIL'}")
if pages > limit:
    print(f"      {pages - limit} page(s) over. Cut prose; never the figure, a table row,")
    print( "      or a result. Recent overruns came from one rewritten paragraph.")
if overfull:
    for line in re.findall(r"Overfull[^\n]*", log)[:5]:
        print("      " + line.strip())
if undef:
    print("      run the full BibTeX build; a [?] in the PDF means a missing key")
sys.exit(0 if ok else 1)
PY
