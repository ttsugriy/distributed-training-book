#!/usr/bin/env bash
# =============================================================================
# Build the whole book as a single, hyperlinked PDF.
#
#   ./scripts/build_pdf.sh
#
# Pipeline:
#   1. mkdocs build -f mkdocs.pdf.yml   -> renders the real site + a combined
#      /print_page/ (math, Mermaid, admonitions all rendered by the browser).
#   2. print_to_pdf.py                  -> headless Chromium prints that page to
#      a tagged PDF with clickable internal links + heading bookmarks.
#
# Output: build/distributed-training-book.pdf
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SITE_DIR="site_pdf"
OUTPUT="build/distributed-training-book.pdf"

# Activate conda base if nothing is active.
if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate base
  fi
fi
PYBIN="$(command -v python || command -v python3)"

# Sanity check: are the build deps present?
if ! "$PYBIN" -c "import mkdocs, playwright" >/dev/null 2>&1; then
  echo "error: build deps missing. Run ./scripts/install_pdf_deps.sh first." >&2
  exit 1
fi

echo "==> [1/2] Building site with print-site plugin -> ${SITE_DIR}/"
"$PYBIN" -m mkdocs build --clean -f mkdocs.pdf.yml -d "$SITE_DIR"

echo "==> [2/2] Printing combined page to PDF"
"$PYBIN" scripts/print_to_pdf.py --site-dir "$SITE_DIR" --output "$OUTPUT"

echo "==> Book ready: ${OUTPUT}"
