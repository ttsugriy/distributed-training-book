#!/usr/bin/env bash
# =============================================================================
# Install the lean toolchain needed to build the PDF / book.
#
# Uses the active conda environment (or activates `base`). Installs only the
# doc-rendering deps -- NOT torch/numpy/etc. -- plus the Chromium browser that
# Playwright drives to print the PDF.
#
#   ./scripts/install_pdf_deps.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Activate conda base if no env is currently active.
if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate base
  fi
fi

echo "==> Using Python: $(command -v python || command -v python3)"
PYBIN="$(command -v python || command -v python3)"

echo "==> Installing PDF build dependencies"
"$PYBIN" -m pip install -r scripts/requirements-pdf.txt

echo "==> Installing Chromium for Playwright (downloads a private browser)"
"$PYBIN" -m playwright install chromium

echo "==> Done. Now run: ./scripts/build_pdf.sh"
