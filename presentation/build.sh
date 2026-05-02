#!/usr/bin/env bash
# Render the Marp deck to PDF, PPTX, and HTML.
#
# Requires: Node.js (npm). On first run installs @marp-team/marp-cli locally.
# On Mac: `brew install node` if not present.
#
# Output: presentation/out/{slides.pdf, slides.pptx, slides.html}

set -euo pipefail
cd "$(dirname "$0")"

OUT="out"
mkdir -p "$OUT"

if ! command -v npx >/dev/null 2>&1; then
  echo "ERROR: npx not found. Install Node.js first (brew install node)."
  exit 1
fi

echo "==> rendering slides.md → PDF, PPTX, HTML"
npx -y @marp-team/marp-cli@latest --allow-local-files --pdf  slides.md -o "$OUT/slides.pdf"
npx -y @marp-team/marp-cli@latest --allow-local-files --pptx slides.md -o "$OUT/slides.pptx"
npx -y @marp-team/marp-cli@latest --allow-local-files --html slides.md -o "$OUT/slides.html"

echo
echo "Outputs:"
ls -lh "$OUT"
