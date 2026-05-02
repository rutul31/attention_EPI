# Source-only helper. Ensures the project's conda env (default: Bioinfomatrics) is active.
# Usage in any other shell script:
#   source "$(dirname "$0")/_activate_env.sh"
#
# Configurable via:
#   EPINTLM_ENV     conda env name to activate (default: Bioinfomatrics)
#   SKIP_ACTIVATE   set to 1 to skip activation entirely (e.g. caller already activated)

EPINTLM_ENV="${EPINTLM_ENV:-Bioinfomatrics}"

if [[ "${SKIP_ACTIVATE:-0}" == "1" ]]; then
  return 0 2>/dev/null || exit 0
fi

# Already in the right env? Skip.
if [[ "${CONDA_DEFAULT_ENV:-}" == "$EPINTLM_ENV" ]]; then
  return 0 2>/dev/null || exit 0
fi

# Try to activate. Load module first if conda isn't already on PATH (HPC pattern).
if ! command -v conda >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1; then
    module load Anaconda3 2>/dev/null || true
  fi
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not on PATH. Run 'module load Anaconda3' (HPC) or install Miniconda." >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$EPINTLM_ENV"; then
  echo "ERROR: conda env '$EPINTLM_ENV' not found. Available:" >&2
  conda env list >&2
  echo "Override with EPINTLM_ENV=<your_env_name>." >&2
  return 1 2>/dev/null || exit 1
fi

conda activate "$EPINTLM_ENV"

# Final sanity check — fail fast with a clear message if torch isn't there.
if ! python -c "import torch" >/dev/null 2>&1; then
  echo "ERROR: torch not importable in env '$EPINTLM_ENV'. Run: pip install -r requirements.txt" >&2
  return 1 2>/dev/null || exit 1
fi
