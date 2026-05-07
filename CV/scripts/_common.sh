#!/bin/bash
# Common setup for benchmark scripts.
# Source this from other scripts via:  source "$(dirname "$0")/_common.sh"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Default device: cuda if available, else mps, else cpu.
DEVICE="${DEVICE:-$(python - <<'PY'
import torch
if torch.cuda.is_available(): print("cuda")
elif torch.backends.mps.is_available(): print("mps")
else: print("cpu")
PY
)}"

echo "Repo: $REPO_ROOT"
echo "Device: $DEVICE"
echo "Python: $(command -v python)"
