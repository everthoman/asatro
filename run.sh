#!/usr/bin/env bash
# Launch the Asatro web app in the `asatro` conda env.
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate asatro
exec uvicorn asatro.app:app --host 0.0.0.0 --port "${ASATRO_PORT:-5023}" "$@"
