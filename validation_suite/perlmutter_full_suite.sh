#!/bin/bash
set -euo pipefail

ROOT=${ROOT:-$HOME/recombination_simulator}
REPO=${REPO:-$ROOT/sparq5}
RESULTS_ROOT=${RESULTS_ROOT:-$REPO/validation_suite/results}
SLIM_BIN_DEFAULT=${SLIM_BIN:-$HOME/software/slim/5.1/bin/slim}
PRESET=${PRESET:-full}
PHASES=${PHASES:-accuracy,speed,scaling,profile}
SIMULATORS=${SIMULATORS:-sparqy,slim}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR=${1:-$RESULTS_ROOT/perlmutter_${PRESET}_$TIMESTAMP}

export OMP_PLACES=${OMP_PLACES:-cores}
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export OMP_STACKSIZE=${OMP_STACKSIZE:-16M}
export SLIM_BIN=$SLIM_BIN_DEFAULT

if command -v module >/dev/null 2>&1; then
    module load python 2>/dev/null || module load cray-python 2>/dev/null || true
fi

echo "=============================================="
echo " sparqy validation suite on Perlmutter"
echo "=============================================="
echo "Repo:      $REPO"
echo "Preset:    $PRESET"
echo "Phases:    $PHASES"
echo "Simulators:$SIMULATORS"
echo "Results:   $RESULTS_DIR"
echo "SLiM:      $SLIM_BIN"
echo "Hostname:  $(hostname)"
echo "Date:      $(date)"
echo "OMP stack: $OMP_STACKSIZE"

NEEDS_SLIM=0
if [[ "$PHASES" == *accuracy* || "$PHASES" == *speed* ]]; then
    if [[ "$SIMULATORS" == *slim* ]]; then
        NEEDS_SLIM=1
    fi
fi

if [ "$NEEDS_SLIM" -eq 1 ] && [ ! -f "$SLIM_BIN" ]; then
    echo "ERROR: SLiM not found at $SLIM_BIN"
    exit 1
fi

cd "$REPO"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j16

python3 validation_suite/run_suite.py run \
    --preset "$PRESET" \
    --phases "$PHASES" \
    --simulators "$SIMULATORS" \
    --results-dir "$RESULTS_DIR" \
    --sparqy-bin "$REPO/build/sparqy" \
    --slim-bin "$SLIM_BIN"

echo ""
echo "Suite complete."
echo "Results written to: $RESULTS_DIR"
echo "Generate local figures later with:"
echo "  python3 validation_suite/plot_results.py $RESULTS_DIR"
