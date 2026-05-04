#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export PRESET=${PRESET:-extreme}
export PHASES=${PHASES:-scaling,profile}
export SIMULATORS=${SIMULATORS:-sparqy}

bash "$SCRIPT_DIR/perlmutter_full_suite.sh" "$@"
