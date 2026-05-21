#!/usr/bin/env bash
# Lock GPU clocks for reproducible benchmarking (docs/benchmarking-methodology.md).
# Boost clocks drift with temperature and load, which silently corrupts the
# microsecond-scale kernel timings in Phase 1+. Lock them before benchmarking.
#
# Needs root:
#   sudo bash scripts/lock_clocks.sh lock          # lock (default SM clock)
#   sudo bash scripts/lock_clocks.sh lock 2520     # lock at a specific SM MHz
#   sudo bash scripts/lock_clocks.sh reset         # restore default behaviour
#
# After locking, record the values shown below into docs/results/env-report.md.
set -euo pipefail

ACTION="${1:-lock}"
SM_MHZ="${2:-2520}"        # sustainable SM clock, below the 3165 MHz max boost
MEM_MHZ=10501              # RTX 4090 max memory clock (see env-report.md)

case "$ACTION" in
  lock)
    nvidia-smi -pm 1                         # persistence mode
    nvidia-smi -lgc "${SM_MHZ},${SM_MHZ}"    # lock SM/graphics clock
    nvidia-smi -lmc "${MEM_MHZ},${MEM_MHZ}"  # lock memory clock
    echo "Locked: SM ${SM_MHZ} MHz / MEM ${MEM_MHZ} MHz"
    ;;
  reset)
    nvidia-smi -rgc                          # reset SM clock
    nvidia-smi -rmc                          # reset memory clock
    echo "Clocks reset to default."
    ;;
  *)
    echo "usage: sudo bash $0 [lock|reset] [sm_mhz]" >&2
    exit 1
    ;;
esac

nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu --format=csv
