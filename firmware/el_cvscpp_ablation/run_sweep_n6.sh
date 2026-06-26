#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CONFIGS=${EL_CVSCPP_SWEEP_CONFIGS:-"8x8 16x8 16x16 32x16 32x32 64x32"}
VARIANTS=${EL_CVSCPP_SWEEP_VARIANTS:-"legacy_c cpp_direct_c_backend cpp_m55 cpp_generic rltools_generic"}

for config in $CONFIGS; do
  for variant in $VARIANTS; do
    printf '\n== EL_C_vsCpp %s %s ==\n' "$config" "$variant"
    sh "$SCRIPT_DIR/flash_and_run_n6.sh" --config "$config" --variant "$variant" "$@"
  done
done
