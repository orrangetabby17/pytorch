#!/bin/bash
# Verify that all .so files in a libtorch lib/ directory have RPATH=$ORIGIN.
# Usage: smoke_test_libtorch_rpath.sh <lib_dir>
set -euo pipefail

lib_dir="${1:?Usage: $0 <lib_dir>}"

failed=0
for so in "$lib_dir"/*.so "$lib_dir"/*.so.*; do
    [ -f "$so" ] || continue
    rpath=$(patchelf --print-rpath "$so" 2>/dev/null || true)
    if [ "$rpath" != '$ORIGIN' ]; then
        echo "FAIL: $(basename "$so") rpath='$rpath' (expected \$ORIGIN)"
        failed=1
    fi
done
[ "$failed" -eq 0 ] || exit 1
echo "All .so files have rpath=\$ORIGIN"
