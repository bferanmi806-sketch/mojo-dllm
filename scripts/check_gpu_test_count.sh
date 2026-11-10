#!/usr/bin/env bash
# Count the `def test_` functions of tests/gpu/test_*.mojo and compare the total with tests/gpu/test-count.txt.
#
# Needs no GPU and no compiler. scripts/run-gpu-tests.sh compares the passing count with the same pin, but it
# refuses to run on a machine without a GPU, and CI only builds the suite, so a test added without re-pinning, or
# a file that stops defining tests, would otherwise pass every gate that can run there.
#
#   scripts/check_gpu_test_count.sh                          tests/gpu against tests/gpu/test-count.txt
#   TEST_DIR=some/dir scripts/check_gpu_test_count.sh        another directory, with its own pin
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
test_dir="${TEST_DIR:-tests/gpu}"
files=("$test_dir"/test_*.mojo)
if [ ! -e "${files[0]}" ]; then
    echo "FAIL: no test files found under $test_dir/" >&2
    exit 1
fi

total=0
for f in "${files[@]}"; do
    # grep -c prints 0 and exits 1 for a file with no match; any other failure is a real one.
    n="$(grep -c '^def test_' "$f")"
    rc=$?
    if [ "$rc" -gt 1 ]; then
        echo "FAIL: grep could not read $f" >&2
        exit 1
    fi
    if [ "$n" -eq 0 ]; then
        echo "FAIL: $f defines no tests (a file that runs zero tests passes silently)" >&2
        exit 1
    fi
    total=$((total + n))
done

pin_file="$test_dir/test-count.txt"
pinned="$(cat "$pin_file" 2>/dev/null)"
if ! [[ "$pinned" =~ ^[0-9]+$ ]]; then
    echo "FAIL: the pin file $pin_file does not hold a test count (found: ${pinned:-nothing})" >&2
    exit 1
fi
if [ "$pinned" != "$total" ]; then
    echo "FAIL: $test_dir defines $total tests but $pin_file pins $pinned." >&2
    echo "      If you added or removed tests on purpose: UPDATE_TEST_COUNT=1 scripts/run-gpu-tests.sh" >&2
    exit 1
fi
echo "OK: $total tests defined in $test_dir (pinned $pinned)"
