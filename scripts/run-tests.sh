#!/usr/bin/env bash
# Build and run every tests/test_*.mojo, then check the total and each file's
# count against the pins.
#
#   scripts/run-tests.sh            run all, compare with tests/test-count.txt and tests/test-counts.txt
#   UPDATE_TEST_COUNT=1 scripts/run-tests.sh   rewrite both pins after adding tests
#   TEST_DIR=tests/gpu scripts/run-tests.sh     another directory, with its own pins
#
# Why the pins: Mojo's TestSuite exits 0 when it runs zero tests, and a renamed
# or mis-imported test file can silently drop out. A suite that passed and a
# suite that never ran look identical unless something counts. The total
# (test-count.txt, one number) catches a lost file; the per-file pin
# (test-counts.txt, one `name count` line per file) also catches tests moved
# from one file to another, which leave the total unchanged.
#
# Tests are compiled with `mojo build` into build/tests/, so a failing file can
# be re-run without recompiling. TestSuite reports per-test times in
# milliseconds.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
# The compiler is the one uv.lock pins; MOJO replaces it (see scripts/build.sh).
MOJO="${MOJO:-uv run --frozen mojo}"
mkdir -p build/tests

test_dir="${TEST_DIR:-tests}"
files=("$test_dir"/test_*.mojo)
if [ "${#files[@]}" -eq 0 ] || [ ! -e "${files[0]}" ]; then
    echo "FAIL: no test files found under $test_dir/" >&2
    exit 1
fi

total_passed=0
declare -A per_file=()
failed_files=()
for f in "${files[@]}"; do
    name="$(basename "$f" .mojo)"
    # Reuse the CLI's explicit accelerator target on CPU-only hosts too.
    if ! build_out="$(MOJO="$MOJO" bash scripts/build.sh "$f" "build/tests/$name" 2>&1)"; then
        printf '%s\n' "$build_out" | grep -v Crashpad >&2
        failed_files+=("$name (build)")
        continue
    fi
    run_out="$("build/tests/$name" 2>&1)"
    rc=$?
    summary="$(grep -E '^Summary' <<<"$run_out" | tail -1)"
    passed="$(sed -n 's/.* \([0-9][0-9]*\) passed .*/\1/p' <<<"$summary")"
    failed="$(sed -n 's/.* \([0-9][0-9]*\) failed .*/\1/p' <<<"$summary")"
    if [ "$rc" -ne 0 ] || [ -z "$summary" ] || [ "${failed:-1}" != "0" ] || [ "${passed:-0}" -eq 0 ]; then
        grep -E 'FAIL|Error|error' <<<"$run_out" | head -20 >&2
        failed_files+=("$name (rc=$rc, passed=${passed:-?}, failed=${failed:-?})")
        continue
    fi
    printf '  %-22s %3d passed\n' "$name" "$passed"
    total_passed=$((total_passed + passed))
    per_file[$name]=$passed
done

if [ "${#failed_files[@]}" -gt 0 ]; then
    printf 'FAIL: %s\n' "${failed_files[@]}" >&2
    exit 1
fi

pin_file="$test_dir/test-count.txt"
files_pin="$test_dir/test-counts.txt"
actual_lines="$(for name in "${!per_file[@]}"; do printf '%s %s\n' "$name" "${per_file[$name]}"; done | LC_ALL=C sort)"
if [ "${UPDATE_TEST_COUNT:-0}" = "1" ]; then
    echo "$total_passed" >"$pin_file"
    printf '%s\n' "$actual_lines" >"$files_pin"
    echo "pinned test count: $total_passed (per file in $files_pin)"
    exit 0
fi
pinned="$(cat "$pin_file" 2>/dev/null || echo missing)"
if [ "$pinned" != "$total_passed" ]; then
    echo "FAIL: $total_passed tests passed but $pin_file pins $pinned." >&2
    echo "      If you added or removed tests on purpose: UPDATE_TEST_COUNT=1 $0" >&2
    exit 1
fi
echo "OK: $total_passed tests passed (pinned $pinned)"
if [ ! -f "$files_pin" ]; then
    echo "FAIL: $files_pin is missing, so tests cannot be checked file by file." >&2
    echo "      Create it: UPDATE_TEST_COUNT=1 $0" >&2
    exit 1
fi
pinned_lines="$(LC_ALL=C sort "$files_pin")"
if [ "$pinned_lines" != "$actual_lines" ]; then
    echo "FAIL: the tests per file differ from $files_pin:" >&2
    diff <(printf '%s\n' "$pinned_lines") <(printf '%s\n' "$actual_lines") | grep '^[<>]' | sed 's/^< /  pinned: /; s/^> /  ran:    /' >&2 || true
    echo "      If you moved, added or removed tests on purpose: UPDATE_TEST_COUNT=1 $0" >&2
    exit 1
fi
echo "OK: every file runs the number of tests pinned in $files_pin"
