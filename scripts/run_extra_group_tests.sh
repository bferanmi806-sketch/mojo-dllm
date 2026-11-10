#!/usr/bin/env bash
# run_extra_group_tests.sh: the Python tests that need numpy, gguf or PyYAML (the fixtures and lint groups).
#
# Each of these files skips itself when a package is missing, so a skip here means nothing ran: it fails the step.
# CI and ci-local.sh both run this script, so the two cannot list different files.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
rc=0
out="$(uv run --frozen --only-group test --only-group fixtures --only-group lint pytest \
    tests/python/test_parity_compare.py tests/python/test_action_pins.py tests/python/test_dg_bound.py \
    tests/python/test_gen_fixtures.py -q -rs 2>&1)" || rc=$?
printf '%s\n' "$out"
[ "$rc" -eq 0 ] && ! grep -q 'skipped' <<<"$out"
