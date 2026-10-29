#!/usr/bin/env bash
# `run --visual-interval` validates its options before trying to load a model.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
bin=${1:-build/mojo-dllm}
if [ ! -x "$bin" ]; then
    echo "FAIL: $bin is not built" >&2
    exit 1
fi

check_error() {
    local expected="$1"
    shift
    local out rc
    out="$("$bin" run --model tests/fixtures/llada-tokenizer.gguf --prompt x "$@" 2>&1)"
    rc=$?
    if [ "$rc" -ne 1 ]; then
        printf 'FAIL: expected exit 1, got %s\n%s\n' "$rc" "$out" >&2
        return 1
    fi
    if ! grep -Fq -- "$expected" <<<"$out"; then
        printf 'FAIL: exit 1 but output did not contain %s:\n%s\n' "$expected" "$out" >&2
        return 1
    fi
}

check_error 'ERROR: --visual-interval must be at least 1' --visual --visual-interval 0 || exit 1
echo 'ok: --visual-interval rejects values below 1 before model loading'

check_error 'ERROR: --visual-interval needs --visual' --visual-interval 4 || exit 1
echo 'ok: --visual-interval requires --visual before model loading'

check_error '--visual-interval' --visual --visual-interval abc || exit 1
echo 'ok: non-integer --visual-interval errors name the flag before model loading'
