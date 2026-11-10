#!/usr/bin/env bash
# `run --visual-interval` and `--json` with `--visual` are validated before a model loads.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
bin=${1:-build/mojo-dllm}
if [ ! -x "$bin" ]; then
    echo "FAIL: $bin is not built" >&2
    exit 1
fi

check_error() {
    local want_rc="$1" expected="$2"
    shift 2
    local out rc
    out="$("$bin" run --model tests/fixtures/llada-tokenizer.gguf --prompt x "$@" 2>&1)"
    rc=$?
    if [ "$rc" -ne "$want_rc" ]; then
        printf 'FAIL: expected exit %s, got %s\n%s\n' "$want_rc" "$rc" "$out" >&2
        return 1
    fi
    if ! grep -Fq -- "$expected" <<<"$out"; then
        printf 'FAIL: exit %s but output did not contain %s:\n%s\n' "$rc" "$expected" "$out" >&2
        return 1
    fi
}

check_error 1 'ERROR: --visual-interval must be at least 1' --visual --visual-interval 0 || exit 1
echo 'ok: --visual-interval rejects values below 1 before model loading'

check_error 1 'ERROR: --visual-interval needs --visual' --visual-interval 4 || exit 1
echo 'ok: --visual-interval requires --visual before model loading'

check_error 1 '--visual-interval' --visual --visual-interval abc || exit 1
echo 'ok: non-integer --visual-interval errors name the flag before model loading'

check_error 2 'ERROR: --json cannot be combined with --visual' --json --visual || exit 1
echo 'ok: --json with --visual is a usage error (exit 2) before model loading'

# The message goes to stderr: stdout stays empty for a consumer that parses it.
stdout_only="$("$bin" run --model tests/fixtures/llada-tokenizer.gguf --prompt x --json --visual 2>/dev/null)"
if [ -n "$stdout_only" ]; then
    printf 'FAIL: --json with --visual wrote to stdout:\n%s\n' "$stdout_only" >&2
    exit 1
fi
echo 'ok: the usage error leaves stdout empty'

check_error 1 'required metadata key missing' --json || exit 1
echo 'ok: --json alone still reaches model loading'

check_error 1 'required metadata key missing' --visual || exit 1
echo 'ok: --visual alone still reaches model loading'
