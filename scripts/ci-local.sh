#!/usr/bin/env bash
# ci-local.sh: run the same gates as .github/workflows/ci.yml on your machine.
#
#   scripts/ci-local.sh                 everything
#   scripts/ci-local.sh --fast          hygiene + format + build (the pre-commit set)
#   scripts/ci-local.sh --install-hooks point git at .githooks and exit
#
# Every step runs even after an earlier one fails, and the summary lists all
# of them. A required tool that is missing is a FAIL, never a skip: a gate that
# did not run and a gate that passed must not look the same.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root" || exit 1

mode=full
case "${1:-}" in
    "") ;;
    --fast) mode=fast ;;
    --install-hooks)
        git config core.hooksPath .githooks
        echo "ci-local: hooks enabled (pre-commit: --fast, pre-push: full)."
        echo "ci-local: disable with: git config --unset core.hooksPath"
        exit 0
        ;;
    -h | --help)
        sed -n '2,10p' "$0"
        exit 0
        ;;
    *)
        echo "ci-local: unknown flag $1" >&2
        exit 2
        ;;
esac

# uv's installer puts it in ~/.local/bin, which may not be on PATH in a git hook
# or a shell whose rc file the installer was told not to edit.
if ! command -v uv >/dev/null 2>&1 && [ -x "$HOME/.local/bin/uv" ]; then
    PATH="$HOME/.local/bin:$PATH"
fi
# The compiler is the one uv.lock pins; MOJO replaces it (see scripts/build.sh).
MOJO="${MOJO:-uv run --frozen mojo}"
export MOJO

RESULTS=()
failures=0
step() {
    local label="$1"
    shift
    printf '\n==> %s\n' "$label"
    if "$@"; then
        RESULTS+=("PASS  $label")
    else
        RESULTS+=("FAIL  $label")
        failures=$((failures + 1))
    fi
}
need() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "missing tool: $1" >&2
        [ "$1" = uv ] && echo "install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
        return 1
    }
}

hygiene() {
    if [ "$mode" = fast ]; then
        bash scripts/check_hygiene.sh --staged
    else
        bash scripts/check_hygiene.sh --tracked
    fi
}
fmt_check() {
    # Compare the tree before and after formatting, so uncommitted edits that
    # are already formatted pass and only formatter changes fail.
    need "${MOJO%% *}" || return 1
    local before after
    before="$(git diff -- src tests bench | sha256sum)"
    $MOJO format -q src tests bench || return 1
    after="$(git diff -- src tests bench | sha256sum)"
    if [ "$before" != "$after" ]; then
        echo "mojo format changed files; review and commit the result" >&2
        git diff --stat -- src tests bench >&2
        return 1
    fi
}
build() {
    need "${MOJO%% *}" || return 1
    bash scripts/build.sh
}
shell_lint() {
    need shellcheck && shellcheck --severity=warning scripts/*.sh .githooks/*
}
yaml_lint() {
    need uv && uv run --frozen --only-group lint yamllint --strict .github
}
md_lint() {
    need markdownlint-cli2 && markdownlint-cli2 "*.md" "docs/**/*.md" "#docs/spec.md" "#node_modules"
}
launcher_tests() {
    need uv && uv run --frozen --only-group test pytest tests/python -q
}
claims() {
    need uv && uv run --no-project python scripts/bench_table.py --check &&
        uv run --no-project python bench/eval_gsm8k.py --self-test &&
        timeout 300 uv run --frozen --only-group fixtures python scripts/check_weights.py --self-test
}
action_pins() {
    # Resolves each pinned SHA's tag through the GitHub API, so it needs gh.
    need gh && bash scripts/verify-action-pins.sh
}
docs_build() {
    need mdbook && mdbook build && python3 scripts/check_links.py
}

step "hygiene (secrets, employer name)" hygiene
step "mojo format" fmt_check
step "build -Werror" build
if [ "$mode" = full ]; then
    step "tests (count pinned)" bash scripts/run-tests.sh
    step "--device gpu without a GPU fails cleanly" bash scripts/check_no_gpu.sh
    step "--visual-interval errors before model loading" bash scripts/check_visual.sh
    gpus="$(nvidia-smi -L 2>/dev/null)"
    if grep -q '^GPU ' <<<"$gpus"; then
        step "GPU tests (count pinned)" bash scripts/run-gpu-tests.sh
    else
        RESULTS+=("SKIP  GPU tests (no NVIDIA GPU on this machine; nothing was checked)")
    fi
    step "shellcheck" shell_lint
    step "yamllint" yaml_lint
    step "markdownlint" md_lint
    step "launcher tests" launcher_tests
    step "README numbers match bench/results" claims
    step "docs build + links" docs_build
    step "action pins match their tags" action_pins
fi

printf '\n==> summary (%s)\n' "$mode"
printf '  %s\n' "${RESULTS[@]}"
if [ "$failures" -gt 0 ]; then
    printf 'ci-local: %d step(s) failed\n' "$failures"
    exit 1
fi
skipped=0
for r in "${RESULTS[@]}"; do
    [[ "$r" == SKIP* ]] && skipped=$((skipped + 1))
done
echo "ci-local: $((${#RESULTS[@]} - skipped)) steps passed, $skipped skipped"
