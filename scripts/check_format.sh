#!/usr/bin/env bash
# check_format.sh: fail when `mojo format` would change a Mojo file under src, tests or bench.
#
# Mojo has no check mode, so this formats in place and compares the content of every file before and after.
# Uncommitted edits that are already formatted pass, and so does a file git does not track yet: comparing
# `git diff` before and after (as this check once did, and as CI does) never sees an untracked file, so a new
# unformatted file was reformatted silently and the gate passed. Ignored files are left out.
#
#   scripts/check_format.sh
#
# MOJO replaces the formatter command (see scripts/build.sh); tests use a stub.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
MOJO="${MOJO:-uv run --frozen mojo}"

dirs=(src tests bench)
snapshot() {
    # "sha256  path" for every tracked or untracked, not ignored, file; a tracked file deleted from the tree is skipped.
    local path
    git ls-files -z --cached --others --exclude-standard -- "${dirs[@]}" | while IFS= read -r -d '' path; do
        # An unreadable file stops the check: a file that could not be compared must not read as unchanged.
        if [ -f "$path" ]; then sha256sum -- "$path" || exit 1; fi
    done
}

if ! command -v "${MOJO%% *}" >/dev/null 2>&1; then
    echo "missing tool: ${MOJO%% *}" >&2
    exit 1
fi
before="$(mktemp)"
after="$(mktemp)"
trap 'rm -f "$before" "$after"' EXIT
snapshot >"$before" || exit 1
# shellcheck disable=SC2086  # MOJO is a command with arguments
$MOJO format -q "${dirs[@]}" || exit 1
snapshot >"$after" || exit 1
if ! cmp -s "$before" "$after"; then
    echo "mojo format changed files; review and commit the result:" >&2
    diff "$before" "$after" | sed -n 's/^> [0-9a-f]*  /  /p' >&2
    exit 1
fi
