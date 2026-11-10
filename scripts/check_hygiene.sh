#!/usr/bin/env bash
# check_hygiene.sh: refuse to commit a credential or an employer reference.
#
# Adapted from lacs-project/sysknife scripts/check_no_secrets.sh (MIT,
# Copyright (c) 2026 Vladimir Rotariu), which explains the length-bounded
# patterns in full: a bare `sk-` or `ghp_` rule would flag every short test
# fixture and get switched off within a day, so each pattern carries the real
# format's minimum body length.
#
# This repository is a personal project. It also refuses the name of the
# author's employer anywhere in tracked text, so ownership is never in doubt
# once the repository is public. The name is assembled at runtime so this file
# does not match itself.
#
# Usage:
#   check_hygiene.sh --staged      what `git commit` is about to record
#   check_hygiene.sh --tracked     every tracked file (CI)
set -euo pipefail

PATTERNS=(
    "OpenAI key:sk-[A-Za-z0-9]{40,}"
    "OpenAI project key:sk-proj-[A-Za-z0-9_-]{40,}"
    "Anthropic key:sk-ant-[A-Za-z0-9_-]{40,}"
    "GitHub PAT:ghp_[A-Za-z0-9]{36,}"
    "GitHub fine-grained PAT:github_pat_[A-Za-z0-9_]{60,}"
    "Hugging Face token:hf_[A-Za-z0-9]{34,}"
    "Groq key:gsk_[A-Za-z0-9]{40,}"
    "AWS access key:AKIA[0-9A-Z]{16}"
    "Google API key:AIza[A-Za-z0-9_-]{35,}"
    "Slack token:xox[baprs]-[A-Za-z0-9-]{20,}"
    # The two secrets TruffleHog's verified-only scan (secret-scan.yml) cannot confirm and lets pass.
    "Private key block:-----BEGIN [A-Z ]*PRIVATE KEY-----"
    "Database URL with a password:(postgres(ql)?|mysql|mariadb|mongodb(\\+srv)?|redis|amqp)://[^[:space:]:/@]+:[^[:space:]@]+@"
)
EMPLOYER="$(printf '%s%s' 'entr' 'opia')"

found=0

scan() {
    # $1 = label; stdin = text. Binary input is skipped (grep -I).
    local label="$1" content
    content="$(cat)"
    for entry in "${PATTERNS[@]}"; do
        local provider="${entry%%:*}" regex="${entry#*:}" hits
        # grep exits 1 for "no match" and 2 for an error (a bad pattern): only the first means clean.
        rc=0
        hits="$(grep -IoE -e "$regex" <<<"$content")" || rc=$?
        if [ "$rc" -gt 1 ]; then
            echo "check_hygiene: grep failed (exit $rc) on the $provider pattern; nothing was checked" >&2
            exit 1
        fi
        while IFS= read -r hit; do
            [ -n "$hit" ] || continue
            printf 'SECRET: %s in %s, starts %s..., %d chars\n' "$provider" "$label" "${hit:0:7}" "${#hit}" >&2
            found=1
        done <<<"$hits"
    done
    if grep -Iqi "$EMPLOYER" <<<"$content"; then
        printf 'EMPLOYER: %s mentions the employer name; this is a personal project\n' "$label" >&2
        found=1
    fi
}

case "${1:-}" in
    --staged)
        if ! list="$(git diff --cached --name-only --diff-filter=ACMR)"; then
            echo "check_hygiene: git could not list staged files" >&2
            exit 1
        fi
        ;;
    --tracked)
        if ! list="$(git ls-files)"; then
            echo "check_hygiene: git could not list tracked files" >&2
            exit 1
        fi
        ;;
    *)
        echo "usage: check_hygiene.sh --staged|--tracked" >&2
        exit 2
        ;;
esac

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
n=0
binary=0
while IFS= read -r path; do
    [ -n "$path" ] || continue
    n=$((n + 1))
    # The file name itself counts too, binary or not.
    scan "path $path" <<<"$path"
    if [ "${1}" = "--staged" ]; then
        if ! git show ":$path" >"$tmp" 2>/dev/null; then
            echo "check_hygiene: could not read staged bytes for $path" >&2
            exit 1
        fi
        src="$tmp"
    else
        [ -f "$path" ] || continue
        src="$path"
    fi
    # Binary fixtures (GGUF files) are not text a person types a key into.
    if [ -s "$src" ] && ! grep -Iq . "$src"; then
        binary=$((binary + 1))
        continue
    fi
    scan "$path" <"$src"
done <<<"$list"

if [ "$found" != 0 ]; then
    echo "check_hygiene: refusing. Remove the value (and rotate it if it was a real credential)." >&2
    exit 1
fi
echo "check_hygiene: $n file(s) clean ($binary binary skipped)"
