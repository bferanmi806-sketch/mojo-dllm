#!/usr/bin/env bash
# capped.sh: run a command in its own memory-capped systemd scope, or refuse to run it.
#
# usage: capped.sh --mem SIZE [--need SIZE] [--cpus N] [--dry-run] -- COMMAND [ARG...]
#
# Why: a job that needs more memory than the machine has makes the kernel's global OOM killer pick a victim, and
# systemd's default `OOMPolicy=stop` then stops the whole scope the victim lived in. Run from a terminal, that
# scope is the terminal: every session in it ends. A job that runs in a scope of its own, under a hard cap,
# is killed by the cgroup OOM killer alone (`constraint=CONSTRAINT_MEMCG`), and only that scope fails.
#
# What it sets, and why each one:
#   MemoryMax=SIZE    the hard cap (cgroup v2 `memory.max`); the kernel kills inside the scope when it is reached
#   MemorySwapMax=0   no swap: a job over its cap dies instead of dragging the desktop through swap
#   OOMPolicy=kill    one OOM kill ends the whole scope (`memory.oom.group`), not a random half of it
#   CPUQuota=N00%     only with --cpus N, so a job leaves cores for the desktop
# It never sets MemoryHigh. Below the cap it throttles a process with nothing to reclaim to a standstill
# (state D, no progress) instead of killing it.
#
# Gate: it starts the job only when /proc/meminfo reports at least --need (default: --mem) available. A cap
# larger than the available memory does not protect the machine; the global OOM killer still runs.
#
# Fail closed: if the systemd user manager cannot enforce memory limits, it exits without running the command.
#
# SIZE is a whole number with an optional K, M or G (binary units, as systemd reads them).
# Exit status: the command's own; 64 usage; 69 isolation unavailable; 75 not enough memory available.
# Test hooks: CAPPED_MEMINFO (default /proc/meminfo) and CAPPED_CONTROLLERS_FILE (the user manager's
# cgroup.controllers).
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
usage: capped.sh --mem SIZE [--need SIZE] [--cpus N] [--dry-run] -- COMMAND [ARG...]
  --mem SIZE    hard memory cap for the command (whole number plus K, M or G)
  --need SIZE   memory that must be available before it starts (default: --mem)
  --cpus N      limit the command to N CPUs of CPU time
  --dry-run     check everything, print the systemd-run command, run nothing
EOF
}

die() {  # die STATUS MESSAGE
    local status="$1"
    shift
    printf 'capped: %s\n' "$*" >&2
    exit "$status"
}

# parse_size 27G -> bytes. Fails on anything that is not a positive whole number with an optional K, M or G.
parse_size() {
    local text="${1-}" digits unit mult
    [[ "$text" =~ ^([0-9]{1,9})([KMG]?)$ ]] || return 1
    digits="${BASH_REMATCH[1]}"
    unit="${BASH_REMATCH[2]}"
    case "$unit" in
        K) mult=1024 ;;
        M) mult=1048576 ;;
        G) mult=1073741824 ;;
        *) mult=1 ;;
    esac
    (( 10#$digits > 0 )) || return 1
    printf '%s\n' $(( 10#$digits * mult ))
}

# mem_available_bytes [FILE] -> MemAvailable in bytes. Fails when the line is missing or not a number.
mem_available_bytes() {
    local file="${1:-/proc/meminfo}" kb
    kb="$(awk '/^MemAvailable:/ { print $2; found = 1; exit } END { exit !found }' "$file")" || return 1
    [[ "$kb" =~ ^[0-9]+$ ]] || return 1
    printf '%s\n' $(( kb * 1024 ))
}

# gib BYTES -> "12.0G"
gib() {
    awk -v b="$1" 'BEGIN { printf "%.1fG", b / 1073741824 }'
}

memory_controller_delegated() {
    local file="${CAPPED_CONTROLLERS_FILE:-}"
    if [[ -z "$file" ]]; then
        local uid
        uid="$(id -u)"
        file="/sys/fs/cgroup/user.slice/user-${uid}.slice/user@${uid}.service/cgroup.controllers"
    fi
    [[ -r "$file" ]] || return 1
    grep -qw memory "$file"
}

main() {
    local mem="" need="" cpus="" dry=0
    while (( $# > 0 )); do
        case "$1" in
            --mem) (( $# >= 2 )) || { usage; exit 64; }; mem="$2"; shift 2 ;;
            --need) (( $# >= 2 )) || { usage; exit 64; }; need="$2"; shift 2 ;;
            --cpus) (( $# >= 2 )) || { usage; exit 64; }; cpus="$2"; shift 2 ;;
            --dry-run) dry=1; shift ;;
            -h | --help) usage; exit 0 ;;
            --) shift; break ;;
            *) usage; exit 64 ;;
        esac
    done
    [[ -n "$mem" ]] && (( $# > 0 )) || { usage; exit 64; }
    local mem_bytes need_bytes
    mem_bytes="$(parse_size "$mem")" || { usage; exit 64; }
    if [[ -n "$need" ]]; then
        need_bytes="$(parse_size "$need")" || { usage; exit 64; }
    else
        need="$mem"
        need_bytes="$mem_bytes"
    fi
    if [[ -n "$cpus" ]]; then
        [[ "$cpus" =~ ^[1-9][0-9]{0,3}$ ]] || { usage; exit 64; }
    fi

    local meminfo="${CAPPED_MEMINFO:-/proc/meminfo}" avail
    avail="$(mem_available_bytes "$meminfo")" || die 75 "refusing to start: cannot read MemAvailable from $meminfo"
    if (( avail < need_bytes )); then
        die 75 "refusing to start: $(gib "$avail") available, $need needed. Free memory (close browsers) or lower --mem/--need."
    fi
    memory_controller_delegated ||
        die 69 "refusing to start: the memory controller is not delegated to the systemd user manager, so a cap would not be enforced"

    local cmd=(systemd-run --user --scope --quiet --collect "--unit=capped-$$-$RANDOM"
        -p "MemoryMax=$mem" -p MemorySwapMax=0 -p OOMPolicy=kill)
    [[ -n "$cpus" ]] && cmd+=(-p "CPUQuota=$(( cpus * 100 ))%")
    cmd+=(-- "$@")

    if (( dry )); then
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi
    command -v systemd-run >/dev/null || die 69 "refusing to start: systemd-run not found"

    printf 'capped: MemoryMax=%s MemorySwapMax=0 OOMPolicy=kill (%s available)\n' "$mem" "$(gib "$avail")" >&2
    local rc=0
    "${cmd[@]}" || rc=$?
    if (( rc == 137 )); then
        printf 'capped: exit 137 is SIGKILL, usually the memory cap (MemoryMax=%s) was reached; raise --mem if the job needs more\n' "$mem" >&2
    fi
    return "$rc"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
