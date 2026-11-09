#!/usr/bin/env bash
# bench-quiet.sh: quiet the machine for a benchmark run, run a command, put everything back.
#
#   sudo scripts/bench-quiet.sh -- python3 bench/eval_engine.py run --engine llamacpp ...
#   sudo scripts/bench-quiet.sh --restore      put back what a run that was killed hard (SIGKILL, power loss)
#                                              left changed, from the state file
#
# What it changes, and only these:
#   * stops the system timers that are active: sysstat-collect, apt-daily, apt-daily-upgrade, fwupd-refresh,
#     anacron, man-db, motd-news (each `NAME.timer`); restores them with `systemctl start`;
#   * turns NVIDIA persistence mode on (`nvidia-smi -pm 1`) and, where the driver allows it, locks the SM
#     clock at its maximum (`-lgc MAX,MAX`). Laptop GPUs often refuse the lock; the script says so and goes on.
#     It restores persistence mode to what it was and resets the clocks (`-rgc`) only if it locked them.
# It never edits cron files. A per-minute user cron job still fires; the isolation window sees it and the
# runner repeats the problem (bench/isolation.py).
#
# Every change is written to the state file as it is made, so --restore can undo a run that could not clean up.
# The command after `--` runs as the user who invoked sudo (SUDO_USER) when there is one. Exit status is the
# command's, and it sees BENCH_QUIET_ACTIVE=1. A trap on EXIT, INT, TERM, HUP and QUIT restores everything, and the script prints what it changed and what
# it restored.
#
# Test seams (environment, all optional):
#   BENCH_QUIET_ASSUME_ROOT=1   skip the "must be root" check, but only when BENCH_QUIET_TESTING=1 is set too,
#                               so one stray variable cannot turn the guard off. Tests set both, with fake
#                               systemctl and nvidia-smi on PATH; do not set either on a real machine.
#   BENCH_QUIET_TESTING=1       the second half of the seam above
#   BENCH_QUIET_STATE=PATH      the state file (default /run/bench-quiet.state)
#
# After a SIGKILL or a power loss nothing can run a trap: run `sudo scripts/bench-quiet.sh --restore`. The
# runners' `--tune` has no state file, so after a SIGKILL the power profile stays at performance: put back the
# profile the arm recorded (`tuned.previous` in ARM.env.json) with `powerprofilesctl set <previous>`.
set -uo pipefail

TIMERS=(sysstat-collect apt-daily apt-daily-upgrade fwupd-refresh anacron man-db motd-news)
STATE="${BENCH_QUIET_STATE:-/run/bench-quiet.state}"

say() { printf 'bench-quiet: %s\n' "$*"; }
err() { printf 'bench-quiet: %s\n' "$*" >&2; }

if { [ "${BENCH_QUIET_ASSUME_ROOT:-}" != 1 ] || [ "${BENCH_QUIET_TESTING:-}" != 1 ]; } && [ "$(id -u)" != 0 ]; then
    err "must run as root: sudo $0 -- COMMAND ..."
    exit 2
fi

mode=run
case "${1:-}" in
    --restore) mode=restore; shift ;;
    --) shift ;;
    -h | --help) sed -n '2,/^set -uo pipefail/p' "$0" | sed '$d'; exit 0 ;;
    *) err "usage: $0 -- COMMAND [ARGS...]   |   $0 --restore"; exit 2 ;;
esac
if [ "$mode" = run ] && [ "$#" -eq 0 ]; then
    err "no command after --"
    exit 2
fi

changed=()
restored=()
failed=()

restore_all() {
    # One line per change in the state file, undone last to first. A failed undo is reported and kept in the
    # state file so --restore can try again; the state file is removed only when everything went back.
    [ -f "$STATE" ] || return 0
    local lines=() line kind arg keep=()
    mapfile -t lines < "$STATE"
    local i
    for ((i = ${#lines[@]} - 1; i >= 0; i--)); do
        line="${lines[i]}"
        kind="${line%% *}"
        arg="${line#* }"
        case "$kind" in
            timer)
                if systemctl start "$arg.timer"; then restored+=("started $arg.timer"); else failed+=("start $arg.timer"); keep+=("$line"); fi ;;
            gpu-persistence)
                if [ "$arg" = Disabled ]; then
                    if nvidia-smi -pm 0 >/dev/null; then restored+=("persistence mode off"); else failed+=("nvidia-smi -pm 0"); keep+=("$line"); fi
                else
                    restored+=("persistence mode left $arg")
                fi ;;
            gpu-clocks)
                if nvidia-smi -rgc >/dev/null; then restored+=("GPU clocks reset"); else failed+=("nvidia-smi -rgc"); keep+=("$line"); fi ;;
            *) err "unknown state line: $line" ;;
        esac
    done
    if [ "${#keep[@]}" -eq 0 ]; then
        rm -f "$STATE"
    else
        printf '%s\n' "${keep[@]}" > "$STATE"
    fi
}

report() {
    say "changed:  ${changed[*]:-nothing}"
    say "restored: ${restored[*]:-nothing}"
    if [ "${#failed[@]}" -gt 0 ]; then
        err "NOT restored: ${failed[*]}; run: sudo $0 --restore"
        return 1
    fi
    return 0
}

if [ "$mode" = restore ]; then
    if [ ! -f "$STATE" ]; then
        say "nothing to restore ($STATE does not exist)"
        exit 0
    fi
    restore_all
    report
    exit $?
fi

if [ -f "$STATE" ]; then
    err "$STATE exists: an earlier run left changes behind. Run: sudo $0 --restore"
    exit 1
fi

child=""
finished=0
cleanup() {
    [ "$finished" = 1 ] && return
    finished=1
    if [ -n "$child" ] && kill -0 "$child" 2>/dev/null; then
        kill -TERM "$child" 2>/dev/null
        wait "$child" 2>/dev/null
    fi
    restore_all
    report
}
on_signal() {
    local sig="$1"
    cleanup
    trap - EXIT
    exit $((128 + sig))
}
trap cleanup EXIT
trap 'on_signal 2' INT
trap 'on_signal 15' TERM
trap 'on_signal 1' HUP
trap 'on_signal 3' QUIT

record() { printf '%s\n' "$1" >> "$STATE"; }

# timers
for t in "${TIMERS[@]}"; do
    if systemctl is-active --quiet "$t.timer"; then
        if systemctl stop "$t.timer"; then
            record "timer $t"
            changed+=("stopped $t.timer")
        else
            err "could not stop $t.timer; leaving it as it is"
        fi
    fi
done

# GPU
if command -v nvidia-smi >/dev/null 2>&1; then
    persistence="$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader 2>/dev/null | head -n 1)"
    case "$persistence" in
        Enabled | Disabled) ;;
        *) err "could not read the GPU persistence mode; not touching it"; persistence="" ;;
    esac
    if [ "$persistence" = Disabled ]; then
        record "gpu-persistence Disabled"
        if nvidia-smi -pm 1 >/dev/null; then
            changed+=("persistence mode on")
        else
            err "nvidia-smi -pm 1 failed"
            sed -i '/^gpu-persistence /d' "$STATE"
        fi
    elif [ "$persistence" = Enabled ]; then
        say "GPU persistence mode was already on"
    fi
    maxclk="$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]')"
    if [[ "$maxclk" =~ ^[0-9]+$ ]]; then
        if nvidia-smi -lgc "$maxclk,$maxclk" >/dev/null 2>&1; then
            record "gpu-clocks locked"
            changed+=("GPU clocks locked at $maxclk MHz")
        else
            say "the driver refused to lock GPU clocks (laptop GPUs often do); clocks left alone"
        fi
    else
        say "could not read the maximum SM clock; GPU clocks left alone"
    fi
else
    say "no nvidia-smi: GPU left alone"
fi

# The runners record this in ARM.env.json (`bench_quiet`), so a row says it was measured on a quieted machine.
export BENCH_QUIET_ACTIVE=1
say "running: $*"
if [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != root ]; then
    runuser -u "$SUDO_USER" -- "$@" &
else
    "$@" &
fi
child=$!
wait "$child"
rc=$?
child=""
cleanup
if [ "$rc" -eq 0 ] && [ "${#failed[@]}" -gt 0 ]; then
    rc=70
fi
exit $rc
