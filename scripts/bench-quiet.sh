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
#   * for each NVIDIA card on its own (`nvidia-smi -i N`): turns persistence mode on (`-pm 1`) and, where the
#     driver allows it, locks the SM clock at that card's maximum (`-lgc MAX,MAX`). Laptop GPUs often refuse the
#     lock; the script says so and goes on. It switches persistence mode off again only on a card where it was
#     off before, and resets the clocks (`-rgc`) only on a card it locked. A lock an administrator set before the
#     run is replaced by the script's and cleared by the reset: nvidia-smi cannot read a lock back, so do not run
#     this on a card with a lock you want to keep.
# It never edits cron files. A per-minute user cron job still fires; the isolation window sees it and the
# runner repeats the problem (bench/isolation.py).
#
# Every change is written to the state file as it is made, so --restore can undo a run that could not clean up.
# The command after `--` runs as the user who invoked sudo (SUDO_USER) when there is one. Exit status is the
# command's, and it sees BENCH_QUIET_ACTIVE=timers:N,pm:N,clocks:N,refused:N: how many timers it stopped, how
# many cards it switched to persistence mode and locked, and how many changes it tried and could not make
# (bench/isolation.py records it as `bench_quiet_changes`, so a row says what was quieted, not only that the
# script ran). Each change is written to the state file before it is made. A trap on EXIT, INT, TERM, HUP and QUIT restores everything, and the script prints what it changed and what
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
n_timers=0
n_pm=0
n_clocks=0
n_refused=0

restore_all() {
    # One line per change in the state file, undone last to first. A failed undo is reported and kept in the
    # state file so --restore can try again; the state file is removed only when everything went back.
    [ -f "$STATE" ] || return 0
    local lines=() line kind arg keep=() idx was
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
                # "INDEX WAS": the state of that card before the run.
                idx="${arg%% *}"
                was="${arg#* }"
                if [ "$was" = Disabled ]; then
                    if nvidia-smi -i "$idx" -pm 0 >/dev/null; then restored+=("GPU $idx persistence mode off"); else failed+=("nvidia-smi -i $idx -pm 0"); keep+=("$line"); fi
                else
                    restored+=("GPU $idx persistence mode left $was")
                fi ;;
            gpu-clocks)
                idx="${arg%% *}"
                if nvidia-smi -i "$idx" -rgc >/dev/null; then restored+=("GPU $idx clocks reset"); else failed+=("nvidia-smi -i $idx -rgc"); keep+=("$line"); fi ;;
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
unrecord() { sed -i "/^$1\$/d" "$STATE"; }

# timers
for t in "${TIMERS[@]}"; do
    if systemctl is-active --quiet "$t.timer"; then
        # Recorded first: a signal between the stop and the record would leave a timer stopped and unknown.
        record "timer $t"
        if systemctl stop "$t.timer"; then
            changed+=("stopped $t.timer")
            n_timers=$((n_timers + 1))
        else
            err "could not stop $t.timer; leaving it as it is"
            unrecord "timer $t"
            n_refused=$((n_refused + 1))
        fi
    fi
done

# GPU: every card on its own, with one state line per index in the state file.
quiet_gpu() {
    local i="$1" out persistence maxclk
    out="$(nvidia-smi -i "$i" --query-gpu=persistence_mode --format=csv,noheader 2>/dev/null)"
    persistence="${out%%$'\n'*}"
    persistence="${persistence//[[:space:]]/}"
    case "$persistence" in
        Enabled | Disabled) ;;
        *) err "could not read the persistence mode of GPU $i; not touching it"; persistence=""; n_refused=$((n_refused + 1)) ;;
    esac
    if [ "$persistence" = Disabled ]; then
        record "gpu-persistence $i Disabled"
        if nvidia-smi -i "$i" -pm 1 >/dev/null; then
            changed+=("GPU $i persistence mode on")
            n_pm=$((n_pm + 1))
        else
            err "nvidia-smi -i $i -pm 1 failed"
            unrecord "gpu-persistence $i Disabled"
            n_refused=$((n_refused + 1))
        fi
    elif [ "$persistence" = Enabled ]; then
        say "GPU $i persistence mode was already on"
    fi
    out="$(nvidia-smi -i "$i" --query-gpu=clocks.max.sm --format=csv,noheader,nounits 2>/dev/null)"
    maxclk="${out%%$'\n'*}"
    maxclk="${maxclk//[[:space:]]/}"
    if [[ "$maxclk" =~ ^[0-9]+$ ]]; then
        # Recorded before the lock, removed if the driver refuses it.
        record "gpu-clocks $i locked"
        if nvidia-smi -i "$i" -lgc "$maxclk,$maxclk" >/dev/null 2>&1; then
            changed+=("GPU $i clocks locked at $maxclk MHz")
            n_clocks=$((n_clocks + 1))
        else
            unrecord "gpu-clocks $i locked"
            say "the driver refused to lock the clocks of GPU $i (laptop GPUs often do); clocks left alone"
            n_refused=$((n_refused + 1))
        fi
    else
        say "could not read the maximum SM clock of GPU $i; its clocks left alone"
        n_refused=$((n_refused + 1))
    fi
}

if command -v nvidia-smi >/dev/null 2>&1; then
    if gpu_list="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)"; then
        while IFS= read -r idx; do
            idx="${idx//[[:space:]]/}"
            [ -n "$idx" ] || continue
            if [[ "$idx" =~ ^[0-9]+$ ]]; then
                quiet_gpu "$idx"
            else
                err "unexpected GPU index '$idx'; leaving it alone"
                n_refused=$((n_refused + 1))
            fi
        done <<<"$gpu_list"
    else
        err "could not list the GPUs; GPU left alone"
        n_refused=$((n_refused + 1))
    fi
else
    say "no nvidia-smi: GPU left alone"
fi

# The runners record this in ARM.env.json (`bench_quiet`, `bench_quiet_changes`), so a row says what was quieted
# on the machine it was measured on, and what could not be.
export BENCH_QUIET_ACTIVE="timers:$n_timers,pm:$n_pm,clocks:$n_clocks,refused:$n_refused"
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
