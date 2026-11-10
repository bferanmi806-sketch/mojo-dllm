#!/usr/bin/env bash
# Build build/mojo-dllm (or the file given as $1 into $2).
#
# Mojo compiles the GPU kernels at build time for one GPU architecture. With
# no GPU in the machine it cannot guess one and the build fails, which would
# lock CPU-only users out. So the target is always named: the local GPU's
# compute capability when nvidia-smi reports one, sm_80 otherwise (the oldest
# architecture the int8 tensor-core kernels support). MOJO_DLLM_GPU_ARCH
# overrides both. The chosen architecture and the reason go to stderr; an
# nvidia-smi that is installed but fails is called out as a warning.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
src="${1:-src/main.mojo}"
out="${2:-build/mojo-dllm}"
arch="${MOJO_DLLM_GPU_ARCH:-}"
if [ -n "$arch" ]; then
    why="MOJO_DLLM_GPU_ARCH"
elif ! command -v nvidia-smi >/dev/null 2>&1; then
    arch=sm_80
    why="no nvidia-smi on PATH, so no GPU to match"
elif caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>&1)"; then
    cap="${caps%%$'\n'*}"
    if [[ "$cap" =~ ^[0-9]+\.[0-9]+$ ]]; then
        arch="sm_${cap/./}"
        why="nvidia-smi reports compute capability $cap"
    else
        arch=sm_80
        why="WARNING: nvidia-smi printed no compute capability ('${cap:0:80}'), so the GPU is unknown"
    fi
else
    arch=sm_80
    why="WARNING: nvidia-smi is installed but failed ('${caps:0:120}'), as it does after a driver mismatch; the GPU is unknown"
fi
# Said every time: a binary built for another card runs the GPU path wrongly or not at all.
echo "build.sh: GPU architecture $arch ($why)" >&2
# The compiler is the one uv.lock pins. MOJO replaces it: the sandbox that runs
# contributor pull requests points it at a read-only copy.
MOJO="${MOJO:-uv run --frozen mojo}"
mkdir -p "$(dirname "$out")"
# shellcheck disable=SC2086  # MOJO is a command with arguments
exec $MOJO build -Werror --target-accelerator "$arch" -I src "$src" -o "$out"
