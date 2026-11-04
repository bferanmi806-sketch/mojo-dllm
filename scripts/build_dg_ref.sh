#!/usr/bin/env bash
# build_dg_ref.sh: build the DiffusionGemma reference, CPU only.
#
#   scripts/build_dg_ref.sh                build llama.cpp at the pin, then build/dg_ref
#   scripts/build_dg_ref.sh --llama-only   stop after the llama.cpp build
#
# mainline llama.cpp cannot load a DiffusionGemma GGUF. Its support is the
# ggml-org/llama.cpp#24423 branch, so the reference is built from that branch at
# one commit, into its own source tree and its own build directory, beside the
# mainline checkout that bench/ uses (~/src/llama.cpp):
#
#   $DG_LLAMA_SRC          default ~/src/llama.cpp-diffusion-gemma
#   $DG_LLAMA_SRC/build    CPU build: llama-tokenize, llama-diffusion-gemma-eval, libllama
#   build/dg_ref           tools/ref/dg_ref.cpp linked against that libllama
#
# The commit is the one the work was written against (issue #66). The branch has
# moved since: its head was 5fcd1b166556b85b8957dd432fcaff5351af59bf on
# 2026-10-29, which differs from the pin only by merges of upstream master. Move
# the pin by editing PINNED_COMMIT here and in
# tests/fixtures/gemma4-tokenizer-golden.provenance.json, and regenerate the goldens.
#
# The build never reaches the network after the fetch: the server, the web UI
# (a prebuilt-UI download), the tests and HTTPS (OpenSSL) are switched off.
#
# Environment:
#   DG_LLAMA_SRC     where the source tree and its build/ live
#   DG_LLAMA_REPO    where to fetch from (default: the PR's head repository)
#   DG_LLAMA_COMMIT  testing only: replaces the pin. The commit is always printed.
#   DG_JOBS          parallel compile jobs (default: all cores)
set -uo pipefail

PINNED_COMMIT="2e715c0639c6f07c57c943f8d5a7919953af8f7c"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

src="${DG_LLAMA_SRC:-$HOME/src/llama.cpp-diffusion-gemma}"
remote="${DG_LLAMA_REPO:-https://github.com/danielhanchen/llama.cpp.git}"
commit="${DG_LLAMA_COMMIT:-$PINNED_COMMIT}"
jobs="${DG_JOBS:-$(nproc)}"
llama_only=0

case "${1:-}" in
    "") ;;
    --llama-only) llama_only=1 ;;
    -h | --help)
        sed -n '2,29p' "$0"
        exit 0
        ;;
    *)
        echo "build_dg_ref: unknown argument $1" >&2
        exit 2
        ;;
esac

die() {
    echo "build_dg_ref: $*" >&2
    exit 1
}

# Run from a git hook, GIT_DIR and its relatives name the repository being pushed, and every `git -C` below
# would act on that one instead of the llama.cpp tree.
if ! local_vars="$(git rev-parse --local-env-vars)"; then
    die "git rev-parse --local-env-vars failed"
fi
for var in $local_vars; do
    unset "$var"
done

[[ $commit =~ ^[0-9a-f]{40}$ ]] || die "expected a commit of 40 hex characters, got: $commit"

started=$SECONDS
echo "build_dg_ref: commit $commit"
echo "build_dg_ref: source $src"

# 1. The source tree, at exactly the pinned commit.
if [ ! -d "$src/.git" ]; then
    mkdir -p "$src" || die "cannot create $src"
    git init -q "$src" || die "git init failed in $src"
fi
if ! git -C "$src" cat-file -e "$commit^{commit}" 2>/dev/null; then
    git -C "$src" remote remove origin 2>/dev/null
    git -C "$src" remote add origin "$remote" || die "cannot add the remote $remote"
    git -C "$src" fetch -q --depth 1 origin "$commit" ||
        die "could not fetch commit $commit from $remote"
fi
git -C "$src" checkout -q --detach "$commit" || die "could not check out $commit in $src"
head="$(git -C "$src" rev-parse HEAD)" || die "git rev-parse failed in $src"
[ "$head" = "$commit" ] || die "HEAD is $head, expected $commit"
# Untracked build output is expected; an edit to a tracked file is not.
if [ -n "$(git -C "$src" status --porcelain --untracked-files=no)" ]; then
    die "$src has modified tracked files; it no longer matches $commit"
fi
echo "build_dg_ref: HEAD $head"

# 2. llama.cpp, CPU only, offline.
cmake -S "$src" -B "$src/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=OFF \
    -DLLAMA_OPENSSL=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF \
    -DLLAMA_BUILD_TESTS=OFF ||
    die "cmake configure failed"
cmake --build "$src/build" -j "$jobs" \
    --target llama-tokenize llama-diffusion-gemma-eval llama-diffusion-cli llama ||
    die "cmake build failed"
echo "build_dg_ref: llama.cpp built in $((SECONDS - started)) s: $src/build/bin"

[ "$llama_only" = 1 ] && exit 0

# 3. The reference tool, linked against the build above (same recipe as
#    llada_ref in docs/correctness.md).
mkdir -p "$repo_root/build" || die "cannot create build/"
c++ -O2 -std=c++17 "$repo_root/tools/ref/dg_ref.cpp" \
    -I"$src/include" -I"$src/ggml/include" \
    -L"$src/build/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$src/build/bin" \
    -o "$repo_root/build/dg_ref" || die "compiling tools/ref/dg_ref.cpp failed"
echo "build_dg_ref: done in $((SECONDS - started)) s: build/dg_ref"
