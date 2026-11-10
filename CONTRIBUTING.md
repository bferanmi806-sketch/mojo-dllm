# Contributing

Thanks for looking. mojo-dllm is small on purpose: a dLLM inference core, not
a serving framework. Changes that make a forward pass faster, a format better
supported, or a number better verified are the ones that land.

## Set up

```bash
./mojo-dllm setup                     # pinned Mojo 1.1.0, build/mojo-dllm, a model
scripts/ci-local.sh --install-hooks   # gates on commit and push
```

## Before you open a pull request

Run `scripts/ci-local.sh`. It runs the CI gates: hygiene scan, `mojo format`,
a `-Werror` build, the tests with their pinned count, the no-GPU and
visual-option checks, the launcher tests, lint, the benchmark-table check with
the eval and weight-checker self-tests, the docs build and the action-pin
check. It runs the GPU tests when `nvidia-smi` lists a GPU and skips them
otherwise, where CI only builds them.
[Development](docs/development.md) explains each one.

## How changes here are made

- **Test first.** Write the failing test, watch it fail, then write the code.
  Fixtures come from scripts in `scripts/`, never from hand-edited bytes.
- **Prove that a check bites.** For a new kernel or guard, break the code on
  purpose and confirm the test fails, then restore it.
- **No hand-typed benchmark numbers.** Tables come from `bench/results/`,
  `bench/parity/`, `bench/quality/` and `bench/eval/` through
  `scripts/bench_table.py`. A change that moves
  performance should come with a new result file from `bench/run_bench.py`.
- **Errors name the thing.** User-facing failures say which key, tensor or
  type is wrong; no assertion-only paths.
- **Toolchain bumps stand alone.** Moving Mojo or max-core in `pyproject.toml`
  (and `uv.lock`) is its own commit with a green run.

## Review

I review a pull request at a named head, and if you push after my review I
review the new commits before I merge. On each one I rerun the CI steps from
your branch in a container with no network, and check the numbers in the
description against the repo. The rest depends on what the change touches:

| The change | What I add |
|---|---|
| A guard: a bounds, type or format check, or a refusal | I break each clause and expect a named test to fail. |
| A bug fix | I revert the fix and expect the new test to fail. |
| Tests only | I break the code that one or two of the new tests cover. |
| Docs or CI configuration only | Nothing beyond the checks above. |

I can merge your branch once it is up to date with `main`. If it trails `main`
and merges cleanly, I press Update branch and wait for CI to pass again. If it
conflicts, I'll ask you to merge `main` into your branch and resolve the
conflict yourself, since you know the change better than I do. The usual
conflict is `tests/test-count.txt` (or `tests/test-counts.txt`, the same
pin for each file): take either side, then run
`UPDATE_TEST_COUNT=1 scripts/run-tests.sh` on the merged tree to pin the new
total and the count of each file.

## Commit messages

An imperative subject under 72 characters, and a body that says why.

## Reporting bugs

Open an issue with the command you ran, the model file (name and sha256), the
output, and `./mojo-dllm doctor` and `lscpu | head -20`. For security issues, see
[SECURITY.md](SECURITY.md).
