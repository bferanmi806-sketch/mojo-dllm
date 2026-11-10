#!/usr/bin/env python3
"""Generate small, deterministic raw expert tensors for test_moe.mojo.

Each file contains four [64, 256] gate/up slices, then four [256, 32]
down slices. Integer weights and f16 scales are exact in the scalar oracle.
No model downloads or third-party dependencies are needed.
"""

import os
from pathlib import Path
import struct


def block(values, scale, kind):
    if kind == "q4":
        unsigned = [v + 7 for v in values]
        packed = bytes(unsigned[64 * half + j] | (unsigned[64 * half + j + 32] << 4) for half in range(4) for j in range(32))
        return struct.pack("<ee", scale, scale) + bytes([1] * 4 + [7] * 4 + [0x71] * 4) + packed
    if kind == "q8":
        return struct.pack("<e32b", scale, *values)
    unsigned = [v + 16 for v in values]
    high = sum((v >> 4) << i for i, v in enumerate(unsigned))
    low = bytes(unsigned[i] & 15 | ((unsigned[i + 16] & 15) << 4) for i in range(16))
    return struct.pack("<eI", scale, high) + low


def tensor(rows, columns, kind, stage):
    result = bytearray()
    for expert in range(4):
        for row in range(rows):
            block_size = 256 if kind == "q4" else 32
            for start in range(0, columns, block_size):
                values = [((expert * 11 + row * 7 + (start + k) * 3 + stage * 5) % 15) - 7 for k in range(block_size)]
                result.extend(block(values, 0.015625 if stage == 0 else 0.03125, kind))
    return result


def main():
    # scripts/check_fixtures.py sets FIXTURES_OUT to regenerate into a scratch directory.
    fixtures = Path(os.environ.get("FIXTURES_OUT", Path(__file__).resolve().parents[1] / "tests" / "fixtures"))
    for kind in ("q5", "q8"):
        data = tensor(64, 256, kind, 0) + tensor(256, 32, kind, 1)
        path = fixtures / f"moe-{kind}.bin"
        path.write_bytes(data)
        print(f"{path.name}: {len(data)} bytes")
        mixed = tensor(64, 256, "q4", 0) + tensor(256, 32, kind, 1)
        path = fixtures / f"moe-q4-{kind}.bin"
        path.write_bytes(mixed)
        print(f"{path.name}: {len(mixed)} bytes")


if __name__ == "__main__":
    main()
