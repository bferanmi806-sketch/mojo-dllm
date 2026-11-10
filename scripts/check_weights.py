#!/usr/bin/env python3
"""Compare GGUF tensor shapes with their safetensors source (headers only).

uv run --only-group fixtures python scripts/check_weights.py MODEL.gguf INDEX.json [--ignore REGEX]
uv run --only-group fixtures python scripts/check_weights.py --self-test
The second argument may also be a single .safetensors file. No tensor data is read.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile

import gguf
import numpy as np


def read_header(path: Path) -> dict:
    """Safetensors: an unsigned little-endian length followed by JSON."""
    if path.exists() and not path.is_file():
        raise ValueError(f"{path}: is not a regular file")
    with path.open("rb") as stream:
        size = stream.read(8)
        if len(size) != 8:
            raise ValueError(f"{path}: missing safetensors header length")
        length = struct.unpack("<Q", size)[0]
        if length > 100_000_000 or length > path.stat().st_size - 8:
            raise ValueError(f"{path}: invalid safetensors header length {length}")
        header = json.loads(stream.read(length))
    if not isinstance(header, dict):
        raise ValueError(f"{path}: header must be an object")
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    for name, info in tensors.items():
        if (not isinstance(info, dict) or not isinstance(info.get("dtype"), str)
                or not isinstance(info.get("shape"), list)
                or any(type(d) is not int or d < 0 for d in info["shape"])):
            raise ValueError(f"{path}: invalid tensor header for {name}")
    return tensors


def read_source(path: Path) -> dict:
    if path.suffix == ".safetensors":
        return read_header(path)
    index = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(index, dict):
        raise ValueError(f"{path}: index must be an object")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{path}: empty or missing weight_map; nothing compared")
    shards = {}
    source = {}
    for name, shard in weight_map.items():
        if not isinstance(shard, str):
            raise ValueError(f"{path}: invalid shard for {name}")
        if (not shard.endswith(".safetensors") or shard == ".safetensors"
                or any(c in shard for c in ("/", "\\", ":"))):
            raise ValueError(f"{path}: shard must be a bare .safetensors filename: {shard}")
        if shard not in shards:
            resolved = (path.parent / shard).resolve()
            if resolved.parent != path.parent.resolve():
                raise ValueError(f"{path}: shard resolves outside index directory: {shard}")
            shards[shard] = read_header(resolved)
        if name not in shards[shard]:
            raise ValueError(f"{path}: {name} absent from declared shard {shard}")
        source[name] = shards[shard][name]
    for shard, header in shards.items():
        for name in header:
            if weight_map.get(name) != shard:
                raise ValueError(f"{path}: {name} in {shard} is not indexed there")
    return source


def name_map(reader):
    architecture = reader.fields["general.architecture"].contents()
    supported = {"llada": gguf.MODEL_ARCH.LLADA, "dream": gguf.MODEL_ARCH.DREAM}
    if architecture not in supported:
        raise ValueError(f"no tensor name map for architecture {architecture}")
    blocks = int(reader.fields[architecture + ".block_count"].contents())
    if not 1 <= blocks <= 1024:
        raise ValueError(f"{architecture}.block_count must be between 1 and 1024, got {blocks}")
    return gguf.get_tensor_name_map(supported[architecture], blocks)


def _say(text: str) -> None:
    """Print a report line; names come from other people's files, so escape
    anything the terminal would act on."""
    print(text if text.isprintable() else text.encode("unicode_escape").decode("ascii"))


def compare(model: Path, source_path: Path, ignore: str | None = None) -> int:
    counts = dict(compared=0, ok=0, mismatch=0, missing_in_gguf=0,
                  missing_in_source=0, unmapped=0, ignored=0)
    try:
        ignored = re.compile(ignore) if ignore is not None else None
        source = read_source(source_path)
        if not source:
            raise ValueError("empty safetensors source; nothing compared")
        reader = gguf.GGUFReader(str(model))
        mapper = name_map(reader)
        targets = {t.name: t for t in reader.tensors}
        seen = set()
        for name, info in sorted(source.items()):
            mapped = mapper.get_name(name, try_suffixes=(".weight", ".bias"))
            if ignored and ignored.search(name):
                counts["ignored"] += 1
                if mapped:
                    seen.add(mapped)
                continue
            if mapped is None:
                _say(f"unmapped source name: {name}")
                counts["unmapped"] += 1
                continue
            if mapped in seen:
                _say(f"unmapped source name: {name} duplicates GGUF target {mapped}")
                counts["unmapped"] += 1
                continue
            seen.add(mapped)
            if mapped not in targets:
                _say(f"missing in GGUF: {name} -> {mapped}")
                counts["missing_in_gguf"] += 1
                continue
            tensor = targets[mapped]
            shape = [int(d) for d in reversed(tensor.shape)]
            counts["compared"] += 1
            types = f"GGUF {tensor.tensor_type.name}, safetensors {info['dtype']}"
            if shape != info["shape"]:
                _say(f"shape mismatch: {name} -> {mapped}: {info['shape']} vs {shape} ({types})")
                counts["mismatch"] += 1
            else:
                _say(f"OK: {name} -> {mapped}: {shape} ({types})")
                counts["ok"] += 1
        for name in sorted(targets.keys() - seen):
            note = "; source may tie its output weights to the embeddings" if name == "output.weight" else ""
            _say(f"missing in safetensors: {name}{note}")
            counts["missing_in_source"] += 1
    except Exception as error:
        _say(f"ERROR: {error}")
        return_code = 2
    else:
        return_code = 2 if counts["compared"] == 0 else int(any(
            counts[k] for k in ("mismatch", "missing_in_gguf", "missing_in_source", "unmapped")))
    _say("compared {compared}, ok {ok}, mismatch {mismatch}, missing-in-gguf {missing_in_gguf}, "
          "missing-in-source {missing_in_source}, unmapped {unmapped}, ignored {ignored}".format(**counts))
    return return_code


def write_shard(path: Path, tensors: dict) -> int:
    """Self-test writer: valid F32 headers with a zero-filled data section."""
    offset = 0
    header = {}
    for name, shape in tensors.items():
        size = math.prod(shape) * 4
        header[name] = dict(dtype="F32", shape=shape, data_offsets=[offset, offset + size])
        offset += size
    data = json.dumps(header).encode()
    data += b" " * (-len(data) % 8)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(data)))
        stream.write(data)
        stream.write(bytes(offset))
    return offset


def write_twin(directory: Path, shapes: dict) -> Path:
    items = list(shapes.items())
    weight_map = {}
    size = 0
    for i in range(2):
        shard = f"model-{i + 1:05d}-of-00002.safetensors"
        tensors = dict(items[i::2])
        size += write_shard(directory / shard, tensors)
        weight_map.update({name: shard for name in tensors})
    index = directory / "model.safetensors.index.json"
    index.write_text(json.dumps(dict(metadata=dict(total_size=size), weight_map=weight_map)))
    return index


def self_test_inputs(fixtures: Path) -> int:
    """Pin diagnostics as well as status, so accidental exceptions do not pass."""
    model = fixtures / "tiny-llada.gguf"
    runs = 0
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        source = directory / "source.safetensors"
        index = directory / "index.json"
        name = "model.transformer.wte.weight"

        def expect(path, message, status=2, gguf_path=model, ignore=None):
            nonlocal runs
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                code = compare(gguf_path, path, ignore)
            assert code == status and message in captured.getvalue(), captured.getvalue()
            runs += 1

        def header(value):
            data = json.dumps(value).encode()
            source.write_bytes(struct.pack("<Q", len(data)) + data)
            return source

        def weights(value):
            index.write_text(json.dumps({"weight_map": value}))
            return index

        source.write_bytes(b"short")
        expect(source, "missing safetensors header length")
        source.write_bytes(struct.pack("<Q", 20) + b"{}")
        expect(source, "invalid safetensors header length")
        # Sparse file: isolate the size cap from the independent file-size check.
        with source.open("wb") as stream:
            stream.write(struct.pack("<Q", 100_000_001) + b"{}")
            stream.truncate(100_000_009)
        expect(source, "invalid safetensors header length")
        expect(header([]), "header must be an object")
        for info in ([], {"dtype": 1, "shape": [512, 256]},
                     {"dtype": "F32", "shape": "512,256"},
                     {"dtype": "F32", "shape": {}},
                     {"dtype": "F32", "shape": [True, 256]},
                     {"dtype": "F32", "shape": [-1, 256]}):
            expect(header({name: info}), "invalid tensor header for " + name)
        expect(header({}), "empty safetensors source; nothing compared")
        index.write_text("[]")
        expect(index, "index must be an object")
        expect(weights({}), "empty or missing weight_map; nothing compared")
        expect(weights([]), "empty or missing weight_map; nothing compared")
        expect(weights({name: 1}), "invalid shard for " + name)

        write_shard(source, {name: [512, 256]})
        expect(weights({"absent.weight": source.name}), "absent from declared shard")
        write_shard(source, {name: [512, 256], "extra.weight": [1]})
        expect(weights({name: source.name}), "extra.weight in source.safetensors is not indexed there")
        for shard in (str(source.resolve()), "../source.safetensors",
                      "sub/source.safetensors", "sub\\source.safetensors",
                      "C:\\source.safetensors", "source.bin"):
            expect(weights({name: shard}), "shard must be a bare .safetensors filename")

        # A legal basename must not escape through a symlink either.
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / "outside.safetensors"
            write_shard(target, {name: [512, 256]})
            link = directory / "linked.safetensors"
            try:
                link.symlink_to(target)
            except OSError as error:
                if getattr(error, "winerror", None) != 1314:
                    raise
                print("SKIP: symlink containment test (Windows requires symlink privileges)")
            else:
                expect(weights({name: link.name}), "shard resolves outside index directory")

        data = b"[" * 200_000 + b"]" * 200_000
        source.write_bytes(struct.pack("<Q", len(data)) + data)
        expect(source, "ERROR:")
        write_shard(source, {name: [512, 256]})
        for blocks in (0, 1025):
            path = directory / f"blocks-{blocks}.gguf"
            writer = gguf.GGUFWriter(str(path), "llada")
            writer.add_uint32("llada.block_count", blocks)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            expect(source, "llada.block_count must be between 1 and 1024", gguf_path=path)

        reader = gguf.GGUFReader(str(model))
        mapper = name_map(reader)
        aliases = [n + ".weight" for n in mapper.mapping
                   if mapper.get_name(n + ".weight", try_suffixes=(".weight",)) == "token_embd.weight"]
        assert len(aliases) >= 2
        write_shard(source, {aliases[0]: [512, 256], aliases[1]: [512, 256]})
        expect(source, "duplicates GGUF target token_embd.weight", status=1)

        # Tensor names come from someone else's file: a terminal escape in one
        # must reach the report as text, not as a command to the terminal.
        write_shard(source, {name + "\x1b[2J": [512, 256]})
        expect(source, "unmapped source name: " + name + "\\x1b[2J")

        # Opening a FIFO for reading waits for a writer, so a FIFO named like a
        # shard would hang the checker before it read a byte.
        if hasattr(os, "mkfifo"):
            fifo = directory / "fifo.safetensors"
            os.mkfifo(fifo)
            expect(fifo, "is not a regular file")
            expect(weights({name: fifo.name}), "is not a regular file")
    return runs


def self_test() -> None:
    fixtures = Path(__file__).resolve().parent.parent / "tests/fixtures"
    runs = 0
    for arch, expected in (("llada", 21), ("dream", 27)):
        model = fixtures / f"tiny-{arch}.gguf"
        reader = gguf.GGUFReader(str(model))
        mapper = name_map(reader)
        shapes = {}
        for tensor in reader.tensors:
            suffix = ".bias" if tensor.name.endswith(".bias") else ".weight"
            candidates = [name + suffix for name in mapper.mapping
                          if name.startswith("model.") or name == "lm_head"]
            name = next(name for name in sorted(candidates)
                        if mapper.get_name(name, try_suffixes=(".weight", ".bias")) == tensor.name)
            shapes[name] = [int(d) for d in reversed(tensor.shape)]
        assert len(shapes) == expected
        embedding = next(n for n in shapes if mapper.get_name(n, try_suffixes=(".weight",)) == "token_embd.weight")
        output = next(n for n in shapes if mapper.get_name(n, try_suffixes=(".weight",)) == "output.weight")
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)

            def expect(source, status, message, ignore=None, gguf_path=None):
                nonlocal runs
                captured = io.StringIO()
                with contextlib.redirect_stdout(captured):
                    code = compare(gguf_path or model, source, ignore)
                assert code == status and message in captured.getvalue(), captured.getvalue()
                runs += 1
                return captured.getvalue()

            expect(write_twin(directory, shapes), 0, f"compared {expected}, ok {expected}")
            single = directory / "single.safetensors"
            write_shard(single, shapes)
            expect(single, 0, f"compared {expected}")
            changed = copy.deepcopy(shapes)
            changed[embedding][0] += 1
            expect(write_twin(directory, changed), 1, "shape mismatch: " + embedding)
            changed = copy.deepcopy(shapes)
            changed[embedding].reverse()
            expect(write_twin(directory, changed), 1, "shape mismatch: " + embedding)
            changed = dict(shapes)
            del changed[output]
            expect(write_twin(directory, changed), 1, "missing in safetensors: output.weight; source may tie")
            changed = dict(shapes)
            changed["unmapped.original.weight"] = changed.pop(embedding)
            expect(write_twin(directory, changed), 1, "unmapped source name: unmapped.original.weight")
            changed = dict(shapes)
            changed["model.layers.99.q_proj.weight"] = [1]
            expect(write_twin(directory, changed), 1, "unmapped source name")
            missing = directory / "missing.gguf"
            writer = gguf.GGUFWriter(str(missing), arch)
            writer.add_uint32(arch + ".block_count", int(reader.fields[arch + ".block_count"].contents()))
            for tensor in reader.tensors:
                if tensor.name != "token_embd.weight":
                    writer.add_tensor(tensor.name, np.zeros(tuple(int(d) for d in reversed(tensor.shape)), dtype=np.float32))
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            expect(write_twin(directory, shapes), 1, "missing in GGUF: " + embedding, gguf_path=missing)
            changed = dict(shapes)
            changed["rotary.inv_freq"] = [2]
            expect(write_twin(directory, changed), 0, f"compared {expected}", r"rotary\.inv_freq$")
            empty = directory / "empty.json"
            empty.write_text('{"weight_map": {}}')
            expect(empty, 2, "empty or missing weight_map; nothing compared")
            expect(directory / "absent.safetensors", 2, "ERROR:")
            expect(single, 2, "compared 0", ".*")
            report = expect(single, 0, f"compared {expected - 1}, ok {expected - 1}", "^" + re.escape(embedding) + "$")
            assert "ignored 1" in report
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                status = compare(fixtures / "kquants.gguf", single)
            assert status == 2 and "no tensor name map" in captured.getvalue()
            runs += 1
    runs += self_test_inputs(fixtures)
    print(f"check_weights self-test: {runs} cases passed (LLaDA 21, Dream 27 tensors)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, nargs="?")
    parser.add_argument("source", type=Path, nargs="?")
    parser.add_argument("--ignore", metavar="REGEX", help="skip matching source tensor names")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.model is None or args.source is None:
        parser.error("MODEL.gguf and INDEX.json or a .safetensors file are required")
    return compare(args.model, args.source, args.ignore)


if __name__ == "__main__":
    raise SystemExit(main())
