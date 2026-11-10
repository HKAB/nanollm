"""Shared JSONL byte ranges and leakage index validation."""

import hashlib
import os
from pathlib import Path

import orjson


def jsonl_rows(filename, start, end):
    """Yield records whose starting byte belongs to [start, end)."""
    if start >= end:
        return
    with open(filename, "rb") as source:
        if start:
            source.seek(start - 1)
            if source.read(1) != b"\n":
                source.readline()
        while (offset := source.tell()) < end:
            line = source.readline()
            if not line:
                break
            yield offset, line


def file_identity(filename):
    stat = os.stat(filename)
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def read_index(path, mode, filenames):
    payload = Path(path).read_bytes()
    index = orjson.loads(payload)
    if index["version"] != 1 or index["mode"] != mode:
        raise ValueError(f"Leakage index format or mode does not match {mode}: {path}")
    offsets = {}
    for filename in filenames:
        entry = index["files"].get(os.path.realpath(filename))
        if entry is None or entry["source"] != file_identity(filename):
            raise ValueError(f"Missing or stale leakage index entry for {filename}")
        values = [match["offset"] for match in entry["matches"]]
        previous = -1
        for offset in values:
            if type(offset) is not int or not previous < offset < entry["source"]["size"]:
                raise ValueError(f"Invalid or unsorted leakage offset for {filename}: {offset!r}")
            previous = offset
        offsets[filename] = values
    return offsets, hashlib.sha256(payload).hexdigest()
