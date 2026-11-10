"""Scan training JSONL for normalized CORE test questions."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
import hashlib
from itertools import islice
import os
from pathlib import Path
import re
import tempfile
import unicodedata

import orjson
from tqdm import tqdm

from scripts.leakage_index import file_identity, jsonl_rows
from scripts.pretokenize import sharegpt_to_messages

WORDS = re.compile(r"\w+")
RANGE_BYTES = 64 * 1024**2


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def build_matcher(grouped):
    items = []
    frequencies = Counter()
    for task, rows in grouped.items():
        for index, row in enumerate(rows):
            query = normalize(row["query"])
            words = set(WORDS.findall(query))
            if not words:
                raise ValueError(f"CORE query has no searchable words: {task}/{index}")
            frequencies.update(words)
            choices = {normalize(choice) for choice in row["choices"]} - {""}
            items.append((task, index, query, words, choices))
    matcher = defaultdict(list)
    for task, index, query, words, choices in items:
        anchor = min(words, key=lambda word: (frequencies[word], -len(word), word))
        question = re.compile(r"(?<!\w)" + re.escape(query) + r"(?!\w)")
        evidence = [
            re.compile(r"(?<!\w)" + re.escape(choice) + r"(?!\w)")
            for choice in sorted(choices)
        ] if len(WORDS.findall(query)) < 8 else []
        matcher[anchor].append((task, index, question, evidence))
    return dict(matcher)


def init_worker(matcher):
    global MATCHER
    MATCHER = matcher


def scan_range(filename, start, end, mode):
    matches = []
    invalid = 0
    for offset, line in jsonl_rows(filename, start, end):
        try:
            doc = orjson.loads(line)
            if mode == "sft":
                messages = sharegpt_to_messages(doc) or []
                fields = [message["content"] for message in messages]
            else:
                fields = [doc.get("text")]
        except (ValueError, AttributeError, TypeError):
            invalid += 1
            continue

        fields = [normalize(field) for field in fields if isinstance(field, str)]
        hits = set()
        for field in fields:
            for word in set(WORDS.findall(field)):
                for task, index, question, choices in MATCHER.get(word, ()):
                    if question.search(field) and (
                        not choices or sum(any(choice.search(text) for text in fields)
                                           for choice in choices) >= 2
                    ):
                        hits.add((task, index))
        if hits:
            matches.append({
                "offset": offset,
                "items": [{"task": task, "index": index} for task, index in sorted(hits)],
                "reason": "normalized_query",
            })
    return matches, invalid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Recursive JSONL glob")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=["pretrain", "sft"], default="pretrain")
    parser.add_argument("--cpu-count", type=int, default=16)
    parser.add_argument("--dataset", default=os.getenv("NANOLLM_CORE_DATASET") or "hkab/nanollm-core")
    parser.add_argument("--revision", default=os.getenv("NANOLLM_CORE_DATASET_REVISION"))
    parser.add_argument("--hf-home")
    parser.add_argument("--ignore", nargs="*", default=[])
    args = parser.parse_args()
    if args.cpu_count < 1:
        parser.error("--cpu-count must be positive")
    files = sorted({
        os.path.realpath(path) for path in glob.glob(args.input_dir, recursive=True)
        if os.path.isfile(path) and not any(ignore in path for ignore in args.ignore)
    })
    if not files:
        parser.error("No input files matched")
    if str(args.output.resolve()) in files:
        parser.error("--output overlaps an input file")
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    os.environ["NANOLLM_CORE_DATASET"] = args.dataset
    if args.revision:
        os.environ["NANOLLM_CORE_DATASET_REVISION"] = args.revision

    print(f"Loading CORE test data from {args.dataset}", flush=True)
    from tasks.pretrain.core import load_core_data

    grouped = load_core_data()
    matcher = build_matcher(grouped)
    manifest = {
        "version": 1, "mode": args.mode, "dataset": args.dataset,
        "revision": args.revision,
        "benchmark_sha256": hashlib.sha256(orjson.dumps(grouped, option=orjson.OPT_SORT_KEYS)).hexdigest(),
        "files": {path: {"source": file_identity(path), "matches": []} for path in files},
    }
    matched = invalid = 0
    total = sum(entry["source"]["size"] for entry in manifest["files"].values())
    with (
        ProcessPoolExecutor(max_workers=args.cpu_count, initializer=init_worker,
                            initargs=(matcher,)) as pool,
        tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc="Scanning JSONL") as progress,
    ):
        for path, entry in manifest["files"].items():
            size = entry["source"]["size"]
            range_size = min(RANGE_BYTES, max(1, (size + args.cpu_count - 1) // args.cpu_count))
            starts = iter(range(0, size, range_size))
            while batch := list(islice(starts, 4 * args.cpu_count)):
                futures = {
                    pool.submit(scan_range, path, start, min(start + range_size, size), args.mode):
                    min(range_size, size - start) for start in batch
                }
                for future in as_completed(futures):
                    hits, errors = future.result()
                    entry["matches"].extend(hits)
                    matched += len(hits)
                    invalid += errors
                    progress.set_postfix(matched=matched, invalid=invalid, refresh=False)
                    progress.update(futures[future])
            entry["matches"].sort(key=lambda match: match["offset"])

    for path, entry in manifest["files"].items():
        if file_identity(path) != entry["source"]:
            raise RuntimeError(f"Input changed during scan: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as output:
        temp_path = Path(output.name)
        try:
            output.write(orjson.dumps(manifest))
            output.close()
            temp_path.replace(args.output)
        finally:
            temp_path.unlink(missing_ok=True)
    print(f"Saved {matched:,} flagged rows across {len(files):,} files to {args.output} "
          f"({invalid:,} invalid rows)")


if __name__ == "__main__":
    main()
