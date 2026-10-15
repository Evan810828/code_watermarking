"""Package, verify, and restore the preserved benchmark evidence using stdlib.

Run from any directory. Archives retain original relative paths and file bytes.
Restore refuses to overwrite differing files, including results of newer runs.
"""

import argparse
from collections import defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile


ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "results"
INVENTORY = STORE / "inventory.json"
PART_BYTES = 64 * 1024 * 1024
GROUPS = {
    **{name: [f"out/{name}"] for name in ("kgw", "semstamp", "sweet", "codewm", "seqmark")},
    "round1": ["archive/out_round1"],
    "robustness": ["out/attacks"],
    "prompt_and_figures": ["out/comment_prompt", "out/figures"],
    "logs": ["logs/baseline", "logs/seqmark"],
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def source_files(prefixes):
    for prefix in prefixes:
        for path in sorted((ROOT / prefix).rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            if {"cache", "__pycache__", ".git"} & set(path.parts):
                continue
            if path.suffix == ".pyc" or "_aug" in path.name:
                continue
            yield path


def build():
    """Create an immutable snapshot; rebuilding requires a new inventory location."""
    if INVENTORY.exists():
        raise ValueError("inventory already exists; preserve this snapshot")
    (STORE / "archives").mkdir(parents=True, exist_ok=True)
    records, archives = [], []
    for group, prefixes in GROUPS.items():
        batches, current, size = [], [], 0
        for path in source_files(prefixes):
            nbytes = path.stat().st_size
            if current and size + nbytes > PART_BYTES:
                batches.append(current)
                current, size = [], 0
            current.append(path)
            size += nbytes
        if current:
            batches.append(current)
        for part, batch in enumerate(batches, 1):
            name = f"archives/{group}-{part:02d}.tar.gz"
            target = STORE / name
            with target.open("xb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                    with tarfile.open(fileobj=compressed, mode="w") as archive:
                        for path in batch:
                            content = path.read_bytes()
                            relative = path.relative_to(ROOT).as_posix()
                            info = tarfile.TarInfo(relative)
                            info.size = len(content)
                            info.mode = 0o644
                            archive.addfile(info, io.BytesIO(content))
                            records.append({"path": relative, "group": group,
                                            "archive": name, "bytes": len(content),
                                            "sha256": sha256(content)})
            packed = target.read_bytes()
            archives.append({"path": name, "group": group, "files": len(batch),
                             "bytes": len(packed), "sha256": sha256(packed)})
            print(f"{name}: {len(batch)} files, {len(packed):,} bytes", flush=True)
    INVENTORY.write_text(json.dumps({
        "schema_version": 1,
        "description": "Historical evidence snapshot; no new experiments or inferred provenance.",
        "excluded": ["model download caches", "Python bytecode", "AST-specific studies"],
        "archives": archives, "files": records,
    }, indent=2) + "\n")


def safe_path(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe archive path: {relative}")
    target = root.joinpath(*path.parts)
    try:
        target.resolve().relative_to(root.resolve())
    except ValueError:
        raise ValueError(f"path escapes destination through a symlink: {relative}")
    return target


def verify(groups=None, destination=None):
    index = json.loads(INVENTORY.read_text())
    records = [r for r in index["files"] if not groups or r["group"] in groups]
    expected = {r["path"]: r for r in records}
    if len(expected) != len(records):
        raise ValueError("duplicate inventory paths")
    by_archive = defaultdict(dict)
    for r in records:
        if PurePosixPath(r["path"]).parts[0] not in {"out", "archive", "logs"}:
            raise ValueError(f"unexpected evidence location: {r['path']}")
        safe_path(ROOT, r["path"])
        by_archive[r["archive"]][r["path"]] = r
    packed_index = {a["path"]: a for a in index["archives"]}
    for name, wanted in by_archive.items():
        archive_path = safe_path(STORE, name)
        content = archive_path.read_bytes()
        meta = packed_index[name]
        if len(content) != meta["bytes"] or sha256(content) != meta["sha256"]:
            raise ValueError(f"archive checksum mismatch: {name}")
        seen = set()
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                if not member.isfile() or member.name not in wanted or member.name in seen:
                    raise ValueError(f"unexpected member in {name}: {member.name}")
                seen.add(member.name)
                data = archive.extractfile(member).read()
                r = wanted[member.name]
                if len(data) != r["bytes"] or sha256(data) != r["sha256"]:
                    raise ValueError(f"file checksum mismatch: {member.name}")
                if destination is not None:
                    target = safe_path(destination, member.name)
                    if target.exists() and (not target.is_file() or sha256(target.read_bytes()) != r["sha256"]):
                        raise ValueError(f"refusing to overwrite differing file: {target}")
        if seen != set(wanted):
            raise ValueError(f"missing archive members: {name}")
    print(f"Verified {len(records)} files in {len(by_archive)} archives.")
    return by_archive


def restore(groups, destination):
    destination = destination.resolve()
    selected = verify(groups, destination)
    restored = 0
    for name in selected:
        with tarfile.open(STORE / name, "r:gz") as archive:
            for member in archive:
                target = safe_path(destination, member.name)
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as handle:
                    handle.write(archive.extractfile(member).read())
                restored += 1
    print(f"Restored {restored} files to {destination}; identical existing files retained.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify", "restore"))
    parser.add_argument("--group", action="append", choices=sorted(GROUPS))
    parser.add_argument("--destination", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.command == "build":
        build()
    elif args.command == "verify":
        verify(args.group)
    else:
        restore(args.group, args.destination)


if __name__ == "__main__":
    main()
