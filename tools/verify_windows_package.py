"""Verify an extracted Soundtrace distribution using only bundled Python."""
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    def verify(item):
        path = ROOT / item["path"]
        if not path.resolve().is_relative_to(ROOT.resolve()) or not path.is_file():
            return item["path"] + ": missing/outside package"
        # The launcher intentionally rewrites these three small configs after moves.
        if item.get("mutable_at_launch"):
            return None
        if path.stat().st_size != item["size"]:
            return item["path"] + ": size mismatch"
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(data)
        if h.hexdigest() != item["sha256"]:
            return item["path"] + ": hash mismatch"
        return None
    failures = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, result in enumerate(pool.map(verify, manifest["files"]), 1):
            if result:
                failures.append(result)
            if index % 10000 == 0:
                print(f"Verified {index}/{len(manifest['files'])} files", flush=True)
    for value in failures:
        print(value)
    print(f"Checked {len(manifest['files'])} files; failures={len(failures)}")
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
