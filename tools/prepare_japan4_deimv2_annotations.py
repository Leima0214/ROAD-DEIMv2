#!/usr/bin/env python3
"""Create DEIMv2-compatible 0-based annotation views without changing source COCO JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


EXPECTED_CATEGORIES = {1: "D00", 2: "D10", 3: "D20", 4: "D40"}
EXPECTED_IMAGES = {"train": 6320, "val": 790, "test": 790}
EXPECTED_ANNOTATIONS = {"train": 13175, "val": 1647, "test": 1647}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(root: Path) -> dict[str, object]:
    source_dir = root / "annotations"
    output_dir = source_dir / "deimv2"
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "source_root": str(root),
        "mapping": {str(key): key - 1 for key in EXPECTED_CATEGORIES},
        "splits": {},
    }
    for split in ("train", "val", "test"):
        source = source_dir / f"instances_{split}.json"
        destination = output_dir / f"instances_{split}.json"
        data = json.loads(source.read_text(encoding="utf-8"))
        categories = {int(item["id"]): item["name"] for item in data["categories"]}
        if categories != EXPECTED_CATEGORIES:
            raise AssertionError(f"Unexpected category contract in {source}: {categories}")
        if len(data["images"]) != EXPECTED_IMAGES[split]:
            raise AssertionError(f"Unexpected image count in {source}")
        if len(data["annotations"]) != EXPECTED_ANNOTATIONS[split]:
            raise AssertionError(f"Unexpected annotation count in {source}")

        for category in data["categories"]:
            category["id"] = int(category["id"]) - 1
        for annotation in data["annotations"]:
            category_id = int(annotation["category_id"])
            if category_id not in EXPECTED_CATEGORIES:
                raise AssertionError(f"Invalid category_id={category_id} in {source}")
            annotation["category_id"] = category_id - 1

        payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=output_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            Path(temporary).replace(destination)
        except Exception:
            Path(temporary).unlink(missing_ok=True)
            raise
        report["splits"][split] = {
            "images": len(data["images"]),
            "annotations": len(data["annotations"]),
            "category_ids": [item["id"] for item in data["categories"]],
            "source_sha256": sha256(source),
            "output_sha256": sha256(destination),
        }

    report_path = output_dir / "conversion_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("/JAPAN4-DETR/JAPAN4-DETR")
    )
    args = parser.parse_args()
    print(json.dumps(convert(args.root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
