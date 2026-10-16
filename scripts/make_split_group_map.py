#!/usr/bin/env python3
"""Write the split group map (ExportConfig.split_group_map_path).

Every source document of a run becomes its own group, and the documents the
room GT annotates are pinned to test -- they are the only room evaluation set
there is, so they must stay out of any training split.

Document names differ between the two sides (Label Studio rewrites spaces and
adds an upload prefix), so GT pages are matched to the run's own documents
through the eval harness's page_key, never by raw file name.

Merging documents into a wider family (the AVI-ON and Electrical PDFs of one
building, or two schools of one design family -- measured to share room names
at Jaccard .39-.59) is a human judgment: add those entries to "groups" by hand
afterwards. This script never invents them.

Usage:
    python scripts/make_split_group_map.py --run <run_dir> --out <map.json>
    python scripts/make_split_group_map.py --run <run_dir> --out <map.json> --gt <result.json>
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "eval"))

from preprocessing_annotations.config import RoomGtEvalConfig
from preprocessing_annotations.export.group_split import document_key

from room_gt_eval import load_gt_polygons, page_key


def build_group_map(run_dir: Path, gt_coco_path: Path) -> dict:
    """One group per run document; groups holding a GT page are pinned to test."""
    gt_pages = set(load_gt_polygons(gt_coco_path))
    groups: dict[str, str] = {}
    pinned: set[str] = set()
    for path in sorted((run_dir / "annotations").glob("*.json")):
        document = document_key(path.name)
        groups[document] = document
        if page_key(path.name) in gt_pages:
            pinned.add(document)
    return {"groups": groups, "pinned_test_groups": sorted(pinned)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write the split group map")
    parser.add_argument("--run", type=Path, required=True, help="Run dir whose annotations name the documents")
    parser.add_argument("--gt", type=Path, default=None, help="GT COCO json (default: RoomGtEvalConfig)")
    parser.add_argument("--out", type=Path, required=True, help="Group map JSON to write")
    args = parser.parse_args()

    group_map = build_group_map(args.run, args.gt or RoomGtEvalConfig().gt_coco_path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(group_map, indent=2) + "\n")
    print(
        f"wrote {args.out}: {len(group_map['groups'])} documents, "
        f"{len(group_map['pinned_test_groups'])} pinned to test"
    )
    for group in group_map["pinned_test_groups"]:
        print(f"  pinned: {group}")
