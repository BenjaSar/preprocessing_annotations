#!/usr/bin/env python3
"""Draft the GT-category -> taxonomy-class map for category scoring.

The room GT uses 93 categories of its own; the pipeline emits the 24
MANDATORY_CLASSES. Scoring whether a located room also has the right class
needs a map between them, and that map is a human judgment.

This drafts it with the pipeline's own resolver and keeps only the answers the
taxonomy explicitly declares -- the category name IS the class, or is a listed
variant of it, or has a VLM_CATEGORY_MAP entry. Anything the resolver reaches
by substring, extended-class or fallback inference is written as null for a
human, because that is where it quietly conflates distinct kinds of room
(measured: `library` reached RESIDENTIAL UNIT that way).

Declared is not the same as correct for scoring: the taxonomy also declares
conflations a reviewer may reject (kitchen is a listed variant of CAFETERIA).
Those surface in the map as several GT categories sharing one class.

The result is a draft: `"reviewed": false` until a human checks the resolved
entries too and flips it.

Usage:
    python scripts/draft_gt_category_map.py --out <map.json>
    python scripts/draft_gt_category_map.py --gt <result.json> --out <map.json>
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from preprocessing_annotations.automation.taxonomy import (
    VLM_CATEGORY_MAP,
    _SURFACE_TO_MANDATORY,
    normalize_to_mandatory,
)
from preprocessing_annotations.config import RoomGtEvalConfig


def category_room_counts(gt_coco_path: Path) -> Counter:
    """GT category name -> number of annotated rooms."""
    data = json.loads(Path(gt_coco_path).read_text())
    name_by_id = {category["id"]: category["name"] for category in data["categories"]}
    return Counter(name_by_id[a["category_id"]] for a in data["annotations"])


def _is_declared(readable: str, resolved: str) -> bool:
    """True when the taxonomy states this category IS that class, rather than
    reaching it by substring, extended-class or fallback inference.

    Withholding the inferred ones is the point: they are where the resolver
    quietly conflates distinct kinds of room (measured: `library` reached
    RESIDENTIAL UNIT that way), and a conflated class silently scores a wrong
    prediction as right.
    """
    return (
        readable == resolved
        or _SURFACE_TO_MANDATORY.get(readable) == resolved
        or VLM_CATEGORY_MAP.get(readable.lower()) == resolved
    )


def draft_mapping(categories: Counter) -> dict:
    """Draft map plus the review queue. Categories the taxonomy does not
    explicitly declare map to None for a human to decide."""
    mapping: dict[str, str | None] = {}
    needs_review: list[dict] = []
    for category, rooms in categories.most_common():
        readable = category.replace("_", " ").replace("-", " ").upper()
        resolved = normalize_to_mandatory(readable)
        if _is_declared(readable, resolved):
            mapping[category] = resolved
        else:
            mapping[category] = None
            needs_review.append({"gt_category": category, "gt_rooms": rooms,
                                 "resolver_would_say": resolved})
    return {"reviewed": False, "mapping": mapping, "needs_review": needs_review}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draft the GT-category -> taxonomy-class map")
    parser.add_argument("--gt", type=Path, default=None, help="GT COCO json (default: RoomGtEvalConfig)")
    parser.add_argument("--out", type=Path, required=True, help="Map JSON to write")
    args = parser.parse_args()

    counts = category_room_counts(args.gt or RoomGtEvalConfig().gt_coco_path)
    draft = draft_mapping(counts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(draft, indent=2) + "\n")

    unresolved_rooms = sum(entry["gt_rooms"] for entry in draft["needs_review"])
    print(
        f"wrote {args.out}: {len(draft['mapping'])} categories "
        f"({sum(counts.values())} GT rooms); "
        f"{len(draft['needs_review'])} categories / {unresolved_rooms} rooms need a human class"
    )
    for entry in draft["needs_review"]:
        print(
            f"  unmapped: {entry['gt_category']} ({entry['gt_rooms']} rooms) "
            f"-- resolver would have said {entry['resolver_would_say']}"
        )
