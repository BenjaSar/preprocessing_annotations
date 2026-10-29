"""Check a finished run's COCO splits for leakage, and re-split when asked.

Two ways one drawing reaches two splits:

  families  -- the group map (ExportConfig.split_group_map_path) merges the
               documents of one building or design family, and every page of
               a group must land in one split. A map edited after the run,
               or a document the map does not name, breaks that.
  copies    -- the same drawing shipped twice (a layered PDF and its
               flattened export). Those are different document names, so the
               group map cannot catch them unless a human named the pair;
               their rasters are near-identical, which this measures.

Pixel comparison catches copies only. A rotated re-export, or two sheets of
one project, look unrelated here and stay a human call from the title block
(see scripts/make_split_group_map.py). Thresholds and their measurements:
SFTDatasetConfig.duplicate_page_max_abs_diff.

A failure needs no rerun: --fix re-runs COCOExporter.export_splits over the
same processed annotations with the current map, which is where the group
assignment is decided.

Usage:
    python eval/split_integrity.py --run <run_dir>
    python eval/split_integrity.py --run <run_dir> --fix
"""

import argparse
import json
import logging
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "eval"))

from preprocessing_annotations.config import (
    ExportConfig,
    PipelineConfig,
    SFTDatasetConfig,
)
from preprocessing_annotations.export.exporters import COCOExporter
from preprocessing_annotations.export.group_split import (
    SPLIT_NAMES,
    GroupMap,
    document_key,
)

from report_io import emit_report

logger = logging.getLogger(__name__)


def split_of_documents(run_dir: Path) -> Dict[str, List[str]]:
    """Document key -> the splits its pages appear in (sorted, deduped)."""
    splits: Dict[str, List[str]] = {}
    for split in SPLIT_NAMES:
        path = run_dir / "coco" / f"{split}.json"
        if not path.exists():
            continue
        for image in json.loads(path.read_text())["images"]:
            document = document_key(image["file_name"])
            if split not in splits.setdefault(document, []):
                splits[document].append(split)
    return splits


def straddling_groups(
    document_splits: Dict[str, List[str]], group_map: GroupMap
) -> List[dict]:
    """Groups whose documents do not all sit in one split."""
    by_group: Dict[str, Dict[str, List[str]]] = {}
    for document, splits in document_splits.items():
        group = group_map.group_of(document)
        by_group.setdefault(group, {})[document] = splits
    return [
        {"group": group, "documents": dict(sorted(documents.items()))}
        for group, documents in sorted(by_group.items())
        if len({split for splits in documents.values() for split in splits}) > 1
    ]


def _thumbnail(path: Path, size: Tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L").resize(size)
    return np.asarray(image, dtype=np.float32) / 255.0


def _page_thumbnails(
    run_dir: Path, documents: Sequence[str], size: Tuple[int, int]
) -> Dict[str, List[np.ndarray]]:
    """Every rendered page of each document, as comparable thumbnails."""
    thumbnails: Dict[str, List[np.ndarray]] = {document: [] for document in documents}
    for path in sorted((run_dir / "images").glob("*")):
        document = document_key(path.name)
        if document in thumbnails:
            try:
                thumbnails[document].append(_thumbnail(path, size))
            except OSError as error:  # not an image, or unreadable
                logger.warning("skipping %s: %s", path.name, error)
    return thumbnails


def duplicate_documents(
    run_dir: Path, document_splits: Dict[str, List[str]], config: SFTDatasetConfig
) -> List[dict]:
    """Document pairs in different splits whose pages are near-identical."""
    cross_split = [
        (a, b)
        for a, b in combinations(sorted(document_splits), 2)
        if not set(document_splits[a]) & set(document_splits[b])
    ]
    if not cross_split:
        return []

    thumbnails = _page_thumbnails(
        run_dir,
        sorted({document for pair in cross_split for document in pair}),
        config.duplicate_thumbnail_px,
    )
    duplicates = []
    for a, b in cross_split:
        pages = [
            float(np.abs(left - right).mean())
            for left, right in zip(thumbnails[a], thumbnails[b])
        ]
        if pages and min(pages) < config.duplicate_page_max_abs_diff:
            duplicates.append({
                "documents": [a, b],
                "splits": [document_splits[a], document_splits[b]],
                "min_abs_diff": round(min(pages), 6),
                "pages_compared": len(pages),
            })
    return duplicates


def check(run_dir: Path, config: SFTDatasetConfig, group_map: GroupMap) -> dict:
    document_splits = split_of_documents(run_dir)
    straddles = straddling_groups(document_splits, group_map)
    duplicates = duplicate_documents(run_dir, document_splits, config)
    return {
        "run": str(run_dir),
        "documents": len(document_splits),
        "duplicate_page_max_abs_diff": config.duplicate_page_max_abs_diff,
        "straddling_groups": straddles,
        "duplicate_documents": duplicates,
        "ok": not straddles and not duplicates,
    }


def resplit(run_dir: Path, pipeline_config: PipelineConfig, group_map: GroupMap) -> dict:
    """Rebuild coco/ from this run's processed annotations and the current map."""
    export = pipeline_config.export
    return COCOExporter().export_splits(
        run_dir / "processed_annotations",
        run_dir / "images",
        run_dir / "coco",
        splits=export.split_ratios,
        seed=export.split_seed,
        min_annotations_per_image=pipeline_config.min_rooms_for_sft,
        group_map=group_map,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check a run's COCO splits for family straddles and duplicate drawings"
    )
    parser.add_argument("--run", type=Path, required=True, help="Run dir with coco/ and images/")
    parser.add_argument(
        "--group-map", type=Path, default=None,
        help="Group map to check against (default: ExportConfig.split_group_map_path)",
    )
    parser.add_argument(
        "--fix", action="store_true",
        help="Re-split coco/ with the current group map, then re-check",
    )
    parser.add_argument("--out", type=Path, default=None, help="Also write report JSON here")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("preprocessing_annotations").setLevel(logging.WARNING)

    pipeline_config = PipelineConfig()
    map_path = args.group_map or pipeline_config.export.split_group_map_path
    group_map = GroupMap.load(map_path)
    sft_config = SFTDatasetConfig()

    report = check(args.run, sft_config, group_map)
    report["group_map"] = str(map_path) if map_path is not None else None

    if args.fix and not report["ok"]:
        report["resplit_counts"] = resplit(args.run, pipeline_config, group_map)
        report["after_fix"] = check(args.run, sft_config, group_map)
        report["ok"] = report["after_fix"]["ok"]

    emit_report(report, args.out)
    sys.exit(0 if report["ok"] else 1)
