"""Phase 1 (2026-10-14): per-page box-convention consistency, without GT.

Why this is separate from room_gt_eval.py: that harness scores predictions
against room polygons, so it only ever sees the 48 pages that have GT.
Measured coverage -- sprint1_verify54 48/50 pages have GT but only 2.7% of
its rooms are vlm_only; sprint1_verify55 has 0/107 pages with GT and 18.2%
vlm_only. Every oversize number this project has recorded therefore came
from the population that barely contains the failure mode, and the page a
reviewer actually flagged (ANIMAL CLINIC ... FLAT_page000) has no GT at all.

What it measures instead is a property of the page itself: the pipeline
emits two box conventions (annotation_schema.box_convention_of -- an
OCR-anchored room's bbox is its label text, a vlm_only room's bbox is the
whole room), and a page carrying both holds boxes whose sizes are not
comparable. On that flagged page, 4500x3000 with nothing above 1% of the
page, LAB (vlm_only, 534x213) is 281x the area of BA (vlm_ocr_merged,
29x15). No area guard can see that, because no box is large in absolute
terms; the defect is the spread within one image.

Measured separation of `area_spread` (max room area / median room area):
mixed-convention pages have median 53.3x on verify54 and 29.2x on
verify55, against 2.1x and 1.5x for single-convention pages.

Deliberately reports the distribution rather than classifying pages: the
separation is strong but not clean (single-convention pages reach 21.6x),
and this project has no measurement that would justify a cutoff. A caller
that needs one supplies it.
"""

import argparse
import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from report_io import emit_report

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageConsistency:
    """One page's box-convention mix and the size spread that comes with it."""

    page: str
    rooms: int
    conventions: Sequence[str]
    untagged_rooms: int
    area_spread: Optional[float]

    @property
    def is_mixed(self) -> bool:
        """More than one convention on one page: boxes of two different kinds."""
        return len(self.conventions) > 1

    @property
    def convention_known(self) -> bool:
        """Whether any room on the page states its convention at all.

        A page from a run predating box_convention states none. Such a page
        is not a single-convention page -- its mix is unknown -- and must be
        reported apart from both, or a whole legacy run reads as clean.
        """
        return bool(self.conventions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page": self.page,
            "rooms": self.rooms,
            "conventions": list(self.conventions),
            "untagged_rooms": self.untagged_rooms,
            "area_spread": self.area_spread,
            "is_mixed": self.is_mixed,
            "convention_known": self.convention_known,
        }


def _room_areas(rooms: Sequence[Dict[str, Any]]) -> List[float]:
    return [
        float(room["bbox"][2]) * float(room["bbox"][3])
        for room in rooms
        if len(room.get("bbox") or []) == 4
    ]


def area_spread(areas: Sequence[float]) -> Optional[float]:
    """Largest room area over the median one, or None when undefined.

    None rather than a sentinel number: a page with fewer than two rooms has
    no spread to speak of, and a zero median cannot be divided by. Both must
    stay out of the distribution instead of landing in it as a 0 or a 1.
    """
    if len(areas) < 2:
        return None
    middle = statistics.median(areas)
    return max(areas) / middle if middle else None


def page_consistency(page: str, annotation: Dict[str, Any]) -> PageConsistency:
    """One processed annotation -> its convention mix and size spread."""
    rooms = [r for r in annotation.get("rooms", []) if len(r.get("bbox") or []) == 4]
    conventions = {r.get("box_convention") for r in rooms}
    return PageConsistency(
        page=page,
        rooms=len(rooms),
        conventions=sorted(c for c in conventions if c is not None),
        untagged_rooms=sum(1 for r in rooms if r.get("box_convention") is None),
        area_spread=area_spread(_room_areas(rooms)),
    )


def scan_run(processed_dir: Path) -> List[PageConsistency]:
    """Every processed annotation in a run, in page order."""
    return [
        page_consistency(path.stem, json.loads(path.read_text()))
        for path in sorted(processed_dir.glob("*.json"))
    ]


def _distribution(spreads: Sequence[float]) -> Dict[str, Any]:
    if not spreads:
        return {"pages": 0}
    return {
        "pages": len(spreads),
        "median": float(np.percentile(spreads, 50)),
        "p90": float(np.percentile(spreads, 90)),
        "max": float(max(spreads)),
    }


def summarize(pages: Sequence[PageConsistency]) -> Dict[str, Any]:
    """Run-level view: how much of the run carries mixed conventions, and how
    the size spread differs between mixed and single-convention pages."""
    measurable = [p for p in pages if p.area_spread is not None]
    known = [p for p in measurable if p.convention_known]
    mixed = [p for p in known if p.is_mixed]
    single = [p for p in known if not p.is_mixed]
    unknown = [p for p in measurable if not p.convention_known]
    untagged = [p for p in pages if p.untagged_rooms]
    return {
        "pages_scanned": len(pages),
        "pages_with_measurable_spread": len(measurable),
        "pages_with_known_convention": len(known),
        "pages_mixed": len(mixed),
        # Denominator is the pages whose convention is actually known: a run
        # predating the field would otherwise report a 0% mixed share, which
        # reads as "clean" when it means "unmeasured".
        "mixed_share_of_known": len(mixed) / len(known) if known else 0.0,
        "pages_with_untagged_rooms": len(untagged),
        "area_spread": {
            "all": _distribution([p.area_spread for p in measurable]),
            "mixed_convention": _distribution([p.area_spread for p in mixed]),
            "single_convention": _distribution([p.area_spread for p in single]),
            "unknown_convention": _distribution([p.area_spread for p in unknown]),
        },
    }


def evaluate(run_dir: Path, worst: int) -> Dict[str, Any]:
    pages = scan_run(run_dir / "processed_annotations")
    ranked = sorted(
        (p for p in pages if p.area_spread is not None),
        key=lambda p: p.area_spread, reverse=True,
    )
    report: Dict[str, Any] = {"run_dir": str(run_dir), **summarize(pages)}
    report["widest_spread_pages"] = [p.to_dict() for p in ranked[:worst]]
    if report["pages_with_untagged_rooms"]:
        logger.warning(
            "%d page(s) carry rooms with no box_convention: these predate the "
            "field, so their convention mix is unknown, not single",
            report["pages_with_untagged_rooms"],
        )
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Phase 1: per-page box-convention consistency (needs no GT)"
    )
    parser.add_argument("--run", type=Path, required=True, help="Pipeline output dir")
    parser.add_argument(
        "--worst", type=int, default=10,
        help="How many widest-spread pages to list (default: 10)",
    )
    parser.add_argument("--out", type=Path, default=None, help="Also write report JSON here")
    args = parser.parse_args()

    emit_report(evaluate(args.run, args.worst), args.out)
