#!/usr/bin/env python3
"""
Regression guard for a bbox-format mismatch across the semantic
reconciliation boundary (found auditing sprint1_verify47, 2026-08-19) --
on BOTH sides of the call, found in two passes:

1. RoomAnnotation.bbox is documented [x, y, width, height]
   (vlm_annotator.py). SemanticReconciler.reconcile() documents and
   unpacks its "bbox" input as [x1, y1, x2, y2] (semantic_reconciler.py's
   _match_room_to_ocr path, via `x1, y1, x2, y2 = bbox`).
   _apply_semantic_reconciliation was passing RoomAnnotation's [x,y,w,h]
   straight through unconverted, so the reconciler read a room's WIDTH
   as an absolute X2 and its HEIGHT as an absolute Y2, and the
   reconciled result was assigned back into RoomAnnotation.bbox without
   converting the reconciler's own [x1,y1,x2,y2] output either. Fixed
   inside _apply_semantic_reconciliation (converts both ways).

2. Audited fix #1 and found the SAME shape of bug on the OTHER input:
   RoomCandidate.bbox (OCR side) is also [x,y,w,h] (ocr_extractor.py),
   passed into _apply_semantic_reconciliation's ocr_results as if
   already [x1,y1,x2,y2]. OCRText.centroid() computes (x1+x2)/2,
   (y1+y2)/2 -- with xywh misread as xyxy, a real OCR label's centroid
   moves hundreds of px, off the actual room polygon, so semantic
   matching finds nothing. Real example (sprint1_verify47): SOCIAL
   SERVICE [1574,633,78,14] -> true centroid (1613,640), buggy centroid
   (826,324). Fixed by extracting the conversion into
   AnnotationPipeline._ocr_candidates_to_reconciler_dicts (previously
   inline in run()'s loop, untestable without a full pipeline run).

Both call sites are SemanticReconciler's only callers (verified: grep
for SemanticReconciler( / .reconcile( / .reconcile_with_config( across
src/ returns exactly pipeline.py's _apply_semantic_reconciliation).
SemanticReconciler itself is untouched -- fixed at both boundaries so
its documented xyxy contract stays intact for any future caller.

Confirmed on the real run: every one of 35 semantically-reconciled
rooms on one page came out with a dimension in the thousands-of-pixels
range (page is 4500x3375); the untouched ocr_rooms list on the same
page stayed correctly small (15-220px). No prior test exercised either
path -- why both shipped undetected.
"""


from preprocessing_annotations.config import PipelineConfig
from preprocessing_annotations.ingestion.ocr_extractor import RoomCandidate
from preprocessing_annotations.orchestration.pipeline import AnnotationPipeline
from preprocessing_annotations.vlm.vlm_annotator import RoomAnnotation, VLMAnnotationResult


def _pipeline() -> AnnotationPipeline:
    config = PipelineConfig()
    config.use_semantic_reconciliation = True
    return AnnotationPipeline(config)


def test_reconciled_room_bbox_keeps_original_dimensions():
    """A room whose width/height are far from its x/y (so a coordinate-
    role swap is unmistakable if it happens) must survive reconciliation
    with its dimensions intact, not swapped into a page-spanning box."""
    pipeline = _pipeline()

    # Realistic tile-local-then-offset room box: x=1146, y=826, far from
    # both w=271 and h=121 -- if w/h get misread as absolute x2/y2, the
    # reconciled box's width/height will come out close to 1146-271=875
    # or similar, nowhere near the true 271x121.
    room = RoomAnnotation(
        room_number="101",
        room_name="STORAGE ROOM",
        category="STORAGE ROOM",
        bbox=[1146, 826, 271, 121],
    )
    result = VLMAnnotationResult(
        image_file="test.png",
        image_size={"width": 4500, "height": 3375},
        rooms=[room],
    )

    # OCR text inside the room's real footprint, so _match_room_to_ocr
    # finds a real match and exercises the fixed conversion both ways
    # (in: RoomAnnotation.bbox -> reconciler; out: reconciler -> bbox).
    ocr_results = [
        {"text": "101", "bbox": [1150, 830, 1200, 850], "confidence": 0.95},
    ]

    reconciled = pipeline._apply_semantic_reconciliation(result, ocr_results)

    assert len(reconciled.rooms) == 1
    x, y, w, h = reconciled.rooms[0].bbox
    assert w == 271, f"width corrupted: got {w}, expected 271"
    assert h == 121, f"height corrupted: got {h}, expected 121"
    assert x == 1146 and y == 826, f"position corrupted: got ({x},{y})"


def test_reconciliation_disabled_leaves_bbox_untouched():
    """Baseline: with the feature off, bbox must pass through byte-
    identical -- isolates the assertion above to the reconciliation path
    itself, not some other bbox mutation."""
    config = PipelineConfig()
    config.use_semantic_reconciliation = False
    pipeline = AnnotationPipeline(config)

    room = RoomAnnotation(
        room_number="101", room_name="STORAGE ROOM", category="STORAGE ROOM",
        bbox=[1146, 826, 271, 121],
    )
    result = VLMAnnotationResult(
        image_file="test.png", image_size={"width": 4500, "height": 3375},
        rooms=[room],
    )

    out = pipeline._apply_semantic_reconciliation(result, [{"text": "101", "bbox": [1150, 830, 1200, 850], "confidence": 0.95}])

    assert out.rooms[0].bbox == [1146, 826, 271, 121]


def test_ocr_candidate_conversion_produces_correct_centroid():
    """The extracted converter (previously inline in run()'s loop) must
    turn RoomCandidate's [x,y,w,h] into [x1,y1,x2,y2] whose midpoint is
    the label's TRUE centroid, not a coordinate-role swap. Uses the real
    sprint1_verify47 numbers: SOCIAL SERVICE [1574,633,78,14]."""
    candidate = RoomCandidate(
        bbox=(1574, 633, 78, 14),
        room_number="",
        room_name="SOCIAL SERVICE",
        confidence=0.9,
    )

    dicts = AnnotationPipeline._ocr_candidates_to_reconciler_dicts([candidate])

    assert len(dicts) == 1
    x1, y1, x2, y2 = dicts[0]["bbox"]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    assert (cx, cy) == (1613, 640), f"centroid corrupted: got ({cx},{cy})"


def test_ocr_label_matches_room_polygon_after_fix():
    """End-to-end (at the SemanticReconciler level, where match success
    is actually observable -- RoomAnnotation carries no field that
    reflects it): an OCR label positioned INSIDE a room's real footprint
    must be matched to that room. Before the fix, SOCIAL SERVICE's buggy
    centroid (826,324) fell outside every real room on its page and the
    label matched nothing -- this is why 'SOCIAL SERVICES' was reported
    missing despite OCR having read the text correctly."""
    from preprocessing_annotations.vlm.semantic_reconciler import SemanticReconciler
    from preprocessing_annotations.config import OCRConfig

    pipeline = _pipeline()
    candidate = RoomCandidate(
        bbox=(1574, 633, 78, 14),  # true centroid (1613,640)
        room_number="", room_name="SOCIAL SERVICE", confidence=0.9,
    )
    ocr_dicts = pipeline._ocr_candidates_to_reconciler_dicts([candidate])

    vlm_room_dict = {
        "room_id": "1", "room_name": "OTHER", "room_type": "OTHER",
        # xywh [1500,600,200,100] -> xyxy, covers x[1500,1700] y[600,700],
        # containing the candidate's true centroid (1613,640).
        "bbox": [1500, 600, 1700, 700],
        "confidence": 1.0, "polygon": None,
    }

    reconciled = SemanticReconciler(OCRConfig()).reconcile_with_config(
        ocr_dicts, [vlm_room_dict], OCRConfig()
    )

    assert len(reconciled) == 1
    assert reconciled[0]["text_labels"] == ["SOCIAL SERVICE"], (
        f"OCR label did not match the room polygon -- centroid still "
        f"wrong (got text_labels={reconciled[0]['text_labels']})"
    )
