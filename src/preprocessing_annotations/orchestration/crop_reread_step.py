"""C2 (2026-10-15): crop re-read as a pipeline step.

Rebuilds one page's Pass-1 room candidates from OCR detections whose text a
VLM has looked at a second time. The gap it closes (measured,
eval/missed_room_ocr_sprint1_verify53.json): 158 GT rooms never get a
candidate although OCR read text inside them, some of it truncated or
misread room words ("TAIR", "CLASSRODM").

Lives in orchestration because the pieces it needs sit in layers that do not
import each other -- the room vocabulary is automation's, the extractor and
re-reader are ingestion's, the model is vlm's. Composing them here keeps all
three decoupled, which is the same split eval/candidate_replay.py uses.

Runs in the assembly step, not during Pass 1: crop re-read needs the VLM, and
Pass 1 deliberately runs with no VLM resident (pipeline.py releases the
PaddleOCR heap first -- the two engines together exceed host RAM).

Crops come from the page scaled to MEPTextExtractor.ocr_scale_size, because
detection coordinates are in OCR's space and the page on disk has not been
resized yet at this point in the run.
"""

import logging
from typing import List, Optional, Sequence, Tuple

from PIL import Image

from ..automation.sft_validator import SemanticRoomValidator
from ..config import CropRereadConfig
from ..ingestion.crop_reread import (
    CropRereader,
    Reread,
    RereadSelector,
    apply_rereads,
)
from ..ingestion.ocr_extractor import MEPTextExtractor, RoomCandidate, TextDetection
from ..vlm.read import CropTextReader

logger = logging.getLogger(__name__)


def reread_selector(
    extractor: MEPTextExtractor, config: CropRereadConfig,
) -> RereadSelector:
    """Room labels per `extractor`'s patterns; room keywords = the semantic
    filter's vocabulary, so a fragment is judged against the same words the
    filter later accepts."""
    return RereadSelector(
        extractor.matches_room_label, SemanticRoomValidator.VALID_ROOM_KEYWORDS, config
    )


def _page_in_ocr_space(image_path, size_of_record: Optional[Tuple[int, int]] = None):
    """The page as a RGB image in the coordinate space OCR read it in."""
    image = Image.open(image_path).convert("RGB")
    target = MEPTextExtractor.ocr_scale_size(size_of_record or image.size)
    if target is None:
        return image
    resized = image.resize(target, Image.Resampling.LANCZOS)
    image.close()
    return resized


class CropRereadStep:
    """Re-reads a page's OCR fragments and rebuilds its room candidates.

    One reason to change: how a page's candidates are derived from re-read
    detections. Owns no model -- the reader is supplied, as with every other
    adapter in this codebase.
    """

    def __init__(
        self, extractor: MEPTextExtractor, reader: CropTextReader,
        config: CropRereadConfig,
    ):
        self._extractor = extractor
        self._config = config
        self._rereader = CropRereader(
            reread_selector(extractor, config), reader, config
        )

    def rebuild_candidates(
        self, image_path, detections: Sequence[TextDetection],
    ) -> Tuple[List[RoomCandidate], List[Reread]]:
        """(candidates, raw re-reads) for one page.

        The re-reads are returned, not the refined detections, because they
        are what a later offline sweep needs: every fragment the VLM looked
        at and what it said, before any consistency guard. Persisted, they
        let eval/candidate_replay.py re-pick the selector bounds on this run
        without a second GPU pass.

        `detections` is never mutated: apply_rereads copies. The caller's OCR
        detections also feed compute_exclusion_zones, which is documented
        unsafe with non-OCR input.
        """
        if not detections:
            return [], []
        page = _page_in_ocr_space(image_path)
        try:
            rereads = self._rereader.reread(page, detections)
        finally:
            page.close()
        refined = apply_rereads(detections, rereads, self._config.consistency_min)
        changed = sum(1 for a, b in zip(detections, refined) if a.text != b.text)
        logger.info(
            "  crop re-read: %d fragment(s) re-read, %d text change(s) accepted",
            len(rereads), changed,
        )
        return self._extractor.find_room_candidates(refined), rereads
