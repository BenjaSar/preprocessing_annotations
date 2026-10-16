"""Crop re-read: a VLM second look at OCR text fragments.

Pass-1 OCR reads some room labels only partly ("TAIR" for "STAIR",
"CLASSRODM" for "CLASSROOM"). Such text fails every room-label pattern and
never becomes a room candidate. Pass 2 (two_pass_ocr_extractor) does not
reach it: Pass 2 fires only for detections that already are candidates and
re-runs room detection on the whole page.

Here a selector picks the fragments, each is cropped from the page image
and transcribed by a CropTextReader, and the OCR text is swapped for the
transcription only when the two are consistent. Boxes never change. The
caller rebuilds candidates from the returned detections with
MEPTextExtractor.find_room_candidates.
"""

import dataclasses
import re
from difflib import SequenceMatcher
from typing import Any, Callable, Iterable, List, Sequence, Tuple

from ..config import CropRereadConfig
from ..vlm.read import CropTextReader
from .ocr_extractor import TextDetection

PixelBox = Tuple[int, int, int, int]


@dataclasses.dataclass(frozen=True)
class Reread:
    """One transcription; `index` is the detection's position in the page's
    OCR detection list, `original` its OCR text."""

    index: int
    original: str
    reread: str


def _words(text: str) -> str:
    return re.sub(r"[^A-Z ]", "", text.upper()).strip()


def _alphanumeric(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


class RereadSelector:
    """Picks detections whose text is not a room label on its own but is a
    room keyword's substring or close to one."""

    def __init__(
        self,
        is_room_label: Callable[[str], bool],
        keywords: Iterable[str],
        config: CropRereadConfig,
    ):
        self._is_room_label = is_room_label
        self._keywords = sorted(keywords)
        self._config = config

    def selects(self, text: str) -> bool:
        words = _words(text)
        if len(words.replace(" ", "")) < self._config.min_text_letters:
            return False
        if self._is_room_label(text.strip()):
            return False
        return any(
            words in keyword or _similarity(words, keyword) >= self._config.keyword_similarity_min
            for keyword in self._keywords
        )

    def select(self, detections: Sequence[TextDetection]) -> List[int]:
        return [i for i, d in enumerate(detections) if self.selects(d.text)]


def crop_box(
    quad: Sequence[Sequence[float]], image_size: Tuple[int, int], padding_box_heights: float,
) -> PixelBox:
    """(left, top, right, bottom) around a detection quad, padded by
    `padding_box_heights` x its height on every side, clamped to the image."""
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    pad = (max(ys) - min(ys)) * padding_box_heights
    width, height = image_size
    return (
        max(0, int(min(xs) - pad)),
        max(0, int(min(ys) - pad)),
        min(width, int(max(xs) + pad)),
        min(height, int(max(ys) + pad)),
    )


def is_consistent(original: str, reread: str, consistency_min: float) -> bool:
    """The re-read may extend or correct the OCR text, not replace it with
    something unrelated."""
    before, after = _alphanumeric(original), _alphanumeric(reread)
    if not after:
        return False
    return before in after or _similarity(before, after) >= consistency_min


class CropRereader:
    """Selects fragments on one page and transcribes each crop."""

    def __init__(self, selector: RereadSelector, reader: CropTextReader, config: CropRereadConfig):
        self._selector = selector
        self._reader = reader
        self._config = config

    def reread(self, image: Any, detections: Sequence[TextDetection]) -> List[Reread]:
        """`image`: the RGB PIL page image the detections were read from."""
        rereads = []
        for index in self._selector.select(detections):
            detection = detections[index]
            box = crop_box(detection.bbox, image.size, self._config.crop_padding_box_heights)
            rereads.append(Reread(index, detection.text, self._reader.read(image.crop(box))))
        return rereads


def apply_rereads(
    detections: Sequence[TextDetection], rereads: Iterable[Reread], consistency_min: float,
) -> List[TextDetection]:
    """Copy of `detections` with each consistent re-read's text swapped in."""
    accepted = {}
    for r in rereads:
        if detections[r.index].text != r.original:
            raise ValueError(
                f"re-read {r.index} was made from {r.original!r}, detection reads "
                f"{detections[r.index].text!r} -- re-reads belong to another OCR pass"
            )
        if is_consistent(r.original, r.reread, consistency_min):
            accepted[r.index] = r.reread
    return [
        dataclasses.replace(d, text=accepted[i]) if i in accepted else d
        for i, d in enumerate(detections)
    ]
