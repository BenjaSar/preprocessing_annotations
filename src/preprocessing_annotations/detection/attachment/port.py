"""AV-3: attachment-policy port.

Confirmed gap: object-detection -> room attachment was a single
hardcoded function (map_detections_to_rooms, window_detector.py), called
directly at both of pipeline.py's call sites -- no alternative policy
was swappable without editing that function itself (OCP violation).
This is the documented root cause of the OCR-only near-zero-attach
problem: in OCR-only mode room boxes are text-label bboxes, not room
extents, so intersection rarely fires
(pipeline.py::_warn_if_yolo_objects_ocr_only's docstring, confirmed on
a real page: 26 OCR rooms + 44 YOLO detections -> 0 attached).

New policies implement this one-method port and register with
AttachmentPolicyFactory (factory.py) instead of pipeline.py branching
on policy type.
"""

from typing import Any, Dict, List, Protocol

from ..window_detector import RoomWindowMapping, WindowDetection


class AttachmentPolicy(Protocol):
    """Any implementation must be swappable for another without the
    caller changing (LSP): same input/output shape, and the contract
    that every room in ``rooms`` gets exactly one RoomWindowMapping back.
    """

    def attach(
        self,
        detections: List[WindowDetection],
        rooms: List[Dict[str, Any]],
    ) -> List[RoomWindowMapping]:
        """Attach detections to rooms, returning one mapping per room."""
        ...
