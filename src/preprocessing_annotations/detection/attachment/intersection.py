"""AV-3: default attachment policy.

Thin wrapper around map_detections_to_rooms (window_detector.py) -- the
function itself is unchanged and stays the module every existing direct
caller (WindowDetector's own Tier 1-3 pipeline, tests) already imports.
This class exists only so the policy is selectable through
AttachmentPolicyFactory alongside alternative policies, with the
default producing byte-identical output to every pre-AV-3 call site.
"""

from typing import Any, Dict, List

from ...config import AttachmentConfig
from ..window_detector import RoomWindowMapping, WindowDetection, map_detections_to_rooms


class IntersectionAttachmentPolicy:
    """Default policy: geometric intersection only, via
    map_detections_to_rooms. Byte-identical to calling that function
    directly with the same strict_type_dispatch value.
    """

    def __init__(self, config: AttachmentConfig):
        self._strict_type_dispatch = config.strict_type_dispatch

    def attach(
        self,
        detections: List[WindowDetection],
        rooms: List[Dict[str, Any]],
    ) -> List[RoomWindowMapping]:
        return map_detections_to_rooms(
            detections, rooms, strict_type_dispatch=self._strict_type_dispatch,
        )
