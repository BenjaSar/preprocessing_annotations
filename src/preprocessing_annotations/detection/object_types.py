"""AV-2 (2026-09-23): object-type -> RoomWindowMapping flag-field registry.

Gap this closes: map_detections_to_rooms's dispatch was a closed if/elif
chain ending in ``else: has_windows = True`` (window_detector.py, pre-AV-2)
-- an unregistered detector class silently miscounted as a window instead
of failing loud or being tracked. That single line is the documented
reason double_door was kept out of YoloObjectDetectorConfig.keep_categories
(config.py comment, "an unhandled category silently miscounts as a
window"). New object classes (fixtures, double_door, wall, ...) now
register here (OCP) instead of editing map_detections_to_rooms.

Default registrations reproduce the pre-AV-2 dispatch chain exactly,
including its catch-all -- flag_for() with strict=False returns
"has_windows" for any unregistered type, byte-identical to the old
`else` branch. strict=True (PipelineConfig.strict_object_type_dispatch)
is opt-in and returns None instead, so the caller can count unregistered
types separately rather than silently mislabeling them.
"""

from typing import Dict, Optional

# RoomWindowMapping boolean flag fields this registry is allowed to target.
# Kept as an explicit tuple (not derived from the dataclass) so a typo in
# register() fails fast instead of silently no-op'ing on a bad field name.
VALID_FLAG_FIELDS = (
    "has_windows",
    "has_skylights",
    "has_openings",
    "has_door",
    "has_toilet",
    "has_bathtub",
    "has_sink",
)


class ObjectTypeRegistry:
    """Maps a detection's ``metadata["type"]`` string to the
    RoomWindowMapping flag field it sets. Extend via register(); the
    dispatch chain this replaces is never edited to add a class.
    """

    _CATCH_ALL_FLAG = "has_windows"

    def __init__(self) -> None:
        self._flag_by_type: Dict[str, str] = {
            "skylight": "has_skylights",
            "side_opening": "has_openings",
            "opening": "has_openings",
            "door": "has_door",
            "toilet": "has_toilet",
            "bathtub": "has_bathtub",
            "sink": "has_sink",
        }

    def register(self, type_name: str, flag_field: str) -> None:
        """Register a new detector class -> RoomWindowMapping flag field.

        Raises ValueError on an unknown flag_field so a typo fails at
        registration time, not silently at mapping time.
        """
        if flag_field not in VALID_FLAG_FIELDS:
            raise ValueError(
                f"Unknown RoomWindowMapping flag field: {flag_field!r}. "
                f"Valid fields: {VALID_FLAG_FIELDS}"
            )
        self._flag_by_type[type_name] = flag_field

    def flag_for(self, type_name: str, strict: bool = False) -> Optional[str]:
        """Resolve type_name to a RoomWindowMapping flag field.

        strict=False (default): unregistered types fall back to
        "has_windows" -- matches the pre-AV-2 dispatch chain's `else`
        branch exactly, so default behavior is unchanged.
        strict=True: unregistered types return None so the caller can
        count them instead of mislabeling them as a window.
        """
        flag = self._flag_by_type.get(type_name)
        if flag is not None:
            return flag
        return None if strict else self._CATCH_ALL_FLAG


# Module-level default instance: shared unless a caller injects its own
# (tests do, to register a fake class without mutating global state).
DEFAULT_OBJECT_TYPE_REGISTRY = ObjectTypeRegistry()
