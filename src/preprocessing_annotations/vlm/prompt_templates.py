"""Centralized VLM prompt templates.

Single source of truth for every prompt string sent to a VLM backend.
Extracted verbatim from vlm_backend.py and vlm_annotator.py (T-A0) so that:

- prompt content lives in one place instead of being duplicated across the
  Claude / Qwen2.5-VL / Unsloth backends,
- the identical window prompt shared by the Claude and Qwen backends is
  defined exactly once (build_window_prompt),
- the maximum-room-count cap is injected from configuration
  (VLMConfig.max_rooms) rather than hardcoded inside the prompt text.

Prompts whose only variable is a plain value use a named token plus
str.replace, because the room/window examples contain literal JSON braces
that an f-string would otherwise try to interpret as replacement fields.
Prompts that were already f-strings in the source (Unsloth room, MEP room)
remain f-strings, with literal braces escaped as ``{{``/``}}`` as before.
"""

_CATEGORIES_TOKEN = "{categories_str}"
_MAX_ROOMS_TOKEN = "{max_rooms}"


# System message shared by every local-backend detection call (room, door,
# window). Kept here — the single source of truth for prompt strings — instead
# of duplicated inline across the backend's message-builders.
_JSON_ARRAY_SYSTEM_PROMPT = (
    "You are a floor plan annotation expert. "
    "Output ONLY valid JSON arrays. "
    "Begin your response with '['. "
    "No greetings, explanations, or text before or after the JSON array. "
    "No markdown."
)


def build_json_array_system_prompt() -> str:
    """System prompt constraining local VLM backends to bare JSON-array output."""
    return _JSON_ARRAY_SYSTEM_PROMPT


_CLAUDE_ROOM_PROMPT = """Analyze this architectural floor plan and identify all rooms and spaces.

For each room or space visible, return a JSON object with:
{
    "room_id": "unique identifier like room_0, room_1",
    "room_type": "{categories_str}",
    "room_name": "extracted room name or label from the plan",
    "bbox": [x1, y1, x2, y2] as percentage of image dimensions (0-100),
    "confidence": 0.0-1.0 confidence in detection,
    "metadata": {}
}

Office Classification Rules:
- For office spaces: classify as PRIVATE OFFICE if visually enclosed with walls/doors,
  or OPEN OFFICE if it's part of an open floor plan with shared/common areas.
- If unclear, assume OPEN OFFICE (more common in modern designs).

General Rules:
- Only include rooms/spaces/areas. Exclude legends, title blocks, schedules, notes, and title sheets.
- Preserve original room labels and numbers from the plan (do NOT expand abbreviations).
- Return valid JSON array only, no markdown or explanation.

Return ONLY a JSON array, e.g.: [{"room_id": "room_0", "room_type": "CONFERENCE", ...}, ...]"""


_QWEN_ROOM_PROMPT = """You are a floor plan annotation expert. Analyze this architectural floor plan image and identify all labeled rooms and spaces that are actually visible in the image.

TASK: For each room/space you can see labeled in the floor plan, extract:
  - room_type: one of PRIVATE OFFICE, OPEN OFFICE, CONFERENCE, LOBBY, CORRIDOR, RESTROOM, STAIRWELL, ELECTRICAL ROOM, STORAGE ROOM, MEETING, or another descriptive type that fits
  - room_name: the exact label text as written on the plan
  - bbox: bounding box as [x1, y1, x2, y2] percentage of image size (0-100)
    where (x1,y1) is top-left corner and (x2,y2) is bottom-right corner.
    ALL values MUST be between 0 and 100.

INCLUDE only: physical rooms and labeled spaces (offices, restrooms, corridors, etc.)
EXCLUDE: legends, title blocks, schedules, notes, equipment labels, panel lists

Office rule: PRIVATE OFFICE = enclosed with walls/doors; OPEN OFFICE = shared open area

Return ONLY a JSON array. Example format (do not copy exact values):
[{"room_id": "room_0", "room_type": "CONFERENCE", "room_name": "Conf Rm A",
  "bbox": [x1, y1, x2, y2], "confidence": 0.9}]

Rules:
- ONLY include rooms that have visible labels in the image
- Do NOT invent or fabricate rooms that are not shown
- All bbox values must be between 0 and 100
- Maximum {max_rooms} rooms
- Return valid JSON only, no markdown"""


_WINDOW_PROMPT = """Analyze this architectural floor plan and identify all windows and openings.

For each window visible, return a JSON array with:
[{
    "bbox": [x1, y1, x2, y2] as percentage of image dimensions (0-100),
    "type": "window" or "skylight" or "side_opening",
    "confidence": 0.0-1.0 confidence in detection
}]

Rules:
- Windows are typically represented as thin lines breaking wall segments
- Skylights are shown as rectangular areas within roof spaces
- Side openings are openings on exterior walls at ground level
- Exclude doors, vents, and other small openings
- Return valid JSON array only

Return ONLY a JSON array: [{"bbox": [10, 20, 30, 40], "type": "window", "confidence": 0.95}, ...]"""


_UNSLOTH_WINDOW_PROMPT = """Analyze this architectural floor plan and identify all windows and openings.

For each window visible, return a JSON array with:
[{
    "bbox": [x1, y1, x2, y2] as percentage of image dimensions (0-100),
    "type": "window" or "skylight" or "side_opening",
    "confidence": 0.0-1.0 confidence in detection
}]

Rules:
- Windows are typically represented as thin lines breaking wall segments
- Skylights are shown as rectangular areas within roof spaces
- Side openings are openings on exterior walls at ground level
- Exclude doors, vents, and other small openings
- Return valid JSON array only

Examples of window symbols in floor plans:
- Parallel thin lines on wall segments
- Double lines at angles (double-hung windows)
- Simple rectangles on wall perimeters
- Repeated grid patterns (curtain walls, glazing)

Return ONLY a JSON array: [{"bbox": [10, 20, 30, 40], "type": "window", "confidence": 0.95}, ...]"""


def build_room_prompt_claude(categories_str: str) -> str:
    """Room-detection prompt for the Claude backend (percentage bbox, 0-100)."""
    return _CLAUDE_ROOM_PROMPT.replace(_CATEGORIES_TOKEN, categories_str)


def build_room_prompt_qwen(max_rooms: int) -> str:
    """Room-detection prompt for the raw Qwen2.5-VL backend (percentage bbox)."""
    return _QWEN_ROOM_PROMPT.replace(_MAX_ROOMS_TOKEN, str(max_rooms))


def build_room_prompt_unsloth(img_width: int, img_height: int, max_rooms: int) -> str:
    """Room-detection prompt for the Unsloth Qwen backend (integer pixel bbox).

    Pixel coordinates match Qwen2.5-VL grounding training; image dimensions
    are injected so the model produces pixel-space bboxes.
    """
    return f"""You are a floor plan annotation expert. The image is {img_width}x{img_height} pixels.

TASK: Identify every labeled room or space visible in this floor plan.

For each room output:
  - room_type: OFFICE, CONFERENCE, CORRIDOR, RESTROOM, LOBBY, KITCHEN, STORAGE, STAIRWELL, ELEVATOR, or OTHER
  - room_name: exact label text from the plan
  - bbox: [x1, y1, x2, y2] in INTEGER PIXELS (top-left origin).
    x values in [0, {img_width}], y values in [0, {img_height}].
  - confidence: 0.0-1.0

INCLUDE: enclosed rooms and labeled spaces that are INSIDE the floor plan boundary lines (walls).
EXCLUDE: legends, title blocks, BOM tables, schedules, notes, panel labels, and any text in the margins or corners of the image.

SPATIAL RULE: The architectural drawing is in the CENTER of the image surrounded by margins.
Do NOT detect anything in margin areas (corners, edges, table blocks) even if they contain room names.
A valid room detection must be inside the floor plan boundary walls, not in a table or text block.

CRITICAL — the bbox must enclose the ENTIRE room: its surrounding walls/boundary,
not just the label text. A room is much larger than its text label. Draw the box
from wall to wall, with the label inside it.

Return ONLY a JSON array, no markdown:
[{{"room_id": "<id>", "room_type": "<type>", "room_name": "<label>", "bbox": [<x1>, <y1>, <x2>, <y2>], "confidence": <0-1>}}]

CRITICAL — bbox values must be the ACTUAL pixel location you observe in the image.
Never reuse any numbers from this prompt text.

Rules:
- Only rooms with visible labels in the image.
- Do NOT fabricate rooms.
- Maximum {max_rooms} rooms.
- Every room must be at a DISTINCT location — no two rooms share the same x1 and y1.
- bbox must have positive width and height (x2 > x1, y2 > y1).
- bbox must span the room's walls, not the label glyphs.
- Return ONLY valid JSON array. No markdown. No code fences. No text outside the JSON array."""


def build_window_prompt() -> str:
    """Window-detection prompt shared by the Claude and Qwen2.5-VL backends."""
    return _WINDOW_PROMPT


def build_window_prompt_unsloth() -> str:
    """Window-detection prompt for the Unsloth Qwen backend (adds symbol examples)."""
    return _UNSLOTH_WINDOW_PROMPT


def build_door_prompt_unsloth(img_width: int, img_height: int) -> str:
    """Build the Unsloth Qwen door-detection prompt (integer pixel bbox).

    Mirrors the room prompt's proven convention: absolute pixel coordinates
    with the image size injected, plus anti-hallucination guards. The earlier
    percentage variant failed because Qwen emits pixel coordinates regardless,
    which a percentage parser then mis-scaled off-image.

    Args:
        img_width: Width in pixels of the image sent to the model.
        img_height: Height in pixels of the image sent to the model.

    Returns:
        The formatted door-detection prompt.
    """
    return f"""You are a floor plan annotation expert. The image is {img_width}x{img_height} pixels.

TASK: Identify every door and door opening visible in this floor plan.

For each door output:
  - bbox: [x1, y1, x2, y2] in INTEGER PIXELS (top-left origin).
    x values in [0, {img_width}], y values in [0, {img_height}].
  - type: "swing", "sliding", or "opening"
  - confidence: 0.0-1.0

A door is a gap in a wall shown as a quarter-circle swing arc (hinged),
two offset parallel panels (sliding), or a plain gap between wall ends (opening).
EXCLUDE windows, fixtures, and any symbol that is not a door.

Return ONLY a JSON array, no markdown:
[{{"bbox": [<x1>, <y1>, <x2>, <y2>], "type": "<type>", "confidence": <0-1>}}]

CRITICAL — bbox values must be the ACTUAL pixel location you observe in the image.
Never reuse any numbers from this prompt text.

Rules:
- Do NOT fabricate doors that are not visible.
- Every door must be at a DISTINCT location — no two doors share the same x1 and y1.
- bbox must have positive width and height (x2 > x1, y2 > y1)."""


def build_room_prompt_mep(categories: str, img_width: int, img_height: int) -> str:
    """Room-detection prompt for the MEP/electrical annotator (fractional xywh bbox)."""
    return f"""You are a floor plan annotation expert. Analyze this MEP/Electrical floor plan image and extract every labeled physical room or functional space.

IMAGE DIMENSIONS: {img_width} x {img_height} pixels

INCLUDE — physical rooms and functional spaces only:
  Offices, conference rooms, restrooms, kitchens, break rooms, lobbies, hallways, corridors,
  mechanical rooms, electrical rooms, storage rooms, server rooms, stairwells, elevator lobbies,
  auditoriums, classrooms, labs, bedrooms, living rooms, compactor rooms, bicycle storage,
  pump rooms, janitor closets, telecom rooms, community facilities.

EXCLUDE — do not output any of these:
  - Electrical panels, switchboards, transformers, circuit breakers (these are equipment, not rooms)
  - Text notes, general notes, symbol lists, legends, disclaimers
  - Compliance statements, code requirements, energy codes
  - Title blocks, revision clouds, approval stamps
  - Schedule tables (door schedules, fixture schedules, panel schedules)
  - Any text that is not labeling a physical space

For each room, report:
  room_number: the room number if visible (e.g. "113"), else ""
  room_name:   the room label as written on the plan (e.g. "MECHANICAL ROOM")
  category:    one of: {categories}
  bbox:        fractional coordinates [x/W, y/H, w/W, h/H] where W={img_width}, H={img_height}
               All values must be in [0.0, 1.0]. (x,y) is the top-left corner.

Output ONLY valid JSON with this exact structure:
{{
  "rooms": [
    {{"room_number": "113", "room_name": "MECHANICAL ROOM", "category": "mechanical", "bbox": [0.42, 0.18, 0.12, 0.08]}}
  ]
}}"""


# Prompt versioning (observability, G-04): a stable hash per builder function,
# computed once at import time, so a run can be attributed to the prompt text
# that produced it. Static-template builders hash their constant string;
# true f-string builders (runtime image dims) hash their function *source*
# instead, since a resolved per-call string would vary by image size and
# defeat the point of a stable version tag.
import hashlib as _hashlib
import inspect as _inspect


def _hash(text: str) -> str:
    return _hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


PROMPT_VERSIONS = {
    "build_json_array_system_prompt": _hash(_JSON_ARRAY_SYSTEM_PROMPT),
    "build_room_prompt_claude": _hash(_CLAUDE_ROOM_PROMPT),
    "build_room_prompt_qwen": _hash(_QWEN_ROOM_PROMPT),
    "build_window_prompt": _hash(_WINDOW_PROMPT),
    "build_window_prompt_unsloth": _hash(_UNSLOTH_WINDOW_PROMPT),
    "build_room_prompt_unsloth": _hash(_inspect.getsource(build_room_prompt_unsloth)),
    "build_door_prompt_unsloth": _hash(_inspect.getsource(build_door_prompt_unsloth)),
    "build_room_prompt_mep": _hash(_inspect.getsource(build_room_prompt_mep)),
}
