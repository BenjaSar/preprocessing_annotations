"""
Configuration classes for the annotation pipeline.

All configurable parameters are centralized here to avoid magic numbers
and enable easy tuning for different floor plan types.

Environment variables:
  ANTHROPIC_API_KEY - Required: Anthropic API key
  VLM_MODEL - Optional: Model to use (default: claude-haiku-4-5-20251001)
  VLM_MAX_TOKENS - Optional: Max response tokens (default: 4096)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None  # Optional dependency

# Project root (the VLM/ dir containing preprocessing_annotations/, test_pcs/,
# yolo_checkpoints/, etc.). This file lives at
# preprocessing_annotations/src/preprocessing_annotations/config/config.py,
# 5 levels below VLM/ -- parents[4], not the 2-level parent.parent that was
# correct when config.py lived directly at the repo root.
_PROJECT_ROOT = Path(__file__).resolve().parents[4]

# SAM checkpoints live in preprocessing_annotations/weights/, a sibling of
# src/ -- not under detection/, which is where SAMConfig.checkpoint's old
# bare-filename default resolved to (sam_segmenter.py's relative-path
# fallback anchors on Path(__file__).parent, i.e. the module's own dir).
SAM_WEIGHTS_ROOT = _PROJECT_ROOT / "preprocessing_annotations" / "weights"


def _get_vlm_model() -> str:
    """Get VLM model from environment or use default Claude Haiku.
    
    Note: This default only applies to the 'claude' backend.
    For 'qwen' and 'unsloth' backends, use qwen_model and unsloth_model fields.
    See VLMConfig.active_model for the resolved model based on the selected backend.
    """
    return os.getenv("VLM_MODEL", "claude-haiku-4-5-20251001")


def _get_vlm_temperature() -> float:
    return float(os.getenv("VLM_TEMPERATURE", "1.0"))

def _get_vlm_top_p() -> float:
    return float(os.getenv("VLM_TOP_P", "0.95"))

def _get_vlm_top_k() -> int:
    return int(os.getenv("VLM_TOP_K", "20"))

def _get_vlm_presence_penalty() -> float:
    return float(os.getenv("VLM_PRESENCE_PENALTY", "0.0"))

def _get_vlm_repetition_penalty() -> float:
    return float(os.getenv("VLM_REPETITION_PENALTY", "1.0"))

def _get_vlm_seed() -> int:
    return int(os.getenv("VLM_SEED", "1234"))

def _get_vlm_do_sample() -> bool:
    val = os.getenv("VLM_DO_SAMPLE", "true")
    return val.lower() in ("true", "1", "yes")


def _detect_device() -> str:
    """Detect available compute device with graceful fallback."""
    if torch is None:
        return "cpu"  # Torch not available, default to CPU
    try:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@dataclass
class PDFConfig:
    """Configuration for PDF extraction."""

    # DPI for rasterization. Higher values capture more detail but increase file size.
    # - 150 DPI: Fast processing, suitable for large room labels
    # - 200 DPI: Balanced (default), good for most MEP plans
    # - 300 DPI: High detail, recommended for small electrical symbols
    dpi: int = 200

    # Minimum longest-edge pixel count per rendered page.
    #
    # A flat `dpi` renders every page at the same pixels-per-inch, so a small
    # physical sheet (e.g. 8.5x11" Letter) that packs an entire multi-unit
    # building into a tiny area produces glyphs too small for OCR/VLM to read,
    # while a large sheet (e.g. ARCH-D) of the same building reads fine at the
    # same dpi. This floor guarantees each page is rasterized to at least this
    # many pixels on its longest edge by raising dpi per-page when needed; it
    # NEVER lowers dpi, so pages that already exceed it (large sheets) are
    # rendered exactly as before.
    #
    # Default matches the established downstream working resolution used across
    # the pipeline — OCR (`MEPTextExtractor._OCR_MAX_DIM_PX`), the VLM pre-resize
    # target, and the bbox validators all operate at 4500px longest edge — so
    # rendering below it starves those stages and rendering above it is discarded
    # by their own down-resize.
    min_longest_edge_px: int = 4500

    # Output image format
    output_format: str = "PNG"

    # Whether to extract all pages or specific page ranges
    page_range: Optional[tuple] = None  # None = all pages


@dataclass
class OCRConfig:
    """Configuration for OCR text extraction."""

    # OCR backend selection: 'paddleocr' (default) or 'easyocr' (legacy)
    # 'paddleocr' is the default for better accuracy and speed
    # 'easyocr' is available as a legacy option via --ocr-backend easyocr
    backend: str = "paddleocr"

    # Languages for OCR
    languages: List[str] = field(default_factory=lambda: ["en"])

    # Device for OCR inference (auto-detected if None)
    device: Optional[str] = None

    # Minimum confidence threshold for OCR results
    confidence_threshold: float = 0.5

    # PaddleOCR text-detection input size limit (longest side).
    # PaddleOCR's default (960) downsamples large floor-plan pages ~10x, shrinking
    # small in-plan room labels below the detector's minimum text size — they are
    # never detected (only large title-block text survives). 4608 keeps labels
    # legible on both 4500px resized and 9600px original extractions (labels scale
    # with resolution). Measured: default(960) → 0 unit labels; 4608 → 69.
    det_limit_side_len: int = 4608

    # OCR tiling: longest-edge (px) above which a page is split into overlapping
    # tiles for detection, then detections are remapped to full-image coordinates.
    # PaddleOCR-CPU memory scales with input area; a full 4500px page peaks ~14.7GB
    # (measured), too close to a 16GB host, while a ~1200px tile peaks ~5.3GB
    # (measured). Tiling bounds peak memory independently of the VLM backend, at
    # full detection resolution per tile (no recall loss from downscaling).
    # 0 disables tiling (single-pass detection). Default 1200 mirrors the VLM
    # tile target and the ~5.3GB/tile measurement, giving comfortable headroom on
    # a 16GB host regardless of VLM backend.
    ocr_tile_max_px: int = 1200
    # Fractional overlap between adjacent tiles so labels on a tile seam are still
    # wholly visible in at least one tile.
    ocr_tile_overlap_pct: float = 0.10

    # Whether to apply image preprocessing before OCR
    preprocess: bool = True

    # CLAHE parameters for contrast enhancement
    clahe_clip_limit: float = 2.0
    clahe_grid_size: tuple = (8, 8)

    # Room number pattern (regex)
    room_number_pattern: str = r"^(\d{3}[A-Z]?|\d{2,3})$"

    # Room name patterns to detect
    # Room name patterns to detect.
    # CRITICAL: All patterns must be anchored (^...$) so they match the
    # complete OCR token, not a substring. Un-anchored patterns like
    # r"ELECTRICAL\s*(ROOM)?" would match "ELECTRICAL GENERAL NOTES" and
    # cause documentation text to leak into room candidates.
    room_name_patterns: List[str] = field(
        default_factory=lambda: [
            # --- Commercial / office spaces ---
            r"^(SUITE|OFFICE|CONFERENCE(\s+ROOM)?|MEETING(\s+ROOM)?)\s*\d*$",
            # --- Utility / MEP rooms (require ROOM suffix or exact standalone) ---
            r"^MECHANICAL(\s+ROOM)?$",
            r"^ELECTRICAL(\s+ROOM)?$",
            r"^MECHANICAL/ELECTRICAL(\s+ROOM)?$",
            r"^SERVER(\s+ROOM)?$",
            r"^BREAK(\s+ROOM)?$",
            r"^MACHINE(\s+ROOM)?$",
            r"^BOILER(\s+ROOM)?$",
            r"^PUMP(\s+ROOM)?$",
            r"^COMPACTOR(\s+ROOM)?$",
            r"^FIRE\s+PUMP(\s+ROOM)?$",
            r"^ELEVATOR(\s+MACHINE(\s+ROOM)?)?$",
            r"^STAIR(WELL|CASE|S)?$",
            r"^CUSTODIAL(\s+CLOSET)?$",
            r"^JANITOR(\s+ROOM)?$",
            # --- Sanitary / amenities ---
            r"^(MEN|WOMEN)('?S)?\s*(RESTROOM|BATHROOM|LOCKER)?$",
            r"^RESTROOM$",
            r"^BATHROOM$",
            # --- Circulation / entry ---
            r"^(ENTRANCE|ENTRY|LOBBY|FOYER|VESTIBULE|RECEPTION)(\s+(AREA|HALL))?$",
            r"^(CORRIDOR|HALLWAY|PASSAGE)$",
            # --- Storage variants ---
            r"^RISER(\s+ROOM)?$",
            r"^STORAGE(\s+ROOM)?$",
            r"^(BUILDING|BICYCLE|COMMERCIAL|GENERAL)\s+STORAGE$",
            r"^(BICYCLE\s+STORAGE|BICYCLE)$",
            # --- Telecom / data ---
            r"^(IDF|MDF|TELECOM)(\s+ROOM)?$",
            r"^CARPENTRY(\s+SHOP)?$",
            r"^COMMUNITY\s+FACILITY$",
            r"^KITCHEN$",
            # --- School spaces ---
            # School sheets (Kennedy, Masonic, Lake Shore) are dominated by
            # CLASSROOM / TOILET / CLOSET labels, which were absent here and so
            # discarded at the OCR gate before reaching the taxonomy — leaving
            # whole school floors with zero detections. Formats verified via OCR:
            # "CLASSROOM5", "CLASSROOM 6A", "CLASSROOM - 117",
            # "SCIENCE CLASSROOM - 116", bare "TOILET", bare "CLOSET".
            r"^(SCIENCE\s+|STUDIO\s+)?CLASSROOM[\s\-]*\d*[A-Z]?$",
            r"^(BOYS|GIRLS|STAFF|STUDENT)?\s*TOILET$",
            r"^CLOSET$",
            # Verified via live OCR (Masonic Heights): "JANITOR CLOSET",
            # "STAFF LOUNGE 19A", "SUPPLY". TEACHER LOUNGE/WORK ROOM and
            # PRINCIPAL per explicit taxonomy directive (not yet OCR-traced).
            r"^JANITOR\s+CLOSET$",
            r"^(STAFF|TEACHER)\s+(LOUNGE|WORK\s+ROOM)(\s*\d*[A-Z]?)?$",
            r"^PRINCIPAL(\s+OFFICE)?$",
            r"^SUPPLY$",
            r"^COMPUTER\s+LAB(\s*\d*)?$",
            # verify41 OCR-confirmed, user-flagged school/amenity labels that the
            # OCR gate dropped (present in taxonomy, but never reached it). Anchored
            # to the whole token to avoid matching documentation/instruction text.
            r"^MEDIA\s+CENTER$",
            r"^COUNTER\s+LAB$",
            r"^CAFETERIA$",
            r"^SPECIAL\s+EDUCATION$",
            r"^GYM\s+OFFICE$",
            r"^(BOYS|GIRLS)\s+COACH$",
            r"^(BOYS|GIRLS)\s+LOCKERS?$",
            r"^PREP(\s+[A-Z]?\d+)?$",
            r"^(ELEC|MECH)\s+ROOM$",
            r"^(MEN|WOMEN)('?S)?\s+TOILET$",
            # Verified real omitted room labels (user-confirmed present on
            # school floor plans; each currently discarded at the OCR gate).
            r"^(TEACHER\s+)?WORK\s+(RM|ROOM)$",
            r"^MAIL\s+(RM|ROOM)$",
            r"^JAN\s+CLOSET$",
            r"^LOUNGE$",
            r"^COUNC(?:IL(?:ING)?)?\s+OFFICE$",
            r"^TEACHER\s+COUNCELING$",
            r"^SECRETARY$",
            r"^SOCIAL\s+WORKER\s+OFFICE$",
            r"^MAIN\s+ENTRANCE\s+LOBBY$",
            r"^OPEN\s+OFFICE$",
            r"^FOOD\s+SERVICE\s+OFFICE$",
            r"^VENDING$",
            r"^SCHOOL\s+STORAGE$",
            r"^VAULT$",
            r"^GYMNASIUM$",
            r"^FITNESS\s+ROOM$",
            r"^LEASING\s+OFFICE$",
            r"^YOGA/MEDITATION\s+ROOM$",
            # User-confirmed room labels the OCR gate discarded before they could
            # reach the taxonomy. Anchored to the whole token, matching the
            # convention above, so documentation/instruction text cannot match.
            # Optional ROOM suffixes cover the bare forms present in the reported
            # detections ("SOCIAL SERVICE", "RECREATION", "WAITING"). The
            # optional-apostrophe group follows the existing (MEN|WOMEN)('?S)?
            # convention for possessive labels.
            r"^SOCIAL\s+SERVICES?$",
            r"^SOCIAL\s+OFFICE$",
            r"^STUDENT\s+SERVICES$",
            r"^RECREATION(\s+ROOM)?$",
            r"^CHILDREN('?S)?\s+PLAY\s+ROOM$",
            r"^COMMUNITY\s+ROOM$",
            r"^WAITING(\s+(ROOM|AREA))?$",
            r"^YOGA(\s+ROOM)?$",
            # Generic numbered room. Anchored digits only — a bare "ROOM" token
            # is not a room label and stays rejected.
            r"^ROOM\s*\d+$",
            # --- Residential unit-type codes (multi-family / mixed-use plans) ---
            # Matches architectural unit-type codes like "TYPE-A1 OBR",
            # "TYPE-B3 1BR", "TYPE-C6.1", "TYPE-A15.2" as a complete token
            # or compound label.  The [O0] group handles the common PaddleOCR
            # confusion between digit-zero and letter-O in "0BR" / "OBR".
            r"^TYPE-[A-Z]\d+(\.\d+)?(\s+[O0-4]BR)?$",
            # --- Residential abbreviations (full-token only) ---
            r"^(BR|BD|BDRM|MBR|MSTR)\s*\d?$",    # bedroom
            r"^([O0]BR|1BR|2BR|3BR|4BR)$",         # unit type (O/0 tolerant)
            r"^(LR|LV)$",                           # living room
            r"^(DR|DIN)$",                          # dining room
            r"^(KIT|K)$",                           # kitchen
            r"^(BA|BATH|MB|PB|PDR)$",              # bathroom
            r"^WIC$",                                # walk-in closet
            r"^(STOR|UTIL|GAR)$",                   # storage/utility/garage
            # --- Abbreviations present in ABBREVIATION_MAP but previously ---
            # --- missing here, causing them to be dropped at the OCR gate ---
            # Without these patterns, find_room_candidates() discards the token
            # before it ever reaches AbbreviationOCRRecovery or SemanticRoomValidator.
            r"^(FR|FAM)$",                           # family room
            r"^(OF|OFC|OFF)$",                       # office
            r"^(CL|CLS)$",                           # closet
            r"^PDR$",                                 # powder room
            r"^(CONF|STE|RECP|WC|TLT|RM)$",         # commercial
            r"^(MECH|BSMT|UTL|LNDRY)$",             # building services
        ]
    )

    def __post_init__(self):
        if self.device is None:
            self.device = _detect_device()


@dataclass
class TemplateConfig:
    """Configuration for template-based symbol detection."""

    # Matching threshold (0.0 to 1.0). Higher = stricter matching.
    threshold: float = 0.7

    # Scale factors to try for multi-scale matching
    scales: List[float] = field(default_factory=lambda: [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])

    # Rotation angles to try (degrees) for rotation-invariant matching
    rotations: List[int] = field(default_factory=lambda: [0, 90, 180, 270])

    # Non-Maximum Suppression IoU threshold
    nms_iou_threshold: float = 0.3

    # Template matching method
    # Options: cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF_NORMED
    match_method: int = 5  # cv2.TM_CCOEFF_NORMED


@dataclass
class VLMConfig:
    """Configuration for VLM (Vision Language Model) annotation.

    Supports three backends:
    1. Claude (Anthropic API) - Requires ANTHROPIC_API_KEY
    2. Qwen2.5-VL (Local inference) - Requires transformers + qwen_vl_utils
    3. Unsloth (Optimized local inference) - ~2x faster, ~70% less VRAM (requires unsloth)

    Backend can be selected via backend field (default: 'claude').
    Model can be set via VLM_MODEL environment variable.

    Claude models:
       - claude-haiku-4-5-20251001 (default) - Fast, cheap, good for room detection
       - claude-sonnet-4-5-20250929 - Balanced performance and cost
       - claude-opus-4-6 - Most capable (expensive)

    Qwen2.5-VL models:
       - Qwen/Qwen2.5-VL-7B-Instruct (default, local, requires ~6GB VRAM at 4-bit)

    Unsloth models (use backend='unsloth', configure via unsloth_model):
       - "qwen2.5-vl-7b" (default) - Optimized Qwen2.5-VL-7B-Instruct
       - "qwen3-vl-2b", "qwen3-vl-4b", "qwen3-vl-8b" - Qwen3-VL variants
    """

    # VLM backend: 'claude' (API), 'qwen' (local), or 'unsloth' (optimized local)
    backend: str = "claude"

    # Model to use for annotation (reads from VLM_MODEL env var)
    # For Claude: claude-haiku-4-5-20251001, claude-sonnet-4-5-20250929, claude-opus-4-6
    # For Qwen: qwen/Qwen2.5-VL-7B (or other Qwen2.5-VL variants)
    model: str = field(default_factory=_get_vlm_model)

    # Maximum tokens for response (reads from VLM_MAX_TOKENS env var).
    # 8192 is the new default: dense floor plans with 20+ rooms routinely
    # truncated at 4096 tokens, causing JSON parse failures and room loss.
    max_tokens: int = field(default_factory=lambda: int(os.getenv("VLM_MAX_TOKENS", "8192")))

    # Max chars of extracted JSON shown in the "Extracted JSON: ..." debug
    # log (vlm_backend.py). Was hardcoded to 200 -- too short to see where
    # a real JSON parse failure occurred (observed failures at char
    # ~2100-2170, e.g. "Expecting ',' delimiter: line 104 column 33").
    # Bounded (not unbounded) to avoid flooding logs on pathological
    # responses; override via VLM_DEBUG_LOG_MAX_CHARS if 4000 isn't enough.
    debug_log_max_chars: int = field(
        default_factory=lambda: int(os.getenv("VLM_DEBUG_LOG_MAX_CHARS", "4000"))
    )

    # Maximum rooms accepted from one VLM room-detection response. Injected
    # into the room prompts ("Maximum N rooms") and enforced by the Claude
    # parser cap, so the prompt instruction and the parser limit stay in sync
    # instead of both hardcoding 25.
    max_rooms: int = 25

    # Number of retries for API calls (Claude only)
    max_retries: int = 3

    # Delay between retries (seconds) (Claude only)
    retry_delay: float = 1.0

    # Quantization for Qwen2.5-VL (8-bit or 4-bit) - reduces VRAM
    qwen_quantization: str = "4bit"  # "8bit" or "4bit" or "none"

    # Device for Qwen2.5-VL inference (auto-detected if None)
    qwen_device: Optional[str] = None

    # HuggingFace model ID for Qwen2.5-VL (only used when backend='qwen')
    # Override via --qwen-model CLI flag or set directly
    qwen_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"

    # Unsloth model key (only used when backend='unsloth')
    # Options:
    #   Instruct: "qwen2.5-vl-7b", "qwen3-vl-2b", "qwen3-vl-4b", "qwen3-vl-8b"
    #   Thinking: "qwen3-vl-2b-thinking", "qwen3-vl-4b-thinking", "qwen3-vl-8b-thinking"
    # Thinking variants add reasoning tokens before structured output.
    # Default is the Instruct 8B variant: this pipeline needs reproducible,
    # directly-parseable JSON (no <think> preamble) and lower latency, which
    # the Instruct variant is documented for. Thinking variants stay opt-in
    # for spatial-reasoning experiments.
    # Override via --unsloth-model CLI flag.
    unsloth_model: str = "qwen3-vl-8b"

    # Generation parameters for local VLM backends (qwen, unsloth).
    # Defaults match Qwen3-VL Thinking recommended sampling values. With the
    # Instruct default model, set the Instruct-recommended values via VLM_*
    # env vars (VLM_TEMPERATURE, VLM_TOP_P, ...); all params are overridable.
    temperature: float = field(default_factory=_get_vlm_temperature)
    top_p: float = field(default_factory=_get_vlm_top_p)
    top_k: int = field(default_factory=_get_vlm_top_k)
    presence_penalty: float = field(
        default_factory=_get_vlm_presence_penalty
    )
    repetition_penalty: float = field(
        default_factory=_get_vlm_repetition_penalty
    )
    seed: int = field(default_factory=_get_vlm_seed)
    do_sample: bool = field(default_factory=_get_vlm_do_sample)

    # Token budgets per task type for local VLM backends.
    # Room budget is tight (768 tokens = ~25 rooms at 30 tokens/room)
    # to truncate autoregressive hallucination loops early.
    # Doors/windows need more because instances are numerous.
    room_max_new_tokens: int = 1024
    door_max_new_tokens: int = 4096
    window_max_new_tokens: int = 4096

    # Room token budget for Thinking-variant Unsloth checkpoints only.
    # Verified 2026-07-24 (t7_stochastic run, qwen3-vl-8b-thinking): the
    # reasoning preamble alone exhausted the 1024-token default with zero
    # JSON emitted (4m17s generation, no '[' in output). Reuses the same
    # 4096 budget already proven for door/window (more generation headroom,
    # not a new magic number) instead of raising the shared default and
    # slowing every Instruct call, which does not reason first.
    room_max_new_tokens_thinking: int = 4096

    # Longest-edge cap (pixels) for images sent to Unsloth room/door detection.
    # Shared by both tiers so a single value governs Unsloth's resize-for-memory
    # step; 4096 matches the prior hardcoded cap (8B model, ~8GB headroom on
    # Tesla T4 — no behavior change from making this configurable).
    unsloth_detection_max_dim_px: int = 4096

    def __post_init__(self):
        if self.qwen_device is None:
            self.qwen_device = _detect_device()

    @property
    def active_model(self) -> str:
        """Return the model ID actually used by the selected backend.
        
        - claude: uses self.model (e.g. claude-haiku-4-5-20251001)
        - qwen:   uses self.qwen_model (e.g. Qwen/Qwen2.5-VL-7B-Instruct)
        - unsloth: uses self.unsloth_model (e.g. qwen3-vl-2b)
        """
        if self.backend == "qwen":
            return self.qwen_model
        elif self.backend == "unsloth":
            return self.unsloth_model
        return self.model

    # Room categories for classification (CV-focused, standardized taxonomy)
    room_categories: List[str] = field(
        default_factory=lambda: [
            "office",
            "open_plan_workspace",
            "executive_office",
            "cubicle_workstation",
            "conference_room",
            "meeting_room",
            "training_room",
            "breakout_space",
            "lobby_reception",
            "hallway_corridor",
            "restroom",
            "kitchen_break_room",
            "storage",
            "mechanical_room",
            "elevator",
            "stairwell",
            "auditorium",
            "data_center",
            "server_room",
            "other",
        ]
    )


@dataclass
class SAMConfig:
    """Configuration for SAM segmentation."""

    # SAM model variant: vit_h, vit_l, vit_b
    model_type: str = "vit_h"

    # Checkpoint path
    checkpoint: str = field(
        default_factory=lambda: str(SAM_WEIGHTS_ROOT / "sam_vit_h_4b8939.pth")
    )

    # Device for inference (auto-detected if None)
    device: Optional[str] = None

    # Minimum accepted expansion area (px²), below which a "successful"
    # expansion is rejected as a label-scale no-op instead of a room.
    # Measured on 830 real rooms (sprint1_verify45): 81% of sam_expanded=True
    # results were <10,000px² on 4500x3375px pages -- a random 8-room visual
    # sample confirmed every room <10,000px² was a pure text-label capture
    # (0/3 real rooms), while 10,000-50,000px² was a genuine mix of real
    # rooms, partial fixture-nook captures, and more label-only boxes (2026-
    # 08-06 spike). This threshold is deliberately conservative: it only
    # catches the confirmed-unambiguous case. A room surviving this check is
    # NOT verified correct -- it is only not-provably-a-label-blob.
    min_expansion_area_px: int = 10_000

    # Whether to output multiple masks
    multimask_output: bool = True

    # Max fraction of image area a single SAM bbox may occupy.
    # Exceeding this triggers the over-segmentation guardrail and keeps the
    # original label bbox instead of the SAM result.
    max_expand_frac: float = 0.25

    # T1: connected-component instance split.
    # When True, splits a SAM mask into per-instance bboxes via
    # cv2.connectedComponentsWithStats. Components < cc_min_area px² are dropped.
    use_cc_split: bool = False
    cc_min_area: int = 40000
    cc_erosion_px: int = 3

    # T2: density-peak grid seeding (EXPERIMENT — [Speculative] LiDAR→raster).
    # Generates supplementary SAM prompts from Sobel gradient peaks,
    # additive to existing label-centroid prompts.
    use_density_prompts: bool = False
    density_tau_factor: float = 0.9
    density_min_spacing: int = 10
    density_max_prompts: int = 25

    # T3: multi-stage mask pool filter (requires use_density_prompts=True).
    # Coarse (multi-component + IoU≥0.8 dedup) + greedy covering set (IoU≤0.01).
    # No-op unless use_density_prompts=True provides a pool.
    use_multistage_filter: bool = False

    def __post_init__(self):
        if self.device is None:
            self.device = _detect_device()


@dataclass
class ExportConfig:
    """Configuration for annotation export."""

    # Default image dimensions if not found in annotations
    default_width: int = 1000
    default_height: int = 1000

    # Percentage of lowest-confidence images to flag for review
    review_threshold: float = 0.25


@dataclass
class EvalConfig:
    """Configuration for ground-truth evaluation against the test_pcs dataset.

    test_pcs (COCO export, label-studio) has verified GT coverage for a
    subset of architectural objects only: door (+door2, sliding door
    variants), window1, toilet, sink1/sink2. It has zero annotations for
    "wall" (category exists, 0 instances) and no category at all for
    stairs/elevators/rooms — those cannot be scored against this dataset.

    KNOWN INCOMPLETE for door/window (2026-08-04, GC0/GC1b): full visual
    census of all 187 door/window predictions with best-IoU==0 against
    this GT (yolo_finetuned_tiled tier) found 107 (57.2%) are real,
    correctly-detected door/window symbols this GT never labeled -- not
    detector false positives. Two images have ZERO window GT at all
    (d0b1bc03, 54a899cb). Concentrated on detail/schedule/site-plan
    sheets (confirmed: 7c208ffc, 2f8c1bd5, 06ee3fd7, d0b1bc03), same
    sheet family that already broke the envelope-distance route
    (EnvelopeConfig, deleted BA4) for an unrelated reason (no single
    building outline).

    Practical effect: raw precision computed against this GT
    UNDERSTATES true detector precision. Recomputed combined door+window
    precision treating the 107 confirmed real detections as true
    positives (not false positives): 0.366 -> 0.624 (conservative, 26
    remaining ambiguous boxes counted as FP) to 0.687 (optimistic, counted
    as real). Recall is correspondingly OVERSTATED (true object count
    exceeds this GT's labeled count). Do not cite this GT's raw
    precision/recall as ground truth without this correction; do treat
    it as a reliable source for door/window BBOX LOCATIONS that are
    labeled (those are real, human-verified).

    A candidate "schedule/legend graphic hallucination" explanation was
    tested and dropped: of the 54 confirmed genuine false positives,
    most are annotation/schedule content (numbered-triangle room
    markers, lighting-fixture-schedule icons, hatch-fill legend
    swatches, dimension-leader callout arrows, bitmap text-rendering
    artifacts) -- not a shape the detector should be expected to
    suppress via any door/window feature (elongation, wall-adjacency:
    both tested, see WindowEvidenceConfig and GC1a). No fix proposed;
    recorded as a real, currently-unaddressed defect.
    """

    # Default points at the repo's own test_pcs export (real asset, same
    # convention as CubiCasa5KDetector.MODEL_PATH's repo-relative default).
    # coco/images/ is empty on disk; coco_w_images/images/ holds the pngs —
    # both result.json files are byte-identical exports, so coco/result.json
    # is read for annotations and coco_w_images/images/ for pixels.
    gt_coco_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "test_pcs" / "coco" / "result.json"
    )
    gt_images_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "test_pcs" / "coco_w_images" / "images"
    )

    # Merge label-studio's split annotation categories onto the vocabulary
    # CubiCasa5KDetector actually predicts (_DETECTABLE in cubicasa5k_detector.py).
    # Categories not listed here (wall, appliance, bed, table*, ...) have no
    # matching detector output and are intentionally left unscored.
    gt_category_merge: Dict[str, str] = field(
        default_factory=lambda: {
            "door": "door",
            "door2": "door",
            "sliding door": "door",
            "window1": "window",
            "toilet": "toilet",
            "sink1": "sink",
            "sink2": "sink",
        }
    )

    # Match threshold for counting a prediction as a true positive.
    # 0.5 mirrors the Accuracy@0.5 convention already used by
    # bbox_metrics.MetricsResult (accuracy_50) elsewhere in this pipeline.
    gt_iou_threshold: float = 0.5

    # Device for CubiCasa5K inference (auto-detected if None, same convention
    # as OCRConfig/SAMConfig/VLMConfig.qwen_device via _detect_device()).
    device: Optional[str] = None

    def __post_init__(self):
        if self.device is None:
            self.device = _detect_device()


# H-2 / I-1 (2026-09-01): second commercial GT source. 11 images, 664 door +
# 252 window instances, label-studio COCO export, disjoint source documents
# from EvalConfig's default set (no eval contamination). Precision measured
# against this set is .822 vs .652 on the old fixture (same detector/config)
# -- corrects the KNOWN INCOMPLETE precision-understatement documented on
# EvalConfig above. Window instances are NOT comparable across the two sets:
# this set's window box excludes wall thickness (short axis 3x thinner) --
# see plan/commercial-dataset-g12-assessment-2026-09-01.md H-3. Do not pool
# window GT from this source with EvalConfig's until that is resolved.
COMMERCIAL_COCO_CATEGORY_MERGE: Dict[str, str] = {
    "door": "door",
    "double_door": "door",
    "bifold_door": "door",
    "bypass_sliding_door": "door",  # declared category, 0 instances
    "window": "window",
    "sliding_window": "window",  # declared category, 0 instances
    "toilet": "toilet",
    "sink1": "sink",
    "sink2": "sink",
}

# I-4/I-6 (2026-09-01, user-confirmed): this set's "window" labels only
# the glazing line -- short axis 1.08 permille of page diagonal, vs
# EvalConfig's old fixture at 3.22 permille (wall-opening, includes
# wall thickness). The pipeline's canonical window convention is
# wall-opening (matches the old fixture AND what the shipped YOLO
# checkpoint was trained to predict, on kaggle-500's convention).
# Dilating this set's window boxes on their short axis by this ratio at
# load time (gt_evaluator.dilate_category_short_axis) recovered window
# TP 6->130, recall .024->.516, precision .017->.365 at conf 0.25 --
# same detector/weights, GT-side fix only, no re-labeling. Do not
# recompute this constant from a smaller/different sample without
# re-running that experiment.
COMMERCIAL_WINDOW_SHORT_AXIS_DILATION: float = 3.22 / 1.08


def commercial_coco_eval_config() -> EvalConfig:
    """EvalConfig pointed at the 2026-08-31 commercial set (11 img, 664
    door / 252 window, test_pcs/commercial/commercial_building_obj_detection_coco_w_images).
    Same COCO reader as the default EvalConfig -- only paths and the
    category merge map differ, so this is a factory, not a second dataclass.
    """
    root = (
        _PROJECT_ROOT
        / "test_pcs"
        / "commercial"
        / "commercial_building_obj_detection_coco_w_images"
    )
    return EvalConfig(
        gt_coco_path=root / "result.json",
        gt_images_dir=root / "images",
        gt_category_merge=dict(COMMERCIAL_COCO_CATEGORY_MERGE),
    )


@dataclass
class WindowEvidenceConfig:
    """Option B (2026-08-03): local wall-context discriminator for
    door-mislabeled-as-window, replacing the envelope-distance route
    (EnvelopeConfig, deleted BA4) killed by BA0/WG1 verification --
    6/10 test_pcs sheets have no single building envelope to measure
    distance to (site plans, multi-unit part-plan sheets, enlarged-unit
    details).

    BA0 spike (2026-08-03, GT crops, test_pcs 363 door/205 window +
    kaggle_floorplans500 355 door/281 window, throwaway script not
    committed) tested 3 local features against a best-threshold
    classification accuracy, not median-gap alone (median gap alone
    overstated separability):

      arc evidence (HoughCircles response): test_pcs acc .639, kaggle
      .558 -- EXACTLY the majority-class-door baseline on both (.639 =
      363/568, .558 = 355/636). Zero real signal. Dropped, not shipped.

      aspect ratio (w/h): test_pcs acc .817 @ threshold 1.714, kaggle
      .805 @ 1.683 -- real signal, both datasets. Superseded by BA1b
      below.

      line-pair evidence (near-parallel near-collinear Hough line-
      segment pairs, the glazing double-line signature): test_pcs acc
      .894 @ threshold 39, kaggle .786 @ 20 on GT -- real signal there,
      but BA2 (real yolo_finetuned_tiled predictions) found it
      collapses kaggle window recall .562->.288 (77 TPs lost) chasing
      1 confused box, and BA3 (10 pixel-confirmed real ROCKAWAY boxes)
      missed 2/4 real windows. Not shippable. A fix attempt (derive
      min_line_length from the crop's LONG axis instead of short, to
      symmetrically help vertical glazing) made it WORSE on both GT
      sets (.894->.808 test_pcs, .786->.673 kaggle). Implementation
      removed as dead code once zero callers remained (no combination
      step was ever shipped) -- numbers kept here, not reimplemented
      without a reason to revisit.

    BA1b (2026-08-03): BA2 found real w/h regressed hard on portrait
    (tall-narrow) windows -- 37-47% of all boxes, portrait-only acc
    only .656/.623 (test_pcs/kaggle), because a vertical window's w/h
    sits near a door's, even though it is exactly as elongated as a
    landscape window. Fixed by making the feature undirected:
    elongation = max(w,h)/min(w,h). Re-measured on the same GT: acc
    .910/.917 overall, .902/.918 portrait-only -- both datasets, no
    regression, portrait failure mode closed. Doors are NOT near 1.0
    (median elong 1.22/1.18, p90 1.79/1.56, max 15.2/2.6) -- the fix
    works because the two DISTRIBUTIONS separate, not because doors
    cluster at a fixed value; do not assume that when reasoning about
    this feature.

    Default below is test_pcs's measured optimum: test_pcs is this
    project's only confirmed commercial-domain door/window GT
    (EvalConfig's own docstring) and the domain carrying the standing
    door/window complaints this option addresses. Kaggle's own optimal
    value (1.8485) is recorded here for audit, not silently discarded.

    BA1b only: exposes the one validated, shipped predicate
    (is_window_shaped, wired into sft_validator.py FIX-9b). No other
    feature or combination rule remains -- both were measured and
    dropped, not invented.
    """

    # BA1b-measured best-threshold on test_pcs (.910 accuracy):
    # elongation = max(w,h)/min(w,h) <= this -> door-like, > this ->
    # window-like (undirected -- orientation-independent, unlike raw
    # w/h). Numerically identical to BA1's w/h threshold (coincidence
    # of this GT, not assumed to hold in general) -- exact value from
    # the sweep, not rounded (a rounded 1.714 was tried first and
    # silently flipped 2/568 boxes' classification vs the measured
    # optimum).
    elongation_threshold: float = 1.7142857142856909


_CUBICASA_CACHE_ROOT = (
    Path.home() / ".cache" / "kagglehub" / "datasets" / "qmarva"
    / "cubicasa5k" / "versions" / "4"
)


@dataclass
class CubicasaEvalConfig:
    """Configuration for ground-truth evaluation against CubiCasa5K.

    CubiCasa5K's COCO export (kagglehub cache) has verified GT coverage for
    "wall" and "room" bounding boxes only (2 categories total; no door/window
    category in this export — those exist only in the raw per-image SVGs).
    Coverage is BBOX LOCALIZATION on RESIDENTIAL floorplans: no room-type
    label (generic "room" class only) and no commercial/MEP coverage — do
    not treat this as ground truth for room-type classification or for the
    project's commercial-plan domain (test_pcs covers commercial doors and
    windows; this dataset does not overlap it).

    Every image in this export shares file_name basenames that repeat across
    the dataset (all 400 test images are literally named "F1_original.png",
    disambiguated only by their parent folder) — resolving by basename would
    silently collide, so this config's resolver preserves the relative path
    instead of using gt_evaluator's default basename resolver.
    """

    coco_test_path: Path = field(
        default_factory=lambda: (
            _CUBICASA_CACHE_ROOT / "cubicasa5k_coco" / "test_coco_pt.json"
        )
    )

    # Root that file_name (with kaggle_path_prefix stripped) resolves under.
    images_root: Path = field(
        default_factory=lambda: _CUBICASA_CACHE_ROOT
    )

    # Prefix baked into every file_name by the export tool (kaggle notebook
    # path), stripped before joining the remainder onto images_root.
    kaggle_path_prefix: str = "/kaggle/input/cubicasa5k/"

    # Both raw COCO categories are already the vocabulary scored — identity
    # map, not a translation (this export has no other categories).
    gt_category_merge: Dict[str, str] = field(
        default_factory=lambda: {"wall": "wall", "room": "room"}
    )

    # Same Accuracy@0.5 convention as EvalConfig.gt_iou_threshold.
    gt_iou_threshold: float = 0.5

    # Cap on images scored per run (None = all). Config-driven so a quick
    # check and a full run use the same code path, no hardcoded loop bound.
    sample_size: Optional[int] = None

    # When True, grow raw VLM room boxes to room walls via the production
    # SAM label-seeded expansion (RoomSegmenter.refine_annotations), matching
    # the pipeline's use_sam path. Default False: raw detect_rooms boxes are
    # measured as-is, so the baseline behavior is unchanged.
    use_room_expansion: bool = False


_KAGGLE_FLOORPLAN_CACHE_ROOT = (
    Path.home() / ".cache" / "kagglehub" / "datasets" / "umairinayat"
    / "floor-plans-500-annotated-object-detection" / "versions" / "1"
)

# Shared by the T-S1 pretrained-baseline tier (kaggle_door_window_eval.py)
# and the T-S2 fine-tune (yolo_train.py) -- single source of truth so an
# absolute checkpoint/output path is used everywhere, never a bare
# Ultralytics model name (which downloads/writes to whatever the CWD
# happens to be when a script runs -- caught during T-S1).
YOLO_CHECKPOINT_ROOT = _PROJECT_ROOT / "yolo_checkpoints"


def yolo_checkpoint_path(filename: str) -> str:
    """Resolve a bare Ultralytics checkpoint filename (e.g. "yolo26s.pt")
    to an absolute path under YOLO_CHECKPOINT_ROOT -- one resolution rule
    reused for every model size, instead of one named constant per size.
    """
    return str(YOLO_CHECKPOINT_ROOT / filename)


YOLO_PRETRAINED_CHECKPOINT = yolo_checkpoint_path("yolo26n.pt")


@dataclass
class KaggleFloorplanEvalConfig:
    """Configuration for ground-truth evaluation against the Kaggle
    floor-plans-500 dataset (Roboflow export, YOLOv11 format, CC BY 4.0
    -- the project's first commercially-licensed door/window GT).

    Verified coverage (2026-07-20, counted per-file in Python -- a first
    shell `cat`-based tally undercounted due to label files lacking a
    trailing newline, merging lines at file-concatenation boundaries;
    do not re-derive this count with `cat *.txt | awk`): door (7282)
    and window (5567) instances across 960 images (train 837 / valid 80
    / test 43). A third class, "zone", is also present (7495 instances)
    but its labeling protocol is unverified -- excluded from
    gt_category_merge by default, so it is loaded but not scored.

    Domain (residential/commercial/MEP) is NOT stated by the source and
    has not been verified against this project's commercial-MEP scope --
    treat scores from this dataset as a supplementary signal, not a
    substitute for test_pcs (EvalConfig), the only confirmed
    commercial-domain door/window GT in this project.
    """

    dataset_root: Path = field(
        default_factory=lambda: _KAGGLE_FLOORPLAN_CACHE_ROOT
    )

    # Which Roboflow split to load: "train", "valid", or "test".
    split: str = "test"

    # "zone" is intentionally excluded (protocol unverified, see class
    # docstring) -- extend this map only after confirming its meaning.
    gt_category_merge: Dict[str, str] = field(
        default_factory=lambda: {"door": "door", "window": "window"}
    )

    # Same Accuracy@0.5 convention as EvalConfig.gt_iou_threshold.
    gt_iou_threshold: float = 0.5


@dataclass
class KaggleFloorplanTrainingConfig:
    """Config for fine-tuning a door/window detector on the Kaggle
    floor-plans-500 dataset (YOLO Pilot, T-D/T-S2 -- see
    VLM/YOLO_FINETUNING_ASSESSMENT.md and the combined plan's Track A).

    Dataset is already native YOLO format (data.yaml + per-image txt
    labels) -- zero conversion needed, unlike CubiCasa5K/test_pcs.
    """

    # Reuses the same cache root KaggleFloorplanEvalConfig resolves images
    # under -- data.yaml sits at this root (verified 2026-07-23).
    data_yaml_path: Path = field(
        default_factory=lambda: _KAGGLE_FLOORPLAN_CACHE_ROOT / "data.yaml"
    )

    # Ultralytics' own documented training defaults (docs.ultralytics.com/
    # usage/cfg/, verified 2026-07-23) -- not this project's invention,
    # kept as the starting point until Spike 2 shows a reason to deviate.
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16

    # Ultralytics' own default (confirmed 1.0 from the T-S2 run's own
    # printed trainer args, 2026-07-24) -- mosaic stitches 4 training
    # images into one composite. YOLO_FINETUNING_ASSESSMENT.md Phase 7
    # flagged this as a risk specific to structured technical drawings
    # (composites can teach false room/wall adjacency) but it was never
    # actually tested with mosaic off until this field existed. Default
    # kept at Ultralytics' 1.0 so adding this field doesn't silently
    # change the already-recorded T-S2 baseline; override via CLI to test
    # the hypothesis.
    mosaic: float = 1.0

    # Base weights to fine-tune from -- the same stock checkpoint T-S1
    # already baselined, so Spike 2's gain is measured against Spike 1,
    # not a different starting point.
    base_checkpoint: str = YOLO_PRETRAINED_CHECKPOINT

    # Ultralytics writes run artifacts under this dir (never CWD -- same
    # stray-file issue T-S1 hit with a bare checkpoint name applies to
    # training output dirs too).
    runs_dir: Path = field(
        default_factory=lambda: YOLO_CHECKPOINT_ROOT / "runs"
    )


@dataclass
class FloorplancadEvalConfig:
    """Config for FloorPlanCAD door/window/wall eval GT (HF Voxel51).

    FloorPlanCAD is a panoptic SYMBOL-spotting CAD dataset covering
    residential AND commercial buildings -- the only commercial-domain
    door/window GT at scale assessed in this project. The HF Voxel51
    mirror is a FiftyOne dataset (5308 samples): a ``samples.json``
    (per-sample ``ground_truth.detections`` with ``label`` and
    ``bounding_box`` normalized ``[x, y, w, h]``, plus ``metadata``
    width/height) alongside a ``data/`` folder of PNGs. Parsed directly
    -- no FiftyOne dependency needed.

    LICENSE: CC BY-NC 4.0 (non-commercial). EVAL-ONLY here; never route
    FloorPlanCAD into shipped SFT training data.

    Category-merge is grounded in the verified label counts (2026-07-20,
    48465 detections, 35 labels): door = single/double/sliding_door
    (12698); window = window/bay_window/blind_window (1952);
    ``opening_symbol`` (2 instances) is EXCLUDED -- negligible and
    semantically ambiguous (opening != window). wall = wall (4710).
    All other labels (stairs, furniture, class_NN, ...) are dropped.
    """

    repo_id: str = "Voxel51/FloorPlanCAD"
    samples_filename: str = "samples.json"
    gt_category_merge: Dict[str, str] = field(
        default_factory=lambda: {
            "single_door": "door",
            "double_door": "door",
            "sliding_door": "door",
            "window": "window",
            "bay_window": "window",
            "blind_window": "window",
            "wall": "wall",
        }
    )

    # Same Accuracy@0.5 convention as EvalConfig.gt_iou_threshold.
    gt_iou_threshold: float = 0.5


# Same checkpoint CubiCasa5KDetector.MODEL_PATH resolves to (detection/
# cubicasa5k_detector.py's own parents[3]-relative computation) -- redefined
# here rather than imported, matching YOLO_CHECKPOINT_ROOT's precedent of
# config.py owning its own path constants instead of depending on detection/.
CUBICASA_MODEL_PATH = _PROJECT_ROOT / "preprocessing_annotations" / "models" / "cubicasa5k_model.pkl"


@dataclass
class CubicasaRoomDetectorConfig:
    """Config for the dormant CubiCasa5K room head (T-A/Technique-1;
    detection/cubicasa5k_detector.py::CubiCasa5KDetector.detect_rooms).

    NOT currently called from pipeline.py. A corroboration-tagging use
    (tag VLM/OCR rooms with independent CNN agreement, unlabeled boxes
    only -- no room_name) was built, wired, and measured this session,
    then removed after evidence found no discriminative value: on the
    correctly-powered test (pipeline's own kept-vs-dropped room split,
    n=259), CNN corroboration rate was statistically indistinguishable
    between kept and dropped rooms (Fisher exact p=0.603). Removed from
    sft_validator.py (was _corroborate_with_cnn_rooms + _bbox_containment,
    called from pipeline.py's Step 4d.5) -- reimplement from this record
    if revisiting, not preserved elsewhere.

    This config and CubiCasa5KDetector.detect_rooms() remain because the
    detector itself is validated independent of that removed use --
    see the localization numbers below -- and stay available for a
    different application (e.g. geometric room-scale upgrade of
    label-sized OCR boxes, not yet built).

    Defaults are the swept-and-verified config (this session, full
    400-image CubiCasa5K test set, class-agnostic room GT, IoU>=0.5):
    per_class=True + min_area_px=1500 + barrier_dilate_px=2 ->
    P.622/R.707/mIoU.863, dominating the no-barrier per-class baseline
    (P.630/R.630/mIoU.843) on recall and mIoU at ~flat precision.
    barrier_dilate_px uses the model's OWN wall (room slice class 2) and
    door/window (icon slice classes 1/2) channels from the SAME forward
    pass to cut same-type adjacent rooms apart before connected-
    components -- see cubicasa5k_detector.py::detect_rooms docstring for
    the FN-cause measurements this was tuned against.

    SCOPE, hence off by default (see use_cubicasa_rooms):
      - Residential-only evidence. CubiCasa5K test set only; test_pcs (this
        project's only commercial GT) has zero "room" category annotations
        -- verified, not assumed -- so commercial performance is unmeasured,
        not merely unmeasured-yet.
      - License: CC BY-NC 4.0 (upstream CubiCasa5k repo's own LICENSE file;
        the vendored seg_model.py header previously and incorrectly claimed
        MIT, corrected this session). Confirmed OK for this project's
        research/non-commercial use. Re-verify before any commercial/
        shipped use -- CC BY-NC still blocks that regardless of the domain
        question above.
    """

    checkpoint_path: str = field(default_factory=lambda: str(CUBICASA_MODEL_PATH))

    # Same convention as OCRConfig/SAMConfig/VLMConfig.qwen_device/
    # CubicasaEvalConfig.device -- None resolves via _detect_device() in
    # __post_init__. Explicit escape hatch to "cpu": use_vlm + use_sam +
    # use_cubicasa_rooms in one run puts 3 models on one GPU; this
    # checkpoint is the smallest of the three (hourglass CNN, single
    # forward pass) so it's the one that can afford to move off-GPU if
    # co-residence becomes a real problem -- untested, not yet needed
    # since default is off and P5's verification run avoids co-residence
    # entirely (use_cubicasa_rooms alone, no --use-vlm/--use-sam).
    device: Optional[str] = None

    per_class: bool = True
    min_area_px: int = 1500
    barrier_dilate_px: int = 2

    def __post_init__(self):
        if self.device is None:
            self.device = _detect_device()


@dataclass
class YoloObjectDetectorConfig:
    """Config for T-I's pipeline-integrated YOLO door/window detector
    (yolo_detector.py::YoloObjectDetector) -- the T-S2 fine-tuned
    checkpoint, wired as an optional object-detection source alongside
    WindowDetector's existing tiers.

    Tag-only by design (T-I plan): detections feed room attributes
    (has_door/has_windows/etc via the shared map_detections_to_rooms,
    window_detector.py), never accept/reject a room -- matches
    WindowDetector's existing behavior, zero new regression surface.
    """

    # Promoted 2026-09-05 (6 -> 7): fine-tuned from -6 with the 11-image
    # commercial GT (test_pcs/commercial/commercial_building_obj_detection_
    # coco_w_images) blended in (9 images oversampled 5x, 2 held out for
    # an honest check), 40 epochs. Verified no regression: Kaggle test
    # door P.86/R.88->P.84/R.87, window P.62/R.70->P.58/R.69 (within
    # noise); test_pcs door P.31/R.31->P.35/R.35, window P.20/R.44->
    # P.36/R.40 (both improved). Held-out (2 images, never trained on):
    # door recall improved R.51->R.61.
    #
    # This retrain ALSO added a 4th class, "double_door" (144 real GT
    # instances, previously merged into "door" for eval, never trained --
    # see plan/room-object-fp-fn-fix-plan-2026-09-05.md SS1c/2b). That part
    # is a NEGATIVE RESULT, not shipped: 0/13 double_door GT recognized on
    # the held-out check, both before and after this retrain -- 45
    # oversampled examples in 40 epochs was not enough signal for a new
    # visual class. Do NOT add "double_door" to keep_categories below --
    # window_detector.py's map_detections_to_rooms (~line 146-160) does
    # strict string-equality type dispatch with an `else: has_windows =
    # True` fallback; an unhandled category silently miscounts as a
    # window, not a door. keep_categories stays ("door", "window") only
    # until that dispatch is updated AND double_door is actually learned.
    # Promoted 2026-09-05 (7 -> 9): E4 electrical-arc-FP mitigation. -8 (an
    # intermediate attempt, not shipped) fine-tuned -7 with 414 hard-negative
    # crops -- one per detection on the 8 pages this session's human QA
    # marked "all detections in category X are FP" (arc-line-vs-electrical-
    # symbol confusion) -- and DID cut the FP count on those pages 301->89
    # (-70%) but caused real collateral regression elsewhere: test_pcs door
    # P.35/R.35->P.26/R.22, held-out door recall R.61->R.43. Root cause:
    # 414 negatives with no fresh positive counterexamples from equally
    # cluttered layouts biased the model toward suppressing real doors in
    # dense drawings generally, not just electrical-arc shapes.
    #
    # -9 retried with the SAME 8 pages' negatives subsampled 414->96
    # (~12/page, same order of magnitude as the 45 commercial-oversample
    # examples already in the training mix). Result: no regression anywhere
    # (test_pcs door P.35/R.35->P.38/R.37, actually improved; held-out door
    # recall R.61->R.59, flat; Kaggle test flat) AND a real 18% further FP
    # drop on the 8 target pages (301->247; -40% from the original -6
    # baseline of 414). Clean win, no tradeoff -- promoted.
    checkpoint_path: str = field(
        default_factory=lambda: str(
            YOLO_CHECKPOINT_ROOT / "runs" / "kaggle_door_window-9"
            / "weights" / "best.pt"
        )
    )

    keep_categories: Tuple[str, ...] = ("door", "window")

    # R2/R4: tiled detection. Full-page inference downscales to the
    # model's ~640px input, shrinking real doors/windows below
    # detectability (measured: 94-96% pure-miss on test_pcs full-page).
    # R3 measurement (2026-07-29) was taken at Ultralytics' implicit
    # conf=0.25 (no threshold plumbed yet) and is superseded -- kept
    # below for provenance only, do not cite as current:
    #   test_pcs: door P.0345->P.307, R.0028->R.311; window
    #   P.119->P.198, R.0244->R.439. Kaggle: byte-identical to
    #   non-tiled (P.8599/R.8817 door, P.6242/R.6975 window) --
    #   because those images (1119-1633px) sit below tile_trigger_px
    #   and detect_objects_tiled falls back to full-image.
    #
    # Re-measured post-R1 at this field's actual default (conf=0.5,
    # 2026-07-30/31): test_pcs door P.4197/R.2231, window P.3198/R.3463;
    # kaggle door P.8896/R.8394, window P.7789/R.5516. Kaggle is NO
    # LONGER byte-identical to non-tiled -- the shift is the conf
    # threshold (0.5 vs the old implicit 0.25), not tiling; those
    # images still fall back to full-image either way.
    #
    # Confidence sweep (CD1, 2026-07-31, --yolo-conf) shows the
    # precision/recall trade flattens past conf~0.25 and costs far
    # more on test_pcs than in-domain: kaggle loses ~11.6pp door
    # precision (.890->.774) for +8.5pp recall; test_pcs loses ~23.8pp
    # (.420->.182) for a similar recall band. Diminishing, domain-
    # dependent -- not evidence for lowering the production default.
    #
    # Near-miss oversizing (R0b's recorded 5.8x median area ratio) does
    # not hold post-tiling/post-R1: door_localization_diagnostic.py
    # measures 4.69 on test_pcs, 0.41 (undersized) on kaggle -- domain-
    # specific, weaker than originally recorded.
    #
    # Bar cleared on both datasets at the time of R4 -- default
    # promoted to True. Trigger/cols/rows/overlap/target_px mirror
    # PipelineConfig's own already-tuned VLM-tiling values (config.py
    # tile_cols/tile_rows/tile_overlap_pct/tile_trigger_px/
    # tile_target_px) -- same convention, not reinvented.
    use_tiling: bool = True
    tile_cols: int = 4
    tile_rows: int = 4
    tile_overlap_pct: float = 0.10
    tile_trigger_px: int = 3000
    tile_target_px: int = 1200

    # NMS IoU for de-duplicating overlapping tile detections. Same value
    # as the VLM room-tiling merge (pipeline.py's splitter.merge call).
    tile_merge_iou: float = 0.30

    # Per-category cap after merge -- NOT the room path's max_rooms
    # (25/40): doors+windows run far denser per page than rooms. test_pcs
    # GT alone reaches ~36 doors/image (363/10); 100 leaves real margin
    # without silently truncating like the room path's tighter cap did.
    tile_max_detections_per_category: int = 100

    # R1: minimum confidence for a detection to count, set by explicit
    # instruction (not measured/invented). Was previously an unconfigured
    # Ultralytics-internal default (0.25) with no way to raise it without
    # editing code. Only affects this production detector's own calls
    # (yolo_detector.py) -- the eval CLI's yolo_pretrained/yolo_finetuned
    # tiers call yolo_infer.detect_objects directly without this config
    # class, so their recorded baseline numbers are untouched.
    #
    # Raised 0.5 -> 0.75 by explicit instruction (2026-08-09). Measured cost
    # on test_pcs GT (conf_sweep, production tiling, IoU 0.5) -- this trades
    # precision for a large recall loss, and window is hit hardest:
    #   door    conf .50: P.394 R.242 F1.300  ->  conf .75: P.621 R.131 F1.216
    #   window  conf .50: P.320 R.346 F1.333  ->  conf .75: P.562 R.044 F1.081
    # Window recall collapses to 9 TP of 205 GT. F1 falls on BOTH classes.
    #
    # LOWERED 0.75 -> 0.25 (I-2, 2026-09-01, user-confirmed after full
    # sweep). Prior note above only compared .50 vs .75; the full sweep
    # (--yolo-conf in {.25,.35,.50,.65,.75}, 3 datasets: kaggle_floorplans500
    # residential 43 img, test_pcs commercial 10 img, test_pcs_commercial
    # commercial 11 img, categories door+window, all tiled) found 0.75 is
    # DOMINATED by every lower value tested -- F1 is worse at 0.75 than at
    # 0.25 in every (dataset, category) cell measured, not a trade:
    #   kaggle   door   conf .25: P.8599 R.8817 F1.8707 -> .75: P.9397 R.7465 F1.8320
    #   kaggle   window conf .25: P.6242 R.6975 F1.6588 -> .75: P.8889 R.1423 F1.2454
    #   test_pcs door   conf .25: P.3071 R.3113 F1.3092 -> .75: P.6515 R.1185 F1.2005
    #   test_pcs window conf .25: P.1982 R.4390 F1.2731 -> .75: P.5625 R.0439 F1.0814
    #   test_pcs_commercial door   conf .25: F1.4618 -> .75: F1.1628
    #   test_pcs_commercial window conf .25: F1.0197 -> .75: F1.0000
    # (test_pcs_commercial window is near-zero at every threshold here
    # because its GT used a different box convention -- see I-4/H-3 below
    # and commercial_coco_eval_config's dilation; not a confidence effect.)
    # 0.25 is not universally the single best cell (e.g. test_pcs window
    # peaks at conf .50, F1.3326) but it is never far off and never worse
    # than .75 anywhere measured. Raw sweep JSON: sweep/*_conf*.json
    # (eval/kaggle_door_window_eval.py --yolo-conf). Revisit if a
    # precision-sensitive consumer starts reading these detections
    # directly -- today they are decorative (no ConfidenceComputer term
    # consumes them, tracked separately as H-7 / the 08-28 plan's G-1),
    # so the wider FP set this lets through has ~no blast radius yet.
    confidence_threshold: float = 0.25

    # D1 (2026-09-05, human QA on sprint1_verify49): degenerate-box floor
    # for door/window detections. NARROWER than originally attempted -- an
    # aspect-ratio cap was tried first and REJECTED after measuring real
    # per-category distributions on this same run: windows are structurally
    # elongated (aspect p05=2.00, median=3.29, up to 7.12 -- a wall window
    # is drawn as a thin rectangle, not a square), so a shared aspect cap
    # dropped 97% of real windows. Doors have their own long tail (p95=2.41,
    # max=5.99) that also overlaps the one confirmed punctuation-glyph FP
    # (a ")" character misread as a door swing, 697px^2, aspect 2.4) closely
    # enough that shape alone cannot cleanly separate it from real doors --
    # that FP is NOT resolved by this floor; catching it needs cross-
    # checking against OCR text-token bboxes (not yet wired, see
    # yolo_detector.py's _passes_shape_floor docstring), not a size/aspect
    # threshold. What remains here only guards against literally-degenerate
    # (near-zero-area) boxes, same convention as ROOM_MIN_AREA_PX/
    # min_expansion_area_px elsewhere -- door p05=756px^2, window p05=310px^2
    # on the measured run, so 150 is a floor well below any real detection,
    # not a precision lever.
    min_area_px: int = 150


# facebook/sam3 is a gated HF repo; weights are downloaded once and cached
# locally under the standard huggingface_hub cache. Shared resolution
# point (mirrors yolo_checkpoint_path's role above) so the production
# detector and the eval harness (kaggle_door_window_eval.py) resolve the
# same file instead of each hardcoding the repo id/filename.
_SAM3_HF_REPO = "facebook/sam3"
_SAM3_CKPT_FILENAME = "sam3.pt"


def resolve_sam3_checkpoint() -> str:
    """Resolve the cached facebook/sam3 checkpoint via huggingface_hub.

    ``local_files_only`` -- the gated repo's HEAD revalidation returns 403
    without a token; the weights are already downloaded, so this never
    needs network access.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return hf_hub_download(
            repo_id=_SAM3_HF_REPO,
            filename=_SAM3_CKPT_FILENAME,
            local_files_only=True,
        )
    except LocalEntryNotFoundError as error:
        raise RuntimeError(
            f"SAM3 checkpoint not cached ({_SAM3_HF_REPO}/"
            f"{_SAM3_CKPT_FILENAME}); download once with a valid HF "
            f"token before offline use"
        ) from error


@dataclass
class Sam3ExemplarDetectorConfig:
    """Config for the tiled SAM3 box-exemplar door detector
    (sam3_exemplar_detector.py::Sam3ExemplarDetector) -- seeded by
    YoloObjectDetector's own high-confidence detections, recovers
    additional doors SAM3's concept mode finds in a native-resolution
    tile around each seed. Door only: the validating spike (tech-eval
    plan, P1) found window seed supply near-empty with no recall gain;
    keep_categories is not exposed here the way YoloObjectDetectorConfig
    exposes it -- this detector does not generalize past door yet.

    Additive only, like YoloObjectDetectorConfig: detections extend the
    same yolo_detections list before it reaches map_detections_to_rooms,
    so they inherit that path's existing tag-only behavior (has_door
    room attribute, never accepts/rejects a room) with no new merge code.

    WITHDRAWN (2026-08-18, P6, numbers superseded same day by P9): the
    paragraph below this note stated a two-dataset validation beating
    YOLO-alone on both precision and recall. That number came from a
    duplicate exemplar implementation that lived in
    eval/kaggle_door_window_eval.py (seed-centred crops, no scale-
    consistency gate) and had silently diverged from THIS class -- it
    was never actually a measurement of the code shipped here. Audit
    #16 found the divergence (P5). Re-measuring THIS class through the
    same eval harness (P6) gave test_pcs P.420/R.223 -> P.414/R.231;
    FloorPlanCAD P.489/R.500 -> P.475/R.609 -- but THAT measurement was
    itself taken under a still-broken grid (tile_cols/tile_rows=4
    capped BELOW what the P0 ceil() fix asked for on real pages, so P6
    silently measured a 1238px-tile/0.81x-downscale grid despite P0
    having already landed -- caught auditing this file's own numbers).
    Cap bumped 4->5 (P9, below); precision was STILL below baseline on
    both datasets at that point.

    RESTORED, on real evidence this time (P13, same day): reconciling
    the per-box census against the scorer's marginal counts exposed a
    dedup defect -- detect_additional deduped additions against the
    high-confidence EXEMPLAR subset rather than every door the caller
    passed in, so doors YOLO found at confidence 0.5-0.75 were
    invisible to the dedup and SAM3 re-finding them produced duplicate
    boxes scoring as false positives (a third of test_pcs's marginal
    FP). Fixed. Current, measured through THIS class:
      test_pcs      P.420/R.223 -> P.423/R.242
      FloorPlanCAD  P.489/R.500 -> P.506/R.580
    Union now beats YOLO-alone on BOTH axes on BOTH datasets. Same
    claim as the withdrawn one, but earned by fixing a defect rather
    than by measuring a fork of the code. Caveats ride with it and are
    NOT optional: test_pcs GT is documented KNOWN INCOMPLETE (see
    EvalConfig) so absolute values are unreliable while the comparison
    holds; P13 costs FloorPlanCAD recall vs P9 (.609->.580);
    sam3_confidence_threshold=0.70 has never been swept against these
    numbers; small n (10/40 images), door only. See
    test/test_sam3_exemplar_gt_regression.py for the pinned numbers and
    the full three-repin history. use_sam3_exemplar remains default
    False -- flipping it is a separate decision needing the stated SFT
    recall-vs-precision objective, not implied by this result.

    THRESHOLD DEPENDENCY, do not drop (found auditing C2, same day):
    every number above is measured at yolo_conf=0.5 (the eval harness's
    pinned baseline). The P13 dedup fix it depends on is a NO-OP when
    the seed detector's own confidence_threshold >= this class's
    seed_confidence_threshold (0.75) -- the two door lists P13 split
    apart (all doors vs >=0.75 doors) are then identical, so there is
    nothing for the wider dedup to catch. STALE (2026-09-01, I-2):
    PipelineConfig's default YoloObjectDetectorConfig.confidence_threshold
    was 0.75 when this was written; it is now 0.25 (see that field's own
    docstring). At 0.25 < seed_confidence_threshold (0.75), P13's dedup
    fix is NOT a no-op in production anymore -- it is live. These numbers
    still describe the yolo_conf=0.5 regime the GT harness measures, not
    whatever production now does at 0.25; that gap has not been
    re-measured under this class specifically.

    Original (now-superseded) claim, for the record: "two-dataset
    validation (test_pcs commercial, FloorPlanCAD residential+
    commercial, tech-eval plan P0/P1): at these defaults the union of
    (YOLO baseline @conf 0.5) + (this detector's additions) beats
    YOLO-alone on BOTH precision and recall on both datasets -- test_pcs
    P.420/R.223 -> P.425/R.251; FloorPlanCAD P.489/R.500 -> P.500/R.659."
    sam3_confidence_threshold=0.70 was picked against that withdrawn
    number and has not been re-swept against the corrected one -- next
    lever, not yet pulled. Native tiling is still not optional
    (unaffected by the withdrawal above): a full-page resize puts a
    ~44px door under one 14px ViT patch token (imgsz 644 -> 0.45 tokens,
    measured max confidence .32); a 1008px native tile puts it at ~3
    tokens (measured max confidence .898 on the same image/exemplar).

    UPDATE (2026-08-12): the original per-seed crop design measured 30%
    page coverage on a real page (7 seeds -> 7 tiles centred on those
    seeds, vs YOLO's systematic grid covering 100%) -- a real door
    anywhere outside a seed's neighborhood was structurally unreachable.
    Same run surfaced a false-positive cascade (up to 39 detections/page)
    where SAM3's same-image concept matching locked onto a repeated
    non-door symbol (light fixture + curved switch-leg wire -- same
    "rectangle + arc" composition as a door swing) instead. Multi-
    exemplar was tested as a fix and made it WORSE (K=1->1 instance,
    K=2->23 instances, same tile) -- refuted, not adopted.
    Fix: tile_cols/tile_rows/tile_overlap_pct/tile_target_px added below,
    mirroring YoloObjectDetectorConfig's own adaptive-grid fields exactly
    (same formula, see Sam3ExemplarDetector._adaptive_grid) so the
    exemplar-producing and exemplar-consuming stages run under the SAME
    tiling regime -- systematic, full-coverage, not seed-centred.
    tile_target_px defaults to tile_px (1008) rather than YOLO's own 1200
    so grid tiles land near SAM3's own native resolution instead of
    YOLO's, keeping the native-resolution property above intact.
    """

    seed_confidence_threshold: float = 0.75
    sam3_confidence_threshold: float = 0.70
    tile_px: int = 1008

    # Systematic tiling grid -- same fields/formula as
    # YoloObjectDetectorConfig (config.py, this file), so the seed
    # detector and this detector search the page under equivalent
    # conditions instead of YOLO's full-coverage grid vs this
    # detector's old seed-centred 30%-coverage crops.
    #
    # tile_cols/tile_rows DIVERGE from YOLO's matching cap (4) as of
    # 2026-08-18 (P9): YOLO's cap is a coverage/cost knob with no
    # resolution guarantee attached. This detector's cap sat below what
    # ceil(full_w/tile_target_px) asks for on real project pages
    # (4500x3375 needs 5 cols, cap was 4 -> _adaptive_grid silently
    # produced 1238px tiles, 0.81x downscale of tile_target_px, even
    # after switching round()->ceil() -- the cap, not the rounding
    # function, was the actual binding constraint). Bumped 4->5:
    # GEOMETRICALLY verified against every real page in both spot-check
    # batches (widest 4500px, tallest 4500px) -- cap no longer binds on
    # any of them, tiles land 807-1100px. That is the ONLY thing
    # "verified" means here -- it is not a claim that 5 beats 4 on real
    # detection quality.
    #
    # CONTESTED (2026-08-18, C2, same day): paired A/B on 2 real pages
    # (same seeds, same code, only this cap varied) found 4->5 LOSING
    # real doors: Lake Shore Electrical p002 lost both of 2 previously-
    # recovered real doors (and its known false-positive cascade went
    # from a partial 5-box scatter to a complete, cleaner 12-box grid of
    # the confusable non-door symbol); Kennedy AVI-ON FLAT p002 lost 3
    # of 4 previously-confirmed real doors. This is the current default
    # and it is NOT validated as an improvement on real pages -- the
    # aggregate GT marginal-precision gain that motivated the bump
    # (test/test_sam3_exemplar_gt_regression.py) cannot see this kind of
    # per-page swap (new config finds a different, non-superset result
    # than old on the same page). C6 (unstarted): paired per-page
    # comparison across all 26 staged pages, yolo_conf held constant,
    # before treating either value as decided.
    tile_cols: int = 5
    tile_rows: int = 5
    tile_overlap_pct: float = 0.10
    # Target px/tile for the adaptive grid. Defaults to tile_px (this
    # detector's own native-resolution ceiling), NOT YoloObjectDetector-
    # Config.tile_target_px (1200) -- using YOLO's own grid size would
    # downscale tiles below SAM3's measured native-resolution requirement.
    tile_target_px: int = 1008

    # NMS IoU for de-duplicating SAM3's own boxes across overlapping
    # seed tiles (multiple seeds can rediscover the same instance).
    dedup_iou: float = 0.6

    # IoU above which a SAM3 box is considered "already found by the
    # seed detector" and dropped rather than double-counted.
    seed_overlap_iou: float = 0.5

    # UPDATE (2026-08-16, Audit #16 P1/P2): fixing _adaptive_grid's
    # round() -> ceil() undershoot (round(4500/1008)=4 cols -> 1238px
    # tiles, non-native res) restored real recall on ROCKAWAY p001, but
    # the exact 1->4 count is NOT attributable to this fix alone -- it
    # sits downstream of the same-day systematic-grid rewrite too, and
    # the two were never measured apart (see detector module docstring).
    # It also made the pre-existing false-positive cascade WORSE, not
    # better (Lake Shore p000: 40 -> 63 additions) -- finer resolution
    # resolves the confusable MEP symbol better too, it does not
    # distinguish it from a real door on its own.
    #
    # 5th guardrail, same all-or-nothing shape as the existing
    # over_segmentation/flood_swallowed_seed/collapse/label_scale_noop
    # family: compare median(addition bbox area) vs median(door seed
    # bbox area) for the page; if the ratio falls below this threshold,
    # discard ALL additions for that page. Calibrated on the P0-fixed,
    # PRE-P9 grid (tile_cols/tile_rows=4): ROCKAWAY p001 (4 real
    # additions) ratio=0.774; Lake Shore p000 (63 cascade additions)
    # ratio=0.376 -- clean separation, 0.5 sits in the gap.
    #
    # STALE SINCE P9 (2026-08-18, same day tile_cols/tile_rows bumped
    # 4->5): neither page above has been re-measured at the shipped
    # cap, and C2 already found this same cap change altering which
    # pages the cascade appears on and how completely (Lake Shore
    # Electrical p002: 7->12 additions, tile_cols docstring). This
    # threshold's separation evidence describes a grid the code no
    # longer runs. Re-derive together with C6 (grid cap decision), not
    # before it -- recalibrating against a cap that may itself revert
    # would be wasted work.
    min_seed_area_ratio: float = 0.5


@dataclass
class ConfidenceWeightsConfig:
    """T-1 (technology-evaluation-2026-08-28.md, G-1): weights for
    ConfidenceComputer.compute(). detection/classification/ocr replace
    the formula's previous inline literals (0.3/0.4/0.3) -- same values,
    now config-driven instead of hardcoded, per that plan's own
    integration note ("weights must become config-driven, not literals").

    corroboration_weight is NOT a fourth share of the same 1.0 pie: the
    detection/classification/ocr weights are still combined as their own
    weighted average (unchanged arithmetic when corroboration_weight=0),
    and corroboration_weight scales an ADDITIVE bonus on top, capped so
    the total never exceeds 1.0. This is deliberate, not an oversight --
    T-1's own risk table requires the term to be asymmetric ("absence of
    evidence never penalizes a room -- no door found != no door"). A
    room with zero object-detection evidence gets corroboration_score=0,
    contributing nothing, so its confidence is identical with the term
    on or off. Reducing detection/classification/ocr's weights to make
    room for a classic 4-way weighted average would instead lower every
    room's floor the moment the term is enabled, penalizing exactly the
    rooms (no nearby door/window) this design must not touch.

    Default 0.0: byte-identical to pre-T-1 behavior until deliberately
    enabled -- same off-by-default precedent as every other object-
    detection feature in this config (use_yolo_objects, use_sam3_exemplar).
    T-1's own verdict is Prototype First, not Adopt: the commercial
    detector is weak (G-4, door R~.22-.41 depending on threshold) so
    corroboration signal may be sparse on real commercial pages -- measure
    confidence-distribution spread and sft_recommended rate before raising
    this above 0.0 (T-1 spike task), do not assume a nonzero value helps.
    """

    detection: float = 0.3
    classification: float = 0.4
    ocr: float = 0.3
    corroboration: float = 0.0


def _get_mlflow_tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


@dataclass
class MlflowConfig:
    """Self-hosted MLflow run tracking. Diagnostic layer only -- disabled
    by default, and every call site that consumes this config must no-op
    (never raise) if mlflow isn't installed or the tracking server is
    unreachable. See orchestration/mlflow_tracking.py for the guard.
    """

    enabled: bool = False
    tracking_uri: str = field(default_factory=_get_mlflow_tracking_uri)
    experiment_name: str = "floorplan-annotation-pipeline"


@dataclass
class PipelineConfig:
    """Master configuration for the full annotation pipeline.

    Supports Solution C (Hybrid) architecture:
    - Step 2: PaddleOCR for text detection/recognition
    - Step 3: Qwen2.5-VL or Claude for room detection/classification
    - New: SemanticReconciler merges OCR text with VLM room polygons
    """

    pdf: PDFConfig = field(default_factory=PDFConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    template: TemplateConfig = field(default_factory=TemplateConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    sam: SAMConfig = field(default_factory=SAMConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    yolo_objects: YoloObjectDetectorConfig = field(
        default_factory=YoloObjectDetectorConfig
    )
    sam3_exemplar: Sam3ExemplarDetectorConfig = field(
        default_factory=Sam3ExemplarDetectorConfig
    )
    cubicasa_rooms: CubicasaRoomDetectorConfig = field(
        default_factory=CubicasaRoomDetectorConfig
    )
    confidence_weights: ConfidenceWeightsConfig = field(
        default_factory=ConfidenceWeightsConfig
    )
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)

    # Optional: ground-truth annotations dir (same format as this pipeline's
    # own annotations/ output). If set, run() calls bbox_metrics.evaluate_dataset
    # after the run completes and logs precision/recall/IoU-family metrics.
    # Not the same as `eval` above (EvalConfig is test_pcs-specific and has no
    # live callers today) -- this is a general-purpose, opt-in GT comparison.
    eval_gt_dir: Optional[str] = None

    # Pipeline options
    # NOTE: use_vlm defaults to False. Use OCR results unless explicitly enabled.
    # This ensures annotations are saved even if VLM API is unavailable.
    use_vlm: bool = False
    use_sam: bool = False
    use_template_matching: bool = False

    # Semantic reconciliation: merge OCR text with VLM room detections (Solution C).
    # Requires use_vlm=True. Enriches VLM rooms with OCR-derived room names/numbers
    # and provides a spatial match signal used by the SFT validator to reject
    # VLM rooms with no OCR corroboration (e.g., BOM/margin detections).
    use_semantic_reconciliation: bool = True

    # Window detection: Tier 1 (PDF layers) + Tier 2 (CubiCasa5K, not yet active)
    # + Tier 3 (VLM prompt). Default False: Tier 2 is not implemented and Tier 3
    # adds a 300 s timeout per image with no quality gain until Tier 2 is ready.
    use_windows: bool = False

    # T-I: fine-tuned YOLO door/window detector (yolo_objects config
    # above). Default False -- same off-by-default precedent as
    # use_windows; flipping this on is the only way pipeline.py's
    # execution path changes at all (T-I regression guard).
    use_yolo_objects: bool = False

    # Tech-eval plan P2/P3: tiled SAM3 box-exemplar door detector
    # (sam3_exemplar config above), seeded by use_yolo_objects's own
    # detections -- has no effect unless use_yolo_objects is also True
    # (no seed source otherwise). Default False, same off-by-default
    # precedent as use_windows/use_yolo_objects: two-dataset validation
    # (P0/P1) beat the YOLO-alone baseline on both P and R, but only at
    # the scale of that validation (test_pcs 10 img, FloorPlanCAD 40
    # img) -- ship default-off, flip only on an explicit go-ahead per
    # the plan's P3 guarded-rollout step.
    use_sam3_exemplar: bool = False

    # Prototype (2026-08-20 audit): recover room labels from a PDF's
    # embedded text layer, additive-only against OCR's own detections
    # (OCR wins every overlap -- see pipeline.py's merge site). Confirmed
    # real on Lake Shore Electrical p002 ('JANITOR' in the PDF text layer,
    # OCR found nothing at the same image location) but NOT a general fix:
    # 4 of 13 project source PDFs carry substantial room-label text (the
    # Electrical-series PDFs); "...FLAT" PDFs and others have zero
    # extractable words (text outlined to curves) -- inert there, not
    # broken.
    #
    # DOWNSTREAM NOISE: MEASURED, SEVERE, NOT JUST THEORETICAL (same day).
    # Merging the PDF text layer into raw_detections on Lake Shore
    # Electrical p002 took that list from 133 to 828 entries (+695) and
    # compute_exclusion_zones's output from 3 zones to 16. Visually
    # verified the 13 new zones: they cover CLASSROOM 142/143/144/146/147,
    # SCIENCE CLASSROOM 128/129/130, PREP ROOM/CLASSROOM 145/COMPUTER LAB,
    # and CLASSROOM 119 -- i.e. most of the real rooms on the page, not
    # BOM tables. sft_validator.py drops any room whose centroid falls in
    # an exclusion zone -- flipping this flag on AS CURRENTLY BUILT would
    # delete legitimate rooms, not just add noise. Root cause: PDF text
    # layers carry small MEP/circuit/dimension tags scattered densely
    # THROUGHOUT room interiors that OCR never surfaced (too small/faint,
    # or below confidence_threshold) but that still match
    # compute_exclusion_zones's excluded-token regexes (dates, equipment
    # codes, generic instruction words) -- the clustering has no way to
    # tell "dense MEP tags inside a real room" from "a real title block".
    #
    # ROUTING FIXED (Y1, same day): the exclusion-zone hazard above was
    # caused by the ORIGINAL merge point (into raw_detections, which
    # feeds compute_exclusion_zones). Corrected: pipeline.py's
    # _pdf_text_layer_detections now only FETCHES PDF detections;
    # MEPTextExtractor.extract_and_find_rooms(extra_detections=...) merges
    # them ONLY into find_room_candidates's input, and its own
    # raw_detections return value stays OCR-only. Verified end-to-end on
    # the real page: exclusion zones stayed at 3 (not 16).
    #
    # BUT TWO NEW, MEASURED PROBLEMS SURFACED FIXING THAT ONE -- still
    # NOT ready to enable:
    #
    # (1) The motivating case (JANITOR) is STILL not recovered, for a
    # DIFFERENT reason than the routing bug. OCR did not miss that
    # region -- it produced a garbled low-confidence read of the same
    # text ('iiAANTTI ii', conf 0.513, centroid 3px from the PDF
    # detection's). The "OCR wins every collision regardless of text
    # content" rule (matching _dedupe_detections's own established
    # convention) treats that garbled read as "OCR already found
    # something here" and correctly-by-design, but wrongly-in-outcome,
    # blocks the accurate PDF text. Not fixed; flagged, not guessed at --
    # any fix here needs its own evidence (e.g. comparing OCR confidence
    # or text plausibility) before implementing.
    #
    # (2) Candidate-quality collapse. On the same page, extra_detections
    # took find_room_candidates's output from 14 to 201 entries -- 185
    # new, but only 6 DISTINCT names among them ('K', 'LV', 'OFF', 'k',
    # 'of', and exactly ONE real room: 'PREP C114'). The other ~184 are
    # repeated MEP circuit/wire-gauge tags the PDF's raw text layer
    # carries at high density; find_room_candidates was built against
    # OCR's naturally sparser output and has no filter tuned for this
    # input's character. One real recovery per ~185 junk entries is not
    # a usable ratio.
    #
    # Default off, same precedent as use_windows/use_yolo_objects/
    # use_sam3_exemplar. Do NOT flip this on. The routing bug is fixed;
    # the feature is further from ready than it looked before that fix,
    # not closer -- promoting past Prototype now needs a real
    # room-name-plausibility filter for PDF-sourced text specifically,
    # evidenced on more than one page, not a threshold guess.
    use_pdf_text_layer: bool = False

    # CubicasaRoomDetectorConfig above. Currently READ BY NOTHING in
    # pipeline.py -- its one caller (Step 4d.5, CNN room corroboration) was
    # removed after evidence found no discriminative value (see that
    # config's docstring). Kept as a placeholder for a future consumer of
    # CubiCasa5KDetector.detect_rooms() so that consumer can reuse this
    # same off-by-default flag rather than adding a new one; flipping it
    # on today is a no-op.
    use_cubicasa_rooms: bool = False

    # BA3 (2026-08-03): drop objectDetections tagged "window" whose bbox
    # fails WindowEvidenceConfig's elongation predicate (window_evidence.
    # is_window_shaped) -- catches door-swing symbols the detector
    # mislabels window (near-square boxes; real windows, portrait or
    # landscape, are elongated). Verified on 10 pixel-confirmed real
    # ROCKAWAY boxes (10/10) + test_pcs/kaggle GT (both precision up, at
    # most 1 TP lost per set, zero on ROCKAWAY).
    #
    # Re-measured 2026-08-30 (T-0 spike, sprint1_verify45's full
    # low-confidence <0.75 "window" population, 85 boxes across 18
    # files/6 drawing sets, manually pixel-classified): catches 25/40
    # confirmed door/other false "window"s (62%), wrongly drops 2/40 real
    # windows (5% -- both on 326 ROCKAWAY, same recurring x2230-2250/
    # y777-798 sliver next to an already-kept wider window box on that
    # wall run, not an independently-lost opening). Clears the >=50%
    # FP-recovery / <5%-ish TP-cost bar this flag was gated on. Residual
    # 38% of FPs are concentrated off the ROCKAWAY drawing family
    # (Kennedy/Violet Elementary Electrical Pages, Bradley Fair) where
    # door-swing bboxes are themselves elongated (2.0-4.1) -- this
    # predicate does not generalize to that CD style; still net-positive
    # everywhere measured, never worse than the prior always-keep
    # behavior. Default promoted to True on this evidence -- same
    # promotion precedent as YoloObjectDetectorConfig.use_tiling.
    use_window_elongation_filter: bool = True

    # A1 (2026-08-21, Prototype First): exempt sam_expanded rooms from
    # FIX-5's ink-density filter (INK_MIN_FRAC=0.09). Measured on 3 pages /
    # 7 visually-verified sam_expanded boxes: ink density has NO
    # discriminative power for this population -- verified-correct boxes
    # measured 0.0053-0.31, verified-wrong boxes measured 0.0869-0.12, and
    # the wrong ones sit inside the correct range. FIX-5 was tuned for
    # label-scale boxes (near-solid glyph ink); a room-scale box over open
    # floor space is mostly white by definition, so the same 0.09 cut
    # conflates "correctly room-scale" with "wrongly placed." Non-expanded
    # rooms are FIX-5's real, measured-effective use (11/15 low-ink drops
    # on ROCKAWAY p001 were genuinely blank boxes, mostly ink=0.000) and
    # are untouched by this flag either way. Default False pending the
    # validation run (re-check room-scale shipped-box counts + visually
    # verify every newly-admitted box across the same 3 pages) before any
    # promotion past Prototype.
    exempt_sam_expanded_from_ink_filter: bool = False

    # Minimum rooms required to mark an image sft_ready=True.
    # Default 1: any image with ≥1 valid room is included.
    # Images with ≥3 rooms also get sft_recommended=True for higher-quality batches.
    min_rooms_for_sft: int = 1

    # Tiled VLM inference: splits large images so each tile has higher pixel
    # density per room, improving spatial grounding.
    # Activated only when max(img_w, img_h) > tile_trigger_px.
    # 25-room cap and hallucination detection run AFTER tile merge, not per-tile.
    use_tiling: bool = True
    tile_cols: int = 4          # max columns (adaptive grid is capped here)
    tile_rows: int = 4          # max rows
    tile_overlap_pct: float = 0.10
    tile_trigger_px: int = 3000
    # Adaptive tiling: tile count derived from image size so each tile is about
    # tile_target_px wide/tall. Dense floors (e.g. 4500px residential) get a
    # finer grid (≈4x3) than the fixed 2x2, improving VLM room localization.
    # 0 disables adaptive sizing (falls back to fixed tile_cols x tile_rows).
    tile_target_px: int = 1200

    # Parallel processing
    num_workers: int = 4

    @classmethod
    def for_high_detail(cls) -> "PipelineConfig":
        """Factory for high-detail extraction (small symbols, dense plans)."""
        config = cls()
        config.pdf.dpi = 300
        config.ocr.preprocess = True
        config.template.scales = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
        return config

    @classmethod
    def for_fast_processing(cls) -> "PipelineConfig":
        """Factory for fast processing (large batches, lower quality OK)."""
        config = cls()
        config.pdf.dpi = 150
        config.ocr.preprocess = False
        config.template.scales = [0.8, 1.0, 1.2]
        config.template.rotations = [0]
        return config
