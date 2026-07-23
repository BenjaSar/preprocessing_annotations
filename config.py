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
from typing import Dict, List, Optional

try:
    import torch
except ImportError:
    torch = None  # Optional dependency


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
            r"^OUTDOOR\s+STORAGE$",
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
    checkpoint: str = "sam_vit_h_4b8939.pth"

    # Device for inference (auto-detected if None)
    device: Optional[str] = None

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
    """

    # Default points at the repo's own test_pcs export (real asset, same
    # convention as CubiCasa5KDetector.MODEL_PATH's repo-relative default).
    # coco/images/ is empty on disk; coco_w_images/images/ holds the pngs —
    # both result.json files are byte-identical exports, so coco/result.json
    # is read for annotations and coco_w_images/images/ for pixels.
    gt_coco_path: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "test_pcs" / "coco" / "result.json"
    )
    gt_images_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "test_pcs" / "coco_w_images" / "images"
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
