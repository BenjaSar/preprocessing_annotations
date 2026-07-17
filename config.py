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
from typing import List, Optional

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
    # Options: "qwen2.5-vl-7b", "qwen3-vl-2b", "qwen3-vl-4b", "qwen3-vl-8b" (default)
    # qwen3-vl-8b has materially better spatial grounding on dense floor plans.
    # Requires ~8-10GB VRAM at 4-bit. Override via --unsloth-model CLI flag.
    unsloth_model: str = "qwen3-vl-8b"

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
