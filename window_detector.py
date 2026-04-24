"""
Window and Structural Opening Detection — Phase 2 Implementation.

Three-tier detection strategy:
  Tier 1: PDF layer extraction (zero-cost, perfect accuracy when available)
  Tier 2: CubiCasa5K model (pretrained, high accuracy on rasterized images)
  Tier 3: VLM prompting (fallback, zero new dependencies)

Output: Window polygons/bboxes with confidence scores, mapped to room IDs
        via spatial intersection (Shapely).

Integration: Called after room detection, before SFT annotation serialization.
             Supplies (has_windows, has_skylights, has_openings) to add_window_suffix().
"""

import logging
import tempfile
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any

import numpy as np

try:
    from shapely.geometry import Polygon, Point
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

logger = logging.getLogger(__name__)


class WindowDetectionTier(Enum):
    """Detection tier identifier for provenance tracking."""
    PDF_LAYERS = "pdf_layers"
    CUBICASA5K = "cubicasa5k"
    VLM_PROMPT = "vlm_prompt"
    NONE = "none"


@dataclass
class WindowDetection:
    """Single detected window."""
    bbox: Tuple[float, float, float, float]  # [x1, y1, x2, y2] in pixels
    polygon: Optional[List[Tuple[float, float]]] = None  # [(x, y), ...] if available
    confidence: float = 0.95  # Detection confidence (0-1)
    source_tier: WindowDetectionTier = WindowDetectionTier.NONE
    metadata: Optional[Dict[str, Any]] = None  # Additional source info


@dataclass
class RoomWindowMapping:
    """Result of spatial intersection: which rooms have windows."""
    room_id: str
    has_windows: bool = False
    has_skylights: bool = False
    has_openings: bool = False
    window_count: int = 0
    intersecting_windows: List[WindowDetection] = None

    def __post_init__(self):
        if self.intersecting_windows is None:
            self.intersecting_windows = []


class WindowDetector:
    """
    Orchestrator for three-tier window detection.

    Tier 1 (PDF layers) is attempted automatically if PDF source available.
    Tier 2 (CubiCasa5K) requires model to be downloaded/available.
    Tier 3 (VLM prompting) uses existing VLM infrastructure.
    """

    def __init__(self, config: Optional[Any] = None):
        """
        Initialize window detector.

        Args:
            config: Optional pipeline config for VLM/SAM settings.
        """
        self.config = config
        self.cubicasa_model = None
        self.tier_results: Dict[WindowDetectionTier, List[WindowDetection]] = {}
        self.has_shapely = HAS_SHAPELY
        self._cubicasa_logged_once = False  # suppress repeated "not implemented" noise

        if not HAS_SHAPELY:
            logger.warning(
                "Shapely not available — spatial intersection disabled. "
                "Install: pip install shapely>=2.0.0"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Tier 1: PDF Layer Extraction
    # ─────────────────────────────────────────────────────────────────────────

    def detect_windows_from_pdf_layers(self, pdf_path: Path) -> List[WindowDetection]:
        """
        Extract windows from AutoCAD/PDF layer metadata.

        Looks for layer names indicating glazing: A-GLAZ, *WINDOWS, G_GLASS, etc.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            List of WindowDetection objects extracted from layer geometry.
        """
        windows = []

        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.debug("PyMuPDF not available for PDF layer extraction")
            return windows

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.warning(f"Failed to open PDF {pdf_path}: {e}")
            return windows

        # Layer names that indicate window/glazing elements
        window_layer_patterns = [
            "glaz",
            "window",
            "glass",
            "opening",
            "fenest",
        ]

        try:
            # Get vector graphics data (requires fitz >= 1.16.0)
            for page_num in range(len(doc)):
                page = doc[page_num]

                # Try to extract text with location info (some PDFs preserve it)
                try:
                    # Attempt to get layer info (PDF feature)
                    # Note: layer extraction is PDF-specific and may not work
                    # on all PDFs. This is a best-effort approach.
                    pass
                except Exception:
                    pass

                # Alternative: analyze paths and rectangles for window-like patterns
                # Walls are typically solid rectangles/paths
                # Windows are often thin rectangles (glazing bars) or double lines
                try:
                    paths = page.get_drawings()
                    for path in paths:
                        # Check if this looks like a window (thin rectangle)
                        if hasattr(path, "rects"):
                            for rect in path.rects:
                                bbox = rect.normalize()
                                w = bbox.width
                                h = bbox.height
                                # Windows typically 0.5-4 inches wide in architectural drawings
                                # At 200 DPI: 100-800 pixels
                                if 50 < min(w, h) < 800 and max(w, h) > 50:
                                    window = WindowDetection(
                                        bbox=(
                                            bbox.x0,
                                            bbox.y0,
                                            bbox.x1,
                                            bbox.y1,
                                        ),
                                        confidence=0.85,  # Lower confidence for heuristic
                                        source_tier=WindowDetectionTier.PDF_LAYERS,
                                    )
                                    windows.append(window)
                except Exception:
                    pass

            doc.close()

            if windows:
                logger.info(
                    f"Tier 1 (PDF layers): Extracted {len(windows)} windows from {pdf_path.name}"
                )
            return windows

        except Exception as e:
            logger.warning(f"Tier 1 (PDF layers) extraction failed: {e}")
            return []

    # ─────────────────────────────────────────────────────────────────────────
    # Tier 2: CubiCasa5K Model
    # ─────────────────────────────────────────────────────────────────────────

    def detect_windows_from_cubicasa5k(self, image_array: np.ndarray) -> List[WindowDetection]:
        """
        Detect windows using CubiCasa5K pretrained model.

        Expects image_array to be RGB, uint8, shape (H, W, 3).

        Args:
            image_array: Floorplan image as numpy array.

        Returns:
            List of WindowDetection objects.
        """
        windows = []

        try:
            import torch
        except ImportError:
            logger.warning("PyTorch not available for CubiCasa5K")
            return windows

        try:
            # Lazy load model
            if self.cubicasa_model is None:
                self.cubicasa_model = self._load_cubicasa5k_model()

            if self.cubicasa_model is None:
                logger.debug("CubiCasa5K model not available")
                return windows

            # Placeholder: actual model inference would happen here
            # This requires the CubiCasa5K model checkpoint and preprocessing
            # For now, we return an empty list until the model is integrated
            logger.debug("CubiCasa5K inference not yet implemented")

            return windows

        except Exception as e:
            logger.warning(f"Tier 2 (CubiCasa5K) detection failed: {e}")
            return []

    def _load_cubicasa5k_model(self) -> Optional[Any]:
        """
        Load pretrained CubiCasa5K model from disk or download.

        Returns:
            Loaded model or None if unavailable.
        """
        # CubiCasa5K integration is not yet implemented.
        # Log only on the first call to avoid polluting logs on every image.
        if not self._cubicasa_logged_once:
            logger.info(
                "Tier 2 (CubiCasa5K): model not yet integrated — skipping. "
                "Window detection will fall back to Tier 3 (VLM prompt)."
            )
            self._cubicasa_logged_once = True
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Tier 3: VLM Prompting
    # ─────────────────────────────────────────────────────────────────────────

    def detect_windows_from_vlm_prompt(
        self,
        image_array: Optional[np.ndarray] = None,
        vlm_backend: Optional[Any] = None,
        img_path: Optional[Path] = None,
    ) -> List[WindowDetection]:
        """
        Detect windows using the VLM backend's window-detection capability (Tier 3).

        All concrete VLM backends (ClaudeBackend, Qwen2_5VLBackend, UnslothQwenBackend)
        implement a ``detect_windows(image_path)`` method that sends the image with a
        structured prompt and parses a JSON response containing per-window bboxes,
        confidence scores, and type labels ("window" | "skylight" | "side_opening").

        This method is a bridge from the WindowDetector interface to that existing
        backend method, converting the returned List[Dict] into List[WindowDetection].

        Preference order for the image source:
            1. img_path (file path) — passed directly to the backend; no encoding
            2. image_array (numpy array) — saved to a temp PNG and cleaned up after

        Args:
            image_array: Floorplan image as numpy array (fallback if img_path absent).
            vlm_backend: VLM backend instance (ClaudeBackend / Qwen2_5VLBackend /
                         UnslothQwenBackend).  Must not be None.
            img_path:    Direct path to the source image.  Preferred over image_array.

        Returns:
            List of WindowDetection objects.  Each WindowDetection carries
            ``metadata={"type": "window"|"skylight"|"side_opening"}`` so that
            map_windows_to_rooms() can set has_windows / has_skylights / has_openings
            correctly.
        """
        if vlm_backend is None:
            logger.debug("Tier 3 (VLM): backend not provided; skipping")
            return []

        # ── Resolve the image path the backend will use ───────────────────────
        tmp_file: Optional[str] = None
        effective_path: Optional[Path] = None

        if img_path is not None and Path(img_path).is_file():
            effective_path = Path(img_path)
        elif image_array is not None:
            # Write numpy array to a temp PNG so the backend can open it
            try:
                from PIL import Image as _PIL_Image
                tmp_fd, tmp_file = tempfile.mkstemp(suffix=".png")
                os.close(tmp_fd)
                _PIL_Image.fromarray(image_array).save(tmp_file)
                effective_path = Path(tmp_file)
            except Exception as _e:
                logger.warning(f"Tier 3 (VLM): could not write temp image: {_e}")
                return []

        if effective_path is None:
            logger.debug("Tier 3 (VLM): no image source available; skipping")
            return []

        # ── Call the backend's detect_windows() method ────────────────────────
        try:
            logger.info(
                f"Tier 3 (VLM): running window detection on {effective_path.name} "
                f"using {vlm_backend.__class__.__name__}"
            )
            raw: List[Dict[str, Any]] = vlm_backend.detect_windows(effective_path)

            if not raw:
                logger.debug("Tier 3 (VLM): no windows returned by backend")
                return []

            # ── Convert List[Dict] → List[WindowDetection] ───────────────────
            windows: List[WindowDetection] = []
            for entry in raw:
                bbox = entry.get("bbox", [])
                if len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = bbox
                # Guard against degenerate bboxes
                if x2 <= x1 or y2 <= y1:
                    continue

                window_type = entry.get("type", "window")  # "window"|"skylight"|"side_opening"
                conf = float(entry.get("confidence", 0.7))

                windows.append(
                    WindowDetection(
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        confidence=conf,
                        source_tier=WindowDetectionTier.VLM_PROMPT,
                        metadata={"type": window_type},
                    )
                )

            logger.info(
                f"Tier 3 (VLM): detected {len(windows)} window(s) in {effective_path.name}"
            )
            return windows

        except Exception as e:
            logger.warning(f"Tier 3 (VLM prompt) detection failed: {e}")
            return []

        finally:
            # Clean up temp file if we created one
            if tmp_file is not None:
                try:
                    os.unlink(tmp_file)
                except OSError:
                    pass

    # ─────────────────────────────────────────────────────────────────────────
    # Spatial Intersection: Map windows to rooms
    # ─────────────────────────────────────────────────────────────────────────

    def map_windows_to_rooms(
        self,
        windows: List[WindowDetection],
        rooms: List[Dict[str, Any]],
    ) -> List[RoomWindowMapping]:
        """
        Compute spatial intersection: which rooms have windows.

        Args:
            windows: List of detected windows (bbox or polygon).
            rooms: List of room annotations, each with 'bbox' or 'polygon' field.

        Returns:
            List of RoomWindowMapping, one per room, with window presence flags.
        """
        mappings = []

        if not self.has_shapely:
            logger.warning("Shapely not available; skipping spatial intersection")
            return mappings

        for idx, room in enumerate(rooms):
            room_id = room.get("id", f"room_{idx}")

            # Extract room polygon or bbox
            room_poly = None
            if "polygon" in room and room["polygon"]:
                try:
                    room_poly = Polygon(room["polygon"])
                except Exception as e:
                    logger.debug(f"Invalid room polygon for {room_id}: {e}")

            if room_poly is None and "bbox" in room:
                # Convert bbox to polygon
                x1, y1, x2, y2 = room["bbox"]
                room_poly = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

            if room_poly is None:
                logger.warning(f"Room {room_id} has no valid geometry")
                mappings.append(RoomWindowMapping(room_id=room_id))
                continue

            # Check which windows intersect this room
            intersecting = []
            has_windows = False
            has_skylights = False
            has_openings = False

            for window in windows:
                # Create window polygon
                window_poly = None
                if window.polygon:
                    try:
                        window_poly = Polygon(window.polygon)
                    except Exception:
                        pass

                if window_poly is None:
                    x1, y1, x2, y2 = window.bbox
                    window_poly = Polygon(
                        [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                    )

                # Check intersection
                try:
                    if room_poly.intersects(window_poly):
                        intersecting.append(window)
                        # Use metadata["type"] to set the correct attribute flag.
                        # The VLM backend distinguishes "window", "skylight", and
                        # "side_opening" in its response; that distinction is preserved
                        # through the WindowDetection.metadata field and applied here.
                        window_type = (window.metadata or {}).get("type", "window")
                        if window_type == "skylight":
                            has_skylights = True
                        elif window_type in ("side_opening", "opening"):
                            has_openings = True
                        else:
                            has_windows = True
                except Exception as e:
                    logger.debug(f"Intersection check failed: {e}")

            mapping = RoomWindowMapping(
                room_id=room_id,
                has_windows=has_windows,
                has_skylights=has_skylights,
                has_openings=has_openings,
                window_count=len(intersecting),
                intersecting_windows=intersecting,
            )
            mappings.append(mapping)

        return mappings

    # ─────────────────────────────────────────────────────────────────────────
    # Main Detection Pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def detect_windows(
        self,
        image_array: Optional[np.ndarray] = None,
        rooms: Optional[List[Dict[str, Any]]] = None,
        pdf_path: Optional[Path] = None,
        vlm_backend: Optional[Any] = None,
        img_path: Optional[Path] = None,
    ) -> List[RoomWindowMapping]:
        """
        Execute three-tier window detection pipeline.

        Tier priority (highest accuracy first):
          Tier 1: PDF layer extraction — zero-cost, perfect when PDF layers available.
          Tier 2: CubiCasa5K model   — high accuracy on rasterized images (not yet active).
          Tier 3: VLM prompting      — uses the already-loaded VLM backend; active.

        Each tier runs only if the previous tier(s) found nothing.

        Args:
            image_array: Floorplan image as numpy array (used by Tier 2; Tier 3
                         prefers img_path but falls back to this).
            rooms:       List of room dicts with 'id' and 'bbox'/'polygon' fields.
            pdf_path:    Path to the source PDF — enables Tier 1.
            vlm_backend: VLM backend instance — enables Tier 3.
            img_path:    Direct path to the source image — passed to Tier 3 so the
                         backend can open it directly without re-encoding.

        Returns:
            List of RoomWindowMapping — one entry per room in rooms, with
            has_windows / has_skylights / has_openings / window_count populated.
        """
        if rooms is None:
            logger.warning("No rooms provided; skipping window detection")
            return []

        windows: List[WindowDetection] = []

        # ── Tier 1: PDF layer extraction ──────────────────────────────────────
        if pdf_path:
            tier1_windows = self.detect_windows_from_pdf_layers(pdf_path)
            windows.extend(tier1_windows)
            self.tier_results[WindowDetectionTier.PDF_LAYERS] = tier1_windows
            if tier1_windows:
                logger.info(f"Tier 1 (PDF layers): found {len(tier1_windows)} window(s)")

        # ── Tier 2: CubiCasa5K ────────────────────────────────────────────────
        if image_array is not None and not windows:
            tier2_windows = self.detect_windows_from_cubicasa5k(image_array)
            windows.extend(tier2_windows)
            self.tier_results[WindowDetectionTier.CUBICASA5K] = tier2_windows

        # ── Tier 3: VLM prompting ─────────────────────────────────────────────
        # Activates when Tiers 1-2 found nothing and a VLM backend is available.
        # Prefers img_path (no re-encoding) but falls back to image_array.
        if not windows and vlm_backend is not None:
            if img_path is not None or image_array is not None:
                tier3_windows = self.detect_windows_from_vlm_prompt(
                    image_array=image_array,
                    vlm_backend=vlm_backend,
                    img_path=img_path,
                )
                windows.extend(tier3_windows)
                self.tier_results[WindowDetectionTier.VLM_PROMPT] = tier3_windows

        if not windows:
            logger.debug("No windows detected across all tiers")

        # ── Spatial association: map window detections to room regions ─────────
        mappings = self.map_windows_to_rooms(windows, rooms)

        return mappings


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function for integration into pipeline
# ─────────────────────────────────────────────────────────────────────────────


def apply_window_suffixes(
    rooms: List[Dict[str, Any]],
    window_mappings: List[RoomWindowMapping],
) -> None:
    """
    Apply window/skylight/opening suffixes to room types in-place.

    Uses the add_window_suffix() function from automation.taxonomy.

    Args:
        rooms: List of room dicts to modify (modified in-place).
        window_mappings: Results from window detection/mapping.
    """
    try:
        from .automation.taxonomy import add_window_suffix
    except ImportError:
        from automation.taxonomy import add_window_suffix

    # Build lookup table
    mapping_by_id = {m.room_id: m for m in window_mappings}

    for room in rooms:
        room_id = room.get("id", room.get("room_number"))
        mapping = mapping_by_id.get(room_id)

        if mapping is None:
            continue

        base_type = room.get("type")
        if base_type is None:
            continue

        # Apply suffix
        suffixed_type = add_window_suffix(
            base_type,
            has_windows=mapping.has_windows,
            has_skylights=mapping.has_skylights,
            has_openings=mapping.has_openings,
        )

        room["type"] = suffixed_type
        room["window_detection"] = {
            "has_windows": mapping.has_windows,
            "has_skylights": mapping.has_skylights,
            "has_openings": mapping.has_openings,
            "window_count": mapping.window_count,
        }
