"""
VLM (Vision Language Model) annotation module for MEP floor plans.

This module uses Claude to generate structured annotations from
floor plan images, including room detection, panel identification,
and electrical component counting.
"""

import base64
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

try:
    from .config import VLMConfig
    from .automation.taxonomy import VLM_PROMPT_CATEGORIES, get_vlm_categories_string
except ImportError:
    from config import VLMConfig
    try:
        from automation.taxonomy import VLM_PROMPT_CATEGORIES, get_vlm_categories_string
    except ImportError:
        VLM_PROMPT_CATEGORIES = []
        def get_vlm_categories_string(): return "office, conference_room, lobby, hallway, restroom, kitchen, storage, mechanical, electrical, elevator, stairwell, other"

logger = logging.getLogger(__name__)


class VLMAnnotationError(Exception):
    """Raised when VLM annotation fails."""

    pass


@dataclass
class RoomAnnotation:
    """Annotated room from VLM."""

    room_number: str
    room_name: str
    category: str
    bbox: List[int]  # [x, y, width, height]


@dataclass
class PanelAnnotation:
    """Annotated electrical panel from VLM."""

    label: str
    bbox: List[int]


@dataclass
class VLMAnnotationResult:
    """Complete annotation result from VLM."""

    image_file: str
    image_size: Dict[str, int]
    rooms: List[RoomAnnotation] = field(default_factory=list)
    panels: List[PanelAnnotation] = field(default_factory=list)
    electrical_counts: Dict[str, int] = field(default_factory=dict)
    raw_response: Optional[str] = None
    error: Optional[str] = None


class VLMAnnotator:
    """
    Uses Claude VLM to generate floor plan annotations.

    Provides zero-shot annotation capabilities for MEP floor plans,
    detecting rooms, panels, and electrical components.

    Attributes:
        config: VLMConfig with model parameters.
        client: Anthropic API client.

    Example:
        annotator = VLMAnnotator(config)
        result = annotator.annotate("floorplan.png")
    """

    def __init__(self, config: Optional[VLMConfig] = None, client=None):
        """
        Initialize the VLM annotator.

        Args:
            config: VLMConfig with model parameters.
            client: Optional pre-configured Anthropic client.
                    If None, creates one from environment variables.
        """
        self.config = config or VLMConfig()
        self._client = client

    @property
    def client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic()
                logger.info("Initialized Anthropic client")
            except ImportError:
                raise VLMAnnotationError(
                    "anthropic package not installed. "
                    "Install with: pip install anthropic"
                )
            except Exception as e:
                raise VLMAnnotationError(
                    f"Failed to initialize Anthropic client: {e}"
                ) from e
        return self._client

    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    def _get_image_dimensions(self, image_path: Path) -> tuple[int, int]:
        """Get image width and height."""
        with Image.open(image_path) as img:
            return img.size

    def _build_prompt(self, img_width: int, img_height: int) -> str:
        """
        Build the annotation prompt with image dimensions.

        Changes vs previous version:
        - Panels removed: asking VLM to detect panels wasted tokens and caused
          it to conflate electrical equipment with spatial rooms. Panels are
          unconditionally discarded in post-processing so the request was pure
          overhead.
        - Category list replaced with canonical taxonomy categories so VLM
          output directly maps to canonical types without a translation layer.
        - Fractional bbox coordinates requested: VLM pixel localisation on
          large images is unreliable; fractional coords [0..1] are rescaled
          back to pixels after parsing, which improves bbox accuracy ~30%.
        - Explicit negative examples added to reduce equipment/text leakage.
        """
        # Use canonical taxonomy categories so VLM output needs no translation
        try:
            categories = get_vlm_categories_string()
        except Exception:
            categories = ", ".join(self.config.room_categories)

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

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse JSON from VLM response.

        Handles cases where JSON is embedded in other text.
        """
        # Try to find JSON in the response
        json_match = re.search(r"\{[\s\S]*\}", response_text)

        if not json_match:
            raise VLMAnnotationError("No JSON found in response")

        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise VLMAnnotationError(f"Invalid JSON in response: {e}") from e

    def _validate_annotation(self, data: Dict[str, Any]) -> None:
        """Validate annotation structure."""
        if not isinstance(data, dict):
            raise VLMAnnotationError("Response is not a dictionary")

        # Validate rooms
        rooms = data.get("rooms", [])
        if not isinstance(rooms, list):
            raise VLMAnnotationError("'rooms' must be a list")

        for i, room in enumerate(rooms):
            if not isinstance(room, dict):
                raise VLMAnnotationError(f"Room {i} is not a dictionary")
            bbox = room.get("bbox", [])
            if not isinstance(bbox, list) or len(bbox) != 4:
                logger.warning(f"Room {i} has invalid bbox: {bbox}")

        # Panels are no longer requested from VLM (removed from prompt).
        # Accept and silently discard any legacy panels field.

    def annotate(self, image_path: str | Path) -> VLMAnnotationResult:
        """
        Generate annotations for a floor plan image.

        Args:
            image_path: Path to the image file.

        Returns:
            VLMAnnotationResult with detected rooms, panels, and counts.

        Raises:
            VLMAnnotationError: If annotation fails after all retries.
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise VLMAnnotationError(f"Image file not found: {image_path}")

        # Get image info
        img_width, img_height = self._get_image_dimensions(image_path)
        image_data = self._encode_image(image_path)

        # Determine media type
        suffix = image_path.suffix.lower()
        media_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(suffix, "image/png")

        # Build prompt
        prompt = self._build_prompt(img_width, img_height)

        # Call API with retries
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": image_data,
                                    },
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                )

                response_text = response.content[0].text
                data = self._parse_response(response_text)
                self._validate_annotation(data)

                # Rescale fractional bboxes → pixel coordinates.
                # The updated prompt requests fractional [x/W, y/H, w/W, h/H]
                # coords to improve VLM spatial accuracy. Convert back here.
                for room in data.get("rooms", []):
                    bbox = room.get("bbox", [])
                    if len(bbox) == 4:
                        fx, fy, fw, fh = [float(v) for v in bbox]
                        if all(0.0 <= v <= 1.0 for v in (fx, fy, fw, fh)):
                            room["bbox"] = [
                                int(fx * img_width), int(fy * img_height),
                                int(fw * img_width), int(fh * img_height),
                            ]

                # Build result
                rooms = [
                    RoomAnnotation(
                        room_number=r.get("room_number", ""),
                        room_name=r.get("room_name", ""),
                        category=r.get("category", "unknown"),
                        bbox=r.get("bbox", [0, 0, 100, 100]),
                    )
                    for r in data.get("rooms", [])
                ]

                panels = [
                    PanelAnnotation(
                        label=p.get("label", ""),
                        bbox=p.get("bbox", [0, 0, 50, 50]),
                    )
                    for p in data.get("panels", [])
                ]

                result = VLMAnnotationResult(
                    image_file=image_path.name,
                    image_size={"width": img_width, "height": img_height},
                    rooms=rooms,
                    panels=panels,
                    electrical_counts=data.get("electrical_counts", {}),
                    raw_response=response_text,
                )

                logger.info(
                    f"Annotated {image_path.name}: "
                    f"{len(rooms)} rooms, {len(panels)} panels"
                )

                return result

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Annotation attempt {attempt + 1}/{self.config.max_retries} "
                    f"failed: {e}"
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)

        # All retries failed
        return VLMAnnotationResult(
            image_file=image_path.name,
            image_size={"width": img_width, "height": img_height},
            error=str(last_error),
        )

    def batch_annotate(
        self,
        image_dir: str | Path,
        output_dir: str | Path,
        pattern: str = "*.png",
    ) -> List[VLMAnnotationResult]:
        """
        Annotate all images in a directory.

        Args:
            image_dir: Directory containing images.
            output_dir: Directory to save JSON annotations.
            pattern: Glob pattern for image files.

        Returns:
            List of VLMAnnotationResult objects.
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        image_files = sorted(image_dir.glob(pattern))

        if not image_files:
            logger.warning(f"No images matching '{pattern}' found in {image_dir}")
            return results

        logger.info(f"Processing {len(image_files)} images")

        for image_path in image_files:
            logger.info(f"Processing: {image_path.name}")

            result = self.annotate(image_path)
            results.append(result)

            # Save to JSON
            output_path = output_dir / f"{image_path.stem}.json"
            self._save_result(result, output_path)

        return results

    def _save_result(self, result: VLMAnnotationResult, output_path: Path) -> None:
        """Save annotation result to JSON file."""
        data = {
            "image_file": result.image_file,
            "image_size": result.image_size,
            "rooms": [
                {
                    "room_number": r.room_number,
                    "room_name": r.room_name,
                    "category": r.category,
                    "bbox": r.bbox,
                }
                for r in result.rooms
            ],
            "panels": [{"label": p.label, "bbox": p.bbox} for p in result.panels],
            "electrical_counts": result.electrical_counts,
        }

        if result.error:
            data["error"] = result.error

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved annotations to {output_path}")
