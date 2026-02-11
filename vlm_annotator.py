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
except ImportError:
    from config import VLMConfig

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
        """Build the annotation prompt with image dimensions."""
        categories = ", ".join(self.config.room_categories)

        return f"""Analyze this MEP/Electrical floor plan and extract structured annotations.

IMAGE DIMENSIONS: {img_width} x {img_height} pixels

CRITICAL FILTERING RULES:

DETECT ONLY PHYSICAL/FUNCTIONAL SPACES:
- INCLUDE: Office, conference room, bathroom, storage, lobby, hallway, elevator, stairwell, mechanical room, electrical room, carpentry shop, classrooms, labs, auditoriums
- INCLUDE: Any clearly labeled functional area or physical space

DO NOT INCLUDE (FILTER OUT):
- Documentation blocks (DOCUMENTATION, REQUIREMENTS, RECOMMENDED, etc.)
- Compliance statements (ENERGY CODE, CODE STATEMENT, COMPLIANCE, etc.)
- Legends, symbols, notes, or drawing annotations
- Plan titles, revision blocks, approval blocks, disclaimers
- Administrative text (SCHEDULE, INDEX, KEY, REFERENCE)
- Header/footer text and metadata

For each VALID ROOM or SPACE:
1. Room number (e.g., "113", "S1.100")
2. Room name (e.g., "MECHANICAL ROOM", "SUITE 102", "STUDENT SERVICES")
3. Bounding box in PIXELS: [x, y, width, height] where (x,y) is top-left corner
4. Category from: {categories}

For ELECTRICAL PANELS:
1. Panel label (e.g., "PANEL H1")
2. Bounding box in pixels

Output ONLY valid JSON:
{{
  "rooms": [
    {{"room_number": "113", "room_name": "MECHANICAL ROOM", "category": "mechanical_room", "bbox": [x, y, w, h]}}
  ],
  "panels": [
    {{"label": "PANEL H1", "bbox": [x, y, w, h]}}
  ],
  "electrical_counts": {{
    "fixtures": 0,
    "receptacles": 0,
    "switches": 0,
    "sensors": 0
  }}
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

        # Validate panels
        panels = data.get("panels", [])
        if not isinstance(panels, list):
            raise VLMAnnotationError("'panels' must be a list")

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
