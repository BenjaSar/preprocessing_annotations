"""
VLM Backend Abstraction — Support multiple VLM backends (Claude, Qwen2.5-VL, Unsloth).

This module provides a pluggable architecture for Vision Language Models:
  - Claude (Anthropic API) - External API, high accuracy, higher latency/cost
  - Qwen2.5-VL (Local inference) - Open source, lower latency, requires GPU
  - Unsloth (Optimized local inference) - ~2x faster, ~70% less VRAM, Qwen2.5-VL or Qwen3-VL

Backend selection via VLMConfig.backend field.
Models are configurable via VLMConfig.model field.

All backends take an image and return room detections in the same format:
    [{
        "room_id": "room_0",
        "polygon": [[x,y], [x,y], ...],
        "room_type": "bedroom | bathroom | kitchen | ...",
        "room_name": "optional inferred name",
        "confidence": 0.0-1.0
    }, ...]
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import json
import base64
import io

from . import prompt_templates

# Try to import torch at module level for VLM inference
# Some VLM models may reference torch directly during generation
try:
    import torch
except ImportError:
    # Torch will be imported locally in methods that need it
    torch = None

logger = logging.getLogger(__name__)

# Generation budget for Unsloth door detection; mirrors the window tier's value.
_DOOR_MAX_NEW_TOKENS = 4096

# Default per-detection confidence when the model omits one.
_DETECTION_DEFAULT_CONFIDENCE = 0.7

_BBOX_LEN = 4
_NO_RESCALE = 1.0


def _extract_json_array(response_text: str) -> Optional[List[Any]]:
    """Extract and repair the first JSON array in a model response.

    Strips markdown fences and trailing commas, and closes unbalanced
    braces/brackets from truncated output.

    Returns:
        The parsed list, or None if no recoverable array is present.
    """
    import re

    # Strip <think> blocks (Qwen3 Thinking model preamble)
    response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)

    match = re.search(r'\[.*\]', response_text, re.DOTALL)
    if not match:
        return None

    text = re.sub(r'^```(?:json)?\s*', '', match.group())
    text = re.sub(r'\s*```$', '', text)
    text = re.sub(r',\s*([}\]])', r'\1', text)

    parsed = _load_json_with_recovery(text)
    if parsed is None:
        return None
    return parsed if isinstance(parsed, list) else [parsed]


def _load_json_with_recovery(text: str) -> Optional[Any]:
    """Parse JSON, retrying once with unbalanced structures closed."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')
        if open_braces <= 0 and open_brackets <= 0:
            return None
        try:
            return json.loads(text + '}' * open_braces + ']' * open_brackets)
        except json.JSONDecodeError:
            return None


def _clamp_bbox(
    bbox: List[float], img_width: int, img_height: int
) -> Optional[List[int]]:
    """Clamp a pixel bbox to image bounds; None if degenerate (zero-area)."""
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(img_width), x1))
    x2 = max(0.0, min(float(img_width), x2))
    y1 = max(0.0, min(float(img_height), y1))
    y2 = max(0.0, min(float(img_height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [int(x1), int(y1), int(x2), int(y2)]


def _pixel_object_from_entry(
    entry: Any,
    img_width: int,
    img_height: int,
    default_type: str,
) -> Optional[Dict[str, Any]]:
    """Convert one detection entry (integer pixel bbox) to a clamped object.

    Returns None for malformed, out-of-range, or degenerate boxes.
    """
    if not isinstance(entry, dict) or len(entry.get("bbox", [])) != _BBOX_LEN:
        return None
    try:
        coords = [float(v) for v in entry["bbox"]]
    except (ValueError, TypeError):
        return None
    clamped = _clamp_bbox(coords, img_width, img_height)
    if clamped is None:
        return None
    confidence = entry.get("confidence", _DETECTION_DEFAULT_CONFIDENCE)
    return {
        "bbox": clamped,
        "confidence": float(confidence),
        "type": entry.get("type", default_type),
    }


def _parse_pixel_bbox_objects(
    response_text: str,
    img_width: int,
    img_height: int,
    default_type: str,
) -> List[Dict[str, Any]]:
    """Parse a JSON array of ``{bbox(px), type, confidence}`` detections.

    Bboxes are absolute integer pixels in the given image space, clamped to
    bounds; degenerate boxes are dropped. Coordinates stay in that image's
    space (rescale separately if the image was resized before inference).

    Args:
        response_text: Raw model output.
        img_width: Width of the image the model saw.
        img_height: Height of the image the model saw.
        default_type: Type label used when an entry omits ``type``.

    Returns:
        Validated detections; empty if none are recoverable.
    """
    raw = _extract_json_array(response_text)
    if not raw:
        return []
    objects = (
        _pixel_object_from_entry(entry, img_width, img_height, default_type)
        for entry in raw
    )
    return [obj for obj in objects if obj is not None]


def _scale_for_max_dim(width: int, height: int, max_dim: int) -> float:
    """Return a downscale factor keeping the longest edge <= max_dim."""
    longest_edge = max(width, height)
    if longest_edge <= max_dim:
        return _NO_RESCALE
    return max_dim / longest_edge


def _rescale_bbox_objects(
    objects: List[Dict[str, Any]], factor: float
) -> None:
    """Scale each object's pixel bbox in place by ``factor``."""
    if factor == _NO_RESCALE:
        return
    for obj in objects:
        obj["bbox"] = [int(value * factor) for value in obj["bbox"]]


class VLMBackend(ABC):
    """Abstract base class for VLM backends."""
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize VLM model (lazy loading allowed)."""
        pass
    
    @abstractmethod
    def detect_rooms(self, image_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Detect rooms from a floor plan image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            List of room detections with format:
            [{
                "room_id": str,
                "polygon": [[x,y], ...] or None,
                "bbox": [x1, y1, x2, y2] or None,
                "room_type": str,
                "room_name": str or None,
                "confidence": float,
                "metadata": dict
            }, ...]
        """
        pass
    
    def detect_windows(self, image_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Detect windows from a floor plan image (Tier 3 fallback).
        
        Sends image to VLM with window-detection prompt.
        
        Args:
            image_path: Path to image file
        
        Returns:
            List of window detections with format:
            [{
                "bbox": [x1, y1, x2, y2],
                "confidence": float,
                "type": "window" | "skylight" | "opening"
            }, ...]
        """
        # Default implementation: not supported by this backend
        logger.debug(f"{self.__class__.__name__} does not implement window detection")
        return []

    def detect_doors(
        self, image_path: Union[str, Path]
    ) -> List[Dict[str, Any]]:
        """Detect doors from a floor plan image (VLM door tier).

        Sends the image with a door-detection prompt and parses a JSON array
        of per-door detections
        ``[{"bbox": [x1, y1, x2, y2], "confidence", "type"}, ...]``.

        Default: not implemented by this backend (returns []), so backends
        without a door tier degrade gracefully — same contract as
        ``detect_windows``.
        """
        logger.debug(
            "%s does not implement door detection", self.__class__.__name__
        )
        return []

    def detect_hallucinations(self, rooms: List[Dict[str, Any]], check_stripes: bool = True) -> List[Dict[str, Any]]:
        """Wrapper around the shared hallucination_detector module."""
        from .hallucination_detector import detect_hallucinations as detect_hallucinations_util
        return detect_hallucinations_util(rooms, check_stripes=check_stripes)

    def detect_rooms_from_image(self, image) -> List[Dict[str, Any]]:
        """
        Detect rooms from an in-memory PIL Image (used by TileSplitter).

        Saves the image to a temp PNG, calls detect_rooms(path), cleans up.
        Backends with native in-memory support can override this method.

        Args:
            image: PIL.Image.Image instance

        Returns:
            Same format as detect_rooms().
        """
        import tempfile, os
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".png")
        try:
            os.close(tmp_fd)
            image.save(tmp_path, "PNG")
            return self.detect_rooms(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


class ClaudeBackend(VLMBackend):
    """Claude (Anthropic API) backend for room detection."""
    
    def __init__(self, config):
        """Initialize Claude backend.
        
        Args:
            config: VLMConfig instance
        """
        self.config = config
        self.client = None
        self.initialized = False
    
    def initialize(self) -> None:
        """Initialize Anthropic client."""
        if self.initialized:
            return
        
        try:
            from anthropic import Anthropic
            self.client = Anthropic()
            self.initialized = True
            logger.info(f"Claude backend initialized with model: {self.config.model}")
        except ImportError:
            logger.error(
                "Anthropic SDK not installed. Install via: pip install anthropic"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Claude backend: {e}")
            raise
    
    def detect_rooms(self, image_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Detect rooms using Claude vision API.
        
        Args:
            image_path: Path to floor plan image
        
        Returns:
            List of room detections
        """
        self.initialize()
        
        # Encode image as base64
        image_path = Path(image_path)
        with open(image_path, 'rb') as f:
            image_data = base64.standard_b64encode(f.read()).decode('utf-8')
        
        # Determine media type
        suffix = image_path.suffix.lower()
        media_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        media_type = media_type_map.get(suffix, 'image/jpeg')
        
        # Build prompt for room detection
        prompt = self._build_room_detection_prompt()
        
        # Call Claude API
        try:
            message = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=0.0,  # Deterministic output for reproducible annotations
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
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ],
                    }
                ]
            )
            
            # Parse response — pass image dimensions for percentage→pixel conversion.
            from PIL import Image as _PIL_Image
            with _PIL_Image.open(image_path) as _img:
                _img_width, _img_height = _img.size
            response_text = message.content[0].text
            rooms = self._parse_room_response(response_text, _img_width, _img_height)
            return rooms
            
        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            return []
    
    def _build_room_detection_prompt(self) -> str:
        """
        Build prompt for room detection using mandatory SFT taxonomy.
        
        Uses VLM_PROMPT_CATEGORIES from automation.taxonomy for consistent
        room type vocabulary across all VLM backends.
        """
        from ..automation.taxonomy import get_vlm_categories_string

        categories_str = get_vlm_categories_string()

        return prompt_templates.build_room_prompt_claude(categories_str)
    
    def _parse_room_response(self, response_text: str, img_width: int = 1000, img_height: int = 1000) -> List[Dict[str, Any]]:
        """Parse JSON response from Claude (bbox values expected as 0-100 percentages)."""
        try:
            # Try to extract JSON array from response
            rooms = json.loads(response_text)
            if not isinstance(rooms, list):
                rooms = [rooms]
            
            # Single consolidated hallucination detector (hallucination_detector.py)
            rooms_before_halluc = len(rooms)
            rooms = self.detect_hallucinations(rooms)
            if rooms_before_halluc != len(rooms):
                logger.debug(
                    f"Stage-transition [after-halluc-detect]: {len(rooms)} rooms "
                    f"(dropped {rooms_before_halluc - len(rooms)})"
                )
             
             # Cap rooms at the configured maximum (kept in sync with the prompt)
            max_rooms = self.config.max_rooms
            if len(rooms) > max_rooms:
                logger.warning(
                    f"VLM generated {len(rooms)} rooms, exceeding prompt limit of {max_rooms}. "
                    f"Capping at {max_rooms}."
                )
                rooms = rooms[:max_rooms]
            
            # Normalize response format with bbox validation
            normalized = []
            dropped_count = 0
            for idx, room in enumerate(rooms):
                # Convert bbox percentages to pixel coordinates using actual image dimensions
                bbox = room.get("bbox")
                if bbox and len(bbox) == 4:
                    try:
                        x1_pct, y1_pct, x2_pct, y2_pct = [float(v) for v in bbox]
                    except (ValueError, TypeError):
                        logger.debug(f"Room {idx}: non-numeric bbox values {bbox}, skipping")
                        dropped_count += 1
                        continue
                    
                    # Validation: all coordinates must be in 0-100 range (with 5% tolerance)
                    if any(v < 0 or v > 105 for v in [x1_pct, y1_pct, x2_pct, y2_pct]):
                        logger.debug(
                            f"Room {idx} ({room.get('room_name', '?')}): out-of-bounds bbox "
                            f"[{x1_pct},{y1_pct},{x2_pct},{y2_pct}]%, skipping"
                        )
                        dropped_count += 1
                        continue
                    
                    # Clamp to valid range for safety
                    x1_pct = max(0, min(100, x1_pct))
                    y1_pct = max(0, min(100, y1_pct))
                    x2_pct = max(0, min(100, x2_pct))
                    y2_pct = max(0, min(100, y2_pct))
                    
                    # Scale to actual pixel coordinates: x-coords use width, y-coords use height
                    x1_px = int(x1_pct * img_width / 100)
                    y1_px = int(y1_pct * img_height / 100)
                    x2_px = int(x2_pct * img_width / 100)
                    y2_px = int(y2_pct * img_height / 100)
                    bbox = [x1_px, y1_px, x2_px, y2_px]
                    
                    # Step 5: Log zero-dimension bbox validation
                    width = x2_px - x1_px
                    height = y2_px - y1_px
                    if width <= 0 or height <= 0:
                        logger.debug(
                            f"Room {idx} ({room.get('room_name', '?')}): zero-dimension bbox "
                            f"[{x1_px},{y1_px},{x2_px},{y2_px}] (w={width}, h={height}), skipping"
                        )
                        dropped_count += 1
                        continue
                    
                    # Step 7: Coordinate-range assertion (pixel coords within image bounds)
                    if not (0 <= x1_px < img_width and 0 <= x2_px <= img_width and x1_px <= x2_px):
                        logger.debug(
                            f"Room {idx}: x-coords [{x1_px},{x2_px}] out of range [0,{img_width}], skipping"
                        )
                        dropped_count += 1
                        continue
                    if not (0 <= y1_px < img_height and 0 <= y2_px <= img_height and y1_px <= y2_px):
                        logger.debug(
                            f"Room {idx}: y-coords [{y1_px},{y2_px}] out of range [0,{img_height}], skipping"
                        )
                        dropped_count += 1
                        continue
                
                normalized.append({
                    "room_id": room.get("room_id", f"room_{idx}"),
                    "polygon": None,  # Qwen returns bbox, not polygon
                    "bbox": bbox,
                    "room_type": room.get("room_type", "other"),
                    "room_name": room.get("room_name"),
                    "confidence": float(room.get("confidence", 0.5)),
                    "metadata": room.get("metadata", {})
                })
            
            if dropped_count > 0:
                logger.info(f"Dropped {dropped_count} rooms with invalid bboxes")
            
            logger.debug(f"Parsed {len(normalized)} rooms from Claude response")
            return normalized
        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            logger.error(f"Failed to parse Claude response: {e}")
            logger.debug(f"Response text: {response_text[:500]}")
            return []
    
    def detect_windows(self, image_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Detect windows in floor plan image using Qwen2.5-VL (Phase 2, Tier 3).

        Args:
            image_path: Path to floor plan image

        Returns:
            List of window detections
        """
        try:
            import torch
            from PIL import Image
            
            self.initialize()
            
            image = Image.open(image_path)
            prompt = self._build_window_detection_prompt()

            logger.debug(f"Running Qwen2.5-VL window detection on {Path(image_path).name}")

            # Prepare inputs
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)

            # Run inference with config-driven generation parameters
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,

                )
            
            # Decode response: only the generated tokens (exclude input prompt echo)
            input_len = inputs["input_ids"].shape[-1]
            generated_ids = output[0][input_len:]
            response_text = self.processor.decode(generated_ids, skip_special_tokens=True)
            
            # Parse windows (pass actual image dimensions for correct bbox scaling)
            img_width, img_height = image.size
            windows = self._parse_window_response(response_text, img_width, img_height)
            
            # GPU memory cleanup
            del inputs, output, generated_ids
            torch.cuda.empty_cache()
            
            return windows
        
        except Exception as e:
            logger.error(f"Window detection failed: {e}")
            return []
    
    def _build_window_detection_prompt(self) -> str:
        """Build prompt for window detection."""
        return prompt_templates.build_window_prompt()

    def _parse_window_response(self, response_text: str, img_width: int = 1000, img_height: int = 1000) -> List[Dict[str, Any]]:
        """Parse window detection response."""
        try:
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if not json_match:
                logger.debug("No windows detected")
                return []
            
            # JSON repair: strip markdown fences and fix common LLM mistakes
            json_str = json_match.group()
            json_str = re.sub(r'^```(?:json)?\s*', '', json_str)
            json_str = re.sub(r'\s*```$', '', json_str)
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            
            windows_raw = json.loads(json_str)
            if not isinstance(windows_raw, list):
                windows_raw = [windows_raw]
            
            windows = []
            for window in windows_raw:
                bbox = window.get("bbox", [])
                if len(bbox) == 4:
                    # Convert percentage (0-100) to pixel coordinates using actual image dimensions
                    x1_pct, y1_pct, x2_pct, y2_pct = bbox
                    bbox = [
                        int(x1_pct * img_width / 100),
                        int(y1_pct * img_height / 100),
                        int(x2_pct * img_width / 100),
                        int(y2_pct * img_height / 100)
                    ]
                
                windows.append({
                    "bbox": bbox,
                    "confidence": float(window.get("confidence", 0.7)),
                    "type": window.get("type", "window"),
                })
            
            logger.debug(f"Detected {len(windows)} windows")
            return windows
        
        except (json.JSONDecodeError, AttributeError) as e:
            logger.debug(f"Failed to parse window response: {e}")
            return []


class Qwen2_5VLBackend(VLMBackend):
    """Qwen2.5-VL (local inference) backend for room detection."""
    
    def __init__(self, config):
        """Initialize Qwen2.5-VL backend.
        
        Args:
            config: VLMConfig instance
        """
        self.config = config
        self.model = None
        self.processor = None
        self.initialized = False
        self.device = config.qwen_device or "cuda"
    
    def initialize(self) -> None:
        """Initialize Qwen2.5-VL model and processor."""
        if self.initialized:
            return
        
        try:
            from transformers import (
                AutoProcessor,
                Qwen2VLForConditionalGeneration,
                Qwen2_5_VLForConditionalGeneration,
                Qwen3VLForConditionalGeneration,
            )
            import torch
            
            # Load model with quantization if configured
            # Resolve model ID: prefer dedicated qwen_model field, fall back to generic model
            # (only if it's not a Claude model ID), otherwise use default
            _DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
            
            model_id = None
            # First priority: dedicated qwen_model field (if present via Fix B)
            if hasattr(self.config, 'qwen_model') and self.config.qwen_model:
                model_id = self.config.qwen_model
            # Second priority: generic model field (only if not a Claude model ID)
            elif self.config.model and not self.config.model.startswith("claude"):
                model_id = self.config.model
            # Last resort: use default Qwen model
            else:
                model_id = _DEFAULT_QWEN_MODEL
                logger.info(f"Using default Qwen model: {model_id}")
            
            # Determine which model class to use based on model ID
            if "2.5" in model_id or "2_5" in model_id:
                ModelClass = Qwen2_5_VLForConditionalGeneration
                logger.info(f"Using Qwen2.5-VL model class")
            elif "3" in model_id.split("/")[-1][:1]:  # Check if version starts with 3
                ModelClass = Qwen3VLForConditionalGeneration
                logger.info(f"Using Qwen3-VL model class")
            else:
                ModelClass = Qwen2VLForConditionalGeneration
                logger.info(f"Using Qwen2-VL model class")
            
            # Configure quantization
            quantization_config = None
            if self.config.qwen_quantization == "4bit":
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            elif self.config.qwen_quantization == "8bit":
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.model = ModelClass.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map=self.device,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
            )
            
            self.initialized = True
            logger.info(
                f"Qwen2.5-VL backend initialized: model={model_id}, "
                f"device={self.device}, quantization={self.config.qwen_quantization}"
            )
        except ImportError as e:
            logger.error(
                f"Required package not installed: {e}. "
                "Install via: pip install transformers torch torchvision"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Qwen2.5-VL backend: {e}")
            raise
    
    def detect_rooms(self, image_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Detect rooms using Qwen2.5-VL.
        
        Args:
            image_path: Path to floor plan image
        
        Returns:
            List of room detections
        """
        self.initialize()
        
        try:
            import torch
            from PIL import Image
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Build prompt
            prompt = self._build_room_detection_prompt()
            
            # Prepare input using processor (handles image preprocessing)
            # Note: Using explicit pad_image=True and padding='max_length' for Qwen2.5-VL compatibility
            inputs = self.processor(
                text=prompt,
                images=[image],
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response with config-driven generation parameters
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,

                )
            
            # Decode response: only the generated tokens (exclude input prompt echo)
            # batch_decode returns list of decoded sequences (one per batch item)
            input_len = inputs["input_ids"].shape[-1]
            generated_ids = output_ids[:, input_len:]
            response_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Parse response (pass actual image dimensions for correct bbox scaling)
            img_width, img_height = image.size
            rooms = self._parse_room_response(response_text, img_width, img_height)
            
            # F6: Detect and truncate hallucinations before returning
            rooms_before = len(rooms)
            rooms = self.detect_hallucinations(rooms)
            rooms_after = len(rooms)
            if rooms_before != rooms_after:
                logger.info(
                    f"Hallucination detection truncated {rooms_before - rooms_after} rooms "
                    f"({rooms_after} remaining)"
                )
            
            # GPU memory cleanup
            del inputs, output_ids, generated_ids
            torch.cuda.empty_cache()
            
            return rooms
            
        except Exception as e:
            logger.error(f"Qwen2.5-VL inference failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []

    def _build_room_detection_prompt(self) -> str:
        """
        Build prompt for room detection to prevent hallucination.
        
        Key improvements:
        - Shortened category list prevents the model from treating it as a checklist
        - No example bbox values (prevents direct copying of [10, 20, 40, 50])
        - Explicit anti-hallucination instruction: "Do NOT invent or fabricate"
        - Room count limit prevents 75-room generation
        """
        return prompt_templates.build_room_prompt_qwen(self.config.max_rooms)

    def _parse_room_response(self, response_text: str, img_width: int = 1000, img_height: int = 1000) -> List[Dict[str, Any]]:
        """Parse JSON response from Qwen2.5-VL."""
        try:
            # Extract JSON from response (may be wrapped in markdown or other text)
            import re
            
            # Try to find JSON array - use greedy matching
            # First try standard JSON array pattern
            json_match = re.search(r'\[\s*\{.*?\}\s*\]', response_text, re.DOTALL)
            
            # If standard pattern fails, try to find any array-like structure
            if not json_match:
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            
            if not json_match:
                logger.error("No JSON array found in Qwen response")
                logger.debug(f"Full response: {response_text[:500]}")
                return []
            
            json_str = json_match.group()
            # Clean up any trailing/leading whitespace
            json_str = json_str.strip()
            _max_chars = self.config.debug_log_max_chars
            _suffix = "..." if len(json_str) > _max_chars else ""
            logger.debug(f"Extracted JSON: {json_str[:_max_chars]}{_suffix}")
            
            # JSON repair: strip markdown fences and fix common LLM mistakes
            json_str = re.sub(r'^```(?:json)?\s*', '', json_str)
            json_str = re.sub(r'\s*```$', '', json_str)
            # Remove trailing commas (common LLM mistake)
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            
            rooms = json.loads(json_str)
            if not isinstance(rooms, list):
                rooms = [rooms]
            
            # Normalize response format with bbox validation
            normalized = []
            dropped_count = 0
            for idx, room in enumerate(rooms):
                # Convert bbox percentages to pixel coordinates using actual image dimensions
                bbox = room.get("bbox")
                if bbox and len(bbox) == 4:
                    try:
                        x1_pct, y1_pct, x2_pct, y2_pct = [float(v) for v in bbox]
                    except (ValueError, TypeError):
                        logger.debug(f"Room {idx}: non-numeric bbox values {bbox}, skipping")
                        dropped_count += 1
                        continue
                    
                    # Validation: all coordinates must be in 0-100 range (with 5% tolerance)
                    if any(v < 0 or v > 105 for v in [x1_pct, y1_pct, x2_pct, y2_pct]):
                        logger.debug(
                            f"Room {idx} ({room.get('room_name', '?')}): out-of-bounds bbox "
                            f"[{x1_pct},{y1_pct},{x2_pct},{y2_pct}]%, skipping"
                        )
                        dropped_count += 1
                        continue
                    
                    # Clamp to valid range for safety
                    x1_pct = max(0, min(100, x1_pct))
                    y1_pct = max(0, min(100, y1_pct))
                    x2_pct = max(0, min(100, x2_pct))
                    y2_pct = max(0, min(100, y2_pct))
                    
                    # Scale to actual pixel coordinates: x-coords use width, y-coords use height
                    x1_px = int(x1_pct * img_width / 100)
                    y1_px = int(y1_pct * img_height / 100)
                    x2_px = int(x2_pct * img_width / 100)
                    y2_px = int(y2_pct * img_height / 100)
                    bbox = [x1_px, y1_px, x2_px, y2_px]
                    
                    # Step 5: Log zero-dimension bbox validation
                    width = x2_px - x1_px
                    height = y2_px - y1_px
                    if width <= 0 or height <= 0:
                        logger.debug(
                            f"Room {idx} ({room.get('room_name', '?')}): zero-dimension bbox "
                            f"[{x1_px},{y1_px},{x2_px},{y2_px}] (w={width}, h={height}), skipping"
                        )
                        dropped_count += 1
                        continue
                    
                    # Step 7: Coordinate-range assertion (pixel coords within image bounds)
                    if not (0 <= x1_px < img_width and 0 <= x2_px <= img_width and x1_px <= x2_px):
                        logger.debug(
                            f"Room {idx}: x-coords [{x1_px},{x2_px}] out of range [0,{img_width}], skipping"
                        )
                        dropped_count += 1
                        continue
                    if not (0 <= y1_px < img_height and 0 <= y2_px <= img_height and y1_px <= y2_px):
                        logger.debug(
                            f"Room {idx}: y-coords [{y1_px},{y2_px}] out of range [0,{img_height}], skipping"
                        )
                        dropped_count += 1
                        continue
                
                normalized.append({
                    "room_id": room.get("room_id", f"room_{idx}"),
                    "polygon": None,  # Qwen returns bbox, not polygon
                    "bbox": bbox,
                    "room_type": room.get("room_type", "other"),
                    "room_name": room.get("room_name"),
                    "confidence": float(room.get("confidence", 0.5)),
                    "metadata": room.get("metadata", {})
                })
            
            if dropped_count > 0:
                logger.info(f"Dropped {dropped_count} rooms with invalid bboxes")
            
            logger.debug(f"Parsed {len(normalized)} rooms from Qwen response")
            return normalized
        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            logger.error(f"Failed to parse Qwen response: {e}")
            logger.debug(f"Response text: {response_text[:500]}")
            return []
    
    def detect_windows(self, image_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Detect windows in floor plan image using Qwen2.5-VL (Phase 2, Tier 3).

        Args:
            image_path: Path to floor plan image

        Returns:
            List of window detections
        """
        try:
            import torch
            from PIL import Image

            self.initialize()

            image = Image.open(image_path)
            prompt = self._build_window_detection_prompt()

            logger.debug(f"Running Qwen2.5-VL window detection on {Path(image_path).name}")

            # Prepare inputs
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)

            # Run inference with config-driven generation parameters
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,

                )
            
            # Decode response: only the generated tokens (exclude input prompt echo)
            input_len = inputs["input_ids"].shape[-1]
            generated_ids = output[0][input_len:]
            response_text = self.processor.decode(generated_ids, skip_special_tokens=True)
            
            # Parse windows (pass actual image dimensions for correct bbox scaling)
            img_width, img_height = image.size
            windows = self._parse_window_response(response_text, img_width, img_height)
            
            # GPU memory cleanup
            del inputs, output, generated_ids
            torch.cuda.empty_cache()
            
            return windows
        
        except Exception as e:
            logger.error(f"Window detection failed: {e}")
            return []
    
    def _build_window_detection_prompt(self) -> str:
        """Build prompt for window detection."""
        return prompt_templates.build_window_prompt()

    def _parse_window_response(self, response_text: str, img_width: int = 1000, img_height: int = 1000) -> List[Dict[str, Any]]:
        """Parse window detection response."""
        try:
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if not json_match:
                logger.debug("No windows detected")
                return []
            
            # JSON repair: strip markdown fences and fix common LLM mistakes
            json_str = json_match.group()
            json_str = re.sub(r'^```(?:json)?\s*', '', json_str)
            json_str = re.sub(r'\s*```$', '', json_str)
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            
            windows_raw = json.loads(json_str)
            if not isinstance(windows_raw, list):
                windows_raw = [windows_raw]
            
            windows = []
            for window in windows_raw:
                bbox = window.get("bbox", [])
                if len(bbox) == 4:
                    # Convert percentage (0-100) to pixel coordinates using actual image dimensions
                    x1_pct, y1_pct, x2_pct, y2_pct = bbox
                    bbox = [
                        int(x1_pct * img_width / 100),
                        int(y1_pct * img_height / 100),
                        int(x2_pct * img_width / 100),
                        int(y2_pct * img_height / 100)
                    ]
                
                windows.append({
                    "bbox": bbox,
                    "confidence": float(window.get("confidence", 0.7)),
                    "type": window.get("type", "window"),
                })
            
            logger.debug(f"Detected {len(windows)} windows")
            return windows
        
        except (json.JSONDecodeError, AttributeError) as e:
            logger.debug(f"Failed to parse window response: {e}")
            return []


class UnslothQwenBackend(VLMBackend):
    """Unsloth-optimized Qwen VL backend (supports Qwen2.5-VL and Qwen3-VL).
    
    Uses Unsloth's FastVisionModel for ~2x faster inference and ~70% less VRAM
    compared to raw transformers. Pre-quantized 4-bit models available.
    
    Supported models via unsloth_model key:
        - "qwen2.5-vl-7b": unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit
        - "qwen3-vl-2b": unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit
        - "qwen3-vl-4b": unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit
        - "qwen3-vl-8b": unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit
        - "qwen3-vl-2b-thinking": unsloth/Qwen3-VL-2B-Thinking-unsloth-bnb-4bit
        - "qwen3-vl-4b-thinking": unsloth/Qwen3-VL-4B-Thinking-unsloth-bnb-4bit
        - "qwen3-vl-8b-thinking": unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit
    """

    # Mapping from short model key to Unsloth HuggingFace model ID
    MODEL_REGISTRY = {
        "qwen2.5-vl-7b": "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
        "qwen3-vl-2b": "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit",
        "qwen3-vl-4b": "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit",
        "qwen3-vl-8b": "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
        "qwen3-vl-2b-thinking": "unsloth/Qwen3-VL-2B-Thinking-unsloth-bnb-4bit",
        "qwen3-vl-4b-thinking": "unsloth/Qwen3-VL-4B-Thinking-unsloth-bnb-4bit",
        "qwen3-vl-8b-thinking": "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit",
    }
    
    def __init__(self, config):
        """Initialize Unsloth Qwen backend.
        
        Args:
            config: VLMConfig instance with unsloth_model field
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.initialized = False
        self.device = config.qwen_device or "cuda"
    
    def initialize(self) -> None:
        """Initialize Qwen model via Unsloth FastVisionModel."""
        if self.initialized:
            return
        
        try:
            import os
            from unsloth import FastVisionModel
            import torch
            
            # Increase HuggingFace Hub timeout for slow connections
            # Default is 10 seconds, we increase to 60 seconds for large model downloads
            os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'
            
            # Also configure huggingface_hub directly if available
            try:
                from huggingface_hub import constants
                constants.HFHUB_DOWNLOAD_TIMEOUT = 60
                logger.info("Set HuggingFace Hub timeout to 60s (via constants)")
            except (ImportError, AttributeError):
                pass
            
            logger.info("Set HuggingFace Hub download timeout to 60s")
            
            # Resolve model ID
            model_key = getattr(self.config, 'unsloth_model', 'qwen3-vl-2b').lower()
            if model_key not in self.MODEL_REGISTRY:
                logger.warning(
                    f"Unknown Unsloth model key: {model_key}. "
                    f"Available: {', '.join(self.MODEL_REGISTRY.keys())}. "
                    f"Using default: qwen3-vl-2b"
                )
                model_key = "qwen3-vl-2b"
            
            model_id = self.MODEL_REGISTRY[model_key]
            logger.info(f"Loading Unsloth model: {model_key} ({model_id})")
            
            # Load model with Unsloth optimization
            self.model, self.tokenizer = FastVisionModel.from_pretrained(
                model_id,
                load_in_4bit=True,  # Use pre-quantized 4-bit model
                use_gradient_checkpointing="unsloth",  # Memory optimization
            )
            
            # Enable inference mode optimization
            FastVisionModel.for_inference(self.model)
            
            self.initialized = True
            logger.info(
                f"Unsloth Qwen backend initialized: model={model_key}, "
                f"device={self.device}"
            )
        
        except ImportError as e:
            logger.error(
                f"Unsloth not installed: {e}. "
                "Install via: pip install unsloth"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Unsloth Qwen backend: {e}")
            raise

    def _generation_kwargs(self, max_new_tokens: int) -> Dict[str, Any]:
        """Config-driven ``model.generate`` kwargs — single decoding source.

        All decoding parameters come from ``VLMConfig`` (no hardcoded
        sampling), so the room, door, and window tiers share one definition
        and the sampling mode is set by configuration, not baked into each
        call site. ``presence_penalty`` is intentionally absent: it is not a
        HuggingFace ``generate`` argument (only ``repetition_penalty`` is).
        """
        return {
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "do_sample": self.config.do_sample,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "repetition_penalty": self.config.repetition_penalty,
        }

    def _seed_generation(self) -> None:
        """Seed the RNG from config so sampled decoding stays reproducible.

        No-op influence under greedy decoding (``do_sample=False``), where
        output is already deterministic; only applied when sampling.
        """
        if not self.config.do_sample:
            return
        import torch
        torch.manual_seed(self.config.seed)

    def _is_thinking_variant(self) -> bool:
        """True if the configured Unsloth model is a Thinking checkpoint.

        Thinking checkpoints generate a full reasoning narrative before any
        JSON (verified 2026-07-24: exhausted the default room token budget
        on reasoning prose alone, emitting no JSON at all) — see
        VLMConfig.room_max_new_tokens_thinking.
        """
        return self.config.unsloth_model.lower().endswith("-thinking")

    def detect_rooms(self, image_path: Union[str, Path]) -> List[Dict[str, Any]]:
         """
         Detect rooms using Unsloth-optimized Qwen VL.
         
         Args:
             image_path: Path to floor plan image
         
         Returns:
             List of room detections
         """
         self.initialize()
         
         try:
             from PIL import Image
             import torch
             
             # Load image
             image = Image.open(image_path).convert("RGB")
             
             # Save original dimensions before resizing
             original_width, original_height = image.size
             
             # Resize large images to reduce GPU memory usage (config-driven cap,
             # shared with door detection — see VLMConfig.unsloth_detection_max_dim_px)
             max_dim = self.config.unsloth_detection_max_dim_px
             width, height = image.size
             resize_scale = 1.0  # Track whether we resized
             if max(width, height) > max_dim:
                 resize_scale = max_dim / max(width, height)
                 new_size = (int(width * resize_scale), int(height * resize_scale))
                 logger.info(f"Resizing image from {width}x{height} to {new_size[0]}x{new_size[1]} (scale={resize_scale:.2f})")
                 image = image.resize(new_size, Image.Resampling.LANCZOS)
             
             # Build room detection prompt with actual pixel dimensions so the
             # model produces pixel-coordinate bboxes (Qwen2.5-VL grounding
             # training used pixel coords, not fractions).
             img_width, img_height = image.size
             prompt = self._build_room_detection_prompt(img_width, img_height)

             # Prepare chat messages in Unsloth format with system message
             messages = [
                 {
                     "role": "system",
                     "content": [
                         {
                             "type": "text",
                             "text": prompt_templates.build_json_array_system_prompt()
                         }
                     ]
                 },
                 {
                     "role": "user",
                     "content": [
                         {"type": "image"},
                         {"type": "text", "text": prompt}
                     ]
                 }
             ]
             
             # Apply chat template
             input_text = self.tokenizer.apply_chat_template(
                 messages,
                 add_generation_prompt=True
             )
             
             # Prepare model inputs
             inputs = self.tokenizer(
                 image,
                 input_text,
                 add_special_tokens=False,
                 return_tensors="pt",
             ).to(self.device)
             
             # Run inference with config-driven generation parameters
             logger.debug(f"Running Unsloth Qwen room detection on {Path(image_path).name}")
             room_max_new_tokens = (
                 self.config.room_max_new_tokens_thinking
                 if self._is_thinking_variant()
                 else self.config.room_max_new_tokens
             )
             self._seed_generation()
             with torch.no_grad():
                 output_ids = self.model.generate(
                     **inputs,
                     **self._generation_kwargs(room_max_new_tokens),
                 )

             # Decode the post-reasoning answer only (see _decode_generated)
             input_len = inputs["input_ids"].shape[-1]
             response_text = self._decode_generated(output_ids, input_len)
             
             # Parse room detections from response (pass actual image dimensions for correct bbox scaling)
             img_width, img_height = image.size
             rooms = self._parse_room_response(response_text, img_width, img_height)
             
             # F6: Detect and truncate hallucinations before rescaling.
             # Skip stripe check here — this runs per-tile; a tile may legitimately
             # hold one grid row. Stripe gate runs once on merged full-page set
             # (pipeline) + sft_validator Fix9 backstop.
             rooms_before = len(rooms)
             rooms = self.detect_hallucinations(rooms, check_stripes=False)
             rooms_after = len(rooms)
             if rooms_before != rooms_after:
                 logger.info(
                     f"Hallucination detection truncated {rooms_before - rooms_after} rooms "
                     f"({rooms_after} remaining)"
                 )
             
             # Rescale bboxes back to original image dimensions if image was resized
             # VLM saw the resized image and generated fractions based on resized dimensions.
             # _parse_room_response converted those fractions to pixel coords in resized space.
             # Now we rescale back to original space so downstream consumers get original-dim coordinates.
             if resize_scale < 1.0:
                 inv_scale = 1.0 / resize_scale  # e.g. 2.1973 if scaled from 4500 to 2048
                 for room in rooms:
                     bbox = room.get("bbox", [])
                     if bbox and len(bbox) == 4:
                         room["bbox"] = [int(v * inv_scale) for v in bbox]
                 logger.debug(f"Rescaled {len(rooms)} room bboxes from resized({img_width}x{img_height}) to original({original_width}x{original_height})")
             
             # GPU memory cleanup - aggressive deletion of all references
             del inputs, output_ids, image, prompt, messages, input_text
             torch.cuda.empty_cache()
             import gc
             gc.collect()
             
             return rooms
         
         except Exception as e:
             logger.error(f"Unsloth Qwen inference failed: {e}")
             import traceback
             logger.debug(traceback.format_exc())
             # Cleanup on error
             import gc
             torch.cuda.empty_cache()
             gc.collect()
             return []

    def _build_room_detection_prompt(self, img_width: int = 1000, img_height: int = 1000) -> str:
         """
         Build prompt for room detection using pixel coordinates.

         Pixel coords match Qwen2.5-VL grounding training distribution,
         which prevents the out-of-range fraction values (e.g. 1.2, 2.7)
         that the old 0-1 fraction prompt consistently produced.

         Args:
             img_width: Actual image width in pixels (after any resizing).
             img_height: Actual image height in pixels (after any resizing).
         """
         return prompt_templates.build_room_prompt_unsloth(
             img_width, img_height, self.config.max_rooms
         )


    def _parse_room_response(self, response_text: str, img_width: int = 1000, img_height: int = 1000) -> List[Dict[str, Any]]:
        """Parse JSON response from Unsloth Qwen."""
        try:
            # Extract JSON array from response (may be wrapped in markdown or other text)
            import re
            
            # Debug: Log the first part of the response
            logger.debug(f"Unsloth response (first 300 chars): {response_text[:300]}")
            
            # Strip <think> blocks (Qwen3 Thinking model preamble)
            response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
            
            # Try to find JSON array - use greedy matching
            # First try standard JSON array pattern
            json_match = re.search(r'\[\s*\{.*?\}\s*\]', response_text, re.DOTALL)
            
            # If standard pattern fails, try to find any array-like structure
            if not json_match:
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            
            if not json_match:
                logger.error("No JSON array found in Unsloth Qwen response")
                logger.debug(f"Full response: {response_text[:500]}")
                return []
            
            json_str = json_match.group()
            # Clean up any trailing/leading whitespace
            json_str = json_str.strip()
            _max_chars = self.config.debug_log_max_chars
            _suffix = "..." if len(json_str) > _max_chars else ""
            logger.debug(f"Extracted JSON: {json_str[:_max_chars]}{_suffix}")
            
            # JSON repair: strip markdown fences and fix common LLM mistakes
            json_str = re.sub(r'^```(?:json)?\s*', '', json_str)
            json_str = re.sub(r'\s*```$', '', json_str)
            # Remove trailing commas (common LLM mistake)
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            
            # Attempt parsing and handle truncation
            try:
                rooms = json.loads(json_str)
            except json.JSONDecodeError as e:
                # If JSON is truncated (incomplete), try to recover by closing structures
                logger.debug(f"Initial JSON parse failed: {e}. Attempting truncation recovery...")
                
                # Count braces/brackets to see if we're missing closing characters
                open_braces = json_str.count('{') - json_str.count('}')
                open_brackets = json_str.count('[') - json_str.count(']')
                
                if open_braces > 0 or open_brackets > 0:
                    # JSON appears truncated, try to close it
                    logger.debug(f"Truncation detected: {open_braces} open braces, {open_brackets} open brackets")
                    recovered_json = json_str + ('}' * open_braces) + (']' * open_brackets)
                    
                    try:
                        rooms = json.loads(recovered_json)
                        logger.info(f"Successfully recovered JSON by closing {open_braces} braces and {open_brackets} brackets")
                    except json.JSONDecodeError as recovery_e:
                        logger.error(f"Truncation recovery failed: {recovery_e}")
                        return []
                else:
                    # Not a truncation issue, re-raise
                    raise
            
            if not isinstance(rooms, list):
                rooms = [rooms]
            
            # Validate and normalise pixel-coordinate bboxes.
            # The prompt now requests integer pixel coords in [0,img_width] x [0,img_height].
            normalized = []
            dropped_count = 0
            for idx, room in enumerate(rooms):
                if not isinstance(room, dict):
                    logger.debug(f"Room {idx}: non-object entry {room!r}, skipping")
                    dropped_count += 1
                    continue

                bbox = room.get("bbox")
                if bbox and len(bbox) == 4:
                    try:
                        x1, y1, x2, y2 = [float(v) for v in bbox]
                    except (ValueError, TypeError):
                        logger.debug(f"Room {idx}: non-numeric bbox values {bbox}, skipping")
                        dropped_count += 1
                        continue

                    # Reject coords that are clearly fractions (all ≤ 1.0) — model
                    # ignored the pixel instruction; discard rather than silently scale.
                    if all(v <= 1.0 for v in [abs(x1), abs(y1), abs(x2), abs(y2)]):
                        logger.debug(
                            f"Room {idx} ({room.get('room_name', '?')}): "
                            f"looks like fractions not pixels [{x1},{y1},{x2},{y2}], skipping"
                        )
                        dropped_count += 1
                        continue

                    # Clamp to image bounds
                    x1 = max(0.0, min(float(img_width), x1))
                    y1 = max(0.0, min(float(img_height), y1))
                    x2 = max(0.0, min(float(img_width), x2))
                    y2 = max(0.0, min(float(img_height), y2))

                    # Reject zero-dimension bboxes
                    if x2 <= x1 or y2 <= y1:
                        logger.debug(
                            f"Room {idx} ({room.get('room_name', '?')}): "
                            f"zero-dimension bbox [{x1},{y1},{x2},{y2}], skipping"
                        )
                        dropped_count += 1
                        continue

                    bbox = [int(x1), int(y1), int(x2), int(y2)]
                
                normalized.append({
                    "room_id": room.get("room_id", f"room_{idx}"),
                    "polygon": None,  # Qwen returns bbox, not polygon
                    "bbox": bbox,
                    "room_type": room.get("room_type", "other"),
                    "room_name": room.get("room_name"),
                    "confidence": float(room.get("confidence", 0.5)),
                    "metadata": room.get("metadata", {})
                })
            
            if dropped_count > 0:
                logger.info(f"Dropped {dropped_count} rooms with invalid bboxes")
            
            logger.debug(f"Parsed {len(normalized)} rooms from Unsloth response")
            # Per-parse (incl. per-tile): skip stripe check — a tile may legitimately
            # hold one grid row. Stripe gate runs once on merged full-page set.
            return self.detect_hallucinations(normalized, check_stripes=False)

        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            logger.error(f"Failed to parse Unsloth Qwen response: {e}")
            logger.debug(f"Response text: {response_text[:500]}")
            return []
    
    def detect_windows(self, image_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Detect windows in floor plan image using Unsloth Qwen (Phase 2, Tier 3).
        
        Args:
            image_path: Path to floor plan image
        
        Returns:
            List of window detections with format:
            [{
                "bbox": [x1, y1, x2, y2],
                "confidence": float,
                "type": "window" | "skylight" | "opening"
            }, ...]
        """
        try:
            import torch
            from PIL import Image
            
            self.initialize()
            
            # Load and prepare image
            image = Image.open(image_path).convert("RGB")
            
            # Build window detection prompt
            prompt = self._build_window_detection_prompt()
            
            # Prepare chat messages in Unsloth format with system message
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_templates.build_json_array_system_prompt()
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Apply chat template
            input_text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True
            )
            
            # Prepare model inputs
            inputs = self.tokenizer(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(self.device)
            
            # Run inference with config-driven generation parameters
            logger.debug(f"Running Unsloth Qwen window detection on {Path(image_path).name}")
            self._seed_generation()
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    **self._generation_kwargs(self.config.window_max_new_tokens),
                )

            # Decode the post-reasoning answer only (see _decode_generated)
            input_len = inputs["input_ids"].shape[-1]
            response_text = self._decode_generated(output_ids, input_len)
            logger.debug(f"Window detection response: {response_text[:200]}...")

            # Parse windows from response (pass actual image dimensions for correct bbox scaling)
            img_width, img_height = image.size
            windows = self._parse_window_response(response_text, img_width, img_height)

            # GPU memory cleanup
            del inputs, output_ids
            torch.cuda.empty_cache()
            
            return windows
        
        except Exception as e:
            logger.error(f"Window detection failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []
    
    def _build_window_detection_prompt(self) -> str:
        """Build prompt for window detection."""
        return prompt_templates.build_window_prompt_unsloth()
    
    def _parse_window_response(self, response_text: str, img_width: int = 1000, img_height: int = 1000) -> List[Dict[str, Any]]:
        """Parse window detection response from Unsloth Qwen."""
        try:
            import re
            # Strip <think> blocks (Qwen3 Thinking model preamble)
            response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
            # Extract JSON array from response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if not json_match:
                logger.debug("No windows detected in response")
                return []
            
            # JSON repair: strip markdown fences and fix common LLM mistakes
            json_str = json_match.group()
            json_str = re.sub(r'^```(?:json)?\s*', '', json_str)
            json_str = re.sub(r'\s*```$', '', json_str)
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            
            # Attempt parsing and handle truncation
            try:
                windows_raw = json.loads(json_str)
            except json.JSONDecodeError as e:
                # If JSON is truncated (incomplete), try to recover by closing structures
                logger.debug(f"Initial JSON parse failed: {e}. Attempting truncation recovery...")
                
                # Count braces/brackets to see if we're missing closing characters
                open_braces = json_str.count('{') - json_str.count('}')
                open_brackets = json_str.count('[') - json_str.count(']')
                
                if open_braces > 0 or open_brackets > 0:
                    # JSON appears truncated, try to close it
                    logger.debug(f"Truncation detected: {open_braces} open braces, {open_brackets} open brackets")
                    recovered_json = json_str + ('}' * open_braces) + (']' * open_brackets)
                    
                    try:
                        windows_raw = json.loads(recovered_json)
                        logger.info(f"Successfully recovered JSON by closing {open_braces} braces and {open_brackets} brackets")
                    except json.JSONDecodeError as recovery_e:
                        logger.error(f"Truncation recovery failed: {recovery_e}")
                        return []
                else:
                    # Not a truncation issue, re-raise
                    raise
            
            if not isinstance(windows_raw, list):
                windows_raw = [windows_raw]
            
            # Normalize response
            windows = []
            for window in windows_raw:
                bbox = window.get("bbox", [])
                if len(bbox) == 4:
                    # Convert percentage (0-100) to pixel coordinates using actual image dimensions
                    x1_pct, y1_pct, x2_pct, y2_pct = bbox
                    bbox = [
                        int(x1_pct * img_width / 100),
                        int(y1_pct * img_height / 100),
                        int(x2_pct * img_width / 100),
                        int(y2_pct * img_height / 100)
                    ]
                
                windows.append({
                    "bbox": bbox,
                    "confidence": float(window.get("confidence", 0.7)),
                    "type": window.get("type", "window"),
                })
            
            logger.debug(f"Detected {len(windows)} windows")
            return windows
        
        except (json.JSONDecodeError, AttributeError) as e:
            logger.debug(f"Failed to parse window response: {e}")
            return []

    def detect_doors(
        self, image_path: Union[str, Path]
    ) -> List[Dict[str, Any]]:
        """Detect doors using Unsloth Qwen (VLM door tier).

        Requests absolute pixel bboxes (Qwen's native convention). Large plans
        are resized for memory/speed; detections are parsed in the resized
        space and rescaled back to the original image dimensions.

        Args:
            image_path: Path to the floor plan image.

        Returns:
            Door detections in the original image's pixel space; [] on failure.
        """
        image = self._open_image_rgb(image_path)
        if image is None:
            return []
        resized, scale = self._resize_for_detection(image)
        doors = self._run_door_inference(resized)
        _rescale_bbox_objects(doors, _NO_RESCALE / scale)
        return doors

    def _open_image_rgb(self, image_path: Union[str, Path]) -> Optional[Any]:
        """Initialize the model and open the image as RGB; None on failure."""
        try:
            from PIL import Image
        except ImportError as error:
            logger.error("Pillow unavailable for door detection: %s", error)
            return None
        try:
            self.initialize()
            return Image.open(image_path).convert("RGB")
        except (OSError, ValueError) as error:
            logger.error("Could not open image: %s", error)
            return None

    def _resize_for_detection(self, image: Any) -> Tuple[Any, float]:
        """Downscale to fit the detection cap; return (image, scale)."""
        from PIL import Image

        width, height = image.size
        scale = _scale_for_max_dim(
            width, height, self.config.unsloth_detection_max_dim_px
        )
        if scale >= _NO_RESCALE:
            return image, scale
        resized = image.resize(
            (int(width * scale), int(height * scale)),
            Image.Resampling.LANCZOS,
        )
        return resized, scale

    def _run_door_inference(self, image: Any) -> List[Dict[str, Any]]:
        """Detect doors on an already-sized image; boxes in its pixel space."""
        width, height = image.size
        prompt = self._build_door_detection_prompt(width, height)
        text = self._generate_for_image(
            image, prompt, _DOOR_MAX_NEW_TOKENS
        )
        return self._parse_door_response(text, width, height)

    def _generate_for_image(
        self, image: Any, prompt: str, max_new_tokens: int
    ) -> str:
        """Run deterministic generation for one image+prompt; return text."""
        import torch

        inputs = self._encode_image_prompt(image, prompt)
        self._seed_generation()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **self._generation_kwargs(max_new_tokens),
            )
        input_len = inputs["input_ids"].shape[-1]
        text = self._decode_generated(output_ids, input_len)
        del inputs, output_ids
        torch.cuda.empty_cache()
        return text

    def _encode_image_prompt(self, image: Any, prompt: str) -> Any:
        """Build tokenized model inputs for one image + text prompt."""
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_templates.build_json_array_system_prompt()
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        return self.tokenizer(
            image, input_text, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)

    def _decode_generated(self, output_ids: Any, input_len: int) -> str:
        """Decode newly generated tokens, keeping only the post-reasoning answer.

        Qwen3 Thinking checkpoints emit ``reasoning </think> answer``: the
        opening ``<think>`` is injected by the chat template's generation
        prompt, so only the closing ``</think>`` appears among generated
        tokens. The reasoning text mentions bbox coordinates and echoes the
        prompt's placeholder example, which the JSON extractor then grabs
        instead of the real answer array. Dropping everything up to and
        including the final ``</think>`` leaves only the answer.

        Non-thinking checkpoints (the Instruct default) never emit
        ``</think>``, so the whole output is kept — no behavior change.
        """
        generated = output_ids[0][input_len:]
        if hasattr(generated, "cpu"):
            generated = generated.cpu()
        generated = self._answer_tokens_after_reasoning(generated)
        decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
        return decoded.strip()

    def _answer_tokens_after_reasoning(self, generated: Any) -> Any:
        """Slice ``generated`` to the tokens after the last ``</think>``.

        Returns ``generated`` unchanged when the reasoning end-token cannot be
        resolved or is absent, so the answer is never dropped for non-thinking
        checkpoints or reasoning that was truncated before it closed.
        """
        tok = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        if not hasattr(tok, "convert_tokens_to_ids"):
            return generated
        end_think_id = tok.convert_tokens_to_ids("</think>")
        unk_id = getattr(tok, "unk_token_id", None)
        if end_think_id is None or end_think_id == unk_id:
            return generated
        ids = generated.tolist() if hasattr(generated, "tolist") else list(generated)
        if end_think_id not in ids:
            return generated
        last = len(ids) - 1 - ids[::-1].index(end_think_id)
        return generated[last + 1:]

    def _build_door_detection_prompt(
        self, img_width: int, img_height: int
    ) -> str:
        """Build the pixel-coordinate door-detection prompt."""
        return prompt_templates.build_door_prompt_unsloth(
            img_width, img_height
        )

    def _parse_door_response(
        self, response_text: str, img_width: int, img_height: int
    ) -> List[Dict[str, Any]]:
        """Parse door detection response (integer pixel bbox)."""
        return _parse_pixel_bbox_objects(
            response_text, img_width, img_height, default_type="door"
        )


class VLMFactory:
    """Factory for creating VLM backend instances."""
    
    _backends = {
        "claude": ClaudeBackend,
        "qwen": Qwen2_5VLBackend,
        "unsloth": UnslothQwenBackend,
    }
    
    @classmethod
    def create(cls, config) -> VLMBackend:
        """Create VLM backend instance.
        
        Args:
            config: VLMConfig instance with backend field
        
        Returns:
            VLMBackend instance
        
        Raises:
            ValueError: If backend is not supported
        """
        backend_name = config.backend.lower()
        if backend_name not in cls._backends:
            raise ValueError(
                f"Unknown VLM backend: {backend_name}. "
                f"Available: {', '.join(cls._backends.keys())}"
            )
        
        backend_class = cls._backends[backend_name]
        logger.info(f"Creating VLM backend: {backend_name}")
        return backend_class(config)
    
    @classmethod
    def register(cls, name: str, backend_class) -> None:
        """Register a new VLM backend.
        
        Args:
            name: Backend identifier
            backend_class: Subclass of VLMBackend
        """
        cls._backends[name] = backend_class
        logger.info(f"Registered VLM backend: {name}")
