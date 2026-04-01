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

logger = logging.getLogger(__name__)


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
            
            # Parse response
            response_text = message.content[0].text
            rooms = self._parse_room_response(response_text)
            return rooms
            
        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            return []
    
    def _build_room_detection_prompt(self) -> str:
        """Build prompt for room detection."""
        return """Analyze this architectural floor plan and identify all rooms and spaces.

For each room or space visible, return a JSON object with:
{
    "room_id": "unique identifier like room_0, room_1",
    "room_type": "bedroom | bathroom | kitchen | living_room | dining_room | office | mechanical | electrical | storage | corridor | lobby | laundry | closet | garage | other",
    "room_name": "extracted room name or label, expand abbreviations",
    "bbox": [x1, y1, x2, y2] as percentage of image dimensions (0-100),
    "confidence": 0.0-1.0 confidence in detection,
    "metadata": {}
}

Rules:
- Only include rooms/spaces/areas. Exclude legends, title blocks, schedules, notes.
- Expand all abbreviations using architectural conventions (BR→Bedroom, KIT→Kitchen).
- Return valid JSON array only, no markdown or explanation.

Return ONLY a JSON array, e.g.: [{"room_id": "room_0", "room_type": "bedroom", ...}, ...]"""
    
    def _parse_room_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse JSON response from Claude."""
        try:
            # Try to extract JSON array from response
            rooms = json.loads(response_text)
            if not isinstance(rooms, list):
                rooms = [rooms]
            
            # Normalize response format
            normalized = []
            for idx, room in enumerate(rooms):
                normalized.append({
                    "room_id": room.get("room_id", f"room_{idx}"),
                    "polygon": None,  # Claude returns bbox, not polygon
                    "bbox": room.get("bbox"),
                    "room_type": room.get("room_type", "other"),
                    "room_name": room.get("room_name"),
                    "confidence": float(room.get("confidence", 0.5)),
                    "metadata": room.get("metadata", {})
                })
            
            return normalized
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response as JSON: {e}")
            logger.debug(f"Response text: {response_text[:200]}...")
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
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
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
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map=self.device,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
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
            from PIL import Image
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Build prompt
            prompt = self._build_room_detection_prompt()
            
            # Prepare input
            inputs = self.processor(
                text=prompt,
                images=[image],
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=1024)
            
            # Decode response
            response_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            
            # Parse response
            rooms = self._parse_room_response(response_text)
            return rooms
            
        except Exception as e:
            logger.error(f"Qwen2.5-VL inference failed: {e}")
            return []
    
    def _build_room_detection_prompt(self) -> str:
        """Build prompt for room detection."""
        return """Analyze this architectural floor plan image and identify all rooms and spaces.

For each room, extract:
- room_type: bedroom, bathroom, kitchen, living_room, dining_room, office, mechanical, electrical, storage, corridor, lobby, laundry, closet, garage, other
- room_name: extracted label, expand abbreviations (BR→Bedroom, KIT→Kitchen)
- approximate bbox coordinates as [x1, y1, x2, y2] where 0-100 is image dimensions

Return ONLY a JSON array, no markdown:
[{"room_id": "room_0", "room_type": "bedroom", "room_name": "Bedroom 1", "bbox": [10, 20, 40, 50], "confidence": 0.95}, ...]

Rules:
- Only rooms/spaces; exclude legends, notes, schedules
- Expand all abbreviations
- Return valid JSON only"""
    
    def _parse_room_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse JSON response from Qwen2.5-VL."""
        try:
            # Extract JSON from response (may be wrapped in markdown)
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if not json_match:
                logger.error("No JSON array found in Qwen response")
                return []
            
            rooms = json.loads(json_match.group())
            if not isinstance(rooms, list):
                rooms = [rooms]
            
            # Normalize response format
            normalized = []
            for idx, room in enumerate(rooms):
                # Convert bbox percentages to pixel coordinates
                bbox = room.get("bbox")
                if bbox and len(bbox) == 4:
                    # Assuming image is max 1000x1000 for normalization
                    # In real use, this would need actual image dimensions
                    bbox = [x * 10 for x in bbox]  # 0-100 -> 0-1000
                
                normalized.append({
                    "room_id": room.get("room_id", f"room_{idx}"),
                    "polygon": None,  # Qwen returns bbox, not polygon
                    "bbox": bbox,
                    "room_type": room.get("room_type", "other"),
                    "room_name": room.get("room_name"),
                    "confidence": float(room.get("confidence", 0.5)),
                    "metadata": room.get("metadata", {})
                })
            
            return normalized
        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"Failed to parse Qwen response: {e}")
            logger.debug(f"Response text: {response_text[:200]}...")
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
    """
    
    # Mapping from short model key to Unsloth HuggingFace model ID
    MODEL_REGISTRY = {
        "qwen2.5-vl-7b": "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
        "qwen3-vl-2b": "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit",
        "qwen3-vl-4b": "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit",
        "qwen3-vl-8b": "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
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
            from unsloth import FastVisionModel
            import torch
            
            # Resolve model ID
            model_key = getattr(self.config, 'unsloth_model', 'qwen2.5-vl-7b').lower()
            if model_key not in self.MODEL_REGISTRY:
                logger.warning(
                    f"Unknown Unsloth model key: {model_key}. "
                    f"Available: {', '.join(self.MODEL_REGISTRY.keys())}. "
                    f"Using default: qwen2.5-vl-7b"
                )
                model_key = "qwen2.5-vl-7b"
            
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
            
            # Build room detection prompt
            prompt = self._build_room_detection_prompt()
            
            # Prepare chat messages in Unsloth format
            messages = [
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
            
            # Generate response
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    use_cache=True,
                    temperature=1.0,
                    top_p=0.9,
                )
            
            # Decode response
            response_text = self.tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True
            )
            
            # Parse room detections from response
            rooms = self._parse_room_response(response_text)
            return rooms
        
        except Exception as e:
            logger.error(f"Unsloth Qwen inference failed: {e}")
            return []
    
    def _build_room_detection_prompt(self) -> str:
        """Build prompt for room detection."""
        return """Analyze this architectural floor plan image and identify all rooms and spaces.

For each room, extract:
- room_type: bedroom, bathroom, kitchen, living_room, dining_room, office, mechanical, electrical, storage, corridor, lobby, laundry, closet, garage, other
- room_name: extracted label, expand abbreviations (BR→Bedroom, KIT→Kitchen)
- approximate bbox coordinates as [x1, y1, x2, y2] where 0-100 is image dimensions

Return ONLY a JSON array, no markdown:
[{"room_id": "room_0", "room_type": "bedroom", "room_name": "Bedroom 1", "bbox": [10, 20, 40, 50], "confidence": 0.95}, ...]

Rules:
- Only rooms/spaces; exclude legends, notes, schedules
- Expand all abbreviations
- Return valid JSON only"""
    
    def _parse_room_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse JSON response from Unsloth Qwen."""
        try:
            # Extract JSON array from response (may be wrapped in markdown or other text)
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if not json_match:
                logger.error("No JSON array found in Unsloth Qwen response")
                return []
            
            rooms = json.loads(json_match.group())
            if not isinstance(rooms, list):
                rooms = [rooms]
            
            # Normalize response format
            normalized = []
            for idx, room in enumerate(rooms):
                # Convert bbox percentages (0-100) to pixel coordinates
                bbox = room.get("bbox")
                if bbox and len(bbox) == 4:
                    # Assuming image is max 1000x1000 for normalization
                    # In real use, this would need actual image dimensions
                    bbox = [x * 10 for x in bbox]  # 0-100 -> 0-1000
                
                normalized.append({
                    "room_id": room.get("room_id", f"room_{idx}"),
                    "polygon": None,  # Qwen returns bbox, not polygon
                    "bbox": bbox,
                    "room_type": room.get("room_type", "other"),
                    "room_name": room.get("room_name"),
                    "confidence": float(room.get("confidence", 0.5)),
                    "metadata": room.get("metadata", {})
                })
            
            return normalized
        
        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"Failed to parse Unsloth Qwen response: {e}")
            logger.debug(f"Response text: {response_text[:200]}...")
            return []


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
