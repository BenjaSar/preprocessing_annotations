"""
SAM3 adapter — the only file in this package allowed to import `sam3`.

Translates Sam3Processor's tensor-ish output into plain Detection objects so
the rest of the codebase never depends on the vendor API shape.
"""

import inspect
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Concept label the model uses when only geometric (exemplar) prompts are set.
_VISUAL_CONCEPT_LABEL = "exemplar"

# Cached SAM3 checkpoint identity (mirrors sam3.model_builder's own values).
# facebook/sam3 is a GATED HF repo; the weights are already downloaded and the
# checkpoint is pinned, so the adapter resolves the cached file directly
# (local_files_only) and passes it to the builder. This avoids the builder's
# default online path, whose HEAD revalidation returns 403 (gated repo, no
# token in-process) before silently falling back to cache -- same load, minus
# the misleading 403 log line and the network round-trip.
_SAM3_HF_REPO = "facebook/sam3"
_SAM3_CKPT_FILENAME = "sam3.pt"

# T-J0a: names the single-object interactive-seed contract is checked
# against (build_sam3_image_model's toggle, SAM3InteractiveImagePredictor's
# constructor/set_image/predict parameters). Verified once by reading
# sam3.model_builder and sam3.model.sam1_task_predictor; re-checked here at
# import time via inspect.signature (no model load, no GPU) so API drift in
# a future sam3 version is caught, not assumed away.
_INTERACTIVE_BUILDER_FLAG = "enable_inst_interactivity"
_INTERACTIVE_INIT_PARAM = "sam_model"
_INTERACTIVE_SET_IMAGE_PARAM = "image"
_INTERACTIVE_PREDICT_PARAMS = ("point_coords", "point_labels", "box")

from .domain import Detection
from .ports import PromptableSegmenter

logger = logging.getLogger(__name__)


class Sam3SegmentationError(Exception):
    """Raised when SAM3 model loading or inference fails."""


@dataclass(frozen=True)
class InteractiveApiReport:
    """T-J0a result: does SAM3 expose a single-object interactive-seed API.

    Each field is one static contract check via ``inspect.signature`` --
    no model weights are loaded and no GPU is used.

    CAVEAT (T-J0b, runtime spike): ``available=True`` means the API
    SHAPE exists, not that it works standalone. Driving
    ``SAM3InteractiveImagePredictor`` via
    ``build_sam3_image_model(enable_inst_interactivity=True)`` crashes --
    ``build_tracker()`` is called without ``with_backbone=True``, so
    ``.set_image()`` hits a None backbone. The vendor's own "interactive"
    example notebook (``sam3_image_interactive.ipynb``) does not use this
    path at all; it drives ``Sam3Processor.add_geometric_prompt`` (the
    same mechanism as ``Sam3Adapter.segment_by_exemplars``). There is no
    separate usable single-object interactive API for images in this
    SAM3 release -- do not build an adapter around it.
    """

    importable: bool
    builder_supports_interactivity: bool
    predictor_has_expected_init: bool
    predictor_has_set_image: bool
    predictor_has_seed_predict: bool
    detail: str = ""

    @property
    def available(self) -> bool:
        """True only if every contract check passed."""
        return (
            self.importable
            and self.builder_supports_interactivity
            and self.predictor_has_expected_init
            and self.predictor_has_set_image
            and self.predictor_has_seed_predict
        )


def _has_params(func: object, names: Tuple[str, ...]) -> bool:
    """True if every name in `names` is a parameter of `func`."""
    params = inspect.signature(func).parameters
    return all(name in params for name in names)


def _unimportable_report(detail: str) -> InteractiveApiReport:
    """Report used when the sam3 package itself is not importable."""
    return InteractiveApiReport(
        importable=False,
        builder_supports_interactivity=False,
        predictor_has_expected_init=False,
        predictor_has_set_image=False,
        predictor_has_seed_predict=False,
        detail=detail,
    )


def _build_report(
    build_fn: object, predictor_cls: type
) -> InteractiveApiReport:
    """Report built from the successfully imported builder/predictor."""
    return InteractiveApiReport(
        importable=True,
        builder_supports_interactivity=_has_params(
            build_fn, (_INTERACTIVE_BUILDER_FLAG,)
        ),
        predictor_has_expected_init=_has_params(
            predictor_cls.__init__, (_INTERACTIVE_INIT_PARAM,)
        ),
        predictor_has_set_image=_has_params(
            predictor_cls.set_image, (_INTERACTIVE_SET_IMAGE_PARAM,)
        ),
        predictor_has_seed_predict=_has_params(
            predictor_cls.predict, _INTERACTIVE_PREDICT_PARAMS
        ),
    )


def check_interactive_api() -> InteractiveApiReport:
    """Statically verify SAM3's single-object interactive-seed API (T-J0a).

    Introspects ``build_sam3_image_model`` and
    ``SAM3InteractiveImagePredictor`` via ``inspect.signature`` only.

    Returns:
        An InteractiveApiReport; ``importable=False`` (all checks False)
        if the ``sam3`` package itself cannot be imported.
    """
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam1_task_predictor import (
            SAM3InteractiveImagePredictor as Predictor,
        )
    except ImportError as error:
        return _unimportable_report(f"sam3 package not importable: {error}")
    return _build_report(build_sam3_image_model, Predictor)


def _resolve_cached_checkpoint() -> str:
    """Resolve the pinned SAM3 checkpoint from the local HF cache.

    Shared by every adapter that builds a SAM3 model (concept-mode
    Sam3Adapter, interactive-mode Sam3InteractiveAdapter) -- one
    resolution path, not one copy per adapter. Uses ``local_files_only``
    so the gated repo is never revalidated over the network (that HEAD
    returns 403 without a token).

    Raises:
        Sam3SegmentationError: if the checkpoint is not cached.
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
        raise Sam3SegmentationError(
            f"SAM3 checkpoint not cached ({_SAM3_HF_REPO}/"
            f"{_SAM3_CKPT_FILENAME}); download once with a valid "
            f"HF token before offline use"
        ) from error


class Sam3Adapter(PromptableSegmenter):
    """Lazy-loaded SAM3 image model, one processor instance per image set."""

    def __init__(self, device: str = "cuda") -> None:
        self._device = device
        self._model = None
        self._processor_cls = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as e:
            raise Sam3SegmentationError(f"sam3 package not importable: {e}") from e

        logger.info("Loading SAM3 image model (device=%s)...", self._device)
        checkpoint_path = _resolve_cached_checkpoint()
        self._model = build_sam3_image_model(checkpoint_path=checkpoint_path)
        self._processor_cls = Sam3Processor
        logger.info("SAM3 model loaded")

    def segment(
        self, image: np.ndarray, prompt: str, score_threshold: float = 0.3
    ) -> List[Detection]:
        import torch  # local: only this adapter is allowed to know sam3 needs torch

        self._ensure_loaded()
        # Sam3Processor filters candidates against confidence_threshold BEFORE
        # upsampling masks to full image resolution — this is a memory guard,
        # not just a result filter (an unfiltered query set OOM'd a T4 on a
        # 4500x3375 image: each surviving candidate gets a full-res mask).
        # So score_threshold must reach the processor, not just our own loop.
        processor = self._processor_cls(self._model, confidence_threshold=score_threshold)

        # SAM3's weights are bf16; every forward pass must run under this
        # autocast or matmuls raise "mat1 and mat2 must have the same dtype".
        with torch.autocast(self._device, dtype=torch.bfloat16):
            pil_image = Image.fromarray(image)
            state = processor.set_image(pil_image)
            # positional args — Sam3Processor.set_text_prompt(self, prompt, state)
            state = processor.set_text_prompt(prompt, state)

        boxes = state.get("boxes")
        masks = state.get("masks")
        scores = state.get("scores")
        if boxes is None or len(boxes) == 0:
            return []

        detections = []
        for i in range(len(boxes)):
            score = float(scores[i]) if scores is not None else 1.0
            if score < score_threshold:
                continue
            x1, y1, x2, y2 = [float(v) for v in _to_list(boxes[i])]
            mask = _to_mask(masks[i]) if masks is not None else None
            detections.append(
                Detection(bbox_xyxy=(x1, y1, x2, y2), label=prompt, score=score, mask=mask)
            )
        return detections

    def segment_by_exemplars(
        self,
        image: np.ndarray,
        exemplar_boxes_cxcywh: List[Tuple[float, float, float, float]],
        score_threshold: float,
    ) -> List[Detection]:
        """Detect all instances matching positive box exemplars (PCS).

        Exemplar boxes are [cx, cy, w, h] normalized to [0, 1]; with no text
        prompt the model treats them as a visual concept and returns every
        matching instance.
        """
        import torch

        if not exemplar_boxes_cxcywh:
            return []
        self._ensure_loaded()
        processor = self._processor_cls(
            self._model, confidence_threshold=score_threshold
        )
        with torch.autocast(self._device, dtype=torch.bfloat16):
            state = processor.set_image(Image.fromarray(image))
            for box in exemplar_boxes_cxcywh:
                processor.add_geometric_prompt(list(box), True, state)
        return _detections_from_state(
            state, _VISUAL_CONCEPT_LABEL, score_threshold
        )


def _detections_from_state(
    state: Dict, label: str, score_threshold: float
) -> List[Detection]:
    """Build Detections from a processor state dict (xyxy pixel boxes)."""
    boxes = state.get("boxes")
    scores = state.get("scores")
    masks = state.get("masks")
    if boxes is None or len(boxes) == 0:
        return []
    detections: List[Detection] = []
    for i in range(len(boxes)):
        score = float(scores[i]) if scores is not None else 1.0
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = [float(v) for v in _to_list(boxes[i])]
        mask = _to_mask(masks[i]) if masks is not None else None
        detections.append(
            Detection(
                bbox_xyxy=(x1, y1, x2, y2),
                label=label,
                score=score,
                mask=mask,
            )
        )
    return detections


def _to_list(value) -> list:
    return value.tolist() if hasattr(value, "tolist") else list(value)


def _to_mask(value) -> Optional[np.ndarray]:
    """SAM3 mask tensors are (1, H, W) bool; squeeze to plain (H, W)."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.astype(bool)


if __name__ == "__main__":
    import json
    from dataclasses import asdict

    logging.basicConfig(level=logging.INFO)
    print(json.dumps(asdict(check_interactive_api()), indent=2))
