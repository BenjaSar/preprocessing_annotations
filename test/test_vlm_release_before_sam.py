#!/usr/bin/env python3
"""
Regression guard for the VLM/SAM GPU-memory collision (found auditing
sprint1_verify47, 2026-08-19): Step 3 (VLM room detection) and Step 3.5
(SAM label->room expansion) ran back-to-back in one process with the
VLM never released. Measured on the real run: 3,093 SAM "CUDA out of
memory" failures, 57/57 pages logging "SAM expanded 0 of N" -- SAM's
expansion never once succeeded because an 8B VLM was still resident.

Fix: AnnotationPipeline._release_vlm_before_sam() calls
self._vlm_annotator.release() before Step 3.5, guarded on the backing
field (not the lazy-create property) so a run that never touched the
VLM doesn't construct one just to no-op release it.

Covers:
  - guard fires only when use_sam=True AND a VLM was actually created
  - guard is a no-op (no crash, no construction) when the VLM was never
    used, or when use_sam=False
  - UnslothQwenBackend.release() actually clears model/tokenizer/
    initialized state (mocked -- no GPU/model load)
  - VLMAnnotator (Claude backend) has a working no-op release(), so the
    polymorphic call site never crashes regardless of which backend
    config.vlm.backend selects
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing_annotations.config import PipelineConfig
from preprocessing_annotations.orchestration.pipeline import AnnotationPipeline
from preprocessing_annotations.vlm.vlm_annotator import VLMAnnotator
from preprocessing_annotations.vlm.vlm_backend import UnslothQwenBackend


def _pipeline(use_sam: bool) -> AnnotationPipeline:
    config = PipelineConfig()
    config.use_sam = use_sam
    return AnnotationPipeline(config)


def test_release_called_when_sam_on_and_vlm_was_used():
    pipeline = _pipeline(use_sam=True)
    fake_vlm = MagicMock()
    pipeline._vlm_annotator = fake_vlm

    pipeline._release_vlm_before_sam()

    fake_vlm.release.assert_called_once()


def test_release_not_called_when_vlm_never_used():
    """No _vlm_annotator was ever created (still None) -- must not
    trigger the lazy-create property just to release nothing."""
    pipeline = _pipeline(use_sam=True)
    assert pipeline._vlm_annotator is None

    pipeline._release_vlm_before_sam()  # must not raise, must not create one

    assert pipeline._vlm_annotator is None


def test_release_not_called_when_sam_disabled():
    pipeline = _pipeline(use_sam=False)
    fake_vlm = MagicMock()
    pipeline._vlm_annotator = fake_vlm

    pipeline._release_vlm_before_sam()

    fake_vlm.release.assert_not_called()


def test_unsloth_backend_release_clears_state():
    """No GPU/model load -- construct the backend and fake an
    already-initialized state, then verify release() clears it."""
    backend = UnslothQwenBackend(config=MagicMock(qwen_device=None))
    backend.model = MagicMock()
    backend.tokenizer = MagicMock()
    backend.initialized = True

    backend.release()

    assert backend.model is None
    assert backend.tokenizer is None
    assert backend.initialized is False


def test_unsloth_backend_release_is_noop_when_not_initialized():
    """Must not touch torch/gc (and must not raise) when there is
    nothing loaded -- this is the path taken for a run where the VLM
    was constructed (property accessed) but initialize() never ran."""
    backend = UnslothQwenBackend(config=MagicMock(qwen_device=None))
    assert backend.initialized is False

    backend.release()  # must not raise

    assert backend.initialized is False


def test_claude_backend_release_is_a_safe_noop():
    annotator = VLMAnnotator()
    annotator.release()  # must not raise
