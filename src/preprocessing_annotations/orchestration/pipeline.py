"""
Main annotation pipeline orchestration for MEP floor plans.

This module provides the AnnotationPipeline class that coordinates
all annotation steps: PDF extraction, OCR, VLM annotation, SAM
segmentation, and export.
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..config import PipelineConfig
from ..ingestion.pdf_extractor import PDFExtractor, PageTypeClassifier
from ..ingestion.ocr_extractor import MEPTextExtractor, RoomCandidate, TextDetection
from ..ingestion.two_pass_ocr_extractor import TwoPassOCRExtractor
from ..vlm.vlm_annotator import VLMAnnotator
from ..vlm.semantic_reconciler import SemanticReconciler
from ..vlm.vlm_backend import VLMFactory
from ..vlm import prompt_templates
from ..detection.sam_segmenter import RoomSegmenter
from ..detection.cubicasa5k_detector import CubiCasa5KDetector
from ..bbox.bbox_visualizer import BboxVisualizer
from ..export.exporters import LabelStudioExporter, ReviewPrioritizer, COCOExporter, CoverageReporter
from ..automation import (
    LabelNormalizer, QualityChecker, RegionExtractor,
    ImageResizer, SemanticRoomValidator, TaxonomyNormalizer,
    filter_by_confidence, validate_for_sft, prepare_sft_annotation,
    SFTAnnotationBuilder, build_annotation_json, ConfidenceComputer
)
from ..automation.sft_validator import _merge_vlm_and_ocr
from ..automation.taxonomy import normalize_to_mandatory, get_extended_type
from ..automation.abbreviation_ocr_recovery import (
    AbbreviationOCRRecovery, ResidentialAbbreviationRecovery
)
from ..detection.window_detector import (
    WindowDetector, WindowDetection, WindowDetectionTier, RoomWindowMapping,
    apply_window_suffixes, map_detections_to_rooms, merge_object_mappings,
)
from ..detection.yolo_detector import YoloObjectDetector
from ..detection.sam3_exemplar_detector import Sam3ExemplarDetector
from .run_lock import single_instance

logger = logging.getLogger(__name__)


# Step 6: Skip-reason tagging helper
def log_skip_reason(image_name: str, reason: str, stage: str = "unknown") -> None:
    """Log why an image/room was skipped with structured tagging."""
    logger.warning(f"SKIP[{stage}] {image_name}: {reason}")


class PipelineError(Exception):
    """Raised when pipeline execution fails."""

    pass


class AnnotationPipeline:
    """
    Orchestrates the full MEP floor plan annotation pipeline.

    Pipeline steps:
    1. PDF Extraction - Convert PDFs to high-resolution images
    2. OCR Extraction - Extract room labels and text
    3. VLM Annotation - Zero-shot annotation with Claude (optional)
    4. Post-Processing - Normalize labels, validate quality, extract regions
    5. SAM Refinement - Refine boundaries with SAM (optional)
    6. Prioritize - Flag low-confidence images for review
    7. Export - Generate Label Studio import file

    Attributes:
        config: PipelineConfig with all component configurations.

    Example:
        pipeline = AnnotationPipeline(PipelineConfig.for_high_detail())
        pipeline.run(input_dir="./pdfs", output_dir="./dataset")
    """

    # All image extensions the pipeline can process.
    # Used by _iter_images() to replace the previous glob("*.png") calls that
    # silently dropped JPG/TIFF inputs copied into images_dir by Form B/D.
    _IMAGE_EXTS: frozenset = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff"})

    def _iter_images(self, images_dir: Path) -> list:
        """
        Return all supported image files in images_dir, sorted by name.

        Handles mixed-case extensions (.PNG, .Jpg, .TIFF) that shutil.copy2
        preserves when the source file has uppercase extensions (Form B / D
        input).  Deduplicates so the same logical file is never yielded twice
        even if the filesystem is case-insensitive.

        Replaces every ``sorted(images_dir.glob("*.png"))`` call in the
        pipeline, which silently omitted any non-PNG image.

        Returns:
            Sorted list of Path objects for all supported images.
        """
        seen: set = set()
        result: list = []
        for p in sorted(images_dir.iterdir()):
            if p.suffix.lower() in self._IMAGE_EXTS and p not in seen:
                seen.add(p)
                result.append(p)
        return result

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        # Initialize components lazily
        self._pdf_extractor = None
        self._ocr_extractor = None
        self._vlm_annotator = None
        self._sam_segmenter = None
        self._cubicasa_room_detector = None
        self._exporter = None
        self._prioritizer = None
        self._label_normalizer = None
        self._quality_checker = None
        self._region_extractor = None
        self._window_detector = None
        self._yolo_object_detector = None
        self._sam3_exemplar_detector = None
        self._bbox_visualizer = None

        # Initialize SFT annotation builder (used for mandatory schema output)
        self.sft_builder = SFTAnnotationBuilder()

        # Set by process() at the start of each run; used by _source_pdf_for()
        self._input_path: Optional[Path] = None

        # Set by run() at the start of each run; used by the bbox_visualizer
        # property to write debug overlays under <output_dir>/debug_overlays/.
        # Previously the visualizer probed self.config for an output_dir
        # attribute that PipelineConfig does not define, so debug overlays
        # were silently never written.
        self._output_dir: Optional[Path] = None

    # ── PDF path reconstruction ───────────────────────────────────────────────

    def _source_pdf_for(self, img_path: Optional[Path]) -> Optional[Path]:
        """
        Attempt to reconstruct the source PDF path from an extracted image path.

        Images are named ``{pdf_stem}_page{N:03d}.png`` by the PDFExtractor.
        This method strips the ``_pageNNN`` suffix and looks for a matching
        ``.pdf`` file in the pipeline's input directory (self._input_path).

        Used by window detection to enable Tier 1 (PDF layer extraction)
        without threading the pdf_path through every save method signature.

        Returns:
            Path to the source PDF if it can be found, otherwise None.
            A None return causes Tier 1 to be skipped gracefully.
        """
        if img_path is None or self._input_path is None:
            return None
        try:
            import re
            stem = img_path.stem  # e.g. "326 ROCKAWAY - AVI-ON LAYOUT_page001"
            pdf_stem = re.sub(r"_page\d+$", "", stem)
            if not pdf_stem or pdf_stem == stem:
                return None  # Not a page-extracted image

            # Look for the PDF in the input path (file or directory)
            candidate = (
                self._input_path
                if self._input_path.suffix.lower() == ".pdf"
                else self._input_path / f"{pdf_stem}.pdf"
            )
            return candidate if candidate.is_file() else None
        except Exception:
            return None

    @property
    def pdf_extractor(self) -> PDFExtractor:
        if self._pdf_extractor is None:
            self._pdf_extractor = PDFExtractor(self.config.pdf)
        return self._pdf_extractor

    def _pdf_text_layer_detections(self, img_path: Path) -> Optional[List[TextDetection]]:
        """Y1 (2026-08-20 audit): fetch PDF text-layer detections for
        img_path, for the caller to pass as find_rooms_pass1's
        extra_detections -- candidate classification ONLY.

        Corrects the original prototype's integration point: that design
        merged PDF detections into raw_detections itself, which also
        feeds compute_exclusion_zones and is CONFIRMED unsafe there
        (measured on Lake Shore Electrical p002: 133->828 detections,
        3->16 exclusion zones, the 13 new zones covering real classrooms,
        not BOM tables -- see PipelineConfig.use_pdf_text_layer's
        docstring for the full incident). This method does no merging at
        all -- it returns PDF-only detections; MEPTextExtractor.
        extract_and_find_rooms does the OCR-wins collision filtering,
        because that is the first point OCR's own raw_detections exists
        to filter against.

        Returns None (not []) when the flag is off, the source PDF can't
        be located, or the filename has no parseable page index --
        find_rooms_pass1's extra_detections=None is its own documented
        byte-identical-to-before path; returning None here keeps that
        contract explicit rather than relying on an empty list behaving
        the same way.
        """
        if not self.config.use_pdf_text_layer:
            return None

        pdf_path = self._source_pdf_for(img_path)
        if pdf_path is None:
            return None

        import re
        match = re.search(r"_page(\d+)$", img_path.stem)
        if not match:
            return None
        page_index = int(match.group(1))

        return self.pdf_extractor.extract_text_layer(pdf_path, page_index) or None

    @property
    def ocr_extractor(self) -> TwoPassOCRExtractor:
        """
        Lazy-loaded OCR extractor with two-pass strategy.
        
        Pass 1: PaddleOCR (all regions, baseline)
        Pass 2: VLM fallback (for low-confidence < 0.7)
        
        Returns TwoPassOCRExtractor which wraps MEPTextExtractor for Pass 1
        and uses VLM backend for Pass 2 fallback.
        """
        if self._ocr_extractor is None:
            vlm_backend = self.vlm_annotator if self.config.use_vlm else None
            self._ocr_extractor = TwoPassOCRExtractor(
                ocr_config=self.config.ocr,
                vlm_backend=vlm_backend,
                confidence_threshold=0.92,  # PaddleOCR routinely returns >0.97; trigger VLM fallback only for genuinely uncertain tokens
            )
        return self._ocr_extractor

    @property
    def vlm_annotator(self):
        """Return appropriate VLM backend (Claude VLMAnnotator, Qwen, or Unsloth)."""
        if self._vlm_annotator is None:
            backend_name = self.config.vlm.backend.lower()
            if backend_name in ("qwen", "unsloth"):
                # Use VLMFactory to create Qwen or Unsloth backend
                self._vlm_annotator = VLMFactory.create(self.config.vlm)
            else:
                # Default to Claude backend (VLMAnnotator)
                self._vlm_annotator = VLMAnnotator(self.config.vlm)
        return self._vlm_annotator

    @property
    def sam_segmenter(self) -> RoomSegmenter:
        if self._sam_segmenter is None:
            self._sam_segmenter = RoomSegmenter(self.config.sam)
        return self._sam_segmenter

    @property
    def cubicasa_room_detector(self) -> CubiCasa5KDetector:
        if self._cubicasa_room_detector is None:
            cfg = self.config.cubicasa_rooms
            self._cubicasa_room_detector = CubiCasa5KDetector(
                model_path=Path(cfg.checkpoint_path), device=cfg.device
            )
        return self._cubicasa_room_detector

    @property
    def exporter(self) -> LabelStudioExporter:
        if self._exporter is None:
            self._exporter = LabelStudioExporter(self.config.export)
        return self._exporter

    @property
    def prioritizer(self) -> ReviewPrioritizer:
        if self._prioritizer is None:
            self._prioritizer = ReviewPrioritizer(self.config.export)
        return self._prioritizer

    @property
    def label_normalizer(self) -> LabelNormalizer:
        if self._label_normalizer is None:
            self._label_normalizer = LabelNormalizer()
        return self._label_normalizer

    @property
    def quality_checker(self) -> QualityChecker:
        if self._quality_checker is None:
            self._quality_checker = QualityChecker()
        return self._quality_checker

    @property
    def region_extractor(self) -> RegionExtractor:
        if self._region_extractor is None:
            self._region_extractor = RegionExtractor(padding_pct=0.1)
        return self._region_extractor

    @property
    def window_detector(self) -> WindowDetector:
        if self._window_detector is None:
            self._window_detector = WindowDetector(config=self.config)
        return self._window_detector

    @property
    def yolo_object_detector(self) -> YoloObjectDetector:
        if self._yolo_object_detector is None:
            self._yolo_object_detector = YoloObjectDetector(
                config=self.config.yolo_objects
            )
        return self._yolo_object_detector

    @property
    def sam3_exemplar_detector(self) -> Sam3ExemplarDetector:
        if self._sam3_exemplar_detector is None:
            self._sam3_exemplar_detector = Sam3ExemplarDetector(
                config=self.config.sam3_exemplar
            )
        return self._sam3_exemplar_detector

    @property
    def bbox_visualizer(self) -> BboxVisualizer:
        if self._bbox_visualizer is None:
            debug_dir = self._output_dir / "debug_overlays" if self._output_dir else None
            self._bbox_visualizer = BboxVisualizer(output_dir=str(debug_dir) if debug_dir else None)
            if debug_dir:
                logger.info(f"Debug bbox overlays enabled → {debug_dir}")
            else:
                logger.warning("Debug bbox overlays disabled (no output_dir set)")
        return self._bbox_visualizer

    def run(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        skip_existing: bool = False,
    ) -> Dict:
        """
        Run the full annotation pipeline.

        Args:
            input_dir: One of:
                - A directory containing PDF files (original behaviour)
                - A directory containing PNG/JPG images (direct image input)
                - A single PDF file path
                - A single image file path (PNG/JPG/TIFF)
              Mixed directories (PDFs + images) prefer PDF extraction.
            output_dir: Output directory for images and annotations.
            skip_existing: Skip images that already have annotations.

        Returns:
            Dictionary with pipeline statistics.

        Raises:
            PipelineError: If a critical step fails.
        """
        import shutil as _shutil
        from datetime import datetime, timezone

        # Q1 (run-manifest provenance): this run's own start time/PID, not
        # borrowed from setup_logging's file-handler timestamp (that's a
        # CLI-only side channel set before this method is even called, and
        # run() must also work when invoked programmatically without it).
        _run_start_utc = datetime.now(timezone.utc)
        _run_timestamp = _run_start_utc.strftime("%Y-%m-%d_%H%M%S")
        _run_pid = os.getpid()

        from .mlflow_tracking import start_run, log_params, log_tags
        _mlflow_run_id = start_run(self.config.mlflow)
        if _mlflow_run_id:
            log_params(self._build_config_dict())
            log_tags({"experiment_name": self.config.mlflow.experiment_name})

        # Per-run accumulator for object-detection corroboration signal
        # (_apply_object_attributes below); logged as an MLflow metric at
        # run-end regardless of whether corroboration_weight is nonzero,
        # so it's visible per run whether the object-evidence path found
        # anything even while inert (weight=0.0 default).
        self._corroboration_totals = {"evidence_count": 0, "evidence_score_sum": 0.0}

        input_path = Path(input_dir)
        output_dir = Path(output_dir)

        # Store input_path on self so save methods can reconstruct PDF paths
        # for Tier 1 window detection without changing every method signature.
        self._input_path = input_path

        # Store output_dir on self so the lazy bbox_visualizer property can
        # locate the debug_overlays subdirectory.
        self._output_dir = output_dir

        # Setup output directories
        images_dir = output_dir / "images"
        annotations_dir = output_dir / "annotations"

        images_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)

        # Per-page per-stage metrics log.
        # Appended throughout run() so a crash mid-run still preserves prior rows.
        _metrics_path = output_dir / "run_metrics.jsonl"
        _metrics_fh = open(_metrics_path, "a")

        def _emit_metric(page_id: str, stage: str, t0: float,
                         outcome: str, reject_reason: str = "",
                         room_count_in: int = 0, room_count_out: int = 0) -> None:
            record = {
                "page_id": page_id,
                "stage": stage,
                "duration_ms": round((time.time() - t0) * 1000),
                "outcome": outcome,
                "reject_reason": reject_reason,
                "room_count_in": room_count_in,
                "room_count_out": room_count_out,
            }
            _metrics_fh.write(json.dumps(record) + "\n")
            _metrics_fh.flush()
            from .mlflow_tracking import log_stage_metric
            log_stage_metric(stage, f"{stage}_duration_ms", record["duration_ms"])
            log_stage_metric(stage, f"{stage}_outcome_{outcome}", 1)

        # Accumulate per-step drop counts across all pages for run summary.
        run_drop_attribution: Dict[str, int] = {}

        stats = {
            "pdfs_processed": 0,
            "images_extracted": 0,
            "images_annotated": 0,
            "rooms_detected": 0,
            "annotations_normalized": 0,
            "annotations_with_issues": 0,
            "total_quality_issues": 0,
            "room_regions_extracted": 0,
            "flagged_for_review": 0,
        }

        # ------------------------------------------------------------------ #
        # Step 1: Resolve input → images_dir                                   #
        #                                                                      #
        # The pipeline accepts four input forms:                               #
        #   A. PDF directory  → batch_extract() → images_dir  (original)     #
        #   B. Image directory → copy images   → images_dir  (NEW)           #
        #   C. Single PDF file → extract_to_directory → images_dir  (NEW)    #
        #   D. Single image file → copy to images_dir  (NEW)                 #
        #                                                                      #
        # Previously: only form A was supported.  Providing a directory of    #
        # PNG images or a single file silently produced zero output because    #
        # batch_extract() globs for *.pdf and returns an empty dict.          #
        # ------------------------------------------------------------------ #
        _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

        self._print_step("STEP 1: Resolving input to images")

        if not input_path.exists():
            raise PipelineError(f"Input path does not exist: {input_path}")

        # --- Form C / D: single file ----------------------------------------
        if input_path.is_file():
            suffix = input_path.suffix.lower()
            if suffix == ".pdf":
                logger.info(f"Single PDF input: {input_path.name}")
                try:
                    extracted = self.pdf_extractor.extract_to_directory(
                        input_path, images_dir
                    )
                    stats["pdfs_processed"] = 1
                    stats["images_extracted"] = len(extracted)
                    logger.info(f"  Extracted {len(extracted)} pages from {input_path.name}")
                except Exception as e:
                    raise PipelineError(f"PDF extraction failed: {e}") from e
            elif suffix in _IMAGE_EXTS:
                logger.info(f"Single image input: {input_path.name}")
                dst = images_dir / input_path.name
                if not (skip_existing and dst.exists()):
                    _shutil.copy2(str(input_path), str(dst))
                stats["images_extracted"] = 1
                logger.info(f"  Copied {input_path.name} to images/")
            else:
                raise PipelineError(
                    f"Unsupported input file type '{suffix}'. "
                    f"Expected .pdf or {sorted(_IMAGE_EXTS)}."
                )

        # --- Form A / B: directory ------------------------------------------
        elif input_path.is_dir():
            pdf_files = sorted(input_path.glob("*.pdf"))

            # Case-insensitive image glob (handles .PNG, .JPG, etc.)
            image_files = sorted(
                p for p in input_path.iterdir()
                if p.suffix.lower() in _IMAGE_EXTS
            )

            if pdf_files:
                # Form A: PDF directory (original flow)
                if image_files:
                    logger.warning(
                        f"Input directory contains {len(image_files)} image file(s) "
                        f"alongside {len(pdf_files)} PDF(s). Only PDFs will be processed. "
                        f"Image files ignored: {[p.name for p in image_files]}"
                    )
                logger.info(
                    f"PDF directory input: {len(pdf_files)} PDF(s) in {input_path}"
                )
                try:
                    extraction_results = self.pdf_extractor.batch_extract(
                        input_path, images_dir
                    )
                    stats["pdfs_processed"] = len(extraction_results)
                    stats["images_extracted"] = sum(
                        len(paths) for paths in extraction_results.values()
                    )
                    logger.info(
                        f"  Extracted {stats['images_extracted']} images "
                        f"from {stats['pdfs_processed']} PDFs"
                    )
                except Exception as e:
                    raise PipelineError(f"PDF extraction failed: {e}") from e

            elif image_files:
                # Form B: image directory (new — user provides PNG/JPG directly)
                logger.info(
                    f"Image directory input: {len(image_files)} image(s) in {input_path}"
                )
                copied = 0
                for src in image_files:
                    dst = images_dir / src.name
                    if not (skip_existing and dst.exists()):
                        _shutil.copy2(str(src), str(dst))
                        copied += 1
                stats["images_extracted"] = len(image_files)
                logger.info(f"  Copied {copied} images to images/")

            else:
                raise PipelineError(
                    f"No PDF or image files found in {input_path}. "
                    f"Expected *.pdf or {sorted(_IMAGE_EXTS)} files."
                )

        else:
            raise PipelineError(f"Input path is neither a file nor a directory: {input_path}")

        # Step 1b: Plan-type classification — skip notes/legend/schedule pages.
        # Runs before OCR to eliminate the largest source of false room candidates
        # at the source, reducing OCR load by 30–50% on typical MEP sets.

        # Per-image status tracking — built BEFORE classification so that
        # images removed by the classifier are still tracked and reported.
        # Updated across steps; inspected at pipeline end to report any image
        # that was loaded but never annotated (silent-skip detection).
        image_status: dict = {}
        for _img in self._iter_images(images_dir):
            image_status[_img.name] = {
                "loaded": True,
                "ocr": False,
                "annotated": False,
                "classified_skip": False,
                "skip_reason": None,
            }

        page_classifier = PageTypeClassifier()
        skipped_dir = output_dir / "skipped_pages"
        skipped_dir.mkdir(parents=True, exist_ok=True)
        skipped_count = 0

        # Re-open original PDFs for embedded-text classification (faster than OCR)
        _classified_skips: set = set()
        try:
            import fitz
            # Resolve which PDFs to classify: single-file or directory input
            if input_path.is_file() and input_path.suffix.lower() == ".pdf":
                _pdfs_to_classify = [input_path]
            elif input_path.is_dir():
                _pdfs_to_classify = sorted(input_path.glob("*.pdf"))
            else:
                _pdfs_to_classify = []  # image-only input — skip classifier
            for pdf_path in _pdfs_to_classify:
                try:
                    doc = fitz.open(str(pdf_path))
                    for page_num, page in enumerate(doc):
                        stem = f"{pdf_path.stem}_page{page_num:03d}"
                        img_candidate = images_dir / f"{stem}.png"
                        if not img_candidate.exists():
                            continue
                        page_type, reason = page_classifier.classify_page(page)
                        if page_type != "floor_plan":
                            import shutil
                            dst = skipped_dir / img_candidate.name
                            shutil.move(str(img_candidate), str(dst))
                            _classified_skips.add(img_candidate.name)
                            skipped_count += 1
                            # Update tracking dict for classified-away images
                            if img_candidate.name in image_status:
                                image_status[img_candidate.name]["classified_skip"] = True
                                image_status[img_candidate.name]["skip_reason"] = reason
                            logger.info(f"  Skipped {img_candidate.name}: {page_type} ({reason})")
                    doc.close()
                except Exception as e:
                    logger.warning(f"  Page classification failed for {pdf_path.name}: {e}")
        except ImportError:
            logger.warning("fitz not available; page classification skipped")

        if skipped_count:
            logger.info(f"  Page classification: skipped {skipped_count} non-floor-plan pages")

        # Step 2: OCR text extraction + abbreviation recovery
        self._print_step("STEP 2: OCR text extraction + abbreviation recovery")

        ocr_results: Dict[str, List[RoomCandidate]] = {}
        # Raw Pass-1 detections retained per image for the deferred assembly step
        # (VLM refine + abbreviation recovery run after the OCR heap is released).
        raw_detections_by_image: Dict[str, List[TextDetection]] = {}
        # FIX-4: per-image BOM/title-block exclusion zones derived from OCR excluded-token clusters.
        exclusion_zones_by_image: Dict[str, list] = {}

        _ocr_first_image = True  # track first image to restore logger after Paddle import
        for img_path in self._iter_images(images_dir):
            try:
                # Pass 1 only: the memory-heavy PaddleOCR sweep runs for every page
                # with no VLM resident. Pass 2 (VLM refine) + abbreviation recovery
                # are deferred to the assembly step below, after the OCR native heap
                # is released, so PaddleOCR and the VLM model are never co-resident
                # (which would exceed host RAM). Order (refine → abbrev) is preserved.
                pdf_text_dets = self._pdf_text_layer_detections(img_path)
                pass1_rooms, raw_detections = self.ocr_extractor.find_rooms_pass1(
                    img_path, extra_detections=pdf_text_dets
                )
                # raw_detections stays OCR-only here -- pdf_text_dets only
                # reached find_room_candidates (inside find_rooms_pass1),
                # never compute_exclusion_zones below. See
                # _pdf_text_layer_detections's docstring for why.

                # FIX-4: Compute exclusion zones from excluded-token clusters.
                try:
                    from PIL import Image
                    with Image.open(img_path) as _im:
                        _img_w, _img_h = _im.size
                    zones = self.ocr_extractor.ocr_extractor.compute_exclusion_zones(
                        raw_detections, _img_w, _img_h
                    )
                    exclusion_zones_by_image[img_path.name] = zones
                    if zones:
                        logger.info(
                            f"  {img_path.name}: {len(zones)} BOM/title-block exclusion zone(s)"
                        )
                except Exception as _ez:
                    logger.debug(f"  {img_path.name}: exclusion zone computation skipped: {_ez}")
                    exclusion_zones_by_image[img_path.name] = []

                # Restore logging immediately after the first PaddleOCR call.
                # PaddlePaddle resets the root logger to WARNING during its first import,
                # silencing all INFO/DEBUG messages for the remaining pages in this loop.
                # Calling verify_logging_handlers() here (not after the loop) ensures
                # pages 001+ are logged correctly.
                if _ocr_first_image:
                    verify_logging_handlers()
                    _ocr_first_image = False

                ocr_results[img_path.name] = pass1_rooms
                raw_detections_by_image[img_path.name] = raw_detections
            except Exception as e:
                logger.error(f"  {img_path.name}: OCR failed - {e}")
                ocr_results[img_path.name] = []
                raw_detections_by_image[img_path.name] = []
            finally:
                if img_path.name in image_status:
                    image_status[img_path.name]["ocr"] = True
                # PaddleOCR's native allocator retains a new high-watermark chunk
                # for every distinct input shape it sees (measured: flat across
                # same-aspect pages, stepped +1-1.2GB on each new page size/tile
                # shape), so across many differently-sized sheets the resident
                # heap ratchets up over the run even though each page's own peak
                # is small. Releasing after every page resets that ratchet before
                # it can accumulate toward the host memory ceiling.
                self.ocr_extractor.release_ocr()

        # Verify logging handlers after Step 2 OCR completes
        # (PaddlePaddle may have reset the root logger level during import)
        verify_logging_handlers()

        # Pass 1 (PaddleOCR) is done for all pages — release its native heap so the
        # ~14GB CPU-arena is returned to the OS before the VLM model loads for
        # Pass 2 / Step 3. Keeps the two engines from being co-resident.
        self.ocr_extractor.release_ocr()

        # Assembly (VLM heap now free): Pass-2 VLM refine, then abbreviation
        # recovery — same order and logic as the original single-pass path, so
        # ocr_results match the pre-change baseline. refine_with_vlm is a no-op
        # when there is no VLM backend, so this is safe with use_vlm disabled too.
        for img_path in self._iter_images(images_dir):
            ocr_results[img_path.name] = self._assemble_ocr_candidates(
                img_path,
                ocr_results.get(img_path.name, []),
                raw_detections_by_image.get(img_path.name, []),
            )

        # Step 3: VLM annotation (optional)
        if self.config.use_vlm:
            self._print_step("STEP 3: VLM zero-shot annotation")
            
            # Verify logging handlers are still in place after VLM model initialization
            verify_logging_handlers()

            for img_path in self._iter_images(images_dir):
                ann_path = annotations_dir / f"{img_path.stem}.json"

                if skip_existing and ann_path.exists():
                    logger.info(f"  Skipping {img_path.name} (annotation exists)")
                    continue

                _t0_vlm = time.time()
                try:
                    # Step 3a: Pre-resize image on disk to reduce I/O for the
                    # VLM annotator.  This is best-effort: if it fails (e.g.
                    # read-only source file, OOM), vlm_annotator.annotate()
                    # enforces its own in-memory dimension limit and will still
                    # produce a valid result.  Log the failure so it is visible
                    # in the pipeline log, but do not abort the annotation step.
                    try:
                        ImageResizer.resize_in_place(img_path, max_kb=4500)
                    except PermissionError:
                        logger.info(
                            f"  {img_path.name}: pre-resize skipped "
                            f"(read-only file — annotator will resize in memory)"
                        )
                    except Exception as resize_error:
                        logger.warning(
                            f"  {img_path.name}: pre-resize failed "
                            f"({type(resize_error).__name__}: {resize_error}) "
                            f"— annotator will resize in memory"
                        )

                    # Get VLM result (handles both Claude and Qwen backends)
                    result = self._get_vlm_result(img_path)

                    # FIX-A: Drop VLM detections matching known non-room patterns.
                    # Applied unconditionally (no OCR-proximity gate).
                    ocr_list = ocr_results.get(img_path.name, [])
                    result.rooms = self._filter_vlm_rooms_by_ocr_proximity(
                        result.rooms, ocr_list, radius_px=300
                    )
                    if self.config.use_semantic_reconciliation and ocr_list:
                        ocr_dicts = self._ocr_candidates_to_reconciler_dicts(ocr_list)
                        result = self._apply_semantic_reconciliation(result, ocr_dicts)
                    
                    stats["images_annotated"] += 1
                    stats["rooms_detected"] += len(result.rooms)
                    _emit_metric(img_path.stem, "vlm_annotation", _t0_vlm, "ok",
                                 room_count_out=len(result.rooms))

                    # Save annotation (img_path needed for window detector)
                    object_detections_for_viz = self._save_annotation(
                        result, ann_path, ocr_results.get(img_path.name, []), img_path,
                        exclusion_zones=exclusion_zones_by_image.get(img_path.name, []))

                    if img_path.name in image_status:
                        image_status[img_path.name]["annotated"] = True

                    logger.info(
                        f"  {img_path.name}: {len(result.rooms)} rooms, "
                        f"{len(result.panels)} panels (VLM)"
                    )
                    
                    # Step 2: Log stage-transition counter (before hallucination detection)
                    logger.debug(f"    Stage-transition [raw-vlm → before-halluc-detect]: {len(result.rooms)} rooms")
                    
                    # Debug: Draw bbox overlays for visual inspection
                    try:
                        self.bbox_visualizer.draw_bboxes(
                            str(img_path),
                            result.rooms,
                            image_name=img_path.stem,
                            stage="raw_vlm",
                            detections=object_detections_for_viz,
                        )
                    except Exception as viz_err:
                        logger.warning(f"Bbox visualization failed: {viz_err}")
                    
                    # GPU memory cleanup between images
                    try:
                        import gc
                        import torch
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception as cleanup_err:
                        logger.debug(f"GPU cleanup warning: {cleanup_err}")
                    
                except Exception as e:
                    logger.error(f"  {img_path.name}: VLM annotation failed - {e}")
                    _emit_metric(img_path.stem, "vlm_annotation", _t0_vlm, "error",
                                 reject_reason=str(e)[:120])
                    # Step 6: Skip-reason tagging
                    log_skip_reason(img_path.name, f"VLM error: {str(e)[:80]}", stage="vlm_annotation")

                    # Fallback: always write OCR annotation (even if empty) so
                    # the image is not silently lost from downstream steps.
                    ocr_rooms = ocr_results.get(img_path.name, [])
                    try:
                        self._save_ocr_annotation(img_path, ocr_rooms, annotations_dir,
                                                   exclusion_zones=exclusion_zones_by_image.get(img_path.name, []))
                        stats["images_annotated"] += 1
                        stats["rooms_detected"] += len(ocr_rooms)
                        if img_path.name in image_status:
                            image_status[img_path.name]["annotated"] = True
                        if ocr_rooms:
                            logger.info(
                                f"  {img_path.name}: Fallback to OCR - "
                                f"saved {len(ocr_rooms)} rooms"
                            )
                        else:
                            logger.warning(
                                f"  {img_path.name}: VLM failed and OCR found 0 rooms "
                                f"— zero-room annotation written for review"
                            )
                    except Exception as fallback_error:
                        logger.error(
                            f"  {img_path.name}: OCR fallback also failed - {fallback_error}"
                        )
        else:
            # Use OCR results as primary annotations
            self._print_step("STEP 3: Generating annotations from OCR")

            for img_path in self._iter_images(images_dir):
                rooms = ocr_results.get(img_path.name, [])
                # Always write the annotation, even when rooms == [].
                # Previously, images with zero OCR rooms were silently dropped:
                # no annotation file was written, no warning was logged, and
                # the image disappeared from all downstream steps and exports.
                # A zero-room annotation is written with sft_ready=False and
                # lands in needs_review.json so a human can inspect it.
                try:
                    self._save_ocr_annotation(img_path, rooms, annotations_dir,
                                              exclusion_zones=exclusion_zones_by_image.get(img_path.name, []))
                    stats["images_annotated"] += 1
                    stats["rooms_detected"] += len(rooms)
                    if img_path.name in image_status:
                        image_status[img_path.name]["annotated"] = True
                    if rooms:
                        logger.info(
                            f"  {img_path.name}: Saved {len(rooms)} rooms from OCR"
                        )
                    else:
                        logger.warning(
                            f"  {img_path.name}: 0 rooms detected — "
                            f"zero-room annotation written for review"
                        )
                except Exception as e:
                    logger.error(
                        f"  {img_path.name}: Failed to save OCR annotation - {e}"
                    )

        self._release_vlm_before_sam()

        # Step 3.5: SAM label→room expansion (optional, pre-filter).
        # Must run BEFORE Step 4 geometric filter so SAM-expanded room-boundary
        # boxes pass the 2.5% / 10000px² thresholds that correctly reject label boxes.
        if self.config.use_sam:
            self._print_step("STEP 3.5: SAM label→room boundary expansion")

            for img_path in self._iter_images(images_dir):
                ann_path = annotations_dir / f"{img_path.stem}.json"
                if not ann_path.exists():
                    continue
                try:
                    with open(ann_path) as _f:
                        ann = json.load(_f)

                    rooms = ann.get("rooms", [])
                    ocr_rooms = ann.get("ocr_rooms", [])

                    if rooms or ocr_rooms:
                        img_w = ann.get("image_size", {}).get("width", 0)
                        img_h = ann.get("image_size", {}).get("height", 0)

                        # FIX-6: Drop out-of-bounds rooms before SAM so OOB anchors
                        # (OCR coords from 9600px space after FIX-5 transition) never
                        # reach SAM and produce garbage expansions.
                        def _in_bounds(room_list, w, h):
                            kept, dropped = [], 0
                            for r in room_list:
                                b = r.get("bbox", [])
                                if len(b) == 4:
                                    bx, by, bw, bh = b
                                    if bx + bw > w * 1.05 or by + bh > h * 1.05 or bx < 0 or by < 0:
                                        dropped += 1
                                        continue
                                kept.append(r)
                            if dropped:
                                logger.warning(f"FIX-6: pre-SAM OOB drop: {dropped} room(s) outside image bounds")
                            return kept

                        if img_w and img_h:
                            rooms = _in_bounds(rooms, img_w, img_h)
                            ocr_rooms = _in_bounds(ocr_rooms, img_w, img_h)

                        img_excl_zones = exclusion_zones_by_image.get(img_path.name, [])

                        if rooms:
                            expanded_rooms = self.sam_segmenter.refine_annotations(
                                img_path, rooms,
                                img_width=img_w, img_height=img_h,
                                max_expand_frac=self.config.sam.max_expand_frac,
                                exclusion_zones=img_excl_zones,
                            )
                            # F-C: dedup near-identical masks from adjacent labels
                            ann["rooms"] = self._dedup_rooms_by_iou(expanded_rooms)

                        if ocr_rooms:
                            expanded_ocr = self.sam_segmenter.refine_annotations(
                                img_path, ocr_rooms,
                                img_width=img_w, img_height=img_h,
                                max_expand_frac=self.config.sam.max_expand_frac,
                                exclusion_zones=img_excl_zones,
                            )
                            ann["ocr_rooms"] = self._dedup_rooms_by_iou(expanded_ocr)

                        ann["sam_expanded"] = True
                        with open(ann_path, "w") as _f:
                            json.dump(ann, _f, indent=2)

                        n_exp = sum(1 for r in ann.get("rooms", []) + ann.get("ocr_rooms", [])
                                    if r.get("sam_expanded"))
                        logger.info(
                            f"  {img_path.name}: SAM expanded {n_exp} of "
                            f"{len(rooms)+len(ocr_rooms)} boxes"
                        )

                except Exception as e:
                    logger.error(f"  {img_path.name}: SAM expansion failed - {e}")

        # Step 4: Annotation post-processing (label normalization, quality checks, region extraction)
        self._print_step("STEP 4: Post-processing annotations")

        processed_dir = output_dir / "processed_annotations"
        regions_dir = output_dir / "room_regions"
        reports_dir = output_dir / "quality_reports"

        processed_dir.mkdir(parents=True, exist_ok=True)
        regions_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        quality_summary = {
            "total_annotations": 0,
            "annotations_normalized": 0,
            "annotations_with_issues": 0,
            "total_issues": 0,
            "regions_extracted": 0,
            # Fix 12: Per-annotation issue details for targeted debugging
            "issue_details": [],
        }

        for ann_path in sorted(annotations_dir.glob("*.json")):
            try:
                with open(ann_path) as f:
                    annotation = json.load(f)

                # Step 4a: SFT-grade filtering and semantic validation
                original_room_count = len(annotation.get("rooms", []) or annotation.get("ocr_rooms", []))
                # Captured before prepare_sft_annotation pops "ocr_rooms" — needed
                # downstream by the quality checker's OCR_NAME_EMPTY check.
                raw_ocr_rooms = annotation.get("ocr_rooms", [])
                _t0_sft = time.time()
                annotation = prepare_sft_annotation(
                    annotation,
                    min_rooms_for_sft=self.config.min_rooms_for_sft,
                    image_dir=images_dir,
                    use_window_elongation_filter=self.config.use_window_elongation_filter,
                    exempt_sam_expanded_from_ink_filter=self.config.exempt_sam_expanded_from_ink_filter,
                )
                filtered_room_count = len(annotation.get("rooms", []))
                _sft_outcome = "sft_ready" if annotation.get("sft_ready") else "filtered"
                _sft_reject = ""
                if annotation.get("degenerate_bbox_reasons"):
                    _sft_reject = "degenerate_bbox"
                elif filtered_room_count == 0:
                    _sft_reject = "no_rooms"
                elif filtered_room_count < 3:
                    _sft_reject = "below_min_rooms"
                _emit_metric(ann_path.stem, "sft_filter", _t0_sft, _sft_outcome,
                             reject_reason=_sft_reject,
                             room_count_in=original_room_count,
                             room_count_out=filtered_room_count)
                for step, count in annotation.get("drop_attribution", {}).items():
                    run_drop_attribution[step] = run_drop_attribution.get(step, 0) + count

                if filtered_room_count < original_room_count:
                    logger.info(
                        f"    {ann_path.name}: SFT filter removed "
                        f"{original_room_count - filtered_room_count} non-spatial annotations"
                    )

                # Step 4b: Normalize room type labels and stamp canonical "type" field.
                # Synthetic rooms (source=synthetic_residential) are excluded from SFT
                # data — they have no visual evidence and would teach the model to hallucinate.
                rooms = annotation.get("rooms", [])
                rooms = [r for r in rooms if r.get("source") != "synthetic_residential"]
                annotation["rooms"] = rooms

                for room in rooms:
                    raw_type = room.get("category") or room.get("type") or "other"
                    canonical = self.label_normalizer.normalize(raw_type)
                    room["type"] = canonical       # canonical field (QualityChecker, RegionExtractor)
                    room["category"] = canonical   # keep for backward compat
                    logger.debug(f"    {ann_path.name}: '{raw_type}' → '{canonical}'")

                # Step 4c: Check annotation quality
                issues = self.quality_checker.check_annotation(annotation, ocr_rooms=raw_ocr_rooms)
                if issues:
                    quality_summary["annotations_with_issues"] += 1
                    quality_summary["total_issues"] += len(issues)

                    # Fix 12: Record per-annotation issue details
                    for issue in issues:
                        quality_summary["issue_details"].append({
                            "image": ann_path.name,
                            "issue_type": getattr(issue, "type", str(type(issue).__name__)),
                            "description": str(issue),
                            "resolution": "pending",
                        })

                    # Save issue report
                    report = self.quality_checker.report(issues)
                    report_path = reports_dir / f"{ann_path.stem}_issues.txt"
                    with open(report_path, "w") as f:
                        f.write(report)

                    logger.warning(
                        f"    {ann_path.name}: Found {len(issues)} quality issues"
                    )

                # Step 4d: Extract room regions for training
                # Find the actual image file — do not assume .png extension.
                # Images may be .jpg, .tiff, etc. when copied from Form B/D input.
                img_path = None
                for _ext in self._IMAGE_EXTS:
                    _candidate = images_dir / f"{ann_path.stem}{_ext}"
                    if _candidate.exists():
                        img_path = _candidate
                        break
                if img_path is not None and rooms:
                    logger.info(
                        f"    {ann_path.name}: Starting region extraction "
                        f"({len(rooms)} rooms) ..."
                    )
                    extracted = self.region_extractor.extract_regions(
                        str(img_path), annotation, str(regions_dir), prefix="room"
                    )
                    quality_summary["regions_extracted"] += len(extracted)
                    logger.info(
                        f"    {ann_path.name}: "
                        f"Extracted {len(extracted)} room regions"
                    )

                # Step 4e: Save processed annotation
                # Remediation Fix #5: Skip writing empty annotations (sft_ready=false AND zero rooms)
                # These are non-floorplan pages (cover sheets, title blocks, etc.) that should
                # not clutter processed_annotations/ or inflate image counts.
                room_count = len(annotation.get("rooms", []))
                is_sft_ready = annotation.get("sft_ready", False)
                localization_failed = annotation.get("localization_failed", False)

                # DV-2: rebuild roomsRecognized from rooms[] survivors BEFORE the
                # write-path branch below. prepare_sft_annotation shrinks rooms[]
                # but never updates roomsRecognized, leaving stale (oversized/
                # mislocated) entries. This must run unconditionally — not just in
                # the sft_ready branch — because localization_failed pages
                # (room_count==0) write processed_annotations too (FIX-B) and were
                # previously left with the raw, unfiltered roomsRecognized (e.g.
                # page005: 36 stale giants, page006: 16), which the overview
                # visualization draws directly. Empty survivors -> empty rr.
                surviving_rooms = annotation.get("rooms", [])
                new_rr = []
                for idx, room in enumerate(surviving_rooms, start=1):
                    name = room.get("room_name") or room.get("name", "")
                    number = room.get("room_number", "")
                    cat = room.get("type") or room.get("category") or name
                    raw_bbox = room.get("bbox", [0, 0, 0, 0])
                    if len(raw_bbox) == 4:
                        bx, by, bw, bh = [int(v) for v in raw_bbox]
                        bbox_xyxy = [bx, by, bx + bw, by + bh]
                    else:
                        bbox_xyxy = [int(v) for v in raw_bbox]
                    sft_room = self.sft_builder.build_room(
                        room_id=idx,
                        mandatory_type=cat,
                        original_name=name,
                        room_number=number,
                        bbox=bbox_xyxy,
                        detection_score=room.get("detection_score", 0.9),
                        classification_match_type=room.get("classification_match_type", "ocr"),
                        ocr_confidence=room.get("ocr_confidence", room.get("confidence", 0.9)),
                        source=room.get("source", "processed"),
                        extended_type=room.get("extended_type", ""),
                        detection_method=room.get("detection_method", "ocr"),
                        detection_model=room.get("detection_model", "pipeline"),
                    )
                    new_rr.append(self.sft_builder.to_dict(sft_room))

                # G-A fix (2026-08-11): the rebuild above discards whatever
                # door/window/fixture evidence _apply_object_attributes had
                # already written onto the PRE-filter roomsRecognized (at
                # coordinates.attributes -- has_door/has_toilet/has_bathtub/
                # has_sink; the window-only fields on SFTRoom.attributes get
                # reset to defaults too, since build_room() above is never
                # given a window_mapping). Verified on real output
                # (sprint1_verify46): 0/786 processed rooms carried any
                # evidence despite 746/2593 (28.8%) carrying real evidence
                # pre-rebuild -- total loss, not partial.
                #
                # Cannot reattach the OLD evidence by matching old<->new
                # room entries: verified rooms[] and roomsRecognized[] are
                # NEVER the same length even before this filtering step (not
                # a positional/order-preserving correspondence at all), so
                # no safe join key exists between them. Recomputing from the
                # persisted objectDetections (which DOES survive filtering
                # untouched) against the FINAL surviving room geometry is
                # correct by construction instead of a guess -- same
                # map_detections_to_rooms + _apply_object_attributes calls
                # Step 3 already uses, applied fresh to the filtered set.
                reconstructed_detections = []
                for det in annotation.get("objectDetections", []):
                    try:
                        tier = WindowDetectionTier(det.get("source_tier", "none"))
                    except ValueError:
                        tier = WindowDetectionTier.NONE
                    reconstructed_detections.append(WindowDetection(
                        bbox=tuple(det["bbox"]),
                        confidence=det.get("confidence", 1.0),
                        source_tier=tier,
                        metadata={"type": det.get("category")},
                    ))
                rooms_for_remapping = [
                    {"id": idx, "bbox": r["coordinates"]["bbox"]}
                    for idx, r in enumerate(new_rr)
                ]
                new_mappings = map_detections_to_rooms(reconstructed_detections, rooms_for_remapping)
                new_rr = self._apply_object_attributes(
                    new_rr, new_mappings, **self._confidence_weight_kwargs()
                )
                annotation["roomsRecognized"] = new_rr

                if not is_sft_ready and room_count == 0 and localization_failed:
                    # FIX-B: rooms WERE detected but all mislocalized (boxes over
                    # blank space). The page has real rooms needing manual
                    # annotation — keep the processed annotation (with its dropped
                    # boxes recorded) and let the review prioritizer flag it.
                    # Do NOT move to skipped_pages/ — that loses a real floor plan.
                    processed_path = processed_dir / ann_path.name
                    with open(processed_path, "w") as f:
                        json.dump(annotation, f, indent=2)
                    logger.warning(
                        f"    {ann_path.name}: localization failed "
                        f"(rooms detected but all mislocalized) — kept for review, not skipped"
                    )
                    quality_summary["total_annotations"] += 1

                elif not is_sft_ready and room_count == 0:
                    # Genuinely empty annotation — move to skipped_pages/
                    # (cover page, title block, roof/site sheet, or OCR failure)
                    image_file = None
                    for _ext in self._IMAGE_EXTS:
                        _candidate = images_dir / f"{ann_path.stem}{_ext}"
                        if _candidate.exists():
                            image_file = _candidate
                            break

                    if image_file:
                        skipped_dir = output_dir / "skipped_pages"
                        skipped_dir.mkdir(parents=True, exist_ok=True)
                        try:
                            import shutil
                            shutil.move(str(image_file), str(skipped_dir / image_file.name))
                            logger.warning(
                                f"    {ann_path.name}: Moved to skipped_pages/ "
                                f"(sft_ready=false, 0 rooms — non-floorplan page)"
                            )
                        except Exception as move_err:
                            logger.warning(
                                f"    {ann_path.name}: Failed to move to skipped_pages/: {move_err}"
                            )
                else:
                    # roomsRecognized already rebuilt above (DV-2), unconditionally,
                    # before this branch. Normal case: write processed annotation.
                    processed_path = processed_dir / ann_path.name
                    with open(processed_path, "w") as f:
                        json.dump(annotation, f, indent=2)
                    quality_summary["total_annotations"] += 1
                    quality_summary["annotations_normalized"] += 1

            except Exception as e:
                logger.error(
                    f"    {ann_path.name}: Post-processing failed - {e}"
                )

        # Log post-processing summary
        logger.info(
            f"\nPost-processing complete:\n"
            f"  Total annotations: {quality_summary['total_annotations']}\n"
            f"  Normalized: {quality_summary['annotations_normalized']}\n"
            f"  With issues: {quality_summary['annotations_with_issues']}\n"
            f"  Total issues found: {quality_summary['total_issues']}\n"
            f"  Room regions extracted: {quality_summary['regions_extracted']}"
        )

        # Save post-processing summary
        summary_path = output_dir / "post_processing_summary.json"
        with open(summary_path, "w") as f:
            json.dump(quality_summary, f, indent=2)

        # Update main stats with post-processing results
        stats["annotations_normalized"] = quality_summary["annotations_normalized"]
        stats["annotations_with_issues"] = quality_summary["annotations_with_issues"]
        stats["total_quality_issues"] = quality_summary["total_issues"]
        stats["room_regions_extracted"] = quality_summary["regions_extracted"]

        # Step 5: (removed) SAM now runs in Step 3.5, before the geometric filter,
        # so expanded room-boundary boxes pass the size thresholds that correctly
        # reject label-sized boxes.

        # Step 6: Prioritize for review
        self._print_step("STEP 6: Prioritizing for human review")

        # FIX: Read from processed_dir (filtered, SFT-validated) not raw annotations_dir.
        # The raw dir still contains panels and OCR noise; confidence scores computed
        # from it were meaningless.
        review_items = self.prioritizer.prioritize(processed_dir)
        stats["flagged_for_review"] = len(review_items)

        # Save review list
        self.prioritizer.save_review_list(
            review_items, output_dir / "needs_review.json"
        )
        self.prioritizer.print_summary(review_items)

        # Step 7: Export to Label Studio + COCO + coverage report
        self._print_step("STEP 7: Exporting annotations")

        # 7a: Label Studio export (human review UI)
        num_exported = self.exporter.export(
            processed_dir, images_dir, output_dir / "label_studio_import.json"
        )
        logger.info(f"  Label Studio: {num_exported} tasks exported")

        # 7b: COCO JSON export (VLM fine-tuning input)
        coco_exporter = COCOExporter()
        coco_exporter.export_splits(
            processed_dir, images_dir, output_dir / "coco",
            splits=(0.70, 0.15, 0.15), seed=42,
            min_annotations_per_image=self.config.min_rooms_for_sft,
        )

        # 7c: Coverage report (class imbalance visibility before training)
        reporter = CoverageReporter()
        reporter.generate(processed_dir, output_dir / "coverage_report.json")

        # Save config for reproducibility
        config_path = output_dir / "pipeline_config.json"
        self._save_config(config_path)

        # Print summary
        self._print_summary(stats, output_dir)

        # ── Pipeline integrity check ───────────────────────────────────────────
        # Report classified skips (moved to skipped_pages/ by Step 1b) and
        # true silent skips (images that reached images_dir but never got an
        # annotation file) as separate categories.

        _classified_skips_list = [
            name for name, s in image_status.items()
            if s.get("classified_skip")
        ]
        if _classified_skips_list:
            logger.warning(
                f"Page classifier skipped {len(_classified_skips_list)} image(s). "
                f"Review skipped_pages/ for false positives. "
                f"Affected files: {_classified_skips_list}"
            )
            for name in _classified_skips_list:
                reason = image_status[name].get("skip_reason", "unknown")
                logger.warning(f"  - {name}: {reason}")
            stats["images_classified_skip"] = len(_classified_skips_list)

        # Any image that was loaded but NOT classified-away AND NOT annotated
        # represents a true silent data loss.
        _never_annotated = [
            name for name, s in image_status.items()
            if s["loaded"] and not s["annotated"] and not s.get("classified_skip")
        ]
        if _never_annotated:
            logger.error(
                f"PIPELINE INTEGRITY FAILURE: {len(_never_annotated)} image(s) "
                f"were loaded into images_dir but produced no annotation file. "
                f"These images are absent from all downstream steps and exports. "
                f"Affected files: {_never_annotated}"
            )
            stats["images_skipped_silently"] = len(_never_annotated)
        else:
            total_tracked = len(image_status)
            total_classified = len(_classified_skips_list)
            total_annotated = total_tracked - total_classified
            logger.info(
                f"Pipeline integrity OK: {total_annotated} of {total_tracked} "
                f"image(s) produced an annotation file"
                + (f" ({total_classified} skipped by classifier)." if total_classified else ".")
            )

        # Write run-level drop attribution summary and close metrics log.
        if run_drop_attribution:
            _summary_record = {
                "page_id": "__run_summary__",
                "stage": "drop_attribution_totals",
                "duration_ms": 0,
                "outcome": "summary",
                "reject_reason": "",
                "drop_attribution": run_drop_attribution,
            }
            _metrics_fh.write(json.dumps(_summary_record) + "\n")
        _metrics_fh.close()
        stats["drop_attribution"] = run_drop_attribution
        logger.info(
            f"run_metrics.jsonl written: {_metrics_path}"
        )
        if run_drop_attribution:
            logger.info(f"Drop attribution totals: {run_drop_attribution}")

        # Q1: run-manifest provenance. Timestamped filename (this run's own
        # start time, never overwritten) so concurrent/repeated launches
        # into the same output_dir stay individually attributable -- this
        # is what F5 (two overlapping runs corrupting one output_dir's
        # provenance) needed and pipeline_config.json/logs alone don't give:
        # PID, argv, and per-image status (incl. skip_reason, previously
        # log-only -- F4) survive after the log file is gone or ambiguous.
        manifest = {
            "pid": _run_pid,
            "argv": sys.argv,
            "start_utc": _run_start_utc.isoformat(),
            "end_utc": datetime.now(timezone.utc).isoformat(),
            "config": self._build_config_dict(),
            "image_status": image_status,
            "mlflow_run_id": _mlflow_run_id,
            "mlflow_experiment_name": (
                self.config.mlflow.experiment_name if _mlflow_run_id else None
            ),
        }
        manifest_path = output_dir / f"run_manifest_{_run_timestamp}.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Run manifest written: {manifest_path}")

        # Optional GT-scored quality metrics (opt-in via --eval-gt-dir).
        # Wired here, not made mandatory: most production runs have no GT.
        if self.config.eval_gt_dir is not None:
            from ..bbox.bbox_metrics import evaluate_dataset
            from .mlflow_tracking import log_params as _log_eval_params
            try:
                eval_results = evaluate_dataset(
                    annotations_dir, Path(self.config.eval_gt_dir), output_dir=output_dir
                )
                _n = len(eval_results) or 1
                _agg = {
                    "mean_iou": sum(r.mean_iou for r in eval_results.values()) / _n,
                    "mean_giou": sum(r.mean_giou for r in eval_results.values()) / _n,
                    "mean_diou": sum(r.mean_diou for r in eval_results.values()) / _n,
                    "mean_ciou": sum(r.mean_ciou for r in eval_results.values()) / _n,
                    "map_50": sum(r.map_50 for r in eval_results.values()) / _n,
                    "accuracy_50": sum(r.accuracy_50 for r in eval_results.values()) / _n,
                }
                eval_metrics_path = output_dir / "eval_metrics.json"
                with open(eval_metrics_path, "w") as f:
                    json.dump(_agg, f, indent=2)
                logger.info(f"GT eval metrics written: {eval_metrics_path} -> {_agg}")
                if _mlflow_run_id:
                    _log_eval_params({"gt_eval": _agg})
            except Exception as e:
                logger.warning(f"GT evaluation failed (non-fatal): {e}")

        from .mlflow_tracking import log_stage_metric as _log_corrob_metric, end_run
        _corrob = self._corroboration_totals
        _log_corrob_metric("corroboration", "total_evidence_count", _corrob["evidence_count"])
        if _corrob["evidence_count"] > 0:
            _log_corrob_metric(
                "corroboration", "mean_evidence_score",
                _corrob["evidence_score_sum"] / _corrob["evidence_count"],
            )
        end_run()

        logger.info("Pipeline run() complete — all steps finished.")
        return stats

    def _assemble_ocr_candidates(
        self,
        img_path: Path,
        pass1_rooms: List[RoomCandidate],
        raw_detections: List[TextDetection],
    ) -> List[RoomCandidate]:
        """Finalize OCR candidates for one page: Pass-2 VLM refine, then residential
        abbreviation recovery.

        Runs after the PaddleOCR heap is released so the VLM can load without
        co-residence. Order (refine → abbrev) and the per-candidate logic match the
        original in-loop two-pass path, so outputs are unchanged.
        """
        refined = self.ocr_extractor.refine_with_vlm(img_path, pass1_rooms)

        recovered_abbrevs: List[RoomCandidate] = []
        for det in raw_detections:
            text = det.text.strip().upper()
            if ResidentialAbbreviationRecovery.is_residential_abbreviation(text):
                expanded = ResidentialAbbreviationRecovery.expand_abbreviation(text)
                x_coords = [p[0] for p in det.bbox]
                y_coords = [p[1] for p in det.bbox]
                x, y = min(x_coords), min(y_coords)
                w, h = max(x_coords) - x, max(y_coords) - y
                recovered_abbrevs.append(
                    RoomCandidate(
                        bbox=(x, y, w, h),
                        room_number="",
                        room_name=expanded,
                        confidence=det.confidence,
                        raw_text=text,
                    )
                )

        # Merge, deduplicate by proximity (50px threshold)
        all_rooms = list(refined)
        for abbrev_cand in recovered_abbrevs:
            ax, ay = abbrev_cand.bbox[0], abbrev_cand.bbox[1]
            already_covered = any(
                abs(r.bbox[0] - ax) < 50 and abs(r.bbox[1] - ay) < 50
                for r in all_rooms
            )
            if not already_covered:
                all_rooms.append(abbrev_cand)

        logger.info(
            f"  {img_path.name}: {len(refined)} rooms + "
            f"{len(recovered_abbrevs)} abbreviations recovered"
        )
        return all_rooms

    def _get_vlm_result(self, img_path: Path):
        """Get VLM annotation result, with optional tiled inference for local backends."""
        vlm = self.vlm_annotator
        backend_name = self.config.vlm.backend.lower()
        is_local_vlm = backend_name in ("qwen", "unsloth")

        if is_local_vlm:
            rooms_dicts = self._detect_rooms_maybe_tiled(img_path, vlm)
            result = self._qwen_dicts_to_vlm_result(rooms_dicts, img_path)
        else:
            # Claude: tiling would quadruple API cost; skip silently unless explicit
            if self.config.use_tiling:
                logger.debug("Tiling skipped for Claude backend (API cost); using full-image")
            result = vlm.annotate(img_path)

        return result

    def _dedup_rooms_by_iou(self, rooms: List[dict], iou_threshold: float = 0.9) -> List[dict]:
        """Drop only genuine duplicate room boxes — near-identical geometry (F-C).

        The F-C case is two label points inside one room that SAM segments into the
        *same mask*, yielding near-identical boxes. Those are merged. Distinct rooms
        that merely overlap — including SAM over-expansions (flood) and distinct
        units that share a generic label (e.g. several "0BR")— have IoU well below
        the near-identical threshold and are kept. The default threshold is high on
        purpose: only same-mask duplicates (IoU→1.0) are dropped; ordinary overlap
        (~0.5-0.8) is preserved so real rooms are never collapsed. Operates on xywh.
        """
        def _iou_xywh(a, b):
            ax1, ay1, aw, ah = a
            bx1, by1, bw, bh = b
            ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            if ix2 <= ix1 or iy2 <= iy1:
                return 0.0
            inter = (ix2 - ix1) * (iy2 - iy1)
            union = aw * ah + bw * bh - inter
            return inter / union if union > 0 else 0.0

        def _priority(r):
            b = r.get("bbox", [])
            area = b[2] * b[3] if len(b) == 4 else 0
            return (1 if r.get("sam_expanded") else 0, area)

        ordered = sorted(rooms, key=_priority, reverse=True)
        kept: List[dict] = []
        dropped = 0
        for room in ordered:
            b = room.get("bbox", [])
            if len(b) != 4:
                kept.append(room)
                continue
            if any(
                len(k.get("bbox", [])) == 4 and _iou_xywh(b, k["bbox"]) >= iou_threshold
                for k in kept
            ):
                dropped += 1
                continue
            kept.append(room)
        if dropped:
            logger.info(f"F-C dedup: removed {dropped} near-identical duplicate box(es)")
        return kept

    # Non-room fixture/tag patterns observed in production data.
    # Drop unconditionally regardless of OCR proximity.
    # Patterns are anchored and case-insensitive; match on stripped uppercase name.
    _NON_ROOM_PATTERNS = [
        r"^[A-Z]{1,3}-\d+$",    # LT-06, T-06, F-3, etc. (circuit/fixture tags)
        r"^F\.D\.$",             # F.D. (floor drain)
        r"\bPLAN$",              # CELLAR PLAN, FLOOR PLAN, etc. (sheet titles)
        r"^B \(EM\)$",           # B (EM) (emergency branch label)
    ]

    @classmethod
    def _is_non_room_by_name(cls, name: str) -> bool:
        import re
        n = (name or "").strip().upper()
        if not n:
            return False
        return any(re.search(p, n) for p in cls._NON_ROOM_PATTERNS)

    def _filter_vlm_rooms_by_ocr_proximity(self, vlm_rooms, ocr_rooms, radius_px: int = 300):
        """FIX-A: Drop VLM detections that match known non-room fixture/tag patterns.

        Replaces the OCR-proximity gate which dropped real rooms on pages where
        PaddleOCR underperforms (cellar/MEP sheets with few legible labels).

        Known non-room tags (circuit labels, drain markers, sheet titles) are
        rejected by name pattern.  Everything else is kept regardless of whether
        OCR found a nearby anchor.

        Args:
            vlm_rooms:  List of RoomAnnotation objects from the VLM backend.
            ocr_rooms:  Unused (kept for call-site compatibility).
            radius_px:  Unused (kept for call-site compatibility).

        Returns:
            Filtered list of RoomAnnotation objects.
        """
        kept = []
        dropped = 0
        for room in vlm_rooms:
            nm = room.room_name or ""
            if self._is_non_room_by_name(nm):
                logger.debug(f"FIX-A name: dropped '{nm}'")
                dropped += 1
                continue
            # Strip-geometry filter: aspect > 5:1 → label strip, not a room
            bbox = room.bbox
            if len(bbox) == 4:
                bw, bh = float(bbox[2]), float(bbox[3])
                if bw > 0 and bh > 0 and max(bw / bh, bh / bw) > 5.0:
                    logger.debug(
                        f"FIX-A strip: dropped '{nm}' "
                        f"aspect={max(bw/bh,bh/bw):.1f} bbox={list(bbox)}"
                    )
                    dropped += 1
                    continue
            kept.append(room)

        if dropped:
            logger.info(
                f"FIX-A filter: {dropped} dropped (name-pattern or strip geometry), "
                f"{len(kept)} kept"
            )
        return kept

    def _detect_rooms_maybe_tiled(self, img_path: Path, vlm) -> List[dict]:
        """Run tiled or full-image room detection for local VLM backends.

        Tiling activates when:
          - config.use_tiling is True, AND
          - max(img_w, img_h) > config.tile_trigger_px

        Hallucination detection and the 25-room cap run AFTER tile merge (global),
        not per-tile, so the limits apply to the full-image result.
        """
        if not self.config.use_tiling:
            return vlm.detect_rooms(img_path)

        from PIL import Image as _PIL
        try:
            full_img = _PIL.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Tiling: could not open {img_path.name}: {e}. Falling back.")
            return vlm.detect_rooms(img_path)

        full_w, full_h = full_img.size
        if max(full_w, full_h) <= self.config.tile_trigger_px:
            logger.debug(
                f"Tiling skipped: {full_w}x{full_h} <= trigger {self.config.tile_trigger_px}px"
            )
            return vlm.detect_rooms(img_path)

        try:
            from ..ingestion.tile_splitter import TileSplitter
        except ImportError:
            logger.warning("tile_splitter not importable — falling back to full-image")
            return vlm.detect_rooms(img_path)

        # Adaptive grid: derive tile count from image size so each tile is about
        # tile_target_px, capped at tile_cols x tile_rows. Finer grid on dense
        # floors improves VLM room localization. tile_target_px=0 → fixed grid.
        if self.config.tile_target_px > 0:
            cols = max(2, min(self.config.tile_cols,
                              round(full_w / self.config.tile_target_px)))
            rows = max(2, min(self.config.tile_rows,
                              round(full_h / self.config.tile_target_px)))
        else:
            cols, rows = self.config.tile_cols, self.config.tile_rows

        splitter = TileSplitter(
            cols=cols,
            rows=rows,
            overlap_pct=self.config.tile_overlap_pct,
        )
        tiles = splitter.split(full_img)
        logger.info(
            f"{img_path.name}: tiling {full_w}x{full_h} -> "
            f"{len(tiles)} tiles ({cols}x{rows}, "
            f"overlap={self.config.tile_overlap_pct:.0%})"
        )

        all_rooms: List[dict] = []
        for tile_img, meta in tiles:
            try:
                tile_rooms = vlm.detect_rooms_from_image(tile_img)
                tile_rooms = splitter.rescale_rooms(tile_rooms, meta)
                logger.debug(
                    f"  Tile ({meta.col},{meta.row}): {len(tile_rooms)} rooms detected"
                )
                all_rooms.extend(tile_rooms)
            except Exception as e:
                logger.warning(f"  Tile ({meta.col},{meta.row}) failed: {e} — skipping")

        # Global hallucination check + NMS merge AFTER all tiles collected.
        # Cap 40 (not 25): dense residential floors carry 30+ units; a 25 cap
        # would discard real rooms that finer adaptive tiling just recovered.
        all_rooms = vlm.detect_hallucinations(all_rooms)
        merged = splitter.merge(all_rooms, iou_threshold=0.30, max_rooms=40)
        logger.info(
            f"{img_path.name}: tiling complete — {len(all_rooms)} raw -> "
            f"{len(merged)} after NMS"
        )
        return merged

    def _qwen_dicts_to_vlm_result(self, rooms_dicts: List[dict], img_path: Path):
        """Convert Qwen backend dict output to VLMAnnotationResult."""
        from PIL import Image
        
        # Get image dimensions
        try:
            img = Image.open(img_path)
            width, height = img.size
        except Exception:
            width, height = 1000, 1000  # fallback
        
        # Convert dict rooms to RoomAnnotation objects
        rooms = []
        for idx, room_dict in enumerate(rooms_dicts):
            bbox = room_dict.get("bbox", [0, 0, 100, 100])
            
            # bbox format: [x1, y1, x2, y2] (from Qwen) → [x, y, w, h] (for RoomAnnotation)
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)
            else:
                x, y, w, h = 0, 0, 100, 100
            
            from ..vlm.vlm_annotator import RoomAnnotation
            room = RoomAnnotation(
                room_number=room_dict.get("room_number", f"room_{idx}"),
                room_name=room_dict.get("room_name", room_dict.get("room_type", "Room")),
                category=room_dict.get("room_type", "other"),
                bbox=[x, y, w, h]
            )
            rooms.append(room)
        
        # Create VLMAnnotationResult
        from ..vlm.vlm_annotator import VLMAnnotationResult
        return VLMAnnotationResult(
            image_file=img_path.name,
            image_size={"width": width, "height": height},
            rooms=rooms,
            panels=[],
            electrical_counts={}
        )

    def _release_vlm_before_sam(self) -> None:
        """Free the VLM's GPU memory before Step 3.5 (SAM) runs.

        Confirmed gap (sprint1_verify47 audit, 2026-08-19): Step 3 (VLM)
        and Step 3.5 (SAM) run back-to-back in one process with the VLM
        never freed -- an 8B model left resident starved SAM's
        allocations, measured as 3,093 "CUDA out of memory" failures and
        57/57 pages logging "SAM expanded 0 of N" (SAM's expansion never
        once succeeded in that run).

        Guarded on self._vlm_annotator (the backing field), not
        self.vlm_annotator (the lazy-create property) -- reading the
        property here would construct a backend just to no-op release()
        it. Safe regardless of use_windows: every VLMBackend.detect_*
        call starts with self.initialize(), so a later stage needing the
        VLM again reloads automatically. Both VLMAnnotator (Claude) and
        every VLMBackend subclass implement release() (no-op for
        VLMAnnotator, which holds no local GPU weights), so this needs
        no type check.
        """
        if self.config.use_sam and self._vlm_annotator is not None:
            self._vlm_annotator.release()

    @staticmethod
    def _ocr_candidates_to_reconciler_dicts(ocr_list: List) -> List[dict]:
        """Convert OCR RoomCandidate objects to the dict format
        _apply_semantic_reconciliation/SemanticReconciler expect.

        RoomCandidate.bbox is [x,y,w,h] (ocr_extractor.py's own contract);
        SemanticReconciler's OCRText.bbox is documented [x1,y1,x2,y2] and
        OCRText.centroid() computes (x1+x2)/2 -- passing xywh through
        unconverted put every OCR text's centroid hundreds of px off (same-
        shaped bug as the VLM-room side of _apply_semantic_reconciliation;
        found together auditing sprint1_verify47, 2026-08-19). Real
        example: SOCIAL SERVICE [1574,633,78,14] -> true centroid
        (1613,640), buggy centroid (826,324) -- centroid landed outside
        every real room polygon, so the label matched nothing.

        Extracted to its own method (previously inline in run()'s loop)
        so this conversion is unit-testable without a full pipeline run.
        """
        ocr_dicts = []
        for room in ocr_list:
            x, y, w, h = room.bbox
            ocr_dicts.append({
                "text": room.room_name,
                "bbox": [x, y, x + w, y + h],
                "confidence": room.confidence,
            })
        return ocr_dicts

    def _apply_semantic_reconciliation(self, result, ocr_results: List[dict]):
        """Apply semantic reconciliation to VLM result if enabled."""
        if not self.config.use_semantic_reconciliation:
            return result
        
        if not ocr_results:
            return result
        
        try:
            reconciler = SemanticReconciler(self.config.ocr)

            # Convert VLM rooms to dict format for reconciler. SemanticReconciler.
            # reconcile() documents its "bbox" input as [x1,y1,x2,y2] and unpacks
            # it that way (semantic_reconciler.py's _match_room_to_ocr path) --
            # but RoomAnnotation.bbox is [x,y,w,h] (see _qwen_dicts_to_vlm_result's
            # own comment above). Passing it through unconverted made the
            # reconciler read a box's WIDTH as an absolute X2 coordinate and its
            # HEIGHT as an absolute Y2 coordinate -- confirmed corrupting every
            # semantically-reconciled room on a real run (sprint1_verify47 audit,
            # 2026-08-19): all 35 VLM rooms on one page came out with one
            # dimension in the thousands-of-px range, while the untouched
            # ocr_rooms list on the same page stayed correctly small. Convert at
            # the boundary instead of changing the reconciler's own contract --
            # this is SemanticReconciler's only call site (verified), but its
            # docstring's xyxy contract may be relied on by a future caller.
            vlm_rooms_dicts = []
            for room in result.rooms:
                x, y, w, h = room.bbox
                vlm_rooms_dicts.append({
                    "room_id": room.room_number,
                    "room_name": room.room_name,
                    "room_type": room.category,
                    "bbox": [x, y, x + w, y + h],
                    "confidence": 1.0,
                    "polygon": None
                })

            # Reconcile with OCR results
            enriched = reconciler.reconcile_with_config(
                ocr_results, vlm_rooms_dicts, self.config
            )

            # Update result.rooms with enriched data. Reconciler returns
            # [x1,y1,x2,y2] (SemanticReconciler._polygon_bounds) -- convert back
            # to RoomAnnotation's [x,y,w,h] contract (same boundary fix as above).
            from ..vlm.vlm_annotator import RoomAnnotation
            new_rooms = []
            for enriched_room in enriched:
                ex1, ey1, ex2, ey2 = enriched_room.get("bbox", [0, 0, 100, 100])
                room = RoomAnnotation(
                    room_number=enriched_room.get("room_number", ""),
                    room_name=enriched_room.get("room_name", "Room"),
                    category=enriched_room.get("room_type", "other"),
                    bbox=[ex1, ey1, ex2 - ex1, ey2 - ey1]
                )
                new_rooms.append(room)

            result.rooms = new_rooms
            return result
        except Exception as e:
            logger.warning(f"Semantic reconciliation failed: {e}. Using VLM results as-is.")
            return result

    def _save_annotation(
        self,
        result,
        output_path: Path,
        ocr_rooms: List[RoomCandidate],
        img_path: Optional[Path] = None,
        exclusion_zones: Optional[List] = None,
    ) -> List[Dict[str, Any]]:
        """
        Save VLM annotation result, merging with OCR data.

        Outputs both legacy format ("rooms", "ocr_rooms") and SFT format
        ("roomsRecognized") for backward compatibility during gradual migration.

        CRITICAL: room_name field is IMMUTABLE — original OCR/VLM label preserved.
        name_expanded field tracks abbreviation expansions separately.

        Args:
            result:      VLM annotation result.
            output_path: Destination JSON path.
            ocr_rooms:   OCR-detected room candidates for this image.
            img_path:    Source image path — used to load image_array for the
                         window detector (Tier 2/3 require pixel data).

        Returns:
            The page's objectDetections list (Q4 -- so the caller's debug
            visualizer can draw them without re-running detection). Empty
            list when neither use_windows nor use_yolo_objects is enabled,
            or on object-detection failure -- existing callers ignoring
            this return value see no change (was -> None, purely additive).
        """
        object_detections: List[Dict[str, Any]] = []
        # Build legacy format (for backward compatibility)
        data = {
            "image_file": result.image_file,
            "image_size": result.image_size,
            "rooms": [
                {
                    "room_number": r.room_number,
                    "room_name": r.room_name,  # ← UNCHANGED from OCR/VLM
                    "name_expanded": getattr(r, 'name_expanded', None),  # ← Expansion if available
                    "category": r.category,
                    "bbox": r.bbox,
                }
                for r in result.rooms
            ],
            "panels": [{"label": p.label, "bbox": p.bbox} for p in result.panels],
            "electrical_counts": result.electrical_counts,
            "ocr_rooms": [
                {
                    "room_number": r.room_number,
                    "room_name": r.room_name,  # ← UNCHANGED
                    "name_expanded": r.name_expanded,  # ← Expansion if available
                    "bbox": list(r.bbox),
                    "confidence": r.confidence,
                }
                for r in ocr_rooms
            ],
        }

        if result.error:
            data["vlm_error"] = result.error

        # Build SFT format ("roomsRecognized") from the merged VLM+OCR room list.
        # When the VLM produces 0 valid rooms (stripe rejection), OCR rooms are
        # used so that raw annotations reflect what OCR actually found.
        merged_room_dicts = _merge_vlm_and_ocr(data["rooms"], data["ocr_rooms"])

        sft_rooms = []
        for idx, room_dict in enumerate(merged_room_dicts, start=1):
            name = room_dict.get("room_name") or room_dict.get("name", "")
            number = room_dict.get("room_number", "")
            cat = room_dict.get("category") or room_dict.get("type") or name
            mandatory_type = normalize_to_mandatory(cat)
            extended_type = get_extended_type(cat)
            source = room_dict.get("source", "vlm_only")
            ocr_conf = float(room_dict.get("confidence", 1.0))

            # Normalise bbox to [x1,y1,x2,y2].  Legacy VLM rooms store xywh;
            # OCR rooms store xywh tuples.  Both are in data["rooms"]/["ocr_rooms"].
            raw_bbox = room_dict.get("bbox", [0, 0, 0, 0])
            if len(raw_bbox) == 4:
                bx, by, bw, bh = [int(v) for v in raw_bbox]
                # xywh → xyxy
                bbox_xyxy = [bx, by, bx + bw, by + bh]
            else:
                bbox_xyxy = [int(v) for v in raw_bbox]

            sft_room = self.sft_builder.build_room(
                room_id=idx,
                mandatory_type=mandatory_type,
                original_name=name,
                room_number=number,
                bbox=bbox_xyxy,
                detection_score=0.9 if "confidence" not in room_dict else ocr_conf,
                classification_match_type="vlm" if source == "vlm_only" else "ocr",
                ocr_confidence=ocr_conf,
                source=source,
                extended_type=extended_type,
                detection_method="vlm" if source == "vlm_only" else "ocr",
                detection_model=self.config.vlm.backend if source == "vlm_only" else "PaddleOCR",
            )
            sft_rooms.append(self.sft_builder.to_dict(sft_room))

        # Add roomsRecognized to data
        data["roomsRecognized"] = sft_rooms

        # Bbox-format-sync invariant: verify that VLM-sourced entries in
        # roomsRecognized carry xyxy bboxes that match the xywh in rooms[].
        # Count divergence is expected when OCR-only rooms augment roomsRecognized,
        # so skip the per-room check in that case.
        _legacy_rooms = data.get("rooms", [])
        _sft_rooms = data.get("roomsRecognized", [])
        if _legacy_rooms and len(_legacy_rooms) == len(_sft_rooms):
            for _i, (_lr, _sr) in enumerate(zip(_legacy_rooms, _sft_rooms)):
                _lb = _lr.get("bbox", [])
                _sb = (_sr.get("coordinates") or {}).get("bbox", [])
                if len(_lb) == 4 and len(_sb) == 4:
                    _x, _y, _w, _h = _lb
                    _expected = [_x, _y, _x + _w, _y + _h]
                    if [float(v) for v in _sb] != [float(v) for v in _expected]:
                        raise ValueError(
                            f"Bbox-sync invariant violated at index {_i}: "
                            f"rooms[].bbox={_lb} (xywh) → expected "
                            f"roomsRecognized[].coordinates.bbox={_expected}, "
                            f"got {_sb}"
                        )

        # FIX-4: store exclusion zones so prepare_sft_annotation can drop mislocalized rooms
        if exclusion_zones:
            data["exclusion_zones"] = exclusion_zones

        # Add preliminary sft_ready flag (will be recomputed in post-processing step)
        data["sft_ready"] = False

        if self.config.use_windows or self.config.use_yolo_objects:
            try:
                # F11: iterate merged_room_dicts (VLM+OCR, same list that
                # becomes roomsRecognized 1:1 -- built at line ~1676,
                # already in scope), NOT data["rooms"] (VLM-only). A page
                # where VLM finds 0 rooms but OCR contributes real ones
                # previously mapped every YOLO detection against an empty
                # room list, discarding all of them regardless of quality.
                rooms_for_detection = [
                    {
                        "id": idx,
                        "type": r.get("type", "UNKNOWN"),
                        "bbox": (
                            [r["bbox"][0], r["bbox"][1],
                             r["bbox"][0] + r["bbox"][2], r["bbox"][1] + r["bbox"][3]]
                            if len(r.get("bbox", [])) == 4 else r.get("bbox", [])
                        ),
                    }
                    for idx, r in enumerate(merged_room_dicts)
                ]
                mappings = self._detect_object_mappings(
                    img_path=img_path, rooms_for_detection=rooms_for_detection,
                )
                object_detections = self._collect_object_detections(mappings)
                data["objectDetections"] = object_detections
                data["roomsRecognized"] = self._apply_object_attributes(
                    data["roomsRecognized"], mappings,
                    **self._confidence_weight_kwargs()
                )

            except Exception as e:
                logger.warning(f"Object detection failed, continuing without attributes: {e}")

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return object_detections

    def _save_ocr_annotation(
        self, img_path: Path, rooms: List[RoomCandidate], annotations_dir: Path,
        exclusion_zones: Optional[List] = None,
    ) -> None:
        """
        Save OCR-only annotation.
        
        Outputs both legacy format ("rooms") and SFT format ("roomsRecognized")
        for backward compatibility during gradual migration.
        
        CRITICAL: room_name field is IMMUTABLE — original OCR label preserved.
        name_expanded field tracks abbreviation expansions separately.
        """
        from PIL import Image

        with Image.open(img_path) as img:
            width, height = img.size

        data = {
            "image_file": img_path.name,
            "image_size": {"width": width, "height": height},
            "rooms": [
                {
                    "room_number": r.room_number,
                    "room_name": r.room_name,  # ← UNCHANGED from OCR
                    "name_expanded": r.name_expanded,  # ← Expansion if available
                    "category": "unknown",
                    "bbox": list(r.bbox),
                    "confidence": r.confidence,
                }
                for r in rooms
            ],
            "panels": [],
            "electrical_counts": {},
            "source": "ocr_only",
        }

        # Build SFT format ("roomsRecognized" key)
        sft_rooms = []
        for idx, ocr_room in enumerate(rooms, start=1):
            # Normalize room name to mandatory class
            # For OCR-only, we don't have VLM classification, so use the name itself
            mandatory_type = normalize_to_mandatory(ocr_room.room_name)
            extended_type = get_extended_type(ocr_room.room_name)
            
            # Convert bbox from (x, y, w, h) tuple to [x1, y1, x2, y2] list
            bbox = ocr_room.bbox
            if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
                x, y, w, h = bbox
                bbox_x1y1x2y2 = [x, y, x + w, y + h]
            else:
                bbox_x1y1x2y2 = list(bbox) if isinstance(bbox, tuple) else bbox
            
            # Build SFT room with OCR source
            sft_room = self.sft_builder.build_room(
                room_id=idx,
                mandatory_type=mandatory_type,
                original_name=ocr_room.room_name,
                room_number=ocr_room.room_number,
                bbox=bbox_x1y1x2y2,
                detection_score=0.95,  # OCR bbox is high confidence
                classification_match_type="exact" if mandatory_type else "fallback",
                ocr_confidence=ocr_room.confidence,  # From PaddleOCR
                source="ocr_only",
                extended_type=extended_type,
                name_expanded=ocr_room.name_expanded,
                detection_method="ocr",
                detection_model="PaddleOCR",
                ocr_backend="PaddleOCR",
            )
            sft_rooms.append(self.sft_builder.to_dict(sft_room))
        
        # Add roomsRecognized to data
        data["roomsRecognized"] = sft_rooms

        # FIX-4: store exclusion zones
        if exclusion_zones:
            data["exclusion_zones"] = exclusion_zones

        # Add preliminary sft_ready flag (will be recomputed in post-processing step)
        data["sft_ready"] = False

        if self.config.use_windows or self.config.use_yolo_objects:
            try:
                rooms_for_detection = [
                    {
                        "id": idx,
                        "type": r.get("category", "UNKNOWN"),
                        "bbox": (
                            [r["bbox"][0], r["bbox"][1],
                             r["bbox"][0] + r["bbox"][2], r["bbox"][1] + r["bbox"][3]]
                            if len(r.get("bbox", [])) == 4 else r.get("bbox", [])
                        ),
                    }
                    for idx, r in enumerate(data["rooms"])
                ]
                mappings = self._detect_object_mappings(
                    img_path=img_path, rooms_for_detection=rooms_for_detection,
                )
                data["objectDetections"] = self._collect_object_detections(mappings)
                data["roomsRecognized"] = self._apply_object_attributes(
                    data["roomsRecognized"], mappings,
                    **self._confidence_weight_kwargs()
                )

            except Exception as e:
                logger.warning(f"Object detection failed, continuing without attributes: {e}")

        output_path = annotations_dir / f"{img_path.stem}.json"
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _load_image_for_detection(img_path: Path, max_side: int = 1024) -> Optional[np.ndarray]:
        """Load image resized to ≤ max_side for CubiCasa5K inference."""
        try:
            from PIL import Image as _PIL
            with _PIL.open(img_path).convert("RGB") as img:
                w, h = img.size
                scale = min(1.0, max_side / max(w, h))
                if scale < 1.0:
                    img = img.resize((int(w * scale), int(h * scale)), _PIL.LANCZOS)
                return np.array(img)
        except Exception as e:
            logger.warning(f"Could not load image for detection {img_path}: {e}")
            return None

    def _detect_windows_with_timeout(self, image_array, rooms_for_detection,
                                      pdf_path, img_path, vlm_backend=None,
                                      timeout_sec: int = 300):
        """Run detect_windows() with a hard wall-clock timeout.

        Returns list of window mappings, or [] on timeout/error.
        """
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self.window_detector.detect_windows,
                image_array=image_array,
                rooms=rooms_for_detection,
                pdf_path=pdf_path,
                vlm_backend=vlm_backend,
                img_path=img_path,
            )
            try:
                return future.result(timeout=timeout_sec)
            except concurrent.futures.TimeoutError:
                logger.warning(
                    f"Window detection timed out after {timeout_sec}s for "
                    f"{img_path} — skipping window attributes for this image."
                )
                future.cancel()
                return []

    def _detect_object_mappings(
        self, img_path: Path, rooms_for_detection: List[Dict[str, Any]],
    ) -> List[RoomWindowMapping]:
        """Run every enabled object-detection source (WindowDetector's
        tiers, and/or T-I's YoloObjectDetector) and OR-merge their
        per-room flags (merge_object_mappings, window_detector.py).

        Only called when at least one of use_windows/use_yolo_objects is
        True (caller-guarded) -- each source is independently gated here
        too, so flipping one flag never changes the other's behavior.
        """
        mappings_sources: List[List[RoomWindowMapping]] = []

        if self.config.use_windows:
            image_array = self._load_image_for_detection(img_path)
            mappings_sources.append(
                self._detect_windows_with_timeout(
                    image_array=image_array,
                    rooms_for_detection=rooms_for_detection,
                    pdf_path=self._source_pdf_for(img_path),
                    img_path=img_path,
                    vlm_backend=self.vlm_annotator if self.config.use_vlm else None,
                )
            )

        if self.config.use_yolo_objects:
            # R4: detect_objects_tiled honors config.yolo_objects.use_tiling
            # itself (falls back to detect_objects when False) -- calling
            # it here, not detect_objects directly, makes that config flag
            # actually take effect in production (it was previously dead:
            # this call site always used the full-image path regardless
            # of use_tiling's value).
            yolo_detections = self.yolo_object_detector.detect_objects_tiled(img_path)

            if self.config.use_sam3_exemplar:
                # Tech-eval plan P2/P3: net-new door detections only,
                # seeded by yolo_detections' own high-confidence output
                # (see Sam3ExemplarDetectorConfig for the validated
                # numbers). Additive to the same list that already
                # flows through map_detections_to_rooms below -- no new
                # merge path. P3: log seed/addition counts per page --
                # the validating spike found some pages produce zero
                # seeds (>=seed_confidence_threshold), which would
                # otherwise no-op silently and look like coverage.
                additional = self.sam3_exemplar_detector.detect_additional(
                    img_path, yolo_detections
                )
                n_seeds = sum(
                    1
                    for d in yolo_detections
                    if (d.metadata or {}).get("type") == "door"
                    and d.confidence >= self.config.sam3_exemplar.seed_confidence_threshold
                )
                logger.info(
                    "SAM3 exemplar: %d door seeds (conf>=%.2f) -> %d additional doors for %s",
                    n_seeds,
                    self.config.sam3_exemplar.seed_confidence_threshold,
                    len(additional),
                    img_path.name,
                )
                yolo_detections = yolo_detections + additional

            for detection in yolo_detections:
                # Q3: full detection list (bbox+confidence) now persisted
                # via _collect_object_detections/data["objectDetections"];
                # this per-detection line is debug-only console noise, not
                # the record of truth (was logger.info -- unbounded volume
                # on full multi-page runs, e.g. 27 lines for a 7-page set).
                logger.debug(
                    "YOLO object %s: bbox=%s conf=%.2f",
                    (detection.metadata or {}).get("type"),
                    detection.bbox,
                    detection.confidence,
                )
            mappings_sources.append(
                map_detections_to_rooms(yolo_detections, rooms_for_detection)
            )

        return merge_object_mappings(mappings_sources)

    @staticmethod
    def _collect_object_detections(
        mappings: List[RoomWindowMapping],
    ) -> List[Dict[str, Any]]:
        """Flatten+dedupe intersecting_windows across all room mappings
        into the page-level objectDetections list (Q3) -- persists the
        actual bbox/confidence geometry that _apply_object_attributes'
        booleans alone discard (F2). Only detections that intersected at
        least one room are included -- same scope as the room-attribute
        tagging, no new detection source introduced.
        """
        seen_ids = set()
        detections: List[Dict[str, Any]] = []
        for mapping in mappings:
            for wd in mapping.intersecting_windows:
                if id(wd) in seen_ids:
                    continue
                seen_ids.add(id(wd))
                detections.append({
                    "category": (wd.metadata or {}).get("type"),
                    "bbox": list(wd.bbox),
                    "confidence": wd.confidence,
                    "source_tier": wd.source_tier.value,
                })
        return detections

    @staticmethod
    def _evidence_bbox_area_frac(
        room_bbox: Optional[List[float]],
        intersecting_windows: List[WindowDetection],
    ) -> Optional[float]:
        """Bounding-box area of a room's own intersecting object
        detections, as a fraction of the room's own bbox area.

        G-B (tech-eval plan): a raw MEASUREMENT, not a flag or veto -- no
        threshold for "too small" (implying an over-expanded/flooded
        room) has been validated against anything yet. Ships as an
        observable field on coordinates.attributes; using it to change a
        room's bbox or confidence is a separate, gated decision pending
        that validation. Returns None when the room has no evidence at
        all (nothing to compute a fraction against) or no bbox.
        """
        if not intersecting_windows or not room_bbox or len(room_bbox) != 4:
            return None
        xs1 = [wd.bbox[0] for wd in intersecting_windows]
        ys1 = [wd.bbox[1] for wd in intersecting_windows]
        xs2 = [wd.bbox[2] for wd in intersecting_windows]
        ys2 = [wd.bbox[3] for wd in intersecting_windows]
        evidence_area = max(0.0, max(xs2) - min(xs1)) * max(0.0, max(ys2) - min(ys1))
        rx1, ry1, rx2, ry2 = room_bbox
        room_area = max(1.0, (rx2 - rx1) * (ry2 - ry1))
        return evidence_area / room_area

    def _confidence_weight_kwargs(self) -> Dict[str, Any]:
        """T-1: unpack PipelineConfig.confidence_weights into
        _apply_object_attributes' kwarg names, once, for its 3 call sites."""
        weights = self.config.confidence_weights
        return {
            "w_detection": weights.detection,
            "w_classification": weights.classification,
            "w_ocr": weights.ocr,
            "w_corroboration": weights.corroboration,
            "corroboration_totals": self._corroboration_totals,
        }

    @staticmethod
    def _apply_object_attributes(
        sft_rooms: List[Dict[str, Any]],
        mappings: List[RoomWindowMapping],
        w_detection: float = 0.3,
        w_classification: float = 0.4,
        w_ocr: float = 0.3,
        w_corroboration: float = 0.0,
        corroboration_totals: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Tag each sft_room's coordinates.attributes with object-
        detection flags (has_door/has_windows/etc), keyed by its index
        into sft_rooms. Tag-only: every room is returned regardless of
        match, never accept/reject (T-I design decision, unchanged from
        the pre-T-I use_windows-only behavior).

        G-B additions (tech-eval plan, confidence-boost + boundary signal):
        evidence_count/evidence_score are a plain, unweighted tally of the
        5 existing flags. evidence_bbox_area_frac is the boundary-
        consistency measurement (see _evidence_bbox_area_frac).

        T-1 (G-1, 2026-09-03): evidence_score now ALSO feeds
        ConfidenceComputer.apply_corroboration as this method's
        corroboration_score -- the wire G-B's own comment said didn't
        exist yet. w_corroboration defaults to 0.0 (ConfidenceWeightsConfig's
        default), so with no caller override this recomputes each room's
        `confidence` from its own already-stored 3-factor components and
        reproduces the same value -- a no-op until a caller passes a
        nonzero weight. Rooms with no mapping match are untouched, same
        as always: no evidence pass, no confidence change, no penalty.
        """
        mapping_by_id = {m.room_id: m for m in mappings}
        updated = []
        for idx, sft_room in enumerate(sft_rooms):
            mapping = mapping_by_id.get(idx)
            if mapping:
                coords = sft_room.get("coordinates", {})
                attrs = coords.get("attributes", {}) if coords else {}
                evidence_flags = (
                    mapping.has_door, mapping.has_windows, mapping.has_toilet,
                    mapping.has_bathtub, mapping.has_sink,
                )
                evidence_count = sum(1 for f in evidence_flags if f)
                evidence_score = evidence_count / len(evidence_flags)
                if corroboration_totals is not None:
                    corroboration_totals["evidence_count"] += evidence_count
                    corroboration_totals["evidence_score_sum"] += evidence_score
                attrs.update({
                    "has_windows": mapping.has_windows,
                    "has_skylights": mapping.has_skylights,
                    "has_openings": mapping.has_openings,
                    "has_door": mapping.has_door,
                    "has_toilet": mapping.has_toilet,
                    "has_bathtub": mapping.has_bathtub,
                    "has_sink": mapping.has_sink,
                    "window_count": mapping.window_count,
                    "evidence_count": evidence_count,
                    "evidence_score": evidence_score,
                    "evidence_bbox_area_frac": AnnotationPipeline._evidence_bbox_area_frac(
                        coords.get("bbox") if coords else None,
                        mapping.intersecting_windows,
                    ),
                })
                if coords:
                    coords["attributes"] = attrs
                ConfidenceComputer.apply_corroboration(
                    sft_room, evidence_score,
                    w_detection=w_detection, w_classification=w_classification,
                    w_ocr=w_ocr, w_corroboration=w_corroboration,
                )
            updated.append(sft_room)
        return updated

    def _build_config_dict(self) -> Dict[str, Any]:
        """Build the JSON-serializable config snapshot shared by
        pipeline_config.json (_save_config) and the run manifest (Q1) --
        one place to list config fields, not two drifting copies."""
        return {
            "pdf": {"dpi": self.config.pdf.dpi},
            "ocr": {
                "backend": self.config.ocr.backend,
                "confidence_threshold": self.config.ocr.confidence_threshold,
                "preprocess": self.config.ocr.preprocess,
            },
            "vlm": {
                "backend": self.config.vlm.backend,
                "model": self.config.vlm.active_model,
                "qwen_model": self.config.vlm.qwen_model,
                "unsloth_model": self.config.vlm.unsloth_model,
                "max_tokens": self.config.vlm.max_tokens,
            },
            "sam": {
                "model_type": self.config.sam.model_type,
            },
            "pipeline": {
                "use_vlm": self.config.use_vlm,
                "use_sam": self.config.use_sam,
                "use_windows": self.config.use_windows,
                "use_yolo_objects": self.config.use_yolo_objects,
                "use_sam3_exemplar": self.config.use_sam3_exemplar,
                "use_semantic_reconciliation": self.config.use_semantic_reconciliation,
                "min_rooms_for_sft": self.config.min_rooms_for_sft,
                "use_tiling": self.config.use_tiling,
                "tile_cols": self.config.tile_cols,
                "tile_rows": self.config.tile_rows,
                "tile_overlap_pct": self.config.tile_overlap_pct,
                "tile_trigger_px": self.config.tile_trigger_px,
            },
            "confidence_weights": {
                "detection": self.config.confidence_weights.detection,
                "classification": self.config.confidence_weights.classification,
                "ocr": self.config.confidence_weights.ocr,
                "corroboration": self.config.confidence_weights.corroboration,
            },
            "prompt_versions": dict(prompt_templates.PROMPT_VERSIONS),
        }

    def _save_config(self, config_path: Path) -> None:
        """Save pipeline configuration for reproducibility."""
        config_dict = self._build_config_dict()
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

    def _print_step(self, title: str) -> None:
        """Print a step header."""
        print(f"\n{'='*60}")
        print(title)
        print("=" * 60)

    def _print_summary(self, stats: Dict, output_dir: Path) -> None:
        """Print pipeline summary."""
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print("\nExtraction:")
        print(f"  PDFs processed:      {stats['pdfs_processed']}")
        print(f"  Images extracted:    {stats['images_extracted']}")
        print(f"  Images annotated:    {stats['images_annotated']}")
        print(f"  Rooms detected:      {stats['rooms_detected']}")

        print("\nPost-Processing:")
        print(f"  Annotations normalized: {stats['annotations_normalized']}")
        print(f"  Annotations with issues: {stats['annotations_with_issues']}")
        print(f"  Total quality issues: {stats['total_quality_issues']}")
        print(f"  Room regions extracted: {stats['room_regions_extracted']}")

        print(f"\nReview:")
        print(f"  Flagged for review:  {stats['flagged_for_review']}")

        if stats["images_extracted"] > 0:
            review_pct = stats["flagged_for_review"] / stats["images_extracted"] * 100
            print(f"  Review percentage:   {review_pct:.1f}%")

        # Per-image room bbox table from processed_annotations (sft_ready only)
        processed_dir = output_dir / "processed_annotations"
        if processed_dir.exists():
            ann_files = sorted(processed_dir.glob("*.json"))
            if ann_files:
                print("\nDetected rooms (sft_ready pages):")
                MAX_ROOMS_SHOWN = 10  # cap per page to avoid flooding console
                for ann_path in ann_files:
                    try:
                        with open(ann_path) as _f:
                            ann = json.load(_f)
                        if not ann.get("sft_ready"):
                            continue
                        rooms = ann.get("roomsRecognized", [])
                        print(f"\n  {ann_path.stem}  ({len(rooms)} rooms)")
                        for r in rooms[:MAX_ROOMS_SHOWN]:
                            bbox = (r.get("coordinates") or {}).get("bbox", [])
                            name = r.get("name", "?")[:28]
                            rtype = r.get("type", "?")[:20]
                            conf = r.get("confidence", 0.0)
                            bbox_str = (
                                f"[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]"
                                if len(bbox) == 4 else "[]"
                            )
                            print(
                                f"    {rtype:<22} {name:<30} "
                                f"bbox={bbox_str:<26} conf={conf:.2f}"
                            )
                        if len(rooms) > MAX_ROOMS_SHOWN:
                            print(f"    ... +{len(rooms) - MAX_ROOMS_SHOWN} more rooms")
                    except Exception:
                        pass

        print(f"\nOutput directory: {output_dir}")
        print("\nGenerated files:")
        print(f"  - images/                     (extracted floor plan images)")
        print(f"  - annotations/                (raw annotations)")
        print(f"  - processed_annotations/      (normalized annotations)")
        print(f"  - room_regions/               (extracted room images for training)")
        print(f"  - quality_reports/            (detailed quality issues by image)")
        print(f"  - post_processing_summary.json (post-processing statistics)")
        print(f"  - label_studio_import.json    (import into Label Studio)")
        print(f"  - needs_review.json           (images flagged for review)")

        print("\nNext steps:")
        print(f"  1. Review quality_reports/ for any critical issues")
        print(f"  2. Import label_studio_import.json into Label Studio")
        print(f"  3. Review the {stats['flagged_for_review']} flagged images")
        print("  4. Export corrected annotations as COCO JSON")
        print("  5. Use room_regions/ as training examples for fine-tuning")


def setup_logging(verbose: bool = False, output_dir: Union[str, Path] = None) -> None:
    """Configure logging for the pipeline.
    
    Args:
        verbose: If True, console output is DEBUG; otherwise INFO
        output_dir: If provided, also write full DEBUG logs to file in this directory
    """
    from datetime import datetime
    
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)
    
    # Get root logger and set to DEBUG (handlers will filter)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler (respects --verbose flag)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (always DEBUG, written to output directory if provided)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        log_path = output_dir / f"pipeline_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Log the file path to console so user knows where to find logs
        root_logger.info(f"Logging pipeline activity to: {log_path}")
        
        # Store FileHandler reference for later verification
        setup_logging._file_handler = file_handler
        setup_logging._file_path = log_path
    
    # Store root logger level for later verification/restoration
    # (in case paddle or other code changes it)
    setup_logging._root_logger_level = logging.DEBUG


def verify_logging_handlers() -> bool:
    """
    Verify that the logging FileHandler is still present and root logger level is correct.
    Re-add FileHandler if missing (can happen if other code reconfigures logging).
    Restore logger level if it was changed (e.g., by PaddlePaddle import).
    
    Returns:
        True if FileHandler is present/restored, False if no file logging configured
    """
    root_logger = logging.getLogger()
    
    # Check and restore root logger level if it was changed by paddle
    # PaddlePaddle resets root logger to WARNING (30) during import
    stored_level = getattr(setup_logging, '_root_logger_level', logging.DEBUG)
    if root_logger.level != stored_level and root_logger.level > stored_level:
        root_logger.setLevel(stored_level)
        root_logger.debug(f"Restored root logger level to {logging.getLevelName(stored_level)}")
    
    # Check if FileHandler exists
    file_handlers = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
    
    if file_handlers:
        # FileHandler still present
        return True
    
    # Check if we stored a reference to re-add
    if hasattr(setup_logging, '_file_handler') and hasattr(setup_logging, '_file_path'):
        try:
            # Re-add the FileHandler
            file_handler = setup_logging._file_handler
            if not file_handler.stream.closed:
                root_logger.addHandler(file_handler)
                root_logger.warning("Re-added lost FileHandler (logging reconfigured by other code)")
                return True
        except Exception as e:
            root_logger.warning(f"Failed to restore FileHandler: {e}")
            return False
    
    return False


def _warn_if_sam3_exemplar_orphaned(config: PipelineConfig) -> None:
    """use_sam3_exemplar has no seed source without use_yolo_objects --
    warn rather than silently no-op (found while building this feature's
    test coverage: the flag combination compiles and runs, produces zero
    additions on every page, and looks like coverage without being any)."""
    if config.use_sam3_exemplar and not config.use_yolo_objects:
        logger.warning(
            "--use-sam3-exemplar has no effect without --use-yolo-objects "
            "(no seed source) -- this run will silently no-op the SAM3 stage"
        )


def _warn_if_yolo_objects_ocr_only(config: PipelineConfig) -> None:
    """use_yolo_objects tags rooms via geometric bbox intersection
    (map_detections_to_rooms) against whatever room boxes exist at that
    point in the run. In OCR-only mode (use_vlm=False) those boxes are
    the OCR text label's own bbox (e.g. 71x14px -- where "SUITE 301" is
    printed), not the room's spatial extent, so a door/window detection
    elsewhere in the room almost never geometrically intersects it.
    SAM boundary expansion (use_sam) cannot rescue this either: it runs
    in STEP 3.5, strictly after object detection -> room mapping is
    already computed and persisted earlier in the same per-image pass.

    Confirmed on a real page (P-1, 2026-09-03): 26 real OCR rooms + 44
    real YOLO detections -> 0 attached (objectDetections empty). All 14
    real runs on disk instead used use_vlm=True, where room boxes come
    from the VLM (real spatial extent) and this problem does not occur
    -- but the flag combination itself still silently burns a full
    tiled-detector pass with ~zero attached signal if anyone runs it
    OCR-only, so warn rather than let that go unnoticed."""
    if config.use_yolo_objects and not config.use_vlm:
        logger.warning(
            "--use-yolo-objects with no --use-vlm: room boxes in OCR-only "
            "mode are text-label bboxes, not room extents, so object "
            "detections will almost never attach to any room (objectDetections "
            "will likely be empty). --use-sam does not fix this (SAM expansion "
            "runs after object mapping is already computed). See memory "
            "use-yolo-objects-ocr-only-nearly-noop for detail."
        )


def main():
    """CLI entry point for the annotation pipeline."""
    parser = argparse.ArgumentParser(
        description="MEP Floor Plan Annotation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m preprocessing_annotations.pipeline --input ./pdfs --output ./dataset

  # With VLM and SAM
  python -m preprocessing_annotations.pipeline --input ./pdfs --output ./dataset --use-vlm --use-sam

  # High detail mode for small symbols
  python -m preprocessing_annotations.pipeline --input ./pdfs --output ./dataset --high-detail
        """,
    )

    parser.add_argument(
        "--input", "-i", required=True, help="Input directory containing PDF files"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output directory for results"
    )
    parser.add_argument(
        "--use-vlm", action="store_true", help="Use Claude VLM for annotation"
    )
    parser.add_argument(
        "--ocr-backend",
        choices=["easyocr", "paddleocr"],
        default="paddleocr",
        help="OCR backend to use. Default: paddleocr. Use 'easyocr' for legacy "
             "backend (requires: pip install easyocr)",
    )
    parser.add_argument(
        "--use-semantic-reconciliation",
        action="store_true",
        help="Merge OCR text into VLM room polygons via spatial containment "
             "(requires: pip install shapely). Now ON by default; this flag is a no-op.",
    )
    parser.add_argument(
        "--no-semantic-reconciliation",
        action="store_true",
        help="Disable semantic reconciliation (override the default-on setting).",
    )
    parser.add_argument(
        "--vlm-backend",
        choices=["claude", "qwen", "unsloth"],
        default="claude",
        help="VLM backend. Options: 'claude' (API, default), 'qwen' (local inference), "
             "'unsloth' (optimized local, ~2x faster, requires: pip install unsloth)",
    )
    parser.add_argument(
        "--vlm-model",
        default=None,
        help="Model ID for the VLM backend. For Claude: claude-haiku-4-5-20251001 (default), "
             "claude-sonnet-4-5-20250929. Overrides the VLM_MODEL env var.",
    )
    parser.add_argument(
        "--qwen-model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="HuggingFace model ID for 'qwen' backend "
             "(default: Qwen/Qwen2.5-VL-7B-Instruct)",
    )
    parser.add_argument(
        "--unsloth-model",
        default=None,
        help="Model key for 'unsloth' backend. Options: qwen3-vl-8b (default), "
             "qwen3-vl-2b, qwen3-vl-4b, qwen3-vl-2b-thinking, "
             "qwen3-vl-4b-thinking, qwen3-vl-8b-thinking",
    )
    parser.add_argument(
        "--use-sam", action="store_true", help="Use SAM for boundary refinement"
    )
    parser.add_argument(
        "--use-windows",
        action="store_true",
        help="Enable window detection (Tier 1 PDF layers only; Tier 3 VLM disabled "
             "until Tier 2 CubiCasa5K is integrated). Default: off.",
    )
    parser.add_argument(
        "--use-yolo-objects",
        action="store_true",
        help="Enable the T-S2 fine-tuned YOLO door/window detector (T-I). "
             "Tag-only: writes has_door/has_windows room attributes, never "
             "accepts/rejects a room. Checkpoint path from "
             "YoloObjectDetectorConfig.checkpoint_path. Default: off.",
    )
    parser.add_argument(
        "--use-sam3-exemplar",
        action="store_true",
        help="Enable the tiled SAM3 box-exemplar door detector, seeded by "
             "--use-yolo-objects' own high-confidence detections (no "
             "effect unless --use-yolo-objects is also set). GT-measured "
             "win over YOLO-alone (Sam3ExemplarDetectorConfig docstring) "
             "was originally measured at YOLO conf=0.5; this codebase's "
             "default confidence_threshold changed 0.75->0.25 (I-2, "
             "2026-09-01) and the win was re-measured there too -- small "
             "but consistent F1 gain on all 3 GT datasets, recall-favoring "
             "(see memory n3-sam3-union-remeasured-conf025-2026-09-01). "
             "Real project pages (not GT) have also shown this feature "
             "losing real doors under its current tile grid -- see that "
             "docstring's CONTESTED note before enabling in production. "
             "Tag-only, same as --use-yolo-objects. Default: off.",
    )
    parser.add_argument(
        "--corroboration-weight",
        type=float,
        default=None,
        help="T-1 (G-1): weight for an ADDITIVE object-detection "
             "corroboration bonus on top of room confidence (default: "
             "ConfidenceWeightsConfig.corroboration, 0.0 -- inert). Has "
             "no effect unless --use-yolo-objects or --use-windows is "
             "also set (no object mapping to corroborate with "
             "otherwise). Boost-only: a room with no nearby door/window "
             "evidence is unaffected, never penalized. Not validated at "
             "any nonzero value yet -- T-1's own spike task (confidence "
             "distribution spread, sft_recommended rate vs baseline) has "
             "not been run; do not ship a nonzero default without it.",
    )
    parser.add_argument(
        "--enable-mlflow",
        action="store_true",
        help="Enable MLflow run tracking for this run (default: off). "
             "Requires an MLflow tracking server reachable at "
             "--mlflow-tracking-uri (default: http://127.0.0.1:5000 or "
             "$MLFLOW_TRACKING_URI). If mlflow is not installed or the "
             "server is unreachable, the run continues unaffected and a "
             "warning is logged.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="Override MlflowConfig.tracking_uri for this run.",
    )
    parser.add_argument(
        "--eval-gt-dir",
        type=str,
        default=None,
        help="Optional path to a ground-truth annotations directory (same "
             "format as this pipeline's own annotations/ output). If set, "
             "runs bbox_metrics.evaluate_dataset against this run's "
             "predictions after run() completes and logs precision/recall/"
             "IoU-family metrics to MLflow (if enabled) and to "
             "eval_metrics.json in output_dir.",
    )
    parser.add_argument(
        "--high-detail",
        action="store_true",
        help="Use high-detail settings (300 DPI, more scales)",
    )
    parser.add_argument(
        "--fast", action="store_true", help="Use fast processing settings"
    )
    parser.add_argument(
        "--dpi", type=int, default=None, help="Custom DPI for PDF extraction"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images that already have annotations",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--min-rooms",
        type=int,
        default=1,
        help="Minimum rooms per image for sft_ready=True (default: 1). "
             "Images with ≥3 rooms also get sft_recommended=True regardless of this value.",
    )
    parser.add_argument(
        "--no-tiling",
        action="store_true",
        help="Disable tiled VLM inference. Send full image to VLM in one pass. "
             "Tiling is ON by default for local backends when max(w,h) > 3000px.",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="If another pipeline run is active, exit immediately (code 3) instead of "
             "waiting. Default: block and wait (serialize runs to protect host RAM).",
    )

    args = parser.parse_args()

    # Setup logging (with file output to --output directory)
    setup_logging(args.verbose, output_dir=args.output)

    # Create config
    if args.high_detail:
        config = PipelineConfig.for_high_detail()
    elif args.fast:
        config = PipelineConfig.for_fast_processing()
    else:
        config = PipelineConfig()

    # Apply CLI overrides
    config.use_vlm = args.use_vlm
    config.use_sam = args.use_sam
    config.use_windows = args.use_windows
    config.use_yolo_objects = args.use_yolo_objects
    config.use_sam3_exemplar = args.use_sam3_exemplar
    _warn_if_sam3_exemplar_orphaned(config)
    _warn_if_yolo_objects_ocr_only(config)
    if args.corroboration_weight is not None:
        config.confidence_weights.corroboration = args.corroboration_weight
    config.mlflow.enabled = args.enable_mlflow
    if args.mlflow_tracking_uri is not None:
        config.mlflow.tracking_uri = args.mlflow_tracking_uri
    if args.eval_gt_dir is not None:
        config.eval_gt_dir = args.eval_gt_dir
    config.min_rooms_for_sft = args.min_rooms
    if args.no_tiling:
        config.use_tiling = False
    config.ocr.backend = args.ocr_backend
    if args.no_semantic_reconciliation:
        config.use_semantic_reconciliation = False
    # --use-semantic-reconciliation is now a no-op (on by default); kept for compat
    config.vlm.backend = args.vlm_backend
    if args.vlm_model is not None:
        config.vlm.model = args.vlm_model
    config.vlm.qwen_model = args.qwen_model
    if args.unsloth_model is not None:
        config.vlm.unsloth_model = args.unsloth_model

    if args.dpi:
        config.pdf.dpi = args.dpi

    # Serialize runs — VLM + SAM load multi-GB models; two concurrent runs exhaust RAM.
    with single_instance(wait=not args.no_wait):
        pipeline = AnnotationPipeline(config)

        try:
            stats = pipeline.run(
                input_dir=args.input,
                output_dir=args.output,
                skip_existing=args.skip_existing,
            )
            sys.exit(0)
        except PipelineError as e:
            logger.error(f"Pipeline failed: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
            sys.exit(130)


if __name__ == "__main__":
    main()
