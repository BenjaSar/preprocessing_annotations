"""
Main annotation pipeline orchestration for MEP floor plans.

This module provides the AnnotationPipeline class that coordinates
all annotation steps: PDF extraction, OCR, VLM annotation, SAM
segmentation, and export.
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

# Handle both relative and absolute imports for flexibility
try:
    from .config import PipelineConfig
    from .pdf_extractor import PDFExtractor, PageTypeClassifier
    from .ocr_extractor import MEPTextExtractor, RoomCandidate
    from .vlm_annotator import VLMAnnotator
    from .semantic_reconciler import SemanticReconciler
    from .vlm_backend import VLMFactory
    from .sam_segmenter import RoomSegmenter
    from .exporters import LabelStudioExporter, ReviewPrioritizer, COCOExporter, CoverageReporter
    from .automation import (
        LabelNormalizer, QualityChecker, RegionExtractor,
        ImageResizer, SemanticRoomValidator, TaxonomyNormalizer,
        filter_by_confidence, validate_for_sft, prepare_sft_annotation,
        SFTAnnotationBuilder, build_annotation_json
    )
    from .automation.taxonomy import normalize_to_mandatory, get_extended_type
    from .automation.abbreviation_ocr_recovery import (
        AbbreviationOCRRecovery, ResidentialAbbreviationRecovery
    )
except ImportError:
    from config import PipelineConfig
    from pdf_extractor import PDFExtractor, PageTypeClassifier
    from ocr_extractor import MEPTextExtractor, RoomCandidate
    from vlm_annotator import VLMAnnotator
    from semantic_reconciler import SemanticReconciler
    from vlm_backend import VLMFactory
    from sam_segmenter import RoomSegmenter
    from exporters import LabelStudioExporter, ReviewPrioritizer, COCOExporter, CoverageReporter
    from automation import (
        LabelNormalizer, QualityChecker, RegionExtractor,
        ImageResizer, SemanticRoomValidator, TaxonomyNormalizer,
        filter_by_confidence, validate_for_sft, prepare_sft_annotation,
        SFTAnnotationBuilder, build_annotation_json
    )
    from automation.taxonomy import normalize_to_mandatory, get_extended_type
    from automation.abbreviation_ocr_recovery import (
        AbbreviationOCRRecovery, ResidentialAbbreviationRecovery
    )

logger = logging.getLogger(__name__)


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
        self._exporter = None
        self._prioritizer = None
        self._label_normalizer = None
        self._quality_checker = None
        self._region_extractor = None
        
        # Initialize SFT annotation builder (used for mandatory schema output)
        self.sft_builder = SFTAnnotationBuilder()

    @property
    def pdf_extractor(self) -> PDFExtractor:
        if self._pdf_extractor is None:
            self._pdf_extractor = PDFExtractor(self.config.pdf)
        return self._pdf_extractor

    @property
    def ocr_extractor(self) -> MEPTextExtractor:
        if self._ocr_extractor is None:
            self._ocr_extractor = MEPTextExtractor(self.config.ocr)
        return self._ocr_extractor

    @property
    def vlm_annotator(self):
        """Return appropriate VLM backend (Claude VLMAnnotator or Qwen VLMBackend)."""
        if self._vlm_annotator is None:
            if self.config.vlm.backend.lower() == "qwen":
                # Use VLMFactory to create Qwen backend
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

        input_path = Path(input_dir)
        output_dir = Path(output_dir)

        # Setup output directories
        images_dir = output_dir / "images"
        annotations_dir = output_dir / "annotations"

        images_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)

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
                            # copy+delete instead of move: keeps a recovery path
                            # in skipped_pages/ if classification is a false positive
                            # on a scanned or ambiguous page, while still removing
                            # the image from images_dir so OCR does not process it.
                            dst = skipped_dir / img_candidate.name
                            shutil.copy2(str(img_candidate), str(dst))
                            img_candidate.unlink()
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
        abbrev_recovery = AbbreviationOCRRecovery()  # uses pattern matching only

        for img_path in self._iter_images(images_dir):
            try:
                # 2a: Standard OCR room detection (compound merging + number linking built-in).
                # extract_and_find_rooms now returns (candidates, raw_detections) to avoid
                # running OCR twice — raw_detections are reused for abbreviation recovery.
                rooms, raw_detections = self.ocr_extractor.extract_and_find_rooms(img_path)
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
                all_rooms = list(rooms)
                for abbrev_cand in recovered_abbrevs:
                    ax, ay = abbrev_cand.bbox[0], abbrev_cand.bbox[1]
                    already_covered = any(
                        abs(r.bbox[0] - ax) < 50 and abs(r.bbox[1] - ay) < 50
                        for r in all_rooms
                    )
                    if not already_covered:
                        all_rooms.append(abbrev_cand)

                ocr_results[img_path.name] = all_rooms
                logger.info(
                    f"  {img_path.name}: {len(rooms)} rooms + "
                    f"{len(recovered_abbrevs)} abbreviations recovered"
                )
            except Exception as e:
                logger.error(f"  {img_path.name}: OCR failed - {e}")
                ocr_results[img_path.name] = []
            finally:
                if img_path.name in image_status:
                    image_status[img_path.name]["ocr"] = True

        # Step 3: VLM annotation (optional)
        if self.config.use_vlm:
            self._print_step("STEP 3: VLM zero-shot annotation")

            for img_path in self._iter_images(images_dir):
                ann_path = annotations_dir / f"{img_path.stem}.json"

                if skip_existing and ann_path.exists():
                    logger.info(f"  Skipping {img_path.name} (annotation exists)")
                    continue

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
                    
                    # Apply semantic reconciliation if enabled
                    ocr_list = ocr_results.get(img_path.name, [])
                    if self.config.use_semantic_reconciliation and ocr_list:
                        # Convert OCR RoomCandidate objects to dicts for reconciler
                        ocr_dicts = []
                        for room in ocr_list:
                            ocr_dicts.append({
                                "text": room.name,
                                "bbox": room.bbox,
                                "confidence": room.confidence
                            })
                        result = self._apply_semantic_reconciliation(result, ocr_dicts)
                    
                    stats["images_annotated"] += 1
                    stats["rooms_detected"] += len(result.rooms)

                    # Save annotation
                    self._save_annotation(result, ann_path, ocr_results.get(img_path.name, []))

                    if img_path.name in image_status:
                        image_status[img_path.name]["annotated"] = True

                    logger.info(
                        f"  {img_path.name}: {len(result.rooms)} rooms, "
                        f"{len(result.panels)} panels (VLM)"
                    )
                except Exception as e:
                    logger.error(f"  {img_path.name}: VLM annotation failed - {e}")

                    # Fallback: always write OCR annotation (even if empty) so
                    # the image is not silently lost from downstream steps.
                    ocr_rooms = ocr_results.get(img_path.name, [])
                    try:
                        self._save_ocr_annotation(img_path, ocr_rooms, annotations_dir)
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
                    self._save_ocr_annotation(img_path, rooms, annotations_dir)
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
                annotation = prepare_sft_annotation(annotation)
                filtered_room_count = len(annotation.get("rooms", []))

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
                quality_summary["annotations_normalized"] += 1

                # Step 4c: Check annotation quality
                issues = self.quality_checker.check_annotation(annotation)
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

                if not is_sft_ready and room_count == 0:
                    # This is an empty annotation — move source image to skipped_pages/
                    # for human review (may be cover page, title block, or OCR failure)
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
                    # Normal case: write processed annotation
                    processed_path = processed_dir / ann_path.name
                    with open(processed_path, "w") as f:
                        json.dump(annotation, f, indent=2)
                    quality_summary["total_annotations"] += 1

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

        # Step 5: SAM refinement (optional)
        if self.config.use_sam:
            self._print_step("STEP 5: SAM boundary refinement")

            for img_path in self._iter_images(images_dir):
                ann_path = annotations_dir / f"{img_path.stem}.json"

                if not ann_path.exists():
                    continue

                try:
                    with open(ann_path) as f:
                        ann = json.load(f)

                    rooms = ann.get("rooms", [])
                    if rooms:
                        refined = self.sam_segmenter.refine_annotations(
                            img_path, rooms
                        )
                        ann["rooms"] = refined
                        ann["sam_refined"] = True

                        with open(ann_path, "w") as f:
                            json.dump(ann, f, indent=2)

                        logger.info(f"  {img_path.name}: Refined {len(refined)} rooms")

                except Exception as e:
                    logger.error(f"  {img_path.name}: SAM refinement failed - {e}")

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

        return stats

    def _get_vlm_result(self, img_path: Path):
        """Get VLM annotation result, handling both Claude and Qwen backends."""
        vlm = self.vlm_annotator
        is_qwen = self.config.vlm.backend.lower() == "qwen"
        
        if is_qwen:
            # Qwen backend returns list of dicts from detect_rooms()
            rooms_dicts = vlm.detect_rooms(img_path)
            result = self._qwen_dicts_to_vlm_result(rooms_dicts, img_path)
        else:
            # Claude backend returns VLMAnnotationResult from annotate()
            result = vlm.annotate(img_path)
        
        return result

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
            
            from vlm_annotator import RoomAnnotation
            room = RoomAnnotation(
                room_number=room_dict.get("room_number", f"room_{idx}"),
                room_name=room_dict.get("room_name", room_dict.get("room_type", "Room")),
                category=room_dict.get("room_type", "other"),
                bbox=[x, y, w, h]
            )
            rooms.append(room)
        
        # Create VLMAnnotationResult
        from vlm_annotator import VLMAnnotationResult
        return VLMAnnotationResult(
            image_file=img_path.name,
            image_size={"width": width, "height": height},
            rooms=rooms,
            panels=[],
            electrical_counts={}
        )

    def _apply_semantic_reconciliation(self, result, ocr_results: List[dict]):
        """Apply semantic reconciliation to VLM result if enabled."""
        if not self.config.use_semantic_reconciliation:
            return result
        
        if not ocr_results:
            return result
        
        try:
            reconciler = SemanticReconciler(self.config.ocr)
            
            # Convert VLM rooms to dict format for reconciler
            vlm_rooms_dicts = []
            for room in result.rooms:
                vlm_rooms_dicts.append({
                    "room_id": room.room_number,
                    "room_name": room.room_name,
                    "room_type": room.category,
                    "bbox": room.bbox,
                    "confidence": 1.0,
                    "polygon": None
                })
            
            # Reconcile with OCR results
            enriched = reconciler.reconcile_with_config(
                ocr_results, vlm_rooms_dicts, self.config
            )
            
            # Update result.rooms with enriched data
            from vlm_annotator import RoomAnnotation
            new_rooms = []
            for enriched_room in enriched:
                room = RoomAnnotation(
                    room_number=enriched_room.get("room_number", ""),
                    room_name=enriched_room.get("room_name", "Room"),
                    category=enriched_room.get("room_type", "other"),
                    bbox=enriched_room.get("bbox", [0, 0, 100, 100])
                )
                new_rooms.append(room)
            
            result.rooms = new_rooms
            return result
        except Exception as e:
            logger.warning(f"Semantic reconciliation failed: {e}. Using VLM results as-is.")
            return result

    def _save_annotation(
        self, result, output_path: Path, ocr_rooms: List[RoomCandidate]
    ) -> None:
        """
        Save VLM annotation result, merging with OCR data.
        
        Outputs both legacy format ("rooms", "ocr_rooms") and SFT format ("roomsRecognized")
        for backward compatibility during gradual migration.
        
        CRITICAL: room_name field is IMMUTABLE — original OCR/VLM label preserved.
        name_expanded field tracks abbreviation expansions separately.
        """
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

        # Build SFT format ("roomsRecognized" key)
        sft_rooms = []
        for idx, vlm_room in enumerate(result.rooms, start=1):
            # Normalize category to mandatory class
            mandatory_type = normalize_to_mandatory(vlm_room.category)
            extended_type = get_extended_type(vlm_room.category)
            
            # Convert bbox from [x, y, w, h] to [x1, y1, x2, y2]
            bbox = vlm_room.bbox
            if len(bbox) == 4:
                x, y, w, h = bbox
                bbox_x1y1x2y2 = [x, y, x + w, y + h]
            else:
                bbox_x1y1x2y2 = bbox
            
            # Build SFT room with VLM source
            sft_room = self.sft_builder.build_room(
                room_id=idx,
                mandatory_type=mandatory_type,
                original_name=vlm_room.room_name,
                room_number=vlm_room.room_number,
                bbox=bbox_x1y1x2y2,
                detection_score=0.9,  # VLM default
                classification_match_type="vlm",  # VLM classification
                ocr_confidence=1.0,  # VLM-generated, no OCR uncertainty
                source="vlm_only",
                extended_type=extended_type,
                detection_method="vlm",
                detection_model=self.config.vlm.backend,
            )
            sft_rooms.append(self.sft_builder.to_dict(sft_room))
        
        # Add roomsRecognized to data
        data["roomsRecognized"] = sft_rooms

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def _save_ocr_annotation(
        self, img_path: Path, rooms: List[RoomCandidate], annotations_dir: Path
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

        output_path = annotations_dir / f"{img_path.stem}.json"
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def _save_config(self, config_path: Path) -> None:
        """Save pipeline configuration for reproducibility."""
        # Convert dataclass config to dict (simplified)
        config_dict = {
            "pdf": {"dpi": self.config.pdf.dpi},
            "ocr": {
                "backend": self.config.ocr.backend,
                "confidence_threshold": self.config.ocr.confidence_threshold,
                "preprocess": self.config.ocr.preprocess,
            },
            "vlm": {
                "backend": self.config.vlm.backend,
                "model": self.config.vlm.model,
                "qwen_model": self.config.vlm.qwen_model,
                "max_tokens": self.config.vlm.max_tokens,
            },
            "sam": {
                "model_type": self.config.sam.model_type,
            },
            "pipeline": {
                "use_vlm": self.config.use_vlm,
                "use_sam": self.config.use_sam,
                "use_semantic_reconciliation": self.config.use_semantic_reconciliation,
            },
        }

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


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the pipeline."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
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
             "(requires: pip install shapely)",
    )
    parser.add_argument(
        "--vlm-backend",
        choices=["claude", "qwen", "unsloth"],
        default="claude",
        help="VLM backend. Options: 'claude' (API, default), 'qwen' (local inference), "
             "'unsloth' (optimized local, ~2x faster, requires: pip install unsloth)",
    )
    parser.add_argument(
        "--qwen-model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="HuggingFace model ID for 'qwen' backend "
             "(default: Qwen/Qwen2.5-VL-7B-Instruct)",
    )
    parser.add_argument(
        "--unsloth-model",
        default="qwen2.5-vl-7b",
        help="Model key for 'unsloth' backend. Options: qwen2.5-vl-7b (default), "
             "qwen3-vl-2b, qwen3-vl-4b, qwen3-vl-8b",
    )
    parser.add_argument(
        "--use-sam", action="store_true", help="Use SAM for boundary refinement"
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

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

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
    config.ocr.backend = args.ocr_backend
    config.use_semantic_reconciliation = args.use_semantic_reconciliation
    config.vlm.backend = args.vlm_backend
    config.vlm.qwen_model = args.qwen_model
    config.vlm.unsloth_model = args.unsloth_model

    if args.dpi:
        config.pdf.dpi = args.dpi

    # Run pipeline
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
