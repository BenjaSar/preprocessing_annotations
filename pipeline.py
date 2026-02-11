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
    from .pdf_extractor import PDFExtractor
    from .ocr_extractor import MEPTextExtractor, RoomCandidate
    from .vlm_annotator import VLMAnnotator
    from .sam_segmenter import RoomSegmenter
    from .exporters import LabelStudioExporter, ReviewPrioritizer
    from .automation import (
        LabelNormalizer, QualityChecker, RegionExtractor,
        ImageResizer, SemanticRoomValidator, TaxonomyNormalizer,
        filter_by_confidence, validate_for_sft, prepare_sft_annotation
    )
except ImportError:
    from config import PipelineConfig
    from pdf_extractor import PDFExtractor
    from ocr_extractor import MEPTextExtractor, RoomCandidate
    from vlm_annotator import VLMAnnotator
    from sam_segmenter import RoomSegmenter
    from exporters import LabelStudioExporter, ReviewPrioritizer
    from automation import (
        LabelNormalizer, QualityChecker, RegionExtractor,
        ImageResizer, SemanticRoomValidator, TaxonomyNormalizer,
        filter_by_confidence, validate_for_sft, prepare_sft_annotation
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
    def vlm_annotator(self) -> VLMAnnotator:
        if self._vlm_annotator is None:
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
            input_dir: Directory containing PDF files.
            output_dir: Output directory for images and annotations.
            skip_existing: Skip images that already have annotations.

        Returns:
            Dictionary with pipeline statistics.

        Raises:
            PipelineError: If a critical step fails.
        """
        input_dir = Path(input_dir)
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

        # Step 1: Extract images from PDFs
        self._print_step("STEP 1: Extracting images from PDFs")

        try:
            extraction_results = self.pdf_extractor.batch_extract(
                input_dir, images_dir
            )
            stats["pdfs_processed"] = len(extraction_results)
            stats["images_extracted"] = sum(
                len(paths) for paths in extraction_results.values()
            )
            logger.info(
                f"Extracted {stats['images_extracted']} images "
                f"from {stats['pdfs_processed']} PDFs"
            )
        except Exception as e:
            raise PipelineError(f"PDF extraction failed: {e}") from e

        # Step 2: OCR text extraction
        self._print_step("STEP 2: OCR text extraction")

        ocr_results: Dict[str, List[RoomCandidate]] = {}
        for img_path in sorted(images_dir.glob("*.png")):
            try:
                rooms = self.ocr_extractor.extract_and_find_rooms(img_path)
                ocr_results[img_path.name] = rooms
                logger.info(f"  {img_path.name}: {len(rooms)} room labels found")
            except Exception as e:
                logger.error(f"  {img_path.name}: OCR failed - {e}")
                ocr_results[img_path.name] = []

        # Step 3: VLM annotation (optional)
        if self.config.use_vlm:
            self._print_step("STEP 3: VLM zero-shot annotation")

            for img_path in sorted(images_dir.glob("*.png")):
                ann_path = annotations_dir / f"{img_path.stem}.json"

                if skip_existing and ann_path.exists():
                    logger.info(f"  Skipping {img_path.name} (annotation exists)")
                    continue

                try:
                    # Step 3a: Resize image to fit Claude API 5MB limit
                    try:
                        ImageResizer.resize_in_place(img_path, max_kb=4500)
                    except Exception as resize_error:
                        logger.warning(f"  {img_path.name}: Image resize failed - {resize_error}")

                    result = self.vlm_annotator.annotate(img_path)
                    stats["images_annotated"] += 1
                    stats["rooms_detected"] += len(result.rooms)

                    # Save annotation
                    self._save_annotation(result, ann_path, ocr_results.get(img_path.name, []))

                    logger.info(
                        f"  {img_path.name}: {len(result.rooms)} rooms, "
                        f"{len(result.panels)} panels (VLM)"
                    )
                except Exception as e:
                    logger.error(f"  {img_path.name}: VLM annotation failed - {e}")

                    # Fallback: Try OCR if VLM fails
                    ocr_rooms = ocr_results.get(img_path.name, [])
                    if ocr_rooms:
                        try:
                            self._save_ocr_annotation(img_path, ocr_rooms, annotations_dir)
                            stats["images_annotated"] += 1
                            stats["rooms_detected"] += len(ocr_rooms)
                            logger.info(
                                f"  {img_path.name}: Fallback to OCR - "
                                f"saved {len(ocr_rooms)} rooms"
                            )
                        except Exception as fallback_error:
                            logger.error(
                                f"  {img_path.name}: OCR fallback also failed - {fallback_error}"
                            )
        else:
            # Use OCR results as primary annotations
            self._print_step("STEP 3: Generating annotations from OCR")

            for img_path in sorted(images_dir.glob("*.png")):
                rooms = ocr_results.get(img_path.name, [])
                if rooms:
                    try:
                        self._save_ocr_annotation(img_path, rooms, annotations_dir)
                        stats["images_annotated"] += 1
                        stats["rooms_detected"] += len(rooms)
                        logger.info(
                            f"  {img_path.name}: Saved {len(rooms)} rooms from OCR"
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

                # Step 4b: Normalize room type labels
                rooms = annotation.get("rooms", [])
                for room in rooms:
                    if "category" in room:
                        original = room["category"]
                        normalized = self.label_normalizer.normalize(original)
                        if original != normalized:
                            room["category"] = normalized
                            logger.debug(
                                f"    {ann_path.name}: "
                                f"Normalized '{original}' → '{normalized}'"
                            )
                quality_summary["annotations_normalized"] += 1

                # Step 4c: Check annotation quality
                issues = self.quality_checker.check_annotation(annotation)
                if issues:
                    quality_summary["annotations_with_issues"] += 1
                    quality_summary["total_issues"] += len(issues)

                    # Save issue report
                    report = self.quality_checker.report(issues)
                    report_path = reports_dir / f"{ann_path.stem}_issues.txt"
                    with open(report_path, "w") as f:
                        f.write(report)

                    logger.warning(
                        f"    {ann_path.name}: Found {len(issues)} quality issues"
                    )

                # Step 4d: Extract room regions for training
                img_path = images_dir / f"{ann_path.stem}.png"
                if img_path.exists() and rooms:
                    extracted = self.region_extractor.extract_regions(
                        str(img_path), annotation, str(regions_dir), prefix="room"
                    )
                    quality_summary["regions_extracted"] += len(extracted)
                    logger.info(
                        f"    {ann_path.name}: "
                        f"Extracted {len(extracted)} room regions"
                    )

                # Step 4e: Save processed annotation
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

            for img_path in sorted(images_dir.glob("*.png")):
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

        review_items = self.prioritizer.prioritize(annotations_dir)
        stats["flagged_for_review"] = len(review_items)

        # Save review list
        self.prioritizer.save_review_list(
            review_items, output_dir / "needs_review.json"
        )
        self.prioritizer.print_summary(review_items)

        # Step 7: Export to Label Studio
        self._print_step("STEP 7: Exporting to Label Studio")

        num_exported = self.exporter.export(
            annotations_dir, images_dir, output_dir / "label_studio_import.json"
        )

        # Save config for reproducibility
        config_path = output_dir / "pipeline_config.json"
        self._save_config(config_path)

        # Print summary
        self._print_summary(stats, output_dir)

        return stats

    def _save_annotation(
        self, result, output_path: Path, ocr_rooms: List[RoomCandidate]
    ) -> None:
        """Save VLM annotation result, merging with OCR data."""
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
            "ocr_rooms": [
                {
                    "room_number": r.room_number,
                    "room_name": r.room_name,
                    "bbox": list(r.bbox),
                    "confidence": r.confidence,
                }
                for r in ocr_rooms
            ],
        }

        if result.error:
            data["vlm_error"] = result.error

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def _save_ocr_annotation(
        self, img_path: Path, rooms: List[RoomCandidate], annotations_dir: Path
    ) -> None:
        """Save OCR-only annotation."""
        from PIL import Image

        with Image.open(img_path) as img:
            width, height = img.size

        data = {
            "image_file": img_path.name,
            "image_size": {"width": width, "height": height},
            "rooms": [
                {
                    "room_number": r.room_number,
                    "room_name": r.room_name,
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

        output_path = annotations_dir / f"{img_path.stem}.json"
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def _save_config(self, config_path: Path) -> None:
        """Save pipeline configuration for reproducibility."""
        # Convert dataclass config to dict (simplified)
        config_dict = {
            "pdf": {"dpi": self.config.pdf.dpi},
            "ocr": {
                "confidence_threshold": self.config.ocr.confidence_threshold,
                "preprocess": self.config.ocr.preprocess,
            },
            "vlm": {
                "model": self.config.vlm.model,
                "max_tokens": self.config.vlm.max_tokens,
            },
            "sam": {
                "model_type": self.config.sam.model_type,
            },
            "pipeline": {
                "use_vlm": self.config.use_vlm,
                "use_sam": self.config.use_sam,
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
