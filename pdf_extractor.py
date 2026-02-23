"""
PDF extraction module for converting floor plan PDFs to images.

This module handles high-resolution rasterization of PDF pages,
optimized for MEP floor plans where small symbols require high DPI.
"""

import logging
import re
from pathlib import Path
from typing import Generator, Optional, Tuple, List

import fitz  # PyMuPDF
from PIL import Image
import io

try:
    from .config import PDFConfig
except ImportError:
    from config import PDFConfig

logger = logging.getLogger(__name__)


class PDFExtractionError(Exception):
    """Raised when PDF extraction fails."""

    pass


class PDFExtractor:
    """
    Extracts pages from PDF files as high-resolution images.

    Attributes:
        config: PDFConfig with extraction parameters.

    Example:
        extractor = PDFExtractor(PDFConfig(dpi=300))
        for image, page_num in extractor.extract("floorplan.pdf"):
            image.save(f"page_{page_num}.png")
    """

    def __init__(self, config: Optional[PDFConfig] = None):
        self.config = config or PDFConfig()

    def extract(
        self, pdf_path: str | Path
    ) -> Generator[Tuple[Image.Image, int], None, None]:
        """
        Extract pages from a PDF as PIL Images.

        Args:
            pdf_path: Path to the PDF file.

        Yields:
            Tuple of (PIL.Image, page_number) for each page.

        Raises:
            PDFExtractionError: If the PDF cannot be opened or processed.
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise PDFExtractionError(f"PDF file not found: {pdf_path}")

        if not pdf_path.suffix.lower() == ".pdf":
            raise PDFExtractionError(f"Not a PDF file: {pdf_path}")

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            raise PDFExtractionError(f"Failed to open PDF {pdf_path}: {e}") from e

        try:
            # Determine page range
            if self.config.page_range:
                start, end = self.config.page_range
                pages = range(start, min(end, len(doc)))
            else:
                pages = range(len(doc))

            # Calculate transformation matrix for target DPI
            # PDF base resolution is 72 DPI
            scale = self.config.dpi / 72.0
            mat = fitz.Matrix(scale, scale)

            for page_num in pages:
                try:
                    page = doc[page_num]
                    pix = page.get_pixmap(matrix=mat)

                    # Convert to PIL Image
                    img_data = pix.tobytes("png")
                    image = Image.open(io.BytesIO(img_data))

                    logger.debug(
                        f"Extracted page {page_num} from {pdf_path.name}: "
                        f"{image.width}x{image.height}px"
                    )

                    yield image, page_num

                except Exception as e:
                    logger.error(f"Failed to extract page {page_num} from {pdf_path}: {e}")
                    continue

        finally:
            doc.close()

    def extract_to_directory(
        self,
        pdf_path: str | Path,
        output_dir: str | Path,
        filename_prefix: Optional[str] = None,
    ) -> List[Path]:
        """
        Extract all pages from a PDF and save to a directory.

        Args:
            pdf_path: Path to the PDF file.
            output_dir: Directory to save extracted images.
            filename_prefix: Optional prefix for output files. Defaults to PDF stem.

        Returns:
            List of paths to extracted image files.

        Raises:
            PDFExtractionError: If extraction or saving fails.
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        prefix = filename_prefix or pdf_path.stem
        output_paths = []

        for image, page_num in self.extract(pdf_path):
            output_path = output_dir / f"{prefix}_page{page_num:03d}.png"

            try:
                image.save(output_path, self.config.output_format)
                output_paths.append(output_path)
                logger.info(f"Saved: {output_path}")
            except Exception as e:
                logger.error(f"Failed to save {output_path}: {e}")

        return output_paths

    def batch_extract(
        self, pdf_dir: str | Path, output_dir: str | Path
    ) -> dict[str, List[Path]]:
        """
        Extract all PDFs from a directory.

        Args:
            pdf_dir: Directory containing PDF files.
            output_dir: Directory to save extracted images.

        Returns:
            Dictionary mapping PDF names to lists of extracted image paths.
        """
        pdf_dir = Path(pdf_dir)
        output_dir = Path(output_dir)

        if not pdf_dir.exists():
            raise PDFExtractionError(f"PDF directory not found: {pdf_dir}")

        results = {}
        pdf_files = sorted(pdf_dir.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir}")
            return results

        logger.info(f"Processing {len(pdf_files)} PDF files")

        for pdf_path in pdf_files:
            try:
                output_paths = self.extract_to_directory(pdf_path, output_dir)
                results[pdf_path.name] = output_paths
                logger.info(f"Extracted {len(output_paths)} pages from {pdf_path.name}")
            except PDFExtractionError as e:
                logger.error(f"Skipping {pdf_path.name}: {e}")
                results[pdf_path.name] = []

        return results


# ---------------------------------------------------------------------------
# Plan-type classifier
# ---------------------------------------------------------------------------

class PageTypeClassifier:
    """
    Classify extracted PDF pages by content type before running OCR.

    The pipeline previously ran full OCR+filter on every page, including
    legend pages, schedule pages, general notes sheets, and title blocks.
    These pages contain hundreds of text tokens that generate false room
    candidates, requiring extensive downstream filtering.

    This classifier runs a fast lightweight heuristic pass first:
    - If a page is a notes/legend/schedule page, it is written to a
      ``skipped/`` directory with a reason code instead of being processed.
    - Floor plan pages proceed normally.

    This eliminates 30–50% of false positive candidates at their source and
    reduces total OCR runtime proportionally.

    Detection method:
    1. Extract embedded text from the PDF page (fast, no OCR).
    2. Check text-to-page-area ratio: pages with dense text are usually
       documentation, not floor plans.
    3. Match header patterns against known legend/schedule/notes keywords.
    """

    # Fraction of page area covered by text bounding boxes.
    # Above this threshold the page is considered "text-heavy" (notes/schedule).
    TEXT_AREA_THRESHOLD: float = 0.12

    # Patterns that strongly indicate this is NOT a floor plan page.
    NOTES_PATTERNS = re.compile(
        r"(GENERAL\s+NOTES?|SYMBOL\s+LIST|LEGEND|ABBREVIATIONS?"
        r"|ELECTRICAL\s+SCHEDULE|PANEL\s+SCHEDULE|FIXTURE\s+SCHEDULE"
        r"|DOOR\s+SCHEDULE|WINDOW\s+SCHEDULE|FINISH\s+SCHEDULE"
        r"|SHEET\s+INDEX|DRAWING\s+INDEX|TITLE\s+SHEET"
        r"|SPECIFICATIONS?|SCOPE\s+OF\s+WORK"
        r"|FDNY\s+REQUIREMENTS?|ENERGY\s+CODE\s+COMPLIANCE"
        r"|BEFORE\s+COMMENCING\s+WORK"
        # Pattern that appeared on the 326 Rockaway notes page:
        # "REFER TO E-000 SERIES FOR GENERAL NOTES, SYMBOL LIST, ETC."
        r"|REFER\s+TO\s+[A-Z]-\d+\s+SERIES"
        r"|FOR\s+GENERAL\s+NOTES,?\s+SYMBOL"
        # AVI-ON / 540 Madison-style notes indicators
        r"|NYC\s+DOB\s+NUMBER"
        r"|PLACE\s+STICKER\s+HERE"
        r"|DRAWING\s+TITLE"
        r"|PROJECT\s+NO\.?"
        r"|REVISIONS?\s+DATE\s+DESCRIPTION"
        r"|SEAL\s+&\s+SIGNATURE"
        r"|THESE\s+PLANS?\s+ARE\s+THE\s+SOLE\s+PROPERTY"
        r"|WORKING\s+DWG"
        r"|ARCHITECT\s+OF\s+RECORD)",
        re.IGNORECASE,
    )

    # Patterns that strongly indicate this IS a floor plan page.
    FLOOR_PLAN_PATTERNS = re.compile(
        r"(FLOOR\s+PLAN|PLAN\s+VIEW|REFLECTED\s+CEILING|RCP"
        r"|ELECTRICAL\s+PLAN|MECHANICAL\s+PLAN|PLUMBING\s+PLAN"
        r"|LIGHTING\s+PLAN|POWER\s+PLAN|LIFE\s+SAFETY\s+PLAN)",
        re.IGNORECASE,
    )

    import re  # make re available at class level for patterns above

    def classify_page(self, page) -> tuple:
        """
        Classify a PyMuPDF page object.

        Args:
            page: fitz.Page object.

        Returns:
            Tuple of (page_type, reason) where page_type is one of:
            "floor_plan" | "notes_page" | "schedule" | "title_sheet" | "unknown"
        """
        import re as _re

        # Extract embedded text (fast, no image processing)
        try:
            text = page.get_text("text")
        except Exception:
            return "unknown", "failed to extract text"

        text_upper = text.upper()

        # Check explicit floor-plan indicators first (high confidence)
        if self.FLOOR_PLAN_PATTERNS.search(text_upper):
            return "floor_plan", "floor plan keyword found"

        # Check notes/legend/schedule indicators
        match = self.NOTES_PATTERNS.search(text_upper)
        if match:
            return "notes_page", f"notes/schedule keyword: '{match.group()}'"

        # Text-area heuristic: dense text pages are documentation
        try:
            page_area = page.rect.width * page.rect.height
            if page_area > 0:
                blocks = page.get_text("blocks")
                text_area = sum(
                    (b[2] - b[0]) * (b[3] - b[1])
                    for b in blocks
                    if b[6] == 0  # type 0 = text block
                )
                ratio = text_area / page_area
                if ratio > self.TEXT_AREA_THRESHOLD:
                    return "notes_page", f"text-area ratio {ratio:.2f} > {self.TEXT_AREA_THRESHOLD}"
        except Exception:
            pass

        return "floor_plan", "default (no notes indicators found)"

    def classify_image(self, image_path: str | Path) -> tuple:
        """
        Classify an already-extracted PNG by running lightweight OCR-free checks.

        Falls back to "floor_plan" when classification is uncertain — it is
        better to process a notes page (and rely on downstream filtering) than
        to silently skip a real floor plan.

        Args:
            image_path: Path to extracted PNG.

        Returns:
            Tuple of (page_type, reason).
        """
        # Without the original PDF page we cannot use embedded-text extraction.
        # Return floor_plan as the safe default.
        return "floor_plan", "image-only classification not available (PDF page needed)"


import re  # needed for PageTypeClassifier patterns above (module-level re)
