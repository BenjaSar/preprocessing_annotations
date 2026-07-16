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

    # PDF user-space is defined at 72 points per inch.
    _PDF_POINTS_PER_INCH = 72.0

    def __init__(self, config: Optional[PDFConfig] = None):
        self.config = config or PDFConfig()

    def _page_scale(self, page: "fitz.Page") -> float:
        """Rasterization scale: the configured flat dpi, capped so the longest
        edge never exceeds `config.min_longest_edge_px`.

        This reproduces the established baseline output sizes — large sheets are
        capped to the cap value, physically small sheets keep their dpi-driven
        size (they are NOT up-scaled). Up-scaling small sheets was tried to
        recover otherwise-unreadable small-format pages, but on a CPU-only OCR
        host it drove those pages' detection memory past the RAM ceiling; the cap
        keeps memory bounded and matches the known-good baseline.
        """
        base_scale = self.config.dpi / self._PDF_POINTS_PER_INCH
        longest_edge_pts = max(page.rect.width, page.rect.height)
        if longest_edge_pts <= 0:
            return base_scale
        cap_scale = self.config.min_longest_edge_px / longest_edge_pts
        return min(base_scale, cap_scale)

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

            for page_num in pages:
                try:
                    page = doc[page_num]
                    scale = self._page_scale(page)
                    mat = fitz.Matrix(scale, scale)
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
    #
    # IMPORTANT: Only include patterns that are EXCLUSIVELY found on
    # notes/legend/schedule pages.  Standard architectural title block
    # boilerplate (DRAWING TITLE, PROJECT NO., SEAL & SIGNATURE,
    # ARCHITECT OF RECORD, NYC DOB NUMBER, WORKING DWG, etc.) appears
    # on EVERY page of a drawing set and must NOT be included here.
    # Including them caused false-positive classification of valid floor
    # plans as "notes_page", silently removing them from the pipeline.
    NOTES_PATTERNS = re.compile(
        r"(GENERAL\s+NOTES?|SYMBOL\s+LIST|LEGEND|ABBREVIATIONS?"
        r"|ELECTRICAL\s+SCHEDULE|PANEL\s+SCHEDULE|FIXTURE\s+SCHEDULE"
        r"|DOOR\s+SCHEDULE|WINDOW\s+SCHEDULE|FINISH\s+SCHEDULE"
        r"|SHEET\s+INDEX|DRAWING\s+INDEX|TITLE\s+SHEET"
        r"|SCOPE\s+OF\s+WORK"
        r"|FDNY\s+REQUIREMENTS?|ENERGY\s+CODE\s+COMPLIANCE"
        r"|BEFORE\s+COMMENCING\s+WORK)",
        re.IGNORECASE,
    )

    # Title block boilerplate — present on every page of a drawing set.
    # These are NOT classification signals on their own.  They are used
    # only as secondary evidence alongside the text-area heuristic to
    # strengthen a "notes_page" classification that is already indicated
    # by the text density ratio.
    _TITLE_BLOCK_PATTERNS = re.compile(
        r"(NYC\s+DOB\s+NUMBER|PLACE\s+STICKER\s+HERE"
        r"|DRAWING\s+TITLE|PROJECT\s+NO\.?"
        r"|REVISIONS?\s+DATE\s+DESCRIPTION"
        r"|SEAL\s+&\s+SIGNATURE"
        r"|THESE\s+PLANS?\s+ARE\s+THE\s+SOLE\s+PROPERTY"
        r"|WORKING\s+DWG|ARCHITECT\s+OF\s+RECORD"
        r"|REFER\s+TO\s+[A-Z]-\d+\s+SERIES"
        r"|FOR\s+GENERAL\s+NOTES,?\s+SYMBOL"
        r"|SPECIFICATIONS?)",
        re.IGNORECASE,
    )

    # Patterns that strongly indicate this IS a floor plan page.
    # Broadened to handle common drawing-title variants that PDF text
    # extraction may split across blocks (e.g. "ELECTRICAL" in one block,
    # "PLAN" in another).  Also covers "LAYOUT" (used by AVI-ON sets)
    # and MEP drawing-number prefixes.
    FLOOR_PLAN_PATTERNS = re.compile(
        r"(FLOOR\s+PLAN|PLAN\s+VIEW|REFLECTED\s+CEILING|RCP"
        r"|ELECTRICAL\s+(PLAN|LAYOUT)|MECHANICAL\s+(PLAN|LAYOUT)"
        r"|PLUMBING\s+(PLAN|LAYOUT)|LIGHTING\s+(PLAN|LAYOUT)"
        r"|POWER\s+(PLAN|LAYOUT)|LIFE\s+SAFETY\s+(PLAN|LAYOUT)"
        # Common drawing titles without "PLAN" suffix
        r"|\d+\w*\s+FLOOR"               # "29TH FLOOR", "1ST FLOOR", etc.
        r"|[EMP]-\d{3}"                   # MEP drawing numbers: E-229, M-101, P-300
        r"|AVI-ON\s+LAYOUT"              # AVI-ON set naming convention
        # Room label clusters (≥2 present → almost certainly a floor plan)
        r"|CONFERENCE\s+\d{3,4}|OFFICE\s+\d{3,4}|RECEPTION\s+\d{3,4}"
        r"|OPEN\s+AREA\s+\d{3,4}|PANTRY\s+\d{3,4})",
        re.IGNORECASE,
    )

    import re  # make re available at class level for patterns above

    def classify_page(self, page) -> tuple:
        """
        Classify a PyMuPDF page object.

        Classification priority:
        1. FLOOR_PLAN_PATTERNS → "floor_plan" (high confidence, immediate)
        2. NOTES_PATTERNS → "notes_page" (exclusive notes/schedule keywords)
        3. Text-area ratio > threshold AND title block patterns present
           → "notes_page" (secondary evidence, never standalone)
        4. Text-area ratio > elevated threshold (no title block match needed)
           → "notes_page" (very high text density alone is sufficient)
        5. Default → "floor_plan" (safe fallback)

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

        # Check notes/legend/schedule indicators (exclusive keywords only)
        match = self.NOTES_PATTERNS.search(text_upper)
        if match:
            return "notes_page", f"notes/schedule keyword: '{match.group()}'"

        # Text-area heuristic: dense text pages are documentation.
        # Title block boilerplate is used as secondary evidence to lower
        # the confidence threshold — pages with high text density AND
        # title block patterns (but no floor plan keywords) are very
        # likely documentation pages.
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

                has_title_block = bool(self._TITLE_BLOCK_PATTERNS.search(text_upper))

                # Lower threshold when title block boilerplate is the ONLY
                # text signal (no floor plan keywords, no notes keywords).
                # This catches pure-notes pages that only have title block
                # text plus dense paragraph content.
                if has_title_block and ratio > self.TEXT_AREA_THRESHOLD:
                    return "notes_page", (
                        f"text-area ratio {ratio:.2f} > {self.TEXT_AREA_THRESHOLD} "
                        f"with title block boilerplate (secondary evidence)"
                    )

                # Very high text density alone is sufficient — even without
                # any keyword match, a page that is >20% text is almost
                # certainly documentation.
                ELEVATED_THRESHOLD = 0.20
                if ratio > ELEVATED_THRESHOLD:
                    return "notes_page", f"text-area ratio {ratio:.2f} > {ELEVATED_THRESHOLD}"
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
