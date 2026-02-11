"""
PDF extraction module for converting floor plan PDFs to images.

This module handles high-resolution rasterization of PDF pages,
optimized for MEP floor plans where small symbols require high DPI.
"""

import logging
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
