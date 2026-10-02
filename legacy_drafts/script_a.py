#!/usr/bin/env python3
"""
Opticolumn - Surya Edition
OCR pipeline using Surya for layout detection, reading order, and TrOCR for text recognition.

Pipeline:
  LayoutPredictor  → column/region boundaries (handles narrow gutters automatically)
  DetectionPredictor → individual text line bboxes
  OrderPredictor   → reading order informed by layout regions
  TrOCR            → text recognition per line
"""

import sys
import os
import tempfile
from pathlib import Path
import fitz  # PyMuPDF
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import logging
from typing import List, Tuple, Dict, Any
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import re
import platform
import datetime
import shutil
import pikepdf

# Surya imports
# Note: this version of Surya does not ship surya.ordering / OrderPredictor.
# Reading order is derived from LayoutPredictor column regions instead —
# see get_ordered_text_lines() for the full explanation.
from surya.foundation import FoundationPredictor
from surya.detection import DetectionPredictor
from surya.layout import LayoutPredictor
from surya.settings import settings

# ─────────────────────────── Configuration ───────────────────────────────────
INPUT_DIR  = "A"
OUTPUT_DIR = "B"
MODELS_DIR = "mlmodels"
POPPLER_PATH = None

# Raised from 200 → 300 so that ~6pt body text on the 1916 paper
# resolves to ~25 px line height, giving TrOCR enough pixels to work with.
# Modern/wider-column papers are unaffected by the higher DPI.
DPI = 300

TROCR_MODELS = {
    "handwritten":       "microsoft/trocr-base-handwritten",
    "printed":           "microsoft/trocr-base-printed",
    "large_handwritten": "microsoft/trocr-large-handwritten",
    "large_printed":     "microsoft/trocr-large-printed",
}
TROCR_MODEL_NAME = TROCR_MODELS["large_handwritten"]

ENABLE_PREPROCESSING = True   # affects OCR copy only, not stored images

# Slightly more lenient than before to accommodate aged / degraded newsprint ink.
CONFIDENCE_THRESHOLD             = 0.22   # was 0.25
SINGLE_CHAR_CONFIDENCE_THRESHOLD = 0.45   # was 0.50

# Scaled up from 10 → 12 to match the higher DPI
MIN_SEGMENT_HEIGHT = 12

FONT_NAME     = "helv"
FONT_PATH     = "fonts/FreeSans.ttf"
SRGB_ICC_PATH = "srgb.icc"

DEBUG_OCR_LAYER         = False
DEBUG_TEXT_POSITIONS    = False
DEBUG_SAVE_INTERMEDIATE = False

# ─────────────────────────── Logging ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ─────────────────────────── Date helpers ────────────────────────────────────
def get_pdf_date_string(dt=None):
    if dt is None:
        dt = datetime.datetime.now()
    return dt.strftime("D:%Y%m%d%H%M%S")


def get_xmp_date_string(dt=None):
    if dt is None:
        dt = datetime.datetime.now()
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


# ─────────────────────────── Font / ICC setup ────────────────────────────────
def setup_pdfa_resources():
    try:
        font_dir = Path("fonts")
        font_dir.mkdir(exist_ok=True)
        font_path = Path(FONT_PATH)
        if not font_path.exists():
            logger.info("Downloading FreeSans font for embedding...")
            import urllib.request
            urllib.request.urlretrieve(
                "https://github.com/opensourcedesign/fonts/raw/master/gnu-freefont_freesans/FreeSans.ttf",
                str(font_path),
            )
        srgb_path = Path(SRGB_ICC_PATH)
        if not srgb_path.exists():
            logger.info("Obtaining sRGB ICC profile...")
            try:
                import urllib.request
                if platform.system() == "Darwin":
                    system_profile = "/System/Library/ColorSync/Profiles/sRGB Profile.icc"
                    if Path(system_profile).exists():
                        shutil.copy2(system_profile, str(srgb_path))
                        return True
                elif platform.system() == "Windows":
                    system_profile = os.path.join(
                        os.environ.get("WINDIR", "C:\\Windows"),
                        "System32", "spool", "drivers", "color",
                        "sRGB Color Space Profile.icm",
                    )
                    if Path(system_profile).exists():
                        shutil.copy2(system_profile, str(srgb_path))
                        return True
                elif platform.system() == "Linux":
                    system_profile = "/usr/share/color/icc/sRGB.icc"
                    if Path(system_profile).exists():
                        shutil.copy2(system_profile, str(srgb_path))
                        return True
                urllib.request.urlretrieve("https://www.color.org/srgb.xalter", str(srgb_path))
                return True
            except Exception as e:
                logger.warning(f"Could not obtain sRGB ICC profile: {e}")
                return False
        return True
    except Exception as e:
        logger.error(f"Failed to setup PDF/A resources: {e}")
        return False


# ─────────────────────────── XMP Metadata ────────────────────────────────────
def create_xmp_metadata(title, author, subject, creator, producer, creation_date, modify_date):
    try:
        return f"""<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about="" xmlns:pdf="http://ns.adobe.com/pdf/1.3/">
      <pdf:Producer>{producer}</pdf:Producer>
    </rdf:Description>
    <rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">
      <dc:title><rdf:Alt><rdf:li xml:lang="x-default">{title}</rdf:li></rdf:Alt></dc:title>
      <dc:creator><rdf:Seq><rdf:li>{author}</rdf:li></rdf:Seq></dc:creator>
      <dc:description><rdf:Alt><rdf:li xml:lang="x-default">{subject}</rdf:li></rdf:Alt></dc:description>
    </rdf:Description>
    <rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/">
      <xmp:CreatorTool>{creator}</xmp:CreatorTool>
      <xmp:CreateDate>{creation_date}</xmp:CreateDate>
      <xmp:ModifyDate>{modify_date}</xmp:ModifyDate>
    </rdf:Description>
    <rdf:Description rdf:about="" xmlns:pdfaid="http://www.aiim.org/pdfa/ns/id/">
      <pdfaid:part>1</pdfaid:part>
      <pdfaid:conformance>B</pdfaid:conformance>
    </rdf:Description>
    <rdf:Description rdf:about="" xmlns:opt="http://github.com/Scholarly-Projects/opticolumn/">
      <opt:ToolName>Opticolumn</opt:ToolName>
      <opt:Version>2026-Surya</opt:Version>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""
    except Exception as e:
        logger.error(f"Failed to create XMP metadata: {e}")
        return None


# ─────────────────────────── Model Loading ───────────────────────────────────
def load_models():
    """
    Load the Surya pipeline plus TrOCR.

    Surya pipeline (this version):
      FoundationPredictor – shared base model weights (1.34 GB); instantiated
                            once and passed into LayoutPredictor to avoid
                            loading the weights twice
      DetectionPredictor  – individual text line bboxes (own weights)
      LayoutPredictor     – column / region boundaries, driven by
                            FoundationPredictor; used to derive reading order
                            since OrderPredictor is not in this Surya build
    """
    try:
        if not setup_pdfa_resources():
            logger.warning("PDF/A resources setup incomplete.")

        logger.info("Loading Surya DetectionPredictor (text line detection)...")
        detection_predictor = DetectionPredictor()

        logger.info("Loading Surya FoundationPredictor (shared base model)...")
        foundation_predictor = FoundationPredictor()

        logger.info("Loading Surya LayoutPredictor (column / region detection)...")
        layout_predictor = LayoutPredictor(foundation_predictor)

        logger.info(f"Loading TrOCR model: {TROCR_MODEL_NAME}")
        processor   = TrOCRProcessor.from_pretrained(TROCR_MODEL_NAME)
        trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_NAME)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trocr_model.to(device)
        logger.info(f"Using device: {device}")

        return detection_predictor, layout_predictor, processor, trocr_model

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        import traceback
        traceback.print_exc()
        raise


try:
    detection_predictor, layout_predictor, processor, trocr_model = load_models()
except Exception as e:
    logger.error("Model loading failed. Exiting.")
    sys.exit(1)


# ─────────────────────────── Image Preprocessing ─────────────────────────────
def preprocess_for_ocr(pil_image: Image.Image) -> Image.Image:
    """
    Return a preprocessed COPY of pil_image suitable for TrOCR.
    The original is never modified and is NOT stored in the output PDF.
    """
    if not ENABLE_PREPROCESSING:
        return pil_image.copy()
    try:
        gray      = pil_image.convert("L")
        gray      = ImageOps.autocontrast(gray, cutoff=2)
        processed = gray.convert("RGB")
        processed = processed.filter(ImageFilter.SHARPEN)
        return processed
    except Exception as e:
        logger.error(f"Error preprocessing image for OCR: {e}")
        return pil_image.copy()


def page_to_pil(page: fitz.Page, dpi: int = DPI) -> Image.Image:
    """Render *page* at *dpi* and return an RGB PIL Image."""
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


# ─────────────────────────── TrOCR Recognition ───────────────────────────────
def recognize_text_with_trocr(image: Image.Image, processor, model) -> tuple[str, float]:
    try:
        pixel_values = processor(image, return_tensors="pt").pixel_values
        device       = next(model.parameters()).device
        pixel_values = pixel_values.to(device)
        with torch.no_grad():
            out = model.generate(
                pixel_values, output_scores=True, return_dict_in_generate=True
            )
            generated_text = processor.batch_decode(
                out.sequences, skip_special_tokens=True
            )[0]
            if out.scores:
                probs      = [torch.softmax(s, dim=-1) for s in out.scores]
                max_probs  = [torch.max(p).item() for p in probs]
                confidence = sum(max_probs) / len(max_probs)
            else:
                confidence = 0.0
        return generated_text.strip(), confidence
    except Exception as e:
        logger.error(f"Error recognising text with TrOCR: {e}")
        return "", 0.0


# ─────────────────────────── Noise Detection ─────────────────────────────────
def is_likely_noise(text: str, confidence: float, seg_h: int, seg_w: int) -> bool:
    if not text:
        return True
    if seg_h < MIN_SEGMENT_HEIGHT or seg_w < 15:
        return True
    ar = seg_w / seg_h
    if ar < 0.1 or ar > 100:
        return True
    tc = text.strip()
    tl = len(tc)
    if tl == 1:
        return confidence < SINGLE_CHAR_CONFIDENCE_THRESHOLD
    if confidence < CONFIDENCE_THRESHOLD:
        return True
    if len(set(tc)) == 1 and tl > 2:
        return True
    noise_patterns = [r"^[oOlI\.\|]+$", r"^[0-9\.\,]+$", r"^[^a-zA-Z0-9\s]+$"]
    for pat in noise_patterns:
        if re.match(pat, tc) and confidence < SINGLE_CHAR_CONFIDENCE_THRESHOLD:
            return True
    if tl > 3 and not any(c.lower() in "aeiou" for c in tc) and confidence < 0.7:
        return True
    return False


# ─────────────────────── Unified Surya Pipeline ──────────────────────────────
def _bbox_from_box(box) -> List[float] | None:
    """Extract [x0, y0, x1, y1] from a Surya box object regardless of format."""
    if hasattr(box, "bbox"):
        return box.bbox
    if hasattr(box, "polygon") and len(box.polygon) >= 4:
        poly = box.polygon
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return [min(xs), min(ys), max(xs), max(ys)]
    return None


def _iou_x_overlap(line_bbox: List[float], region_bbox: List[float]) -> float:
    """
    Return the fraction of the line's horizontal span that overlaps the region.
    Used to assign a text line to the column it mostly falls inside.
    """
    lx0, lx1 = line_bbox[0], line_bbox[2]
    rx0, rx1 = region_bbox[0], region_bbox[2]
    overlap  = max(0.0, min(lx1, rx1) - max(lx0, rx0))
    line_w   = max(lx1 - lx0, 1)
    return overlap / line_w


def get_ordered_text_lines(
    image: Image.Image,
    detection_predictor: DetectionPredictor,
    layout_predictor: LayoutPredictor,
) -> List[Dict[str, Any]]:
    """
    Layout-region-aware reading order without OrderPredictor.

    Strategy
    ────────
    Because this Surya build does not ship surya.ordering, reading order is
    derived directly from LayoutPredictor's column/region bboxes:

    1. LayoutPredictor returns bboxes for each detected region (columns,
       headers, captions, figures, etc.).  These bboxes define where columns
       begin and end, so their left edges give us a reliable column index even
       when gutters are only a few pixels wide (e.g. the 1916 7-column paper).

    2. DetectionPredictor returns individual text line bboxes.

    3. Each text line is assigned to the layout region whose horizontal span
       it overlaps the most (x-overlap fraction).  Lines that don't overlap
       any region get their own synthetic single-line region so nothing is lost.

    4. Regions are sorted left-to-right by their x0 coordinate.  Within each
       region, lines are sorted top-to-bottom by their y0 coordinate.

    This produces correct column-by-column reading order for any number of
    columns without any hard-coded pixel tolerances.  The LayoutPredictor
    transformer handles gutter-width variation implicitly.

    Fallback
    ────────
    If LayoutPredictor returns no regions, the function falls back to a
    gentle row-bucket sort (tolerance = 15 px) which is still better than
    the old 20 px tolerance because the bucket size is smaller and the
    function never crashes.
    """
    try:
        # ── Step 1: Layout regions ────────────────────────────────────────────
        layout_results = layout_predictor([image])
        layout_page    = layout_results[0] if layout_results else None

        regions: List[List[float]] = []
        if layout_page is not None and hasattr(layout_page, "bboxes"):
            for box in layout_page.bboxes:
                bbox = _bbox_from_box(box)
                if bbox is not None:
                    regions.append(bbox)

        logger.debug(f"LayoutPredictor found {len(regions)} regions.")

        # ── Step 2: Text line detection ───────────────────────────────────────
        detection_results = detection_predictor([image])
        if not detection_results or not hasattr(detection_results[0], "bboxes"):
            logger.warning("DetectionPredictor returned no results.")
            return []

        det_page = detection_results[0]
        if not det_page.bboxes:
            logger.warning("DetectionPredictor found zero text lines.")
            return []

        raw_lines: List[Dict[str, Any]] = []
        for i, box in enumerate(det_page.bboxes):
            bbox = _bbox_from_box(box)
            if bbox is None:
                continue
            raw_lines.append({
                "bbox":       bbox,
                "confidence": getattr(box, "confidence", 1.0),
                "position":   i,
            })

        if not raw_lines:
            return []

        # ── Step 3: Assign lines to regions ──────────────────────────────────
        if regions:
            # Sort regions left-to-right so the region index IS the column index
            regions_sorted = sorted(regions, key=lambda r: r[0])

            # Build buckets: one list of lines per region
            buckets: List[List[Dict]] = [[] for _ in regions_sorted]
            unassigned: List[Dict]    = []

            for line in raw_lines:
                best_region_idx = -1
                best_overlap    = 0.05   # minimum 5% x-overlap to count
                for ri, region in enumerate(regions_sorted):
                    overlap = _iou_x_overlap(line["bbox"], region)
                    if overlap > best_overlap:
                        best_overlap    = overlap
                        best_region_idx = ri
                if best_region_idx >= 0:
                    buckets[best_region_idx].append(line)
                else:
                    unassigned.append(line)

            # Within each bucket sort top-to-bottom
            ordered: List[Dict] = []
            for bucket in buckets:
                ordered.extend(sorted(bucket, key=lambda l: l["bbox"][1]))

            # Unassigned lines go last, sorted by position
            ordered.extend(sorted(unassigned, key=lambda l: l["bbox"][1]))

            logger.info(
                f"Layout-region sort: {len(ordered)} lines across "
                f"{len(regions_sorted)} regions "
                f"({len(unassigned)} unassigned)."
            )
            return ordered

        else:
            # ── Fallback: no layout regions — row-bucket sort ─────────────────
            logger.warning(
                "No layout regions found; using row-bucket sort (tolerance=15 px)."
            )
            tolerance = 15
            raw_lines.sort(key=lambda l: (l["bbox"][1] // tolerance * tolerance, l["bbox"][0]))
            return raw_lines

    except Exception as e:
        logger.error(f"Error in get_ordered_text_lines: {e}")
        import traceback
        traceback.print_exc()
        return []


# ─────────────────────────── OCR Extraction ──────────────────────────────────
def create_ocr_text_elements(
    pil_images: List[Image.Image],
    filename: str,
) -> List[List[dict]]:
    """
    Run the Surya pipeline + TrOCR on each PIL image.
    Returns per-page lists of dicts with pixel-space coordinates.
    Keys: x0, y_baseline, font_size, text
    """
    font_path = Path(FONT_PATH)
    if not font_path.exists():
        raise FileNotFoundError(f"Required font {font_path} is missing.")

    all_pages: List[List[dict]] = []
    total_elements = 0

    for idx, pil_image in enumerate(pil_images):
        page_num = idx + 1
        logger.info(f"Processing page {page_num}/{len(pil_images)} of {filename}")
        page_elements: List[dict] = []
        filtered = 0

        try:
            ocr_image = preprocess_for_ocr(pil_image)   # OCR copy — preprocessed

            # Full Surya pipeline: layout regions → detection → column-aware order
            sorted_lines = get_ordered_text_lines(
                ocr_image,
                detection_predictor,
                layout_predictor,
            )
            logger.info(f"Found {len(sorted_lines)} ordered text lines on page {page_num}.")

            if not sorted_lines:
                logger.warning("No text lines detected. Saving debug image...")
                debug_dir = Path("debug_images")
                debug_dir.mkdir(exist_ok=True)
                ocr_image.save(debug_dir / f"{filename}_page{page_num}_preprocessed.png")

            for i, line in enumerate(sorted_lines):
                try:
                    bbox        = line["bbox"]
                    x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
                    sh, sw      = y1 - y0, x1 - x0

                    if sh < 5 or sw < 5:
                        filtered += 1
                        continue

                    # Crop from the OCR copy (same pixel space)
                    line_img            = ocr_image.crop((x0, y0, x1, y1))
                    text, confidence    = recognize_text_with_trocr(
                        line_img, processor, trocr_model
                    )

                    if is_likely_noise(text, confidence, sh, sw):
                        filtered += 1
                        continue

                    page_elements.append({
                        "x0":         x0,
                        "y_baseline": y1,
                        "font_size":  max(6, min(sh * 0.9, 72)),
                        "text":       text,
                    })

                except Exception as e:
                    logger.error(f"Error on text line {i + 1}: {e}")

            logger.info(
                f"Page {page_num}: {len(page_elements)} elements accepted, "
                f"{filtered} filtered."
            )
            total_elements += len(page_elements)

        except Exception as e:
            logger.error(f"OCR failed for page {page_num}: {e}")
            import traceback
            traceback.print_exc()

        all_pages.append(page_elements)

    logger.info(
        f"OCR extraction complete: {total_elements} total text elements "
        f"across {len(pil_images)} pages."
    )
    return all_pages


# ─────────────────────────── PDF/A Compliance ────────────────────────────────
def setup_pdfa_compliance(pdf_path: str):
    """Embed sRGB OutputIntent into an already-saved PDF using pikepdf."""
    try:
        srgb_path = Path(SRGB_ICC_PATH)
        if not srgb_path.exists():
            logger.error("sRGB ICC profile not found; skipping PDF/A OutputIntent.")
            return
        with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
            if "/OutputIntents" not in pdf.Root:
                pdf.Root["/OutputIntents"] = pikepdf.Array()

            icc_data   = srgb_path.read_bytes()
            icc_stream = pdf.make_stream(icc_data)
            icc_stream.stream_dict["/N"]         = pikepdf.Integer(3)
            icc_stream.stream_dict["/Alternate"] = pikepdf.Name("/DeviceRGB")

            output_intent = pikepdf.Dictionary({
                "/Type":                      pikepdf.Name("/OutputIntent"),
                "/S":                         pikepdf.Name("/GTS_PDFA1"),
                "/Info":                      pikepdf.String("sRGB IEC61966-2.1"),
                "/OutputConditionIdentifier": pikepdf.String("sRGB"),
                "/DestOutputProfile":         pdf.make_indirect(icc_stream),
            })
            pdf.Root["/OutputIntents"].append(pdf.make_indirect(output_intent))
            pdf.save(pdf_path)
        logger.info("PDF/A OutputIntent embedded successfully.")
    except Exception as e:
        logger.error(f"Failed to set up PDF/A compliance: {e}")


# ─────────────────────────── PDF OCR Processing ──────────────────────────────
def process_single_pdf_ocr(input_path: str, output_path: str) -> bool:
    """
    Add an invisible OCR text layer to *input_path* and write the result
    to *output_path*.
    """
    filename = os.path.basename(input_path)
    logger.info(f"Starting OCR for: {filename}")

    try:
        with fitz.open(input_path) as doc:
            logger.info(f"Rendering {len(doc)} pages at {DPI} DPI for OCR…")
            pil_images: List[Image.Image] = []
            for page in doc:
                pil_images.append(page_to_pil(page, dpi=DPI))

            ocr_pages = create_ocr_text_elements(pil_images, filename)

            now           = datetime.datetime.now()
            creation_date = get_pdf_date_string(now)
            doc.set_metadata({
                "title":        filename,
                "author":       "Opticolumn",
                "subject":      "OCR processed document",
                "creator":      "Opticolumn 2026-Surya",
                "producer":     "PyMuPDF",
                "creationDate": creation_date,
                "modDate":      creation_date,
            })
            xmp = create_xmp_metadata(
                title=filename,
                author="Opticolumn",
                subject="OCR processed document",
                creator="Opticolumn 2026-Surya",
                producer="PyMuPDF",
                creation_date=get_xmp_date_string(now),
                modify_date=get_xmp_date_string(now),
            )
            if xmp:
                doc.set_xml_metadata(xmp)

            page_count = min(len(doc), len(ocr_pages))
            logger.info(f"Inserting text into {page_count} pages…")

            for page_num in range(page_count):
                page     = doc[page_num]
                elements = ocr_pages[page_num]
                pil_img  = pil_images[page_num]

                existing_text = page.get_text().strip()
                if existing_text:
                    logger.info(
                        f"Page {page_num+1}: removing existing text layer "
                        f"({len(existing_text)} chars) before inserting OCR."
                    )
                    page.add_redact_annot(page.rect)
                    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

                img_w, img_h = pil_img.size
                page_w = page.rect.width
                page_h = page.rect.height
                sx     = page_w / img_w
                sy     = page_h / img_h

                logger.debug(
                    f"Page {page_num+1}: {img_w}×{img_h}px → "
                    f"{page_w:.1f}×{page_h:.1f}pt  (sx={sx:.4f}, sy={sy:.4f})"
                )

                inserted = 0
                for elem in elements:
                    try:
                        page.insert_text(
                            fitz.Point(elem["x0"] * sx, elem["y_baseline"] * sy),
                            elem["text"],
                            fontsize=max(4, elem["font_size"] * sy),
                            fontname=FONT_NAME,
                            render_mode=3,
                            color=(0, 0, 0),
                        )
                        inserted += 1
                    except Exception as e:
                        logger.error(f"Failed to insert text on page {page_num+1}: {e}")

                logger.info(
                    f"Page {page_num+1}: inserted {inserted}/{len(elements)} text elements."
                )

            doc.save(
                output_path,
                deflate=True,
                garbage=4,
                clean=True,
                deflate_images=False,
                encryption=fitz.PDF_ENCRYPT_KEEP,
            )
            logger.info(f"OCR-enhanced PDF saved: {output_path}")

        srgb_path = Path(SRGB_ICC_PATH)
        if srgb_path.exists():
            logger.info("Applying PDF/A OutputIntent…")
            setup_pdfa_compliance(output_path)
        else:
            logger.warning("sRGB ICC profile not found; PDF/A OutputIntent skipped.")

        logger.info("Verifying OCR layer in final output…")
        try:
            with fitz.open(output_path) as final_pdf:
                total_chars = 0
                for i, pg in enumerate(final_pdf):
                    chars = len(pg.get_text().strip())
                    total_chars += chars
                    logger.info(f"Final PDF page {i+1} extractable text length: {chars}")
                if total_chars > 0:
                    logger.info(
                        f"SUCCESS: Final PDF contains {total_chars} characters of searchable text."
                    )
                else:
                    logger.error("PROBLEM: Final PDF has no extractable text!")
        except Exception as e:
            logger.error(f"Failed to verify final output: {e}")

        return True

    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ─────────────────────────── Compression ─────────────────────────────────────
def compress_to_target_size(input_pdf: Path, output_pdf: Path, original_size: int) -> Path:
    """
    Try to keep the output within 15% of the original size using
    PDF-native deflate compression only.
    """
    max_target    = int(original_size * 1.15)
    current_size  = input_pdf.stat().st_size
    logger.info(
        f"Targeting maximum size: {max_target // 1024} KB "
        f"(15% increase from original {original_size // 1024} KB)"
    )
    logger.info(f"OCR file size before compression: {current_size // 1024} KB")

    if current_size <= max_target:
        shutil.copy2(input_pdf, output_pdf)
        logger.info("File already within target size. No additional compression needed.")
        return output_pdf

    compression_options = [
        {"deflate": True, "garbage": 4, "clean": True, "deflate_images": False},
        {"deflate": True, "garbage": 3, "clean": True, "deflate_images": False},
        {"deflate": True, "garbage": 2, "clean": True, "deflate_images": False},
    ]

    for i, opts in enumerate(compression_options):
        temp_out = output_pdf.with_suffix(f".temp_{i}.pdf")
        try:
            with fitz.open(str(input_pdf)) as doc:
                doc.save(str(temp_out), **opts, encryption=fitz.PDF_ENCRYPT_KEEP)

            compressed_size = temp_out.stat().st_size
            pct = (compressed_size - original_size) / original_size * 100
            logger.info(
                f"Compression option {i+1}: {compressed_size // 1024} KB "
                f"({pct:+.1f}% from original)"
            )

            if compressed_size <= max_target:
                try:
                    with fitz.open(str(temp_out)) as chk:
                        total_chars = sum(len(pg.get_text().strip()) for pg in chk)
                except Exception:
                    total_chars = -1

                if total_chars > 0:
                    shutil.move(str(temp_out), str(output_pdf))
                    logger.info(
                        f"Compression option {i+1} accepted; "
                        f"OCR preserved ({total_chars} chars)."
                    )
                    return output_pdf
                else:
                    logger.error(
                        "OCR lost after compression attempt — "
                        "falling back to uncompressed OCR file."
                    )
                    temp_out.unlink(missing_ok=True)
                    shutil.copy2(input_pdf, output_pdf)
                    return output_pdf
            else:
                temp_out.unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Compression option {i+1} failed: {e}")
            temp_out.unlink(missing_ok=True)

    logger.warning(
        "All deflate options exceeded the 15% size budget. "
        "Returning OCR file as-is (images untouched, quality preserved)."
    )
    shutil.copy2(input_pdf, output_pdf)
    return output_pdf


# ─────────────────────────── Main ────────────────────────────────────────────
def main():
    input_folder  = Path(INPUT_DIR)
    output_folder = Path(OUTPUT_DIR)

    if not input_folder.exists():
        logger.error(f"Input folder '{INPUT_DIR}' not found.")
        sys.exit(1)
    output_folder.mkdir(exist_ok=True)

    pdf_files = list(input_folder.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files in '{INPUT_DIR}'")
        sys.exit(1)

    logger.info(f"Processing {len(pdf_files)} files with TrOCR: {TROCR_MODEL_NAME}")
    logger.info("Layout detection:  Surya LayoutPredictor (column region boundaries)")
    logger.info("Line detection:    Surya DetectionPredictor")
    logger.info("Reading order:     Layout-region column sort (OrderPredictor not in this Surya build)")
    logger.info("Target:            Final size ≤ original + 15% (OCR text layer only; images untouched)")

    for pdf_path in pdf_files:
        original_size = pdf_path.stat().st_size
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing {pdf_path.name} | Original: {original_size // 1024} KB")

        ocr_temp_path = output_folder / f"{pdf_path.stem}_ocr_temp.pdf"

        if not process_single_pdf_ocr(str(pdf_path), str(ocr_temp_path)):
            logger.error(f"Skipping {pdf_path.name} due to OCR failure.")
            continue

        final_path  = output_folder / f"{pdf_path.stem}_final.pdf"
        result_path = compress_to_target_size(ocr_temp_path, final_path, original_size)

        if result_path.exists():
            final_size    = result_path.stat().st_size
            size_increase = (final_size - original_size) / original_size * 100
            logger.info(
                f"SUCCESS: {result_path.name} | {final_size // 1024} KB "
                f"({size_increase:+.1f}% increase from original)"
            )
        else:
            logger.error(f"Failed to generate final output for {pdf_path.name}")

        try:
            ocr_temp_path.unlink()
        except Exception as e:
            logger.warning(f"Could not delete temp file: {e}")

    logger.info(f"\nAll done! Output files in '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()