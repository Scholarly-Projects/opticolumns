#!/usr/bin/env python3
"""
Opticolumn - Surya Edition
OCR pipeline using Surya for layout/reading order and TrOCR for text recognition
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

# Surya imports - use DetectionPredictor for text line detection (like Kraken BLLA)
from surya.detection import DetectionPredictor
from surya.settings import settings

# ---------------- Configuration ----------------
INPUT_DIR  = "A"
OUTPUT_DIR = "B"
MODELS_DIR = "mlmodels"
POPPLER_PATH = None
DPI = 200
TROCR_MODELS = {
    "handwritten":       "microsoft/trocr-base-handwritten",
    "printed":           "microsoft/trocr-base-printed",
    "large_handwritten": "microsoft/trocr-large-handwritten",
    "large_printed":     "microsoft/trocr-large-printed",
}
TROCR_MODEL_NAME               = TROCR_MODELS["large_handwritten"]
ENABLE_PREPROCESSING           = True   # affects OCR copy only, not stored images
CONFIDENCE_THRESHOLD           = 0.25
SINGLE_CHAR_CONFIDENCE_THRESHOLD = 0.5
MIN_SEGMENT_HEIGHT             = 10
FONT_NAME  = "helv"
FONT_PATH  = "fonts/FreeSans.ttf"
SRGB_ICC_PATH = "srgb.icc"
DEBUG_OCR_LAYER         = False
DEBUG_TEXT_POSITIONS    = False
DEBUG_SAVE_INTERMEDIATE = False

# ---------------- Logging Setup ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ── Date helpers ──────────────────────────────────────────────────────────────
def get_pdf_date_string(dt=None):
    if dt is None:
        dt = datetime.datetime.now()
    return dt.strftime("D:%Y%m%d%H%M%S")

def get_xmp_date_string(dt=None):
    if dt is None:
        dt = datetime.datetime.now()
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


# ---------------- Font and ICC Profile Setup ----------------
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
            logger.info("Downloading sRGB ICC profile...")
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


# ---------------- XMP Metadata ----------------
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


# ---------------- Model Loading ----------------
def load_models():
    try:
        if not setup_pdfa_resources():
            logger.warning("PDF/A resources setup incomplete.")
        
        logger.info("Loading Surya detection predictor (text line detection)...")
        # DetectionPredictor detects individual text lines (like Kraken BLLA)
        detection_predictor = DetectionPredictor()
        
        logger.info(f"Loading TrOCR model: {TROCR_MODEL_NAME}")
        processor   = TrOCRProcessor.from_pretrained(TROCR_MODEL_NAME)
        trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_NAME)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trocr_model.to(device)
        logger.info(f"Using device: {device}")
        return detection_predictor, processor, trocr_model
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        import traceback
        traceback.print_exc()
        raise

try:
    detection_predictor, processor, trocr_model = load_models()
except Exception as e:
    logger.error("Model loading failed. Exiting.")
    sys.exit(1)


# ---------------- Image Preprocessing (OCR copy only) ----------------
def preprocess_for_ocr(pil_image: Image.Image) -> Image.Image:
    """
    Return a preprocessed COPY of pil_image suitable for TrOCR.
    The original pil_image is never modified and is NOT stored in the output PDF.
    """
    if not ENABLE_PREPROCESSING:
        return pil_image.copy()
    try:
        gray = pil_image.convert("L")
        gray = ImageOps.autocontrast(gray, cutoff=2)
        processed = gray.convert("RGB")
        processed = processed.filter(ImageFilter.SHARPEN)
        return processed
    except Exception as e:
        logger.error(f"Error preprocessing image for OCR: {e}")
        return pil_image.copy()


# ── Render a PDF page to a PIL image (original quality, used for the PDF) ──
def page_to_pil(page: fitz.Page, dpi: int = DPI) -> Image.Image:
    """Render *page* at *dpi* and return an RGB PIL Image."""
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


# ---------------- Text Recognition with TrOCR ----------------
def recognize_text_with_trocr(image: Image.Image, processor, model) -> tuple[str, float]:
    try:
        pixel_values = processor(image, return_tensors="pt").pixel_values
        device = next(model.parameters()).device
        pixel_values = pixel_values.to(device)
        with torch.no_grad():
            out = model.generate(
                pixel_values, output_scores=True, return_dict_in_generate=True
            )
            generated_text = processor.batch_decode(
                out.sequences, skip_special_tokens=True
            )[0]
            if out.scores:
                probs     = [torch.softmax(s, dim=-1) for s in out.scores]
                max_probs = [torch.max(p).item() for p in probs]
                confidence = sum(max_probs) / len(max_probs)
            else:
                confidence = 0.0
        return generated_text.strip(), confidence
    except Exception as e:
        logger.error(f"Error recognising text with TrOCR: {e}")
        return "", 0.0


# ---------------- Noise Detection ----------------
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


# ---------------- Surya Text Line Detection ----------------
def get_surya_text_lines(image: Image.Image, detection_predictor: DetectionPredictor) -> List[Dict[str, Any]]:
    """
    Get text line bounding boxes from Surya DetectionPredictor.
    Returns list of dicts with bbox in pixel coordinates.
    
    This is the Surya equivalent of Kraken's blla.segment() - detects
    individual text lines, not document-level layout regions.
    """
    try:
        # Surya detection predictor - pass image in a list for batch processing
        detection_results = detection_predictor([image])
        
        if not detection_results or len(detection_results) == 0:
            return []
        
        page_result = detection_results[0]
        lines = []
        
        # Access bboxes from detection result
        # Each box represents a text line
        for i, box in enumerate(page_result.bboxes):
            # Surya bbox format: [x1, y1, x2, y2] or polygon
            if hasattr(box, 'bbox'):
                bbox = box.bbox  # [x1, y1, x2, y2]
            elif hasattr(box, 'polygon') and len(box.polygon) >= 4:
                # Convert polygon to bbox
                poly = box.polygon
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
            else:
                continue
            
            # Get confidence score if available
            confidence = getattr(box, 'confidence', 1.0)
            
            lines.append({
                'bbox': bbox,
                'confidence': confidence,
                'position': i,  # Detection order (top-to-bottom, left-to-right)
            })
        
        # Sort by position (y-first, then x) for reading order
        # Surya detection already returns boxes in reasonable order
        lines.sort(key=lambda x: (x['bbox'][1] // 10 * 10, x['bbox'][0]))
        
        return lines
    except Exception as e:
        logger.error(f"Error getting Surya text lines: {e}")
        import traceback
        traceback.print_exc()
        return []


def order_lines_surya(text_lines: List[Dict[str, Any]], pil_image: Image.Image) -> List[Dict[str, Any]]:
    """
    Return text lines in reading order.
    Surya detection already returns lines in reasonable order, but we
    apply a gentle sort to ensure proper reading order (top-to-bottom,
    left-to-right with tolerance for line height variations).
    """
    if not text_lines:
        return []
    
    # Sort by y-coordinate (with tolerance for slight variations), then x
    # This handles multi-column layouts better than pure y-sort
    tolerance = 20  # pixels - lines within this y-range are considered same "row"
    
    def sort_key(line):
        y = line['bbox'][1]
        x = line['bbox'][0]
        return (y // tolerance * tolerance, x)
    
    logger.debug(f"Using Surya reading order for {len(text_lines)} lines.")
    return sorted(text_lines, key=sort_key)


# ---------------- OCR Text Element Extraction ----------------
def create_ocr_text_elements(
    pil_images: List[Image.Image],
    filename: str,
) -> List[List[dict]]:
    """
    Run Surya text line detection + TrOCR on each PIL image.
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
            ocr_image  = preprocess_for_ocr(pil_image)   # OCR copy — preprocessed
            
            # Get text line boxes from Surya (like Kraken blla.segment)
            text_lines = get_surya_text_lines(ocr_image, detection_predictor)
            logger.info(f"Found {len(text_lines)} text lines on page {page_num}")

            if not text_lines:
                logger.warning("No text lines detected. Saving debug image...")
                debug_dir = Path("debug_images")
                debug_dir.mkdir(exist_ok=True)
                ocr_image.save(debug_dir / f"{filename}_page{page_num}_preprocessed.png")

            sorted_lines = order_lines_surya(text_lines, ocr_image)
            logger.info(f"Sorted {len(sorted_lines)} lines into reading order.")

            for i, line in enumerate(sorted_lines):
                try:
                    bbox = line['bbox']
                    x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]

                    sh, sw = y1 - y0, x1 - x0
                    if sh < 5 or sw < 5:
                        filtered += 1
                        continue

                    # Crop from the OCR copy (same pixel space)
                    line_img = ocr_image.crop((x0, y0, x1, y1))
                    text, confidence = recognize_text_with_trocr(
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

            logger.info(f"Page {page_num}: {len(page_elements)} elements, {filtered} filtered.")
            total_elements += len(page_elements)

        except Exception as e:
            logger.error(f"OCR failed for page {page_num}: {e}")
            import traceback
            traceback.print_exc()

        all_pages.append(page_elements)

    logger.info(
        f"OCR extraction complete: {total_elements} total text elements "
        f"across {len(pil_images)} pages"
    )
    return all_pages


# ---------------- PDF/A Compliance ----------------
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

            icc_data = srgb_path.read_bytes()
            icc_stream = pdf.make_stream(icc_data)
            icc_stream.stream_dict["/N"]         = pikepdf.Integer(3)
            icc_stream.stream_dict["/Alternate"] = pikepdf.Name("/DeviceRGB")

            output_intent = pikepdf.Dictionary({
                "/Type":                     pikepdf.Name("/OutputIntent"),
                "/S":                        pikepdf.Name("/GTS_PDFA1"),
                "/Info":                     pikepdf.String("sRGB IEC61966-2.1"),
                "/OutputConditionIdentifier": pikepdf.String("sRGB"),
                "/DestOutputProfile":         pdf.make_indirect(icc_stream),
            })
            pdf.Root["/OutputIntents"].append(pdf.make_indirect(output_intent))
            pdf.save(pdf_path)
        logger.info("PDF/A OutputIntent embedded successfully.")
    except Exception as e:
        logger.error(f"Failed to set up PDF/A compliance: {e}")


# ---------------- PDF Processing (OCR) ----------------
def process_single_pdf_ocr(input_path: str, output_path: str) -> bool:
    """
    Add an invisible OCR text layer to *input_path* and write the result to
    *output_path*.
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

                logger.info(f"Page {page_num+1}: inserted {inserted}/{len(elements)} text elements")

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
                    logger.info(f"SUCCESS: Final PDF contains {total_chars} characters of searchable text")
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


# ---------------- Compression (Size Targeting) ----------------
def compress_to_target_size(input_pdf: Path, output_pdf: Path, original_size: int) -> Path:
    """
    Try to keep the output within 15% of the original size using
    PDF-native deflate compression only.
    """
    max_target = int(original_size * 1.15)
    logger.info(
        f"Targeting maximum size: {max_target // 1024} KB "
        f"(15% increase from original {original_size // 1024} KB)"
    )

    current_size = input_pdf.stat().st_size
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
            logger.info(f"Compression option {i+1}: {compressed_size // 1024} KB ({pct:+.1f}% from original)")

            if compressed_size <= max_target:
                try:
                    with fitz.open(str(temp_out)) as chk:
                        total_chars = sum(len(pg.get_text().strip()) for pg in chk)
                except Exception:
                    total_chars = -1

                if total_chars > 0:
                    shutil.move(str(temp_out), str(output_pdf))
                    logger.info(f"Compression option {i+1} accepted; OCR preserved ({total_chars} chars).")
                    return output_pdf
                else:
                    logger.error("OCR lost after compression attempt — falling back to uncompressed OCR file.")
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


# ---------------- Main ----------------
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
    logger.info("Text Line Detection: Surya DetectionPredictor (transformer model)")
    logger.info("Target: Final size ≤ original + 15% (OCR text layer only; images untouched)")

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
            final_size   = result_path.stat().st_size
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