#!/usr/bin/env python3
"""
Opticolumn  –  SURYA LAYOUT  /  TrOCR RECOGNITION  (HYBRID EDITION)
======================================================================
Historic Newspaper Page Segmentation, Reading Order & OCR

ARCHITECTURE
------------
Stage 1  Surya LayoutPredictor
         Detects semantic regions (Section-header, Text, Caption, Table,
         Picture, Page-header, etc.) with a native `position` field that
         encodes column-aware reading order.  Identical to the Surya-native
         edition.

Stage 2  Surya DetectionPredictor  +  TrOCR
         Within each layout region crop, DetectionPredictor segments the crop
         into individual text lines.  Each line image is then passed to the
         TrOCR VisionEncoderDecoder model for character recognition.

         Two-pass strategy for single-block regions (Section-header,
         Page-header, Caption, Footnote):
           Pass 1 — DetectionPredictor finds lines; TrOCR reads each one.
           Pass 2 — If Pass 1 returns nothing, the whole crop is fed to TrOCR
                    as a single image.  This handles full-width banner headlines
                    where the crop IS the line and DetectionPredictor finds no
                    internal boundaries.

Stage 3  Assemble & write
         Elements are ordered by Surya layout `position`, then vertical
         baseline within each region.  An invisible text layer is stamped into
         the output PDF for searchability.

WHY THIS COMBINATION
--------------------
* Surya LayoutPredictor handles newspaper column structure far better than any
  histogram-based gutter detection — it was trained on real document layouts.

* TrOCR (microsoft/trocr-large-handwritten) was specifically trained on
  handwritten and aged printed text, making it well-suited to historic
  newspapers with degraded ink, uneven baselines, and period typefaces.

* Keeping Surya for detection and layout while using TrOCR for recognition
  allows direct A/B comparison against the all-Surya edition
  (opticolumn_surya_native.py) on the same layout segmentation.

USAGE
-----
1. Drop PDF files into the A/ folder.
2. pip install surya-ocr pymupdf pikepdf pillow transformers torch
3. python opticolumn_trocr_hybrid.py
Output PDFs land in B/; debug images and reports land in debug/.
"""

import sys
import os
import re
import datetime
import shutil
import platform
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

import fitz          # PyMuPDF
import pikepdf
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from surya.detection import DetectionPredictor
from surya.foundation import FoundationPredictor
from surya.layout import LayoutPredictor
from surya.settings import settings


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

INPUT_DIR  = "A"
OUTPUT_DIR = "B"
DEBUG_DIR  = "debug"

# Render resolution.  300 DPI gives both Surya and TrOCR enough detail on
# aged newsprint without creating impractically large tensors.
DPI = 300

# ── TrOCR model selection ─────────────────────────────────────────────────────
# large_handwritten performs best on aged/degraded historic newspaper type.
# Switch to large_printed for cleaner modern scans.
TROCR_MODELS = {
    "handwritten":       "microsoft/trocr-base-handwritten",
    "printed":           "microsoft/trocr-base-printed",
    "large_handwritten": "microsoft/trocr-large-handwritten",
    "large_printed":     "microsoft/trocr-large-printed",
}
TROCR_MODEL_NAME = TROCR_MODELS["large_handwritten"]

# ── TrOCR noise-filter thresholds ────────────────────────────────────────────
# TrOCR does not expose per-token probabilities in the same way as Surya;
# we approximate confidence from the softmax of the output scores.
CONFIDENCE_THRESHOLD             = 0.25   # minimum mean token confidence
SINGLE_CHAR_CONFIDENCE_THRESHOLD = 0.50   # tighter threshold for 1-char results
MIN_LINE_H                       = 8      # px — skip lines shorter than this
MIN_LINE_W                       = 15     # px — skip lines narrower than this

# ── Layout label taxonomy ─────────────────────────────────────────────────────
# Every label the Surya LayoutPredictor can return is listed explicitly.
# OCR_LABELS  → segment with DetectionPredictor, recognise with TrOCR
# SKIP_LABELS → no OCR
OCR_LABELS = {
    "Text",              # body copy — primary article content
    "Section-header",    # column and article headlines
    "Caption",           # photo / illustration captions
    "Footnote",          # editorial notes, source citations
    "List-item",         # bulleted or numbered list entries
    "Page-footer",       # pagination lines, print datelines
    "Page-header",       # masthead, volume / issue / date strip
    "Table-of-contents", # index entries (text, not grid structure)
    "Handwriting",       # editorial annotations, marginalia
    "Text-inline-math",  # inline mathematical notation within prose
    "Formula",           # display equations
}
SKIP_LABELS = {
    "Picture",   # photographs — no text
    "Figure",    # diagrams / charts — no text
    "Table",     # grid structure — use TableRecPredictor separately
    "Form",      # field/value layout — requires form-aware parsing
}

# Labels that are typically single large text blocks filling the entire crop.
# For these, DetectionPredictor often returns nothing because there are no
# inter-line whitespace boundaries.  They always receive the two-pass treatment.
SINGLE_BLOCK_LABELS = {
    "Section-header",
    "Page-header",
    "Caption",
    "Footnote",
}

# Minimum pixel dimensions for a region to bother processing.
MIN_REGION_W = 40
MIN_REGION_H = 15

# PDF/A font & colour-profile resources
FONT_NAME     = "helv"
FONT_PATH     = "fonts/FreeSans.ttf"
SRGB_ICC_PATH = "srgb.icc"

# Debug colour palette keyed on layout label
LABEL_COLOURS: Dict[str, str] = {
    "Page-header":       "#1565C0",
    "Section-header":    "#C62828",
    "Text":              "#2E7D32",
    "Caption":           "#6A1B9A",
    "Footnote":          "#4E342E",
    "Page-footer":       "#37474F",
    "Table":             "#E65100",
    "Picture":           "#00838F",
    "Figure":            "#00695C",
    "List-item":         "#558B2F",
    "Handwriting":       "#AD1457",
    "Form":              "#FF6F00",
    "Table-of-contents": "#0277BD",
}
DEFAULT_COLOUR = "#9E9E9E"


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

DEBUG_PATH = Path(DEBUG_DIR)
DEBUG_PATH.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(DEBUG_PATH / "run.log"), mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# COLOUR UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _hex_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _label_hex(label: str) -> str:
    return LABEL_COLOURS.get(label, DEFAULT_COLOUR)


def _pil_font(size: int = 16) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_PATH, size=size)
    except Exception:
        return ImageFont.load_default()


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING  –  tuned for aged newsprint
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_newspaper(image: Image.Image) -> Image.Image:
    """
    Tiled adaptive contrast + unsharp mask for historic newsprint.

    Runs autocontrast on overlapping 256 px tiles (CLAHE approximation),
    then applies a 1.5 px unsharp mask to sharpen hairline serifs without
    amplifying the halftone dot pattern common on period newspaper printing.
    """
    try:
        gray    = image.convert("L")
        tile_sz = 256
        overlap = tile_sz // 2
        w, h    = gray.size
        result  = gray.copy()
        for ty in range(0, h, overlap):
            for tx in range(0, w, overlap):
                x1   = min(tx + tile_sz, w)
                y1   = min(ty + tile_sz, h)
                tile = gray.crop((tx, ty, x1, y1))
                tile = ImageOps.autocontrast(tile, cutoff=1)
                result.paste(tile, (tx, ty))
        blurred   = result.filter(ImageFilter.GaussianBlur(radius=1.5))
        sharpened = Image.blend(result, blurred, alpha=-0.35)
        return sharpened.convert("RGB")
    except Exception as exc:
        logger.warning(f"Preprocessing fallback: {exc}")
        return image.convert("RGB")


# ══════════════════════════════════════════════════════════════════════════════
# PDF/A HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _setup_pdf_resources() -> None:
    try:
        Path("fonts").mkdir(exist_ok=True)
        if not Path(FONT_PATH).exists():
            import urllib.request
            logger.info("  Fetching FreeSans.ttf …")
            urllib.request.urlretrieve(
                "https://github.com/opensourcedesign/fonts/raw/master/"
                "gnu-freefont_freesans/FreeSans.ttf",
                FONT_PATH,
            )
        if not Path(SRGB_ICC_PATH).exists():
            import urllib.request
            logger.info("  Fetching sRGB ICC profile …")
            sys_icc = {
                "Darwin":  "/System/Library/ColorSync/Profiles/sRGB Profile.icc",
                "Linux":   "/usr/share/color/icc/sRGB.icc",
                "Windows": os.path.join(
                    os.environ.get("WINDIR", r"C:\Windows"),
                    "System32", "spool", "drivers", "color",
                    "sRGB Color Space Profile.icm",
                ),
            }.get(platform.system(), "")
            if sys_icc and Path(sys_icc).exists():
                shutil.copy2(sys_icc, SRGB_ICC_PATH)
            else:
                urllib.request.urlretrieve("https://www.color.org/srgb.xalter", SRGB_ICC_PATH)
    except Exception as exc:
        logger.warning(f"PDF/A resource setup incomplete: {exc}")


def _make_xmp(title, author, subject, creator, producer, cdate, mdate) -> str:
    return (
        '<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '<rdf:Description rdf:about="" xmlns:pdf="http://ns.adobe.com/pdf/1.3/">'
        f'<pdf:Producer>{producer}</pdf:Producer></rdf:Description>\n'
        '<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">'
        f'<dc:title><rdf:Alt><rdf:li xml:lang="x-default">{title}</rdf:li></rdf:Alt></dc:title>'
        f'<dc:creator><rdf:Seq><rdf:li>{author}</rdf:li></rdf:Seq></dc:creator>'
        f'<dc:description><rdf:Alt><rdf:li xml:lang="x-default">{subject}'
        f'</rdf:li></rdf:Alt></dc:description></rdf:Description>\n'
        '<rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/">'
        f'<xmp:CreatorTool>{creator}</xmp:CreatorTool>'
        f'<xmp:CreateDate>{cdate}</xmp:CreateDate>'
        f'<xmp:ModifyDate>{mdate}</xmp:ModifyDate></rdf:Description>\n'
        '<rdf:Description rdf:about="" xmlns:pdfaid="http://www.aiim.org/pdfa/ns/id/">'
        '<pdfaid:part>1</pdfaid:part>'
        '<pdfaid:conformance>B</pdfaid:conformance></rdf:Description>\n'
        '</rdf:RDF></x:xmpmeta>\n<?xpacket end="w"?>'
    )


def _embed_output_intent(pdf_path: str) -> None:
    try:
        icc = Path(SRGB_ICC_PATH)
        if not icc.exists():
            return
        with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
            if "/OutputIntents" not in pdf.Root:
                pdf.Root["/OutputIntents"] = pikepdf.Array()
            stream = pdf.make_stream(icc.read_bytes())
            stream.stream_dict["/N"]         = pikepdf.Integer(3)
            stream.stream_dict["/Alternate"] = pikepdf.Name("/DeviceRGB")
            pdf.Root["/OutputIntents"].append(pdf.make_indirect(pikepdf.Dictionary({
                "/Type":                      pikepdf.Name("/OutputIntent"),
                "/S":                         pikepdf.Name("/GTS_PDFA1"),
                "/Info":                      pikepdf.String("sRGB IEC61966-2.1"),
                "/OutputConditionIdentifier": pikepdf.String("sRGB"),
                "/DestOutputProfile":         pdf.make_indirect(stream),
            })))
            pdf.save(pdf_path)
        logger.info("  PDF/A OutputIntent embedded.")
    except Exception as exc:
        logger.warning(f"  PDF/A OutputIntent skipped: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_models():
    """
    Load all models:

    Surya  — LayoutPredictor (layout backbone) + DetectionPredictor (line segmentation)
    TrOCR  — TrOCRProcessor + VisionEncoderDecoderModel (text recognition)

    RecognitionPredictor is NOT loaded in this edition; TrOCR replaces it
    entirely for the character-recognition step.

    Layout path (per Surya docs):
        LayoutPredictor(FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT))
    """
    logger.info("=" * 62)
    logger.info("  LOADING MODELS  (Surya Layout + TrOCR Recognition)")
    logger.info("=" * 62)

    _setup_pdf_resources()

    # ── Surya DetectionPredictor ───────────────────────────────────────────────
    logger.info("  DetectionPredictor (line segmentation) …")
    det_predictor = DetectionPredictor()

    # ── Surya LayoutPredictor ─────────────────────────────────────────────────
    logger.info(f"  FoundationPredictor (layout: {settings.LAYOUT_MODEL_CHECKPOINT}) …")
    foundation_lay   = FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    logger.info("  LayoutPredictor …")
    layout_predictor = LayoutPredictor(foundation_lay)

    # ── TrOCR ─────────────────────────────────────────────────────────────────
    logger.info(f"  TrOCR processor + model: {TROCR_MODEL_NAME} …")
    trocr_processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_NAME)
    trocr_model     = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_NAME)
    device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trocr_model.to(device)
    logger.info(f"  TrOCR device: {device}")

    logger.info("  All models ready.\n")
    return det_predictor, layout_predictor, trocr_processor, trocr_model


# ══════════════════════════════════════════════════════════════════════════════
# LABEL NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

# The Surya docs specify hyphenated label strings ("Section-header", etc.).
# Some installed versions emit camelCase ("SectionHeader").  This map converts
# every known variant to the canonical documented form.
_LABEL_ALIAS: Dict[str, str] = {
    "SectionHeader":    "Section-header",
    "PageHeader":       "Page-header",
    "PageFooter":       "Page-footer",
    "ListItem":         "List-item",
    "TableOfContents":  "Table-of-contents",
    "InlineMath":       "Text-inline-math",
    "TextInlineMath":   "Text-inline-math",
    "Header":           "Page-header",
    "Footer":           "Page-footer",
    "Heading":          "Section-header",
    "Title":            "Section-header",
    "Handwritten":      "Handwriting",
}


def _normalise_label(raw: str) -> str:
    return _LABEL_ALIAS.get(raw, raw)


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_layout_result(result) -> Tuple[List[Dict], Optional[List[float]]]:
    """
    Normalise one LayoutPredictor page result.

    Surya LayoutPredictor output per page (docs):
        result.bboxes     — list of bbox objects
        result.image_bbox — [x1, y1, x2, y2] coordinate space of the page image
        result.page       — 0-based page number

    Each bbox object:
        .bbox     [x1, y1, x2, y2]
        .polygon  [(x1,y1)…(x4,y4)] clockwise from top-left
        .position int   reading-order index (column-aware, model-native)
        .label    str   e.g. "Text", "SectionHeader", "Section-header" …
                  NOTE: may be camelCase; normalised via _normalise_label()
        .top_k    dict  {label: confidence} for OTHER candidate labels only
                  (the primary label is NOT in top_k — it is the argmax)

    Returns (regions list, image_bbox or None).
    """
    regions: List[Dict] = []
    if result is None or not hasattr(result, "bboxes"):
        return regions, None

    image_bbox: Optional[List[float]] = (
        list(result.image_bbox)
        if hasattr(result, "image_bbox") and result.image_bbox
        else None
    )

    for box in result.bboxes:
        if hasattr(box, "bbox") and box.bbox:
            bbox = [float(v) for v in box.bbox]
        elif hasattr(box, "polygon") and len(box.polygon) >= 4:
            xs   = [float(p[0]) for p in box.polygon]
            ys   = [float(p[1]) for p in box.polygon]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
        else:
            continue

        polygon = (
            [(float(p[0]), float(p[1])) for p in box.polygon]
            if hasattr(box, "polygon") and box.polygon
            else None
        )

        raw_label = getattr(box, "label", "Text")
        label     = _normalise_label(raw_label)
        top_k     = {
            _normalise_label(k): v
            for k, v in (getattr(box, "top_k", {}) or {}).items()
        }
        position = int(getattr(box, "position", 0))

        regions.append({
            "bbox":     bbox,
            "polygon":  polygon,
            "label":    label,
            "position": position,
            "top_k":    top_k,
        })

    return regions, image_bbox


# ══════════════════════════════════════════════════════════════════════════════
# TROCR RECOGNITION  +  NOISE FILTER
# ══════════════════════════════════════════════════════════════════════════════

def _trocr_read(
    image: Image.Image,
    processor: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
) -> Tuple[str, float]:
    """
    Run TrOCR on a single image crop.

    Returns (text, confidence) where confidence is the mean max-token
    probability across all generated tokens (0–1).  This approximates
    per-character certainty without access to TrOCR's internal beam scores.
    """
    try:
        pixel_values = processor(image.convert("RGB"), return_tensors="pt").pixel_values
        device       = next(model.parameters()).device
        pixel_values = pixel_values.to(device)
        with torch.no_grad():
            out = model.generate(
                pixel_values,
                output_scores=True,
                return_dict_in_generate=True,
            )
        text = processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
        if out.scores:
            probs      = [torch.softmax(s, dim=-1) for s in out.scores]
            max_probs  = [torch.max(p).item() for p in probs]
            confidence = sum(max_probs) / len(max_probs)
        else:
            confidence = 0.0
        return text, confidence
    except Exception as exc:
        logger.debug(f"    TrOCR error: {exc}")
        return "", 0.0


def _is_noise(text: str, confidence: float, h: int, w: int) -> bool:
    """
    Heuristic noise filter for TrOCR output.

    Returns True if the recognised text is likely artefact, ruling line,
    punctuation noise, or a confidence-floor rejection.
    """
    if not text:
        return True
    if h < MIN_LINE_H or w < MIN_LINE_W:
        return True
    ar = w / h
    if ar < 0.1 or ar > 100:
        return True
    tc = text.strip()
    tl = len(tc)
    if tl == 1:
        return confidence < SINGLE_CHAR_CONFIDENCE_THRESHOLD
    if confidence < CONFIDENCE_THRESHOLD:
        return True
    if len(set(tc)) == 1 and tl > 2:    # repeated single character
        return True
    noise_pats = [r"^[oOlI\.\|]+$", r"^[0-9\.\,]+$", r"^[^a-zA-Z0-9\s]+$"]
    for pat in noise_pats:
        if re.match(pat, tc) and confidence < SINGLE_CHAR_CONFIDENCE_THRESHOLD:
            return True
    if tl > 3 and not any(c.lower() in "aeiou" for c in tc) and confidence < 0.7:
        return True
    return False


def _surya_line_bboxes(
    crop: Image.Image,
    det_predictor: DetectionPredictor,
) -> List[List[float]]:
    """
    Run Surya DetectionPredictor on a crop image.

    Returns a list of [x0, y0, x1, y1] bboxes in crop-relative coordinates,
    sorted top-to-bottom.
    """
    try:
        results = det_predictor([crop])
        if not results or not hasattr(results[0], "bboxes"):
            return []
        bboxes = []
        for box in results[0].bboxes:
            if hasattr(box, "bbox") and box.bbox:
                bboxes.append([float(v) for v in box.bbox])
            elif hasattr(box, "polygon") and len(box.polygon) >= 4:
                xs = [float(p[0]) for p in box.polygon]
                ys = [float(p[1]) for p in box.polygon]
                bboxes.append([min(xs), min(ys), max(xs), max(ys)])
        bboxes.sort(key=lambda b: b[1])   # top-to-bottom
        return bboxes
    except Exception as exc:
        logger.debug(f"    DetectionPredictor error: {exc}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# PER-REGION OCR  (two-pass: detect → TrOCR per line, fallback whole-crop)
# ══════════════════════════════════════════════════════════════════════════════

def ocr_region(
    page_image: Image.Image,
    region: Dict,
    det_predictor: DetectionPredictor,
    trocr_processor: TrOCRProcessor,
    trocr_model: VisionEncoderDecoderModel,
) -> List[Dict]:
    """
    Two-pass OCR for a single layout region.

    Pass 1 — DetectionPredictor + TrOCR per line
    ─────────────────────────────────────────────
    The region crop is passed to DetectionPredictor to find text line bboxes.
    Each line is then cropped and read by TrOCR independently.  Noise lines
    are filtered by _is_noise().  This pass works well for multi-line body
    text regions.

    Pass 2 — Whole-crop TrOCR fallback
    ────────────────────────────────────
    Triggered when:
      (a) label is in SINGLE_BLOCK_LABELS  (always attempt whole-crop), OR
      (b) Pass 1 returns zero accepted lines for any label.

    Without detection, TrOCR receives the entire region crop as one image.
    This is the correct approach for a single-line banner headline or masthead
    where the crop IS the text and DetectionPredictor finds no line boundaries.

    When both passes yield results, the one with more accepted lines wins.

    All returned bbox coordinates are in full-page (absolute) pixel space.
    """
    x0, y0, x1, y1 = [int(c) for c in region["bbox"]]
    iw, ih          = page_image.size
    label           = region["label"]

    x0 = max(0, x0);  y0 = max(0, y0)
    x1 = min(iw, x1); y1 = min(ih, y1)

    rw, rh = x1 - x0, y1 - y0
    if rw < MIN_REGION_W or rh < MIN_REGION_H:
        return []

    crop = page_image.crop((x0, y0, x1, y1))

    # ── Pass 1: line detection → TrOCR per line ───────────────────────────────
    line_bboxes = _surya_line_bboxes(crop, det_predictor)
    pass1_elems: List[Dict] = []

    for lb in line_bboxes:
        lx0, ly0, lx1, ly1 = [int(v) for v in lb]
        lh, lw = ly1 - ly0, lx1 - lx0
        if lh < MIN_LINE_H or lw < MIN_LINE_W:
            continue
        line_crop        = crop.crop((lx0, ly0, lx1, ly1))
        text, confidence = _trocr_read(line_crop, trocr_processor, trocr_model)
        if _is_noise(text, confidence, lh, lw):
            logger.debug(
                f"      [NOISE] conf={confidence:.2f} "
                f"{lw}×{lh}px | {text[:40]}"
            )
            continue
        # Absolute page coordinates
        abs_bbox = [lx0 + x0, ly0 + y0, lx1 + x0, ly1 + y0]
        pass1_elems.append({
            "text":             text,
            "bbox":             abs_bbox,
            "confidence":       confidence,
            "font_size":        max(6.0, min(lh * 0.85, 72.0)),
            "source_label":     label,
            "reading_position": region["position"],
        })

    n_pass1 = len(pass1_elems)

    if n_pass1 > 0 and label not in SINGLE_BLOCK_LABELS:
        logger.debug(f"      Pass 1 (det+TrOCR): {n_pass1} line(s)")
        return pass1_elems

    # ── Pass 2: whole-crop TrOCR ──────────────────────────────────────────────
    text_wb, conf_wb = _trocr_read(crop, trocr_processor, trocr_model)
    pass2_elems: List[Dict] = []

    if not _is_noise(text_wb, conf_wb, rh, rw):
        # The whole-crop read is treated as a single line spanning the region.
        pass2_elems.append({
            "text":             text_wb,
            "bbox":             [float(x0), float(y0), float(x1), float(y1)],
            "confidence":       conf_wb,
            "font_size":        max(6.0, min(rh * 0.85, 72.0)),
            "source_label":     label,
            "reading_position": region["position"],
        })

    n_pass2 = len(pass2_elems)

    if n_pass2 > 0:
        logger.debug(
            f"      Pass 2 (whole-crop TrOCR): {n_pass2} line(s)  "
            f"[Pass 1 had {n_pass1}]"
        )
        # Prefer whichever pass produced more lines
        if n_pass2 >= n_pass1:
            return pass2_elems

    if n_pass1 > 0:
        logger.debug(f"      Kept Pass 1 ({n_pass1} lines) over Pass 2 ({n_pass2})")
        return pass1_elems

    logger.debug(f"      Both passes empty for label={label}")
    return []


# ══════════════════════════════════════════════════════════════════════════════
# DEBUG VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def save_layout_debug(image: Image.Image, regions: List[Dict], path: Path) -> None:
    img      = image.copy().convert("RGB")
    draw     = ImageDraw.Draw(img, "RGBA")
    lbl_font = _pil_font(15)
    for region in sorted(regions, key=lambda r: r.get("position", 999)):
        x0, y0, x1, y1 = region["bbox"]
        label = region.get("label", "?")
        pos   = region.get("position", "?")
        rgb   = _hex_rgb(_label_hex(label))
        tag   = f"[{pos}] {label}"
        tag_w = len(tag) * 9 + 6
        draw.rectangle([x0, y0, x1, y1], outline=rgb + (210,), fill=rgb + (22,), width=2)
        draw.rectangle([x0, y0, x0 + tag_w, y0 + 20], fill=rgb + (175,))
        draw.text((x0 + 3, y0 + 2), tag, fill=(255, 255, 255, 255), font=lbl_font)
    img.save(str(path), "JPEG", quality=90)
    logger.info(f"    [DEBUG] Layout map → {path.name}")


def save_ocr_debug(image: Image.Image, elements: List[Dict], path: Path) -> None:
    img  = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    font = _pil_font(12)
    for elem in elements:
        x0, y0, x1, y1 = elem["bbox"]
        rgb = _hex_rgb(_label_hex(elem.get("source_label", "Text")))
        draw.rectangle([x0, y0, x1, y1], outline=rgb + (180,), width=1)
        draw.text((x0 + 1, y0), elem["text"][:55], fill=rgb + (220,), font=font)
    img.save(str(path), "JPEG", quality=88)
    logger.info(f"    [DEBUG] OCR overlay → {path.name}")


def save_layout_report(regions: List[Dict], image_bbox: Optional[List[float]],
                       path: Path, filename: str, page_num: int) -> None:
    lines = [
        f"FILE: {filename}   PAGE: {page_num}",
        f"Regions detected: {len(regions)}",
        (f"image_bbox (coord space): {[round(v) for v in image_bbox]}"
         if image_bbox else "image_bbox: not reported"),
        "=" * 80,
        f"{'POS':>4}  {'LABEL':<22}  {'X0':>6} {'Y0':>6} {'X1':>6} {'Y1':>6}"
        f"  TOP-K ALTERNATIVES",
        "-" * 80,
    ]
    for r in sorted(regions, key=lambda x: x.get("position", 999)):
        x0, y0, x1, y1 = r["bbox"]
        top_k     = r.get("top_k", {})
        top_k_str = "  ".join(
            f"{lbl}:{conf:.2f}"
            for lbl, conf in sorted(top_k.items(), key=lambda kv: -kv[1])
        ) if top_k else "—"
        lines.append(
            f"{r.get('position','?'):>4}  "
            f"{r.get('label','?'):<22}  "
            f"{x0:>6.0f} {y0:>6.0f} {x1:>6.0f} {y1:>6.0f}"
            f"  {top_k_str}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"    [DEBUG] Layout report → {path.name}")


def save_ocr_report(elements: List[Dict], path: Path,
                    filename: str, page_num: int) -> None:
    lines = [
        f"FILE: {filename}   PAGE: {page_num}",
        f"OCR elements: {len(elements)}",
        "=" * 80,
    ]
    for i, e in enumerate(elements):
        x0, y0, x1, y1 = e["bbox"]
        lines.append(
            f"[{e.get('reading_position', i):>3}] "
            f"{e.get('source_label','?'):<18}  "
            f"trocr_conf={e.get('confidence', 0.0):.2f}  "
            f"({x0:.0f},{y0:.0f}→{x1:.0f},{y1:.0f})  "
            f"| {e['text']}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"    [DEBUG] OCR report → {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def process_page(
    pil_image: Image.Image,
    page_num: int,
    filename: str,
    stem: str,
    det_predictor: DetectionPredictor,
    layout_predictor: LayoutPredictor,
    trocr_processor: TrOCRProcessor,
    trocr_model: VisionEncoderDecoderModel,
) -> List[Dict]:
    """
    Full pipeline for a single newspaper page.

    1. Preprocess  — tiled CLAHE-approx + unsharp mask
    2. Layout      — LayoutPredictor → semantic regions + reading order positions
    3. OCR         — DetectionPredictor (line segmentation) + TrOCR per line
    4. Sort        — by Surya layout position, then vertical baseline within region

    Returns a flat list of element dicts in reading order.
    """
    pfx = DEBUG_PATH / f"{stem}_p{page_num:03d}"
    logger.info("")
    logger.info("─" * 62)
    logger.info(f"  PAGE {page_num}  [{pil_image.width}×{pil_image.height}]  {filename}")
    logger.info("─" * 62)

    # ── Stage 1: Preprocess ───────────────────────────────────────────────────
    processed = preprocess_newspaper(pil_image)
    processed.save(str(Path(str(pfx) + "_00_preprocessed.jpg")), "JPEG", quality=85)

    # ── Stage 2: Surya layout detection ──────────────────────────────────────
    logger.info("  [LAYOUT] Running LayoutPredictor …")
    layout_regions: List[Dict]         = []
    image_bbox: Optional[List[float]]  = None

    try:
        layout_out = layout_predictor([processed])
        if layout_out:
            layout_regions, image_bbox = parse_layout_result(layout_out[0])
            page_reported = getattr(layout_out[0], "page", "n/a")
            logger.debug(f"  [LAYOUT] result.page={page_reported}  image_bbox={image_bbox}")
    except Exception as exc:
        logger.error(f"  LayoutPredictor failed: {exc}")
        import traceback; traceback.print_exc()

    if not layout_regions:
        logger.warning("  No layout regions — falling back to full-page single region.")
        layout_regions = [{
            "bbox":     [0.0, 0.0, float(pil_image.width), float(pil_image.height)],
            "polygon":  None,
            "label":    "Text",
            "position": 0,
            "top_k":    {},
        }]
    else:
        label_counts = Counter(r["label"] for r in layout_regions)
        logger.info(
            f"  [LAYOUT] {len(layout_regions)} regions  image_bbox={image_bbox}: "
            + "  ".join(f"{lbl}×{n}" for lbl, n in sorted(label_counts.items()))
        )
        save_layout_debug(pil_image, layout_regions,
                          Path(str(pfx) + "_01_layout.jpg"))
        save_layout_report(layout_regions, image_bbox,
                           Path(str(pfx) + "_01_layout_report.txt"),
                           filename, page_num)

    # ── Stage 3: Per-region DetectionPredictor + TrOCR ────────────────────────
    text_regions    = [r for r in layout_regions if r["label"] in OCR_LABELS]
    skip_regions    = [r for r in layout_regions if r["label"] in SKIP_LABELS]
    unknown_regions = [
        r for r in layout_regions
        if r["label"] not in OCR_LABELS and r["label"] not in SKIP_LABELS
    ]
    if unknown_regions:
        unk_lbls = sorted({r["label"] for r in unknown_regions})
        logger.warning(
            f"  [LAYOUT] {len(unknown_regions)} region(s) with unrecognised "
            f"label(s) {unk_lbls} — skipping OCR for these."
        )

    logger.info(
        f"  [OCR] {len(text_regions)} regions to OCR "
        f"(TrOCR model: {TROCR_MODEL_NAME}), "
        f"{len(skip_regions)} region(s) skipped."
    )

    all_elements: List[Dict] = []

    for ri, region in enumerate(sorted(text_regions, key=lambda r: r["position"])):
        lbl  = region["label"]
        bbox = region["bbox"]
        logger.debug(
            f"    Region {ri+1}/{len(text_regions)}: {lbl} "
            f"pos={region['position']}  "
            f"bbox=({bbox[0]:.0f},{bbox[1]:.0f}→{bbox[2]:.0f},{bbox[3]:.0f})"
        )
        elems = ocr_region(
            processed, region,
            det_predictor, trocr_processor, trocr_model,
        )
        logger.debug(f"      → {len(elems)} element(s) accepted.")
        all_elements.extend(elems)

    # ── Stage 4: Final reading order sort ────────────────────────────────────
    all_elements.sort(key=lambda e: (e["reading_position"], e["bbox"][1]))

    logger.info(f"  [RESULT] {len(all_elements)} OCR element(s) on page {page_num}.")

    if all_elements:
        save_ocr_debug(pil_image, all_elements,
                       Path(str(pfx) + "_02_ocr_overlay.jpg"))
        save_ocr_report(all_elements,
                        Path(str(pfx) + "_02_ocr_report.txt"),
                        filename, page_num)

    return all_elements


# ══════════════════════════════════════════════════════════════════════════════
# PDF PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def process_pdf(
    input_path: str,
    output_path: str,
    det_predictor: DetectionPredictor,
    layout_predictor: LayoutPredictor,
    trocr_processor: TrOCRProcessor,
    trocr_model: VisionEncoderDecoderModel,
) -> bool:
    filename = os.path.basename(input_path)
    stem     = Path(input_path).stem

    logger.info(f"\n{'━'*62}")
    logger.info(f"  PROCESSING: {filename}")
    logger.info(f"{'━'*62}")

    try:
        with fitz.open(input_path) as doc:
            n_pages = len(doc)
            logger.info(f"  {n_pages} page(s) — rendering at {DPI} DPI …")

            pil_images: List[Image.Image] = []
            for page in doc:
                pix = page.get_pixmap(dpi=DPI)
                pil_images.append(
                    Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                )

            all_page_elements: List[List[Dict]] = []
            for idx, pil_img in enumerate(pil_images):
                elems = process_page(
                    pil_img, idx + 1, filename, stem,
                    det_predictor, layout_predictor,
                    trocr_processor, trocr_model,
                )
                all_page_elements.append(elems)

            # ── PDF metadata ──────────────────────────────────────────────────
            now      = datetime.datetime.now()
            pdf_date = now.strftime("D:%Y%m%d%H%M%S")
            xmp_date = now.strftime("%Y-%m-%dT%H:%M:%S")
            doc.set_metadata({
                "title":        filename,
                "author":       "Opticolumn",
                "subject":      "OCR-processed historic newspaper (TrOCR hybrid)",
                "creator":      "Opticolumn-TrOCR-Hybrid 2026",
                "producer":     "PyMuPDF",
                "creationDate": pdf_date,
                "modDate":      pdf_date,
            })
            xmp = _make_xmp(
                title=filename, author="Opticolumn",
                subject="OCR-processed historic newspaper (TrOCR hybrid)",
                creator="Opticolumn-TrOCR-Hybrid 2026", producer="PyMuPDF",
                cdate=xmp_date, mdate=xmp_date,
            )
            if xmp:
                doc.set_xml_metadata(xmp)

            # ── Insert invisible text layer ───────────────────────────────────
            for page_num, (page, elements, pil_img) in enumerate(
                zip(doc, all_page_elements, pil_images), start=1
            ):
                if not elements:
                    logger.info(f"  Page {page_num}: no elements.")
                    continue

                if page.get_text().strip():
                    page.add_redact_annot(page.rect)
                    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

                iw, ih = pil_img.size
                pw, ph = page.rect.width, page.rect.height
                sx, sy = pw / iw, ph / ih

                writer   = fitz.TextWriter(page.rect)
                inserted = 0

                for elem in elements:
                    bx0, by0, bx1, by1 = elem["bbox"]
                    try:
                        writer.append(
                            fitz.Point(bx0 * sx, by1 * sy),
                            elem["text"],
                            font=fitz.Font(FONT_NAME),
                            fontsize=max(4.0, elem["font_size"] * sy),
                        )
                        inserted += 1
                    except Exception as exc:
                        logger.debug(f"    Text insert skip p{page_num}: {exc}")

                if inserted:
                    writer.write_text(
                        page, overlay=True, render_mode=3, color=(0, 0, 0)
                    )
                logger.info(
                    f"  Page {page_num}: inserted {inserted}/{len(elements)} element(s)."
                )

            doc.save(
                output_path,
                deflate=True, garbage=4, clean=True,
                deflate_images=False, encryption=fitz.PDF_ENCRYPT_KEEP,
            )
            logger.info(f"  Saved: {output_path}")

        _embed_output_intent(output_path)

        with fitz.open(output_path) as chk:
            total_chars = sum(len(p.get_text().strip()) for p in chk)
        status = "SUCCESS" if total_chars > 0 else "WARNING — no extractable text"
        logger.info(f"  {status}: {total_chars} characters in output PDF.")
        return total_chars > 0

    except Exception as exc:
        logger.error(f"  process_pdf failed: {exc}")
        import traceback; traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# COMPRESSION
# ══════════════════════════════════════════════════════════════════════════════

def compress_output(src: Path, dst: Path, orig_bytes: int) -> Path:
    budget  = int(orig_bytes * 1.15)
    current = src.stat().st_size
    logger.info(
        f"  Compression: {current // 1024} KB → budget ≤{budget // 1024} KB "
        f"(orig {orig_bytes // 1024} KB)"
    )
    if current <= budget:
        shutil.copy2(src, dst)
        logger.info("  Within budget — no compression applied.")
        return dst
    for i, opts in enumerate([
        {"deflate": True, "garbage": 4, "clean": True, "deflate_images": False},
        {"deflate": True, "garbage": 3, "clean": True, "deflate_images": False},
    ]):
        tmp = dst.with_suffix(f".tmp{i}.pdf")
        try:
            with fitz.open(str(src)) as d:
                d.save(str(tmp), **opts, encryption=fitz.PDF_ENCRYPT_KEEP)
            if tmp.stat().st_size <= budget:
                with fitz.open(str(tmp)) as chk:
                    chars = sum(len(p.get_text().strip()) for p in chk)
                if chars > 0:
                    shutil.move(str(tmp), str(dst))
                    logger.info(
                        f"  Option {i+1} accepted "
                        f"({dst.stat().st_size // 1024} KB, {chars} chars)."
                    )
                    return dst
                else:
                    logger.error(f"  Option {i+1}: text lost — rejecting.")
            tmp.unlink(missing_ok=True)
        except Exception as exc:
            logger.error(f"  Option {i+1} error: {exc}")
            tmp.unlink(missing_ok=True)
    logger.warning("  No compression option met budget — writing as-is.")
    shutil.copy2(src, dst)
    return dst


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║  OPTICOLUMN  –  Surya Layout  /  TrOCR Recognition (Hybrid) ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")
    logger.info(f"  Input    : {INPUT_DIR}")
    logger.info(f"  Output   : {OUTPUT_DIR}")
    logger.info(f"  Debug    : {DEBUG_PATH.resolve()}")
    logger.info(f"  DPI      : {DPI}")
    logger.info(f"  TrOCR    : {TROCR_MODEL_NAME}")
    logger.info("")

    input_folder  = Path(INPUT_DIR)
    output_folder = Path(OUTPUT_DIR)

    if not input_folder.exists():
        logger.error(f"Input folder '{INPUT_DIR}' not found.")
        sys.exit(1)
    output_folder.mkdir(exist_ok=True)

    pdf_files = sorted(input_folder.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files in '{INPUT_DIR}'.")
        sys.exit(1)

    logger.info(f"  Found {len(pdf_files)} PDF file(s).\n")

    # Load all models once; reuse across every file
    det_predictor, layout_predictor, trocr_processor, trocr_model = load_models()

    summary: List[Tuple[str, str, int]] = []

    for pdf_path in pdf_files:
        orig_size = pdf_path.stat().st_size
        logger.info(f"\n{'━'*62}")
        logger.info(f"  FILE: {pdf_path.name}  ({orig_size // 1024} KB)")

        tmp_path   = output_folder / f"{pdf_path.stem}_ocr_tmp.pdf"
        final_path = output_folder / f"{pdf_path.stem}_final.pdf"

        ok = process_pdf(
            str(pdf_path), str(tmp_path),
            det_predictor, layout_predictor,
            trocr_processor, trocr_model,
        )

        if not ok:
            logger.error("  Skipping — OCR stage failed.")
            summary.append((pdf_path.name, "FAILED", 0))
            continue

        result = compress_output(tmp_path, final_path, orig_size)
        fsz    = result.stat().st_size
        delta  = (fsz - orig_size) / orig_size * 100.0
        logger.info(f"\n  ✓ {result.name}  {fsz // 1024} KB ({delta:+.1f}%)")
        summary.append((pdf_path.name, "OK", fsz))

        try:
            tmp_path.unlink()
        except Exception:
            pass

    logger.info(f"\n{'═'*62}")
    logger.info("  SUMMARY")
    logger.info(f"{'═'*62}")
    for name, status, sz in summary:
        logger.info(f"  {status:<8}  {name}  ({sz // 1024} KB)")

    debug_files = sorted(DEBUG_PATH.iterdir())
    logger.info(
        f"\n  Debug artefacts : {len(debug_files)} files in {DEBUG_PATH.resolve()}"
    )
    logger.info(f"  Output PDFs     : {output_folder.resolve()}")


if __name__ == "__main__":
    main()