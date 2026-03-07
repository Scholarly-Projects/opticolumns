#!/usr/bin/env python3
"""
Opticolumn  –  SURYA-NATIVE EDITION
=====================================
Historic Newspaper Page Segmentation, Reading Order & OCR

ARCHITECTURE
------------
Stage 1  LayoutPredictor
         Detects semantic regions (Section-header, Text, Caption, Table,
         Picture, Page-header, etc.) with a native `position` field that
         encodes column-aware reading order.  Replaces the histogram gutter
         heuristic entirely.

Stage 2  RecognitionPredictor  (driven by DetectionPredictor internally)
         Surya's own OCR pipeline: line detection + recognition in a single
         call per region crop.  Replaces TrOCR and its hand-rolled noise
         filters.

Stage 3  Assemble & write
         Elements are ordered by layout `position`, then by vertical baseline
         within each region.  An invisible text layer is stamped into the
         output PDF for searchability.

WHY THIS IS BETTER FOR HISTORIC NEWSPAPERS
-------------------------------------------
* Histogram gutter detection assumes evenly spaced columns; old newspapers
  break that assumption constantly (varied column widths, rule lines,
  merged headline columns, boxed articles).  LayoutPredictor was trained on
  real document structure and handles these cases natively.

* TrOCR was designed for short line snippets; Surya RecognitionPredictor is
  trained end-to-end on multi-language document text and benchmarks ahead of
  cloud OCR on degraded historical material.

* Surya's `position` field encodes reading order across columns without any
  knowledge of gutter coordinates — it is derived from the model's spatial
  attention, not from a histogram.

* Region-scoped detection (cropping each layout bbox before running line
  detection) prevents lines in adjacent columns from bleeding into each
  other, a major failure mode of the original full-page detection pass.

PREPROCESSING CHOICES
----------------------
Historic newspapers benefit from a tiled CLAHE-approximation (autocontrast
on 256 px tiles) rather than a global autocontrast, which tends to crush
detail in partially-inked regions.  A lightweight unsharp mask recovers the
thin strokes of period typefaces without amplifying halftone dot noise.

USAGE
-----
1. Drop PDF files into the A/ folder.
2. pip install surya-ocr pymupdf pikepdf pillow
3. python opticolumn_surya_native.py
Output PDFs land in B/; debug images and reports land in debug/.
"""

import sys
import os
import datetime
import shutil
import platform
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz          # PyMuPDF
import pikepdf
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from surya.detection import DetectionPredictor
from surya.foundation import FoundationPredictor
from surya.layout import LayoutPredictor
from surya.recognition import RecognitionPredictor
from surya.settings import settings


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

INPUT_DIR  = "A"
OUTPUT_DIR = "B"
DEBUG_DIR  = "debug"

# Render resolution.  300 DPI gives Surya enough detail on aged newsprint
# without creating impractically large tensors.  Lower to 200 if RAM-limited.
DPI = 300

# ISO 639-1 language codes for OCR.  Add any languages your papers contain.
OCR_LANGUAGES: List[str] = ["en"]

# ── Layout label taxonomy ─────────────────────────────────────────────────────
# Every label the Surya LayoutPredictor can return is listed explicitly in one
# of exactly two sets so that a new label never silently falls through.
#
#   OCR_LABELS  → run RecognitionPredictor and include in the PDF text layer
#   SKIP_LABELS → no OCR; region is purely visual or structural
#
# Excluded from OCR (SKIP_LABELS):
#   Picture / Figure — photographic or illustrated content; no text to extract
#   Table            — tabular cell structure; plain line OCR produces garbled
#                      output.  Use surya_table / TableRecPredictor separately
#                      if machine-readable table data is needed.
#   Form             — structured field/value layout; OCR without field-aware
#                      parsing produces unordered fragments of limited value.
#
# Everything else carries readable prose or display text and is OCR'd:
OCR_LABELS = {
    "Text",              # body copy — the primary article content
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
    "Picture",   # photographs, illustrations — no text content
    "Figure",    # diagrams, charts — no text content
    "Table",     # grid structure — use TableRecPredictor for this
    "Form",      # field/value layout — requires form-aware parsing
}

# Minimum pixel dimensions for a region to bother OCR-ing.
# Note: layout label confidence is NOT read from top_k (which contains only
# ALTERNATIVE labels).  The model's primary label is trusted directly.
MIN_REGION_W = 40
MIN_REGION_H = 15

# PDF/A font & colour-profile resources
FONT_NAME     = "helv"
FONT_PATH     = "fonts/FreeSans.ttf"
SRGB_ICC_PATH = "srgb.icc"

# Debug colour palette keyed on layout label
LABEL_COLOURS: Dict[str, str] = {
    "Page-header":       "#1565C0",   # deep blue
    "Section-header":    "#C62828",   # deep red
    "Text":              "#2E7D32",   # dark green
    "Caption":           "#6A1B9A",   # purple
    "Footnote":          "#4E342E",   # brown
    "Page-footer":       "#37474F",   # blue-grey
    "Table":             "#E65100",   # burnt orange
    "Picture":           "#00838F",   # teal
    "Figure":            "#00695C",   # dark teal
    "List-item":         "#558B2F",   # olive green
    "Handwriting":       "#AD1457",   # pink
    "Form":              "#FF6F00",   # amber
    "Table-of-contents": "#0277BD",   # light blue
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
    Tiled adaptive contrast  +  unsharp mask for historic newsprint.

    PIL lacks a native CLAHE, so we approximate it by running autocontrast on
    overlapping 256 px tiles then blending the results back.  This handles the
    uneven illumination, foxing stains, and ink-spread common on scanned
    historic papers far better than a single global autocontrast pass.

    The unsharp mask uses a 1.5 px Gaussian radius — sufficient to sharpen
    hairline serifs without amplifying the halftone dot pattern typical of
    late-19th / early-20th century newspaper printing.
    """
    try:
        gray      = image.convert("L")
        tile_sz   = 256
        overlap   = tile_sz // 2
        w, h      = gray.size
        result    = gray.copy()

        for ty in range(0, h, overlap):
            for tx in range(0, w, overlap):
                x1 = min(tx + tile_sz, w)
                y1 = min(ty + tile_sz, h)
                tile  = gray.crop((tx, ty, x1, y1))
                tile  = ImageOps.autocontrast(tile, cutoff=1)
                result.paste(tile, (tx, ty))

        # Unsharp mask: subtract mild Gaussian blur from sharpened blend
        blurred   = result.filter(ImageFilter.GaussianBlur(radius=1.5))
        sharpened = Image.blend(result, blurred, alpha=-0.35)   # overshoot

        return sharpened.convert("RGB")

    except Exception as exc:
        logger.warning(f"Preprocessing fallback (returning as-is): {exc}")
        return image.convert("RGB")


# ══════════════════════════════════════════════════════════════════════════════
# PDF/A HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _setup_pdf_resources() -> None:
    """Download FreeSans font and obtain sRGB ICC profile if absent."""
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
                urllib.request.urlretrieve(
                    "https://www.color.org/srgb.xalter", SRGB_ICC_PATH
                )
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

def load_surya_models():
    """
    Instantiate all three Surya predictors exactly as the documentation specifies.

    Recognition path  (docs: surya/recognition.py example):
        foundation_rec = FoundationPredictor()           ← default OCR checkpoint
        RecognitionPredictor(foundation_rec)
        DetectionPredictor()                             ← passed as det_predictor

    Layout path  (docs: surya/layout.py example):
        LayoutPredictor(FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT))

    The layout checkpoint is a completely different set of weights from the
    recognition backbone.  Falling back to the recognition backbone when the
    layout checkpoint fails would silently produce garbage layout predictions,
    so we let any load error propagate and abort.
    """
    logger.info("=" * 62)
    logger.info("  LOADING SURYA MODELS")
    logger.info("=" * 62)

    _setup_pdf_resources()

    # ── Recognition backbone (default checkpoint) ─────────────────────────────
    logger.info("  FoundationPredictor (recognition backbone) …")
    foundation_rec = FoundationPredictor()

    logger.info("  DetectionPredictor …")
    det_predictor = DetectionPredictor()

    logger.info("  RecognitionPredictor …")
    rec_predictor = RecognitionPredictor(foundation_rec)

    # ── Layout backbone (LAYOUT_MODEL_CHECKPOINT) ─────────────────────────────
    # Docs: LayoutPredictor(FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT))
    logger.info(f"  FoundationPredictor (layout backbone: {settings.LAYOUT_MODEL_CHECKPOINT}) …")
    foundation_lay = FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)

    logger.info("  LayoutPredictor …")
    layout_predictor = LayoutPredictor(foundation_lay)

    logger.info("  All models ready.\n")
    return det_predictor, rec_predictor, layout_predictor


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT PARSING
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# LABEL NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

# The Surya docs specify hyphenated label strings ("Section-header",
# "Page-header", "List-item", etc.).  In practice the model may emit camelCase
# variants ("SectionHeader", "PageHeader", "ListItem") depending on the version
# installed.  All downstream logic — OCR_LABELS, SKIP_LABELS,
# SINGLE_BLOCK_LABELS, LABEL_COLOURS — uses the documented hyphenated form.
# This map converts every known camelCase variant to its documented equivalent
# so label matching works regardless of the installed Surya version.
#
# Keys   = what the model actually emits (camelCase / abbreviated)
# Values = what the docs specify (canonical form used everywhere in this file)
_LABEL_ALIAS: Dict[str, str] = {
    # camelCase → hyphenated (observed in real output files)
    "SectionHeader":    "Section-header",
    "PageHeader":       "Page-header",
    "PageFooter":       "Page-footer",
    "ListItem":         "List-item",
    "TableOfContents":  "Table-of-contents",
    "InlineMath":       "Text-inline-math",
    "TextInlineMath":   "Text-inline-math",
    # Abbreviated / alternate spellings seen in some builds
    "Header":           "Page-header",
    "Footer":           "Page-footer",
    "Heading":          "Section-header",
    "Title":            "Section-header",
    "Handwritten":      "Handwriting",
}


def _normalise_label(raw: str) -> str:
    """Return the canonical hyphenated label for *raw*, falling back to *raw* unchanged."""
    return _LABEL_ALIAS.get(raw, raw)


def parse_layout_result(result) -> Tuple[List[Dict], Optional[List[float]]]:
    """
    Normalise one LayoutPredictor page result into a plain list of region dicts
    and the page-level image_bbox.

    Surya LayoutPredictor output per page (from docs):
        result.bboxes        — list of bbox objects
        result.image_bbox    — [x1, y1, x2, y2] coordinate space of the image
        result.page          — 0-based page number

    Each bbox object has:
        .bbox       [x1, y1, x2, y2]
        .polygon    [(x1,y1),(x2,y2),(x3,y3),(x4,y4)] clockwise from top-left
        .position   int   reading-order index (column-aware, from the model)
        .label      str   e.g. "Text", "SectionHeader", "Section-header" …
                    NOTE: the model may return camelCase variants; all are
                    normalised to the documented hyphenated form via
                    _normalise_label() before being stored in the region dict.
        .top_k      dict  {label: confidence} for OTHER candidate labels
                    NOTE: top_k contains *alternatives*, not the primary label.
                    The model's chosen label is already the argmax prediction;
                    do not try to look it up in top_k (it won't be there).

    Region dict keys:
        bbox        [x0, y0, x1, y1]  — axis-aligned rectangle, page pixels
        polygon     list of (x,y)     — exact quad, useful for skewed columns
        label       str               — semantic label (model's top prediction)
        position    int               — reading order index from Surya
        top_k       dict              — alternative labels and their confidences
                                        (kept for debug reporting only)
    """
    regions: List[Dict] = []
    if result is None or not hasattr(result, "bboxes"):
        return regions, None

    # Page-level coordinate space — all bbox coordinates are relative to this.
    image_bbox: Optional[List[float]] = (
        list(result.image_bbox)
        if hasattr(result, "image_bbox") and result.image_bbox
        else None
    )

    for box in result.bboxes:
        # ── Axis-aligned bbox (primary coordinate form) ───────────────────────
        if hasattr(box, "bbox") and box.bbox:
            bbox = [float(v) for v in box.bbox]
        elif hasattr(box, "polygon") and len(box.polygon) >= 4:
            xs   = [float(p[0]) for p in box.polygon]
            ys   = [float(p[1]) for p in box.polygon]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
        else:
            continue  # skip malformed entries

        # ── Polygon (exact quad for skewed / rotated regions) ─────────────────
        polygon = (
            [(float(p[0]), float(p[1])) for p in box.polygon]
            if hasattr(box, "polygon") and box.polygon
            else None
        )

        # ── Semantic label — trust the model's argmax directly ────────────────
        # top_k holds OTHER candidate labels; the primary label is NOT in top_k.
        # Do not gate or replace the primary label using top_k values — doing so
        # would always trigger (primary absent → confidence 0.0 < threshold)
        # and replace every label with the runner-up prediction.
        raw_label = getattr(box, "label", "Text")
        label     = _normalise_label(raw_label)           # canonical form
        top_k     = {
            _normalise_label(k): v
            for k, v in (getattr(box, "top_k", {}) or {}).items()
        }   # normalise alternatives too; kept for debug only

        # ── Reading-order position (column-aware, model-native) ───────────────
        position = int(getattr(box, "position", 0))

        regions.append({
            "bbox":     bbox,
            "polygon":  polygon,
            "label":    label,
            "position": position,
            "top_k":    top_k,      # alternatives; for debug logging only
        })

    return regions, image_bbox


# ══════════════════════════════════════════════════════════════════════════════
# PER-REGION OCR
# ══════════════════════════════════════════════════════════════════════════════

# Labels that are typically single large text blocks occupying the full crop.
# For these, DetectionPredictor often returns zero lines because the entire
# image IS one glyph cluster with no multi-line structure to segment.
# They receive the two-pass treatment described in _run_recognition().
SINGLE_BLOCK_LABELS = {
    "Section-header",    # article / column headlines — one large bold line
    "Page-header",       # masthead, volume/issue strip — one or two big lines
    "Caption",           # photo caption — often a single short sentence
    "Footnote",          # often a single short line at the foot of a column
}


def _run_recognition(
    crop: Image.Image,
    rec_predictor: RecognitionPredictor,
    det_predictor: DetectionPredictor,
    label: str,
    region_bbox: Tuple[int, int, int, int],
) -> List:
    """
    Two-pass recognition strategy that prevents silent blank output for
    single-block header regions.

    PASS 1 — Detection-driven (standard path)
    ──────────────────────────────────────────
    RecognitionPredictor is called with det_predictor so it can segment the
    crop into individual text lines before recognising each one.  This is
    correct for body-text columns that contain many lines.

    PASS 2 — Whole-crop fallback (single-block path)
    ─────────────────────────────────────────────────
    For regions in SINGLE_BLOCK_LABELS, OR whenever Pass 1 returns zero
    text_lines for any label, RecognitionPredictor is called WITHOUT
    det_predictor.  Without a line detector, Surya treats the entire crop
    image as a single recognition unit.  This is the right approach when
    the layout region IS the text (e.g. a full-width 72pt headline where
    DetectionPredictor's receptive field finds no internal line boundaries).

    Fallback is also triggered for any label whose Pass 1 returns nothing —
    this handles edge cases like a very short caption that the detector
    cannot confidently segment.

    Returns the raw text_lines list from the winning pass (may be empty
    if both passes fail).
    """
    x0, y0, x1, y1 = region_bbox

    def _call_rec(with_det: bool) -> List:
        try:
            if with_det:
                results = rec_predictor(
                    [crop], [OCR_LANGUAGES], det_predictor=det_predictor
                )
            else:
                results = rec_predictor([crop], [OCR_LANGUAGES])
        except TypeError:
            # Older Surya versions without a languages argument
            try:
                if with_det:
                    results = rec_predictor([crop], det_predictor=det_predictor)
                else:
                    results = rec_predictor([crop])
            except Exception as exc:
                logger.debug(f"      _call_rec(with_det={with_det}) TypeError path failed: {exc}")
                return []
        except Exception as exc:
            logger.debug(f"      _call_rec(with_det={with_det}) failed: {exc}")
            return []

        if not results:
            return []
        return getattr(results[0], "text_lines", []) or []

    # ── Pass 1: detection-driven ──────────────────────────────────────────────
    lines = _call_rec(with_det=True)
    n_pass1 = len([ln for ln in lines if getattr(ln, "text", "").strip()])

    if n_pass1 > 0 and label not in SINGLE_BLOCK_LABELS:
        # Body text: detection found lines and this is not a header type — done.
        logger.debug(f"      Pass 1 (det): {n_pass1} line(s)")
        return lines

    # ── Pass 2: whole-crop fallback ───────────────────────────────────────────
    # Triggered when:
    #   (a) label is in SINGLE_BLOCK_LABELS (always attempt whole-crop), OR
    #   (b) Pass 1 returned 0 non-empty lines for any label
    lines_fb = _call_rec(with_det=False)
    n_pass2  = len([ln for ln in lines_fb if getattr(ln, "text", "").strip()])

    if n_pass2 > 0:
        logger.debug(
            f"      Pass 2 (whole-crop fallback): {n_pass2} line(s)  "
            f"[Pass 1 had {n_pass1}]"
        )
        # For single-block labels, prefer whichever pass yielded more text.
        # For body labels that had 0 in Pass 1, the fallback is always better.
        if n_pass2 >= n_pass1:
            return lines_fb

    if n_pass1 > 0:
        logger.debug(f"      Kept Pass 1 result ({n_pass1} lines) over Pass 2 ({n_pass2})")
        return lines

    logger.debug(f"      Both passes returned 0 lines for label={label}")
    return []


def ocr_region(
    page_image: Image.Image,
    region: Dict,
    rec_predictor: RecognitionPredictor,
    det_predictor: DetectionPredictor,
) -> List[Dict]:
    """
    Crop a layout region from the full page and run two-pass Surya recognition.

    Why crop first?
    ───────────────
    Running DetectionPredictor on the full newspaper page and then filtering
    lines by bbox containment is prone to cross-column line bleeding because
    old newspaper columns sit very close together.  By cropping to the exact
    layout region before detection we constrain the receptive field to one
    semantic unit (one article, one headline block, etc.).

    Why two passes?
    ───────────────
    DetectionPredictor segments crops into individual text lines by looking for
    transitions between text and whitespace.  For a single-line headline that
    fills the entire crop (e.g. "SOPHS KICK OFF GALA YULETIDE WEEKEND" at 72pt
    spanning the full page width), there is no inter-line whitespace — the crop
    IS the line.  DetectionPredictor returns zero segments, RecognitionPredictor
    sees no lines to process, and the headline silently disappears from the
    output.  Pass 2 bypasses detection and presents the whole crop directly to
    RecognitionPredictor, which then recognises it as a single text unit.

    Returns a list of element dicts with coordinates in full-page pixel space.
    """
    x0, y0, x1, y1 = [int(c) for c in region["bbox"]]
    iw, ih          = page_image.size
    label           = region["label"]

    # Clamp to image bounds
    x0 = max(0, x0);  y0 = max(0, y0)
    x1 = min(iw, x1); y1 = min(ih, y1)

    rw, rh = x1 - x0, y1 - y0
    if rw < MIN_REGION_W or rh < MIN_REGION_H:
        return []

    crop  = page_image.crop((x0, y0, x1, y1))
    lines = _run_recognition(crop, rec_predictor, det_predictor, label,
                             (x0, y0, x1, y1))

    elements: List[Dict] = []
    for line in lines:
        text = getattr(line, "text", "").strip()
        if not text:
            continue

        conf = float(getattr(line, "confidence", 0.0))

        # Convert crop-relative bbox back to full-page coordinates
        lb = getattr(line, "bbox", None)
        if lb and len(lb) == 4:
            abs_bbox = [lb[0] + x0, lb[1] + y0, lb[2] + x0, lb[3] + y0]
        else:
            abs_bbox = [float(x0), float(y0), float(x1), float(y1)]

        lh = abs_bbox[3] - abs_bbox[1]
        elements.append({
            "text":             text,
            "bbox":             abs_bbox,
            "confidence":       conf,
            "font_size":        max(6.0, min(lh * 0.85, 72.0)),
            "source_label":     label,
            "reading_position": region["position"],
        })

    return elements


# ══════════════════════════════════════════════════════════════════════════════
# DEBUG VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def save_layout_debug(image: Image.Image, regions: List[Dict], path: Path) -> None:
    """Colour-coded bounding boxes annotated with label and reading-order index."""
    img  = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    lbl_font = _pil_font(15)

    for region in sorted(regions, key=lambda r: r.get("position", 999)):
        x0, y0, x1, y1 = region["bbox"]
        label  = region.get("label", "?")
        pos    = region.get("position", "?")
        rgb    = _hex_rgb(_label_hex(label))
        fill   = rgb + (22,)
        border = rgb + (210,)
        tag    = f"[{pos}] {label}"
        tag_w  = len(tag) * 9 + 6

        draw.rectangle([x0, y0, x1, y1], outline=border, fill=fill, width=2)
        draw.rectangle([x0, y0, x0 + tag_w, y0 + 20], fill=rgb + (175,))
        draw.text((x0 + 3, y0 + 2), tag, fill=(255, 255, 255, 255), font=lbl_font)

    img.save(str(path), "JPEG", quality=90)
    logger.info(f"    [DEBUG] Layout map → {path.name}")


def save_ocr_debug(image: Image.Image, elements: List[Dict], path: Path) -> None:
    """Overlay recognised text lines on the page image."""
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
    """
    Write a human-readable layout report.  image_bbox is the page-level
    coordinate space reported by Surya (result.image_bbox); all bbox values
    are contained within it.  top_k shows the alternative label candidates
    the model considered — kept here for diagnostic use only.
    """
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
        top_k = r.get("top_k", {})
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
            f"ocr_conf={e.get('confidence', 0.0):.2f}  "
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
    rec_predictor: RecognitionPredictor,
    layout_predictor: LayoutPredictor,
) -> List[Dict]:
    """
    Full pipeline for a single newspaper page.

    1. Preprocess  — tiled CLAHE-approx + unsharp mask
    2. Layout      — LayoutPredictor → semantic regions + reading order
    3. OCR         — RecognitionPredictor per text/table region crop
    4. Sort        — by layout position, then vertical baseline within region

    Returns a flat list of text-element dicts, in reading order.
    """
    pfx = DEBUG_PATH / f"{stem}_p{page_num:03d}"
    logger.info("")
    logger.info("─" * 62)
    logger.info(f"  PAGE {page_num}  [{pil_image.width}×{pil_image.height}]  {filename}")
    logger.info("─" * 62)

    # ── Stage 1: Preprocess ───────────────────────────────────────────────────
    processed = preprocess_newspaper(pil_image)
    processed.save(str(Path(str(pfx) + "_00_preprocessed.jpg")),
                   "JPEG", quality=85)

    # ── Stage 2: Layout detection ─────────────────────────────────────────────
    # Docs: layout_predictions = layout_predictor([image])
    # Each result has .bboxes (list), .image_bbox, .page
    # Each bbox has .bbox, .polygon, .label, .position (reading order), .top_k
    logger.info("  [LAYOUT] Running LayoutPredictor …")
    layout_regions: List[Dict] = []
    image_bbox: Optional[List[float]] = None

    try:
        layout_out = layout_predictor([processed])
        if layout_out:
            layout_regions, image_bbox = parse_layout_result(layout_out[0])
            page_reported = getattr(layout_out[0], "page", "n/a")
            logger.debug(f"  [LAYOUT] result.page={page_reported}  "
                         f"image_bbox={image_bbox}")
    except Exception as exc:
        logger.error(f"  LayoutPredictor failed: {exc}")
        import traceback; traceback.print_exc()

    if not layout_regions:
        logger.warning(
            "  No layout regions returned.  "
            "Falling back to full-page single-region OCR."
        )
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
        save_layout_debug(
            pil_image, layout_regions,
            Path(str(pfx) + "_01_layout.jpg"),
        )
        save_layout_report(
            layout_regions, image_bbox,
            Path(str(pfx) + "_01_layout_report.txt"),
            filename, page_num,
        )

    # ── Stage 3: Per-region OCR ───────────────────────────────────────────────
    # Regions to OCR: everything in OCR_LABELS
    # Skipped regions (Picture, Figure, Table, Form): no RecognitionPredictor call
    text_regions = [r for r in layout_regions if r["label"] in OCR_LABELS]
    skip_regions = [r for r in layout_regions if r["label"] in SKIP_LABELS]
    unknown_regions = [
        r for r in layout_regions
        if r["label"] not in OCR_LABELS and r["label"] not in SKIP_LABELS
    ]
    if unknown_regions:
        unknown_labels = sorted({r["label"] for r in unknown_regions})
        logger.warning(
            f"  [LAYOUT] {len(unknown_regions)} region(s) with unrecognised "
            f"label(s) {unknown_labels} — skipping OCR for these."
        )

    logger.info(
        f"  [OCR] {len(text_regions)} regions to OCR, "
        f"{len(skip_regions)} picture/figure region(s) skipped."
    )

    all_elements: List[Dict] = []

    for ri, region in enumerate(
        sorted(text_regions, key=lambda r: r["position"])
    ):
        lbl  = region["label"]
        bbox = region["bbox"]
        logger.debug(
            f"    Region {ri+1}/{len(text_regions)}: {lbl} "
            f"pos={region['position']}  "
            f"bbox=({bbox[0]:.0f},{bbox[1]:.0f}→{bbox[2]:.0f},{bbox[3]:.0f})"
        )
        elems = ocr_region(processed, region, rec_predictor, det_predictor)
        logger.debug(f"      → {len(elems)} line(s) recognised.")
        all_elements.extend(elems)

    # ── Stage 4: Final reading order sort ────────────────────────────────────
    # Primary:   Surya's native layout position   (column-aware)
    # Secondary: top-to-bottom within each region (baseline Y)
    all_elements.sort(key=lambda e: (e["reading_position"], e["bbox"][1]))

    logger.info(
        f"  [RESULT] {len(all_elements)} OCR element(s) on page {page_num}."
    )

    if all_elements:
        save_ocr_debug(
            pil_image, all_elements,
            Path(str(pfx) + "_02_ocr_overlay.jpg"),
        )
        save_ocr_report(
            all_elements,
            Path(str(pfx) + "_02_ocr_report.txt"),
            filename, page_num,
        )

    return all_elements


# ══════════════════════════════════════════════════════════════════════════════
# PDF PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def process_pdf(
    input_path: str,
    output_path: str,
    det_predictor: DetectionPredictor,
    rec_predictor: RecognitionPredictor,
    layout_predictor: LayoutPredictor,
) -> bool:
    """
    OCR a single PDF and write a PDF/A-1b compliant output with an invisible
    text overlay for searchability.
    """
    filename = os.path.basename(input_path)
    stem     = Path(input_path).stem

    logger.info(f"\n{'━'*62}")
    logger.info(f"  PROCESSING: {filename}")
    logger.info(f"{'━'*62}")

    try:
        with fitz.open(input_path) as doc:
            n_pages = len(doc)
            logger.info(f"  {n_pages} page(s) — rendering at {DPI} DPI …")

            # Render all pages to PIL images first
            pil_images: List[Image.Image] = []
            for page in doc:
                pix = page.get_pixmap(dpi=DPI)
                pil_images.append(
                    Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                )

            # OCR all pages
            all_page_elements: List[List[Dict]] = []
            for idx, pil_img in enumerate(pil_images):
                elems = process_page(
                    pil_img, idx + 1, filename, stem,
                    det_predictor, rec_predictor, layout_predictor,
                )
                all_page_elements.append(elems)

            # ── PDF metadata ─────────────────────────────────────────────────
            now      = datetime.datetime.now()
            pdf_date = now.strftime("D:%Y%m%d%H%M%S")
            xmp_date = now.strftime("%Y-%m-%dT%H:%M:%S")
            doc.set_metadata({
                "title":        filename,
                "author":       "Opticolumn",
                "subject":      "OCR-processed historic newspaper",
                "creator":      "Opticolumn-Surya 2026",
                "producer":     "PyMuPDF",
                "creationDate": pdf_date,
                "modDate":      pdf_date,
            })
            xmp = _make_xmp(
                title=filename, author="Opticolumn",
                subject="OCR-processed historic newspaper",
                creator="Opticolumn-Surya 2026", producer="PyMuPDF",
                cdate=xmp_date, mdate=xmp_date,
            )
            if xmp:
                doc.set_xml_metadata(xmp)

            # ── Insert invisible text layer ───────────────────────────────────
            for page_num, (page, elements, pil_img) in enumerate(
                zip(doc, all_page_elements, pil_images), start=1
            ):
                if not elements:
                    logger.info(f"  Page {page_num}: no elements to insert.")
                    continue

                # Clear any existing text layer (scanned-in machine text etc.)
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
                            fitz.Point(bx0 * sx, by1 * sy),   # bottom-left baseline
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
                    f"  Page {page_num}: inserted {inserted}/{len(elements)} "
                    "text element(s)."
                )

            doc.save(
                output_path,
                deflate=True, garbage=4, clean=True,
                deflate_images=False, encryption=fitz.PDF_ENCRYPT_KEEP,
            )
            logger.info(f"  Saved: {output_path}")

        # PDF/A compliance stamp
        _embed_output_intent(output_path)

        # Sanity-check: count extractable characters
        with fitz.open(output_path) as check_doc:
            total_chars = sum(len(p.get_text().strip()) for p in check_doc)

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
    """
    Attempt deflate compression options to keep output within 115% of input
    size.  Verifies the text layer survives each attempt before committing.
    """
    budget = int(orig_bytes * 1.15)
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
    logger.info("║   OPTICOLUMN  –  Surya-Native  Historic Newspaper OCR        ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")
    logger.info(f"  Input   : {INPUT_DIR}")
    logger.info(f"  Output  : {OUTPUT_DIR}")
    logger.info(f"  Debug   : {DEBUG_PATH.resolve()}")
    logger.info(f"  DPI     : {DPI}")
    logger.info(f"  Languages: {OCR_LANGUAGES}")
    logger.info("")

    input_folder  = Path(INPUT_DIR)
    output_folder = Path(OUTPUT_DIR)

    if not input_folder.exists():
        logger.error(f"Input folder '{INPUT_DIR}' not found.")
        sys.exit(1)
    output_folder.mkdir(exist_ok=True)

    pdf_files = sorted(input_folder.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in '{INPUT_DIR}'.")
        sys.exit(1)

    logger.info(f"  Found {len(pdf_files)} PDF file(s).\n")

    # Load models ONCE, reuse across all files
    det_predictor, rec_predictor, layout_predictor = load_surya_models()

    summary: List[Tuple[str, str, int]] = []   # (name, status, final_bytes)

    for pdf_path in pdf_files:
        orig_size = pdf_path.stat().st_size
        logger.info(f"\n{'━'*62}")
        logger.info(f"  FILE: {pdf_path.name}  ({orig_size // 1024} KB)")

        tmp_path   = output_folder / f"{pdf_path.stem}_ocr_tmp.pdf"
        final_path = output_folder / f"{pdf_path.stem}_final.pdf"

        ok = process_pdf(
            str(pdf_path), str(tmp_path),
            det_predictor, rec_predictor, layout_predictor,
        )

        if not ok:
            logger.error(f"  Skipping — OCR stage failed.")
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

    # ── Summary ───────────────────────────────────────────────────────────────
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