#!/usr/bin/env python3
"""
Opticolumns  –  debug_script_h.py
======================================================================
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
from xml.sax.saxutils import escape as xml_escape

# (h) Let PyTorch run any operation Apple's MPS backend lacks on the CPU instead
# of raising.  Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

try:
    import pymupdf as fitz      # current PyMuPDF (avoids the deprecated 'fitz' alias)
except ImportError:             # older PyMuPDF releases
    import fitz
import pikepdf
from PIL import Image, ImageChops, ImageCms, ImageDraw, ImageFilter, ImageFont, ImageOps

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

# Skip any input PDF whose <stem>.pdf already exists in OUTPUT_DIR.
SKIP_EXISTING_OUTPUT = True

# Render resolution.  300 DPI gives both Surya and TrOCR enough detail on
# aged newsprint without creating impractically large tensors.
DPI = 300

# ── Branding / document metadata ──────────────────────────────────────────────
APP_NAME      = "Opticolumns"
APP_VERSION   = "2026"
APP_CREATOR   = f"{APP_NAME} {APP_VERSION}"          # "Opticolumns 2026"
DOC_SUBJECT   = "OCR-processed historic newspaper"
DOC_LANGUAGE  = "en-US"                              # BCP-47; written to XMP + /Lang
# Namespace for the custom opt: XMP properties (declared via a PDF/A extension schema)
OPT_NAMESPACE = "http://github.com/Scholarly-Projects/opticolumn/"

# ── TrOCR model selection ─────────────────────────────────────────────────────
# large_handwritten performs best on aged/degraded historic newspaper type.
# Switch to large_printed for cleaner modern scans.
TROCR_MODELS = {
    "handwritten":       "microsoft/trocr-base-handwritten",
    "printed":           "microsoft/trocr-base-printed",
    "large_handwritten": "microsoft/trocr-large-handwritten",
    "large_printed":     "microsoft/trocr-large-printed",
}
TROCR_MODEL_NAME = TROCR_MODELS["large_printed"]

# ── TrOCR noise-filter thresholds ────────────────────────────────────────────

CONFIDENCE_THRESHOLD             = 0.25   # minimum mean token confidence
SINGLE_CHAR_CONFIDENCE_THRESHOLD = 0.50   # tighter threshold for 1-char results
MIN_LINE_H                       = 5      # px — skip lines shorter than this
MIN_LINE_W                       = 10     # px — skip lines narrower than this
SPARSE_LINE_WIDTH_RATIO          = 2.0    # width / (height * char count) above this

# TrOCR generation cap (tokens).  Without an explicit cap, long column lines can
# be truncated by the checkpoint's default generation length.
MAX_NEW_TOKENS = 192

# ── Recognition input (revision g) ────────────────────────────────────────────
# True  → body-text recognition crops come from the RAW page render with a
#         per-line autocontrast only (layout + line detection still use the
#         tile-preprocessed image).
# False → previous behaviour: recognition crops come from the preprocessed
#         page.  Kept as a switch so the two can be A/B tested.
# Header crops (HEADER_LABELS) always came from the raw render; unchanged.
RECOGNIZE_FROM_RAW       = True
LINE_AUTOCONTRAST_CUTOFF = 1       # cutoff % for the per-line autocontrast

# Recognition crops are expanded by this many px into the surrounding page
# (clamped to the page edge) so tight detector boxes don't clip ascenders,
# descenders or the first/last letter.  Only the image TrOCR sees is padded;
# element boxes, noise-filter geometry and the text layer use the unpadded box.
# Keep LINE_PAD_Y below the typical inter-line gap (a few px at 300 DPI on
# newsprint) or neighbouring lines start to intrude.
LINE_PAD_X = 6
LINE_PAD_Y = 3

# (h) How the LINE_PAD_X horizontal padding is filled:
#   "context" → expand into the real page (g behaviour);
#   "border"  → crop the line at its detected x-extent and add LINE_PAD_X px
#               of plain paper tone left and right, so a nearby column rule or
#               gutter ink can't be read as a leading 'I', '(' or '"'.
# Vertical padding (LINE_PAD_Y) always uses real page context.
LINE_PAD_X_MODE = "context"

# (h) Normalise TrOCR's IAM-style spacing around punctuation
# ('Moscow , Wednesday' → 'Moscow, Wednesday'; '( alas' → '(alas').
NORMALIZE_PUNCT_SPACING = True

# ── Selective beam search (revision g) ────────────────────────────────────────
# Every crop is decoded greedily first (explicit num_beams=1, whatever the
# checkpoint's generation defaults).  A beam retry happens only when ALL hold:
#   - greedy confidence >= max(BEAM_RETRY_ABOVE, the caller's noise floor)
#     (CONFIDENCE_THRESHOLD for regions, SWEEP_MIN_CONFIDENCE for the sweep),
#     so beam can never lift a read that was going to be discarded as noise
#     over the filter;
#   - greedy confidence <  BEAM_RETRY_BELOW;
#   - the greedy text has at least BEAM_MIN_CHARS non-space characters.
# The beam result is adopted only if its confidence is >= greedy's AND it
# differs in more than case / spacing / punctuation (see _trocr_read).
# The g run showed beam helping mostly for greedy confidence ~0.40–0.75 and
# mostly hurting at 0.80+.
BEAM_RETRY_ENABLED  = True
BEAM_NUM_BEAMS      = 5
BEAM_RETRY_BELOW    = 0.80
BEAM_RETRY_ABOVE    = 0.35
BEAM_MIN_CHARS      = 4
BEAM_LENGTH_PENALTY = 1.0

# (h) Device for TrOCR: "auto" = CUDA if present, else Apple MPS, else CPU.
# Or force "cuda" / "mps" / "cpu".
TROCR_DEVICE = "auto"

# ── Recall sweep (safety-net for text the layout model never boxed) ──────────
# After region OCR, the whole page is tiled with overlap; every detected text
# line that is not already covered by an existing element is OCR'd and added.
SWEEP_ENABLED        = True
SWEEP_TILE           = 1920   # px per tile side (300 DPI ≈ 6.4 in)
SWEEP_OVERLAP        = 960    # px; must exceed the longest line the sweep should recover
                              # (broadsheet columns at 300 DPI run up to ~715 px)
SWEEP_EDGE_MARGIN    = 6      # px; boxes touching an interior tile edge are ignored
SWEEP_COVERED_FRAC   = 0.50   # candidate counts as "already read" above this overlap
SWEEP_MIN_CONFIDENCE = 0.40   # stricter than CONFIDENCE_THRESHOLD: sweep also sees photos

# ── Column bands (give recovered lines a sensible reading position) ──────────
# Every non-page-wide layout region, of any height, claims its own x-range as
# a "column body" — see _column_bands().  A run of recovered lines centred in
# a genuinely unclaimed x-gap (a whole column the layout model skipped) is
# slotted into reading order between the columns on either side of it,
# instead of being attached to a neighbor.
COLUMN_REGION_MAX_W_FRAC = 0.40   # regions wider than this share of the page (mastheads) are ignored
COLUMN_GAP_MIN           = 150    # px; an x-gap at least this wide between claimed bands = unclaimed column

# A cluster of recovered lines in an unclaimed gap is only treated as a whole
# missed column (and relocated in reading order) if it clears BOTH of these —
# a handful of stray lines (a coverage false-negative, a marginal duplicate)
# stays at its original nearest-neighbor position instead, which is always
# at worst a harmless local duplicate rather than being moved to an unrelated
# part of the page.
MIN_UNCLAIMED_COLUMN_LINES        = 5      # fewer recovered lines than this are not relocated
MIN_UNCLAIMED_COLUMN_HEIGHT_FRAC  = 0.25   # cluster must span at least this fraction of the page height

# Same-position elements are grouped into a visual row only if consecutive
# lines' boxes overlap by more than this fraction of the smaller line's
# height, AND their x-overlap is below ROW_MAX_X_OVERLAP (genuinely side by
# side, not stacked) -- see _assign_visual_rows.
ROW_OVERLAP_FRACTION = 0.5
ROW_MAX_X_OVERLAP    = 0.5

# ── Banner/masthead reading-order correction ──────────────────────────────────

BANNER_BAND_OVERLAP_THRESHOLD = 0.18   # y-range overlap (as a fraction of the narrower
                                        # region's height) required to count as one band
BANNER_BAND_MIN_GAP_PX        = max(10, round(DPI * 0.06))  # px gap still counted as
                                                              # the same band (scales with DPI)
                                       # to resolve, instead of being repositioned next to it
                                       # (see the in-loop comment in _reorder_banner_regions)

SWEEP_COVERAGE_PAD = 6

# ── Coverage audit (diagnostic; never changes the output) ────────────────────

AUDIT_ENABLED     = True
AUDIT_SCALE       = 8      # analyse the page at 1/8 size
AUDIT_INK_LEVEL   = 215    # grey level (0-255) below which a reduced block counts as ink
AUDIT_WARN_FRAC   = 0.08   # WARN if more than this share of the ink is uncovered
AUDIT_MIN_BAND_PX = 150    # report uncovered vertical bands at least this wide

# ── Debug output ──────────────────────────────────────────────────────────────
# True  → one rolling set of debug files (latest_*.jpg, latest_*.txt),
#         overwritten by every page, so the debug folder does not grow with
#         batch size — covers the JPEGs AND the layout/OCR .txt reports.
# False → separate files for every file/page.
DEBUG_OVERWRITE = True

# ── Layout label taxonomy ─────────────────────────────────────────────────────

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
    "Table",             # stock quotes, box scores, schedules — full of text
    "Form",              # coupons, order forms, notices
}
SKIP_LABELS = {
    "Picture",   # photographs — sweep recovers any embedded text
    "Figure",    # diagrams / charts — sweep recovers any embedded text
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

# Labels treated as bold/stylized display type (headlines, masthead) rather
# than body copy — a subset of SINGLE_BLOCK_LABELS.  Caption/Footnote are
# normal-weight text at body-ish size and don't need the treatment below.
HEADER_LABELS = {
    "Section-header",
    "Page-header",
}

# "Page furniture" — headline/masthead-type content.  Used to break ties when

FURNITURE_LABELS = HEADER_LABELS | {"Page-footer", "Table-of-contents"}

DEDUPE_CONTAINMENT = 0.5

DEDUPE_MAX_AREA_RATIO = 3.0

# (h) In a duplicate pair where exactly one read is page furniture, the
# furniture read wins UNLESS the other read is more confident by more than
# this margin.
DEDUPE_FURNITURE_MARGIN = 0.15

HEADER_AUTOCONTRAST_CUTOFF = 1     # cutoff % for ImageOps.autocontrast

MAX_HEADER_AR         = 6.0    # width / height threshold that triggers splitting
HEADER_SEGMENT_OVERLAP = 0.20  # fraction of segment width shared with the next segment
# (h) Only crops at least this tall are segment-split — i.e. genuine display
# type such as the masthead.  Thin single-line strips (date line, most
# section heads) read better whole, as body lines do.  0.3 in at the render DPI.
HEADER_SPLIT_MIN_H     = round(DPI * 0.30)

ELEMENT_SEPARATOR = " "

# Minimum pixel dimensions for a region to bother processing.
MIN_REGION_W = 40
MIN_REGION_H = 15

# ── PDF/A font & colour-profile resources ─────────────────────────────────────
# EMBED_FONT = True  → hidden text uses FreeSans; PyMuPDF embeds it and, with
#                      fonttools installed, subsets it to the glyphs used.
#                      If the font file cannot be obtained the script falls back
#                      to PyMuPDF's built-in "helv" substitute, which TextWriter
#                      also embeds (bulkier: unsubset) — still PDF/A-safe.
#                      NEVER switch the text layer to page.insert_text(
#                      fontname="helv"): that leaves a non-embedded base-14
#                      reference, which PDF/A forbids.
EMBED_FONT    = True
MIN_FONT_PT   = 4.0    # smallest font size for the hidden text
FONT_NAME     = "helv"                       # fallback font only
FONT_PATH     = "fonts/FreeSans.ttf"
FONT_URL      = ("https://github.com/opensourcedesign/fonts/raw/master/"
                 "gnu-freefont_freesans/FreeSans.ttf")
SRGB_ICC_PATH = "srgb.icc"

# ── Text-layer geometry (revision g) ──────────────────────────────────────────
# Each line is stretched horizontally so its invisible text spans exactly the
# printed line's box.  The stretch factor is clamped to this range purely as a
# guard against degenerate input (an empty-looking string in a huge box); in
# normal output it never binds, because the noise filter already rejects
# strings far too short for their box (SPARSE_LINE_WIDTH_RATIO).
MIN_TEXT_STRETCH = 0.05
MAX_TEXT_STRETCH = 20.0

# ── Article threads (revision g) ──────────────────────────────────────────────
# Write Surya's reading order as a native PDF article thread: one thread per
# page, one bead per reading-order block (a layout region, or a relocated
# column of recovered lines), chained in reading order.  Any pre-existing
# threads in the input PDF are replaced.  Permitted in PDF/A-1b.
ARTICLE_THREADS_ENABLED = True

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
    "Recovered":         "#FFD600",   # lines found by the recall sweep
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

# Per-page TrOCR counters (reset in process_page, reported at the end of it).
_READ_STATS: Counter = Counter()


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

    Used for layout analysis and line DETECTION.  Since revision g, body-text
    RECOGNITION reads the raw render instead (see RECOGNIZE_FROM_RAW): the
    tiles are pasted without blending, leaving contrast steps every 128 px
    that a full column line crosses several times.
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


def preprocess_header_crop(crop: Image.Image) -> Image.Image:
    """
    Light-touch preprocessing for a bold/stylized header crop (Section-header,
    Page-header — see HEADER_LABELS).

    preprocess_newspaper()'s unsharp mask is tuned to bring out thin body-text
    serifs; applied to already-thick bold headline strokes it tends to
    over-sharpen and blob adjacent strokes together, which hurts recognition
    rather than helping it.  This crops from the ORIGINAL render (never the
    globally-unsharpened page) and applies autocontrast only.
    """
    try:
        gray = crop.convert("L")
        gray = ImageOps.autocontrast(gray, cutoff=HEADER_AUTOCONTRAST_CUTOFF)
        return gray.convert("RGB")
    except Exception as exc:
        logger.warning(f"Header preprocessing fallback: {exc}")
        return crop.convert("RGB")


def preprocess_line_crop(crop: Image.Image) -> Image.Image:
    """
    Recognition preprocessing for a single body-text line cut from the RAW
    render: one autocontrast over the whole line (no tiling, so no seams, and
    no unsharp mask, which TrOCR — trained on natural-looking text images —
    does not benefit from).
    """
    try:
        gray = crop.convert("L")
        gray = ImageOps.autocontrast(gray, cutoff=LINE_AUTOCONTRAST_CUTOFF)
        return gray.convert("RGB")
    except Exception as exc:
        logger.warning(f"Line preprocessing fallback: {exc}")
        return crop.convert("RGB")


def _passthrough_crop(crop: Image.Image) -> Image.Image:
    """Recognition 'preprocessing' for crops already cut from the preprocessed page."""
    return crop.convert("RGB")


def _recognition_crop(source: Image.Image, box: List[float], prep) -> Image.Image:
    """
    Cut the image TrOCR will read for `box` ([x0, y0, x1, y1], absolute page
    pixels) out of `source`, expanded by LINE_PAD_X / LINE_PAD_Y into the
    surrounding page (clamped to its edges), then apply `prep`.
    (h) With LINE_PAD_X_MODE = "border" the horizontal padding is plain paper
    tone added after `prep` instead of real page content.
    """
    iw, ih = source.size
    border = LINE_PAD_X_MODE == "border"                      # (h)
    pad_x  = 0 if border else LINE_PAD_X
    x0 = max(0,  int(box[0]) - pad_x)
    y0 = max(0,  int(box[1]) - LINE_PAD_Y)
    x1 = min(iw, int(box[2]) + pad_x)
    y1 = min(ih, int(box[3]) + LINE_PAD_Y)
    img = prep(source.crop((x0, y0, x1, y1)))
    if border and LINE_PAD_X > 0:
        img = ImageOps.expand(img, border=(LINE_PAD_X, 0, LINE_PAD_X, 0),
                              fill=_paper_tone(img))
    return img


def _paper_tone(img: Image.Image) -> Tuple[int, int, int]:
    """(h) Background colour of a text crop: its 90th-percentile grey level."""
    hist   = img.convert("L").histogram()
    target = 0.90 * sum(hist)
    acc    = 0
    for level, n in enumerate(hist):
        acc += n
        if acc >= target:
            return (level, level, level)
    return (255, 255, 255)


def page_to_pil(page: "fitz.Page", dpi: int = DPI) -> Image.Image:
    """Render one PDF page at `dpi` to an RGB PIL image (OCR input only)."""
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


# ══════════════════════════════════════════════════════════════════════════════
# PDF/A  +  ACCESSIBILITY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_pdf_date_string(dt: Optional[datetime.datetime] = None) -> str:
    dt = dt or datetime.datetime.now()
    return dt.strftime("D:%Y%m%d%H%M%S")


def get_xmp_date_string(dt: Optional[datetime.datetime] = None) -> str:
    dt = dt or datetime.datetime.now()
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _valid_icc(path: Path) -> bool:
    """True if `path` looks like an RGB ICC profile (checks the 'acsp' signature)."""
    try:
        data = path.read_bytes()
        return len(data) > 128 and data[36:40] == b"acsp" and data[16:20] == b"RGB "
    except Exception:
        return False


def setup_pdfa_resources() -> bool:
    """
    Make sure the two PDF/A resources exist locally:

      fonts/FreeSans.ttf  — downloaded once (embedded into every output PDF)
      srgb.icc            — copied from the OS if present, otherwise generated
                            locally with Pillow (no download).  An existing
                            srgb.icc that is not a real ICC profile (e.g. an
                            HTML page saved by an earlier version of this
                            tool) is detected and replaced.
    """
    ok = True

    # ── Font ──────────────────────────────────────────────────────────────────
    try:
        Path(FONT_PATH).parent.mkdir(parents=True, exist_ok=True)
        if not Path(FONT_PATH).exists():
            import urllib.request
            logger.info("  Fetching FreeSans.ttf …")
            part = FONT_PATH + ".part"
            urllib.request.urlretrieve(FONT_URL, part)
            os.replace(part, FONT_PATH)
    except Exception as exc:
        logger.warning(f"  Could not obtain FreeSans font: {exc}")
        ok = False

    # ── sRGB ICC profile ──────────────────────────────────────────────────────
    try:
        icc = Path(SRGB_ICC_PATH)
        if icc.exists() and not _valid_icc(icc):
            logger.warning(f"  {SRGB_ICC_PATH} is not a valid ICC profile — regenerating.")
            icc.unlink()
        if not icc.exists():
            candidates = {
                "Darwin":  ["/System/Library/ColorSync/Profiles/sRGB Profile.icc"],
                "Linux":   ["/usr/share/color/icc/sRGB.icc",
                            "/usr/share/color/icc/colord/sRGB.icc"],
                "Windows": [os.path.join(
                    os.environ.get("WINDIR", r"C:\Windows"),
                    "System32", "spool", "drivers", "color",
                    "sRGB Color Space Profile.icm")],
            }.get(platform.system(), [])
            source = next((c for c in candidates
                           if Path(c).exists() and _valid_icc(Path(c))), None)
            if source:
                shutil.copy2(source, icc)
                logger.info(f"  sRGB profile copied from {source}")
            else:
                icc.write_bytes(
                    ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
                )
                logger.info("  sRGB profile generated locally (Pillow/LittleCMS).")
    except Exception as exc:
        logger.warning(f"  Could not obtain sRGB ICC profile: {exc}")
        ok = False

    return ok


def load_text_font() -> Tuple["fitz.Font", bool]:
    """
    Return (font, using_freesans).  With FreeSans, PyMuPDF embeds the font
    file (and it can be subset).  Otherwise PyMuPDF's built-in "helv"
    substitute is used; TextWriter embeds that too, so the output stays
    PDF/A-safe, just slightly larger.
    """
    if EMBED_FONT and Path(FONT_PATH).exists():
        try:
            return fitz.Font(fontfile=FONT_PATH), True
        except Exception as exc:
            logger.warning(f"  Could not load {FONT_PATH}: {exc}")
    logger.warning(
        "  FreeSans unavailable — using PyMuPDF's built-in font "
        "(embedded, but unsubset, so the output is a little larger)."
    )
    return fitz.Font(FONT_NAME), False


def create_xmp_metadata(
    title: str,
    author: str,
    subject: str,
    creator: str,
    producer: str,
    creation_date: str,
    modify_date: str,
    language: str = DOC_LANGUAGE,
) -> Optional[str]:
    """
    Build the XMP packet: Dublin Core (title / creator / description /
    language), PDF producer, XMP dates, PDF/A-1b identification, and the
    custom opt: properties together with the PDF/A extension schema that
    declares them.  All text values are XML-escaped.
    """
    try:
        t, a, s, c, p, lang = (
            xml_escape(str(v)) for v in (title, author, subject, creator, producer, language)
        )
        app, ver = xml_escape(APP_NAME), xml_escape(APP_VERSION)
        ns = xml_escape(OPT_NAMESPACE)
        # NB: "\ufeff" (the real BOM character) is required in the xpacket header.
        return f"""<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about="" xmlns:pdf="http://ns.adobe.com/pdf/1.3/">
      <pdf:Producer>{p}</pdf:Producer>
    </rdf:Description>
    <rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">
      <dc:title><rdf:Alt><rdf:li xml:lang="x-default">{t}</rdf:li></rdf:Alt></dc:title>
      <dc:creator><rdf:Seq><rdf:li>{a}</rdf:li></rdf:Seq></dc:creator>
      <dc:description><rdf:Alt><rdf:li xml:lang="x-default">{s}</rdf:li></rdf:Alt></dc:description>
      <dc:language><rdf:Bag><rdf:li>{lang}</rdf:li></rdf:Bag></dc:language>
    </rdf:Description>
    <rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/">
      <xmp:CreatorTool>{c}</xmp:CreatorTool>
      <xmp:CreateDate>{creation_date}</xmp:CreateDate>
      <xmp:ModifyDate>{modify_date}</xmp:ModifyDate>
    </rdf:Description>
    <rdf:Description rdf:about="" xmlns:pdfaid="http://www.aiim.org/pdfa/ns/id/">
      <pdfaid:part>1</pdfaid:part>
      <pdfaid:conformance>B</pdfaid:conformance>
    </rdf:Description>
    <rdf:Description rdf:about="" xmlns:opt="{ns}">
      <opt:ToolName>{app}</opt:ToolName>
      <opt:Version>{ver}</opt:Version>
    </rdf:Description>
    <rdf:Description rdf:about=""
        xmlns:pdfaExtension="http://www.aiim.org/pdfa/ns/extension/"
        xmlns:pdfaSchema="http://www.aiim.org/pdfa/ns/schema#"
        xmlns:pdfaProperty="http://www.aiim.org/pdfa/ns/property#">
      <pdfaExtension:schemas>
        <rdf:Bag>
          <rdf:li rdf:parseType="Resource">
            <pdfaSchema:schema>{app} processing metadata</pdfaSchema:schema>
            <pdfaSchema:namespaceURI>{ns}</pdfaSchema:namespaceURI>
            <pdfaSchema:prefix>opt</pdfaSchema:prefix>
            <pdfaSchema:property>
              <rdf:Seq>
                <rdf:li rdf:parseType="Resource">
                  <pdfaProperty:name>ToolName</pdfaProperty:name>
                  <pdfaProperty:valueType>Text</pdfaProperty:valueType>
                  <pdfaProperty:category>internal</pdfaProperty:category>
                  <pdfaProperty:description>Name of the tool that produced the OCR text layer</pdfaProperty:description>
                </rdf:li>
                <rdf:li rdf:parseType="Resource">
                  <pdfaProperty:name>Version</pdfaProperty:name>
                  <pdfaProperty:valueType>Text</pdfaProperty:valueType>
                  <pdfaProperty:category>internal</pdfaProperty:category>
                  <pdfaProperty:description>Version of the tool that produced the OCR text layer</pdfaProperty:description>
                </rdf:li>
              </rdf:Seq>
            </pdfaSchema:property>
          </rdf:li>
        </rdf:Bag>
      </pdfaExtension:schemas>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""
    except Exception as exc:
        logger.error(f"Failed to create XMP metadata: {exc}")
        return None


def apply_document_metadata(doc: "fitz.Document", filename: str) -> None:
    """
    Write Info-dict metadata, the XMP packet and the accessibility catalog
    entries (/Lang, /ViewerPreferences /DisplayDocTitle).  The Info dict and
    XMP carry identical title / author / producer / dates, as PDF/A requires.
    """
    now      = datetime.datetime.now()
    pdf_date = get_pdf_date_string(now)
    xmp_date = get_xmp_date_string(now)
    producer = f"PyMuPDF {fitz.VersionBind}"

    doc.set_metadata({
        "title":        filename,
        "author":       APP_NAME,
        "subject":      DOC_SUBJECT,
        "creator":      APP_CREATOR,
        "producer":     producer,
        "creationDate": pdf_date,
        "modDate":      pdf_date,
    })

    xmp = create_xmp_metadata(
        title=filename, author=APP_NAME, subject=DOC_SUBJECT,
        creator=APP_CREATOR, producer=producer,
        creation_date=xmp_date, modify_date=xmp_date,
        language=DOC_LANGUAGE,
    )
    if xmp:
        doc.set_xml_metadata(xmp)

    # Accessibility: document language + show the title (not the file name)
    # in the viewer's title bar.
    cat = doc.pdf_catalog()
    doc.xref_set_key(cat, "ViewerPreferences", "<</DisplayDocTitle true>>")
    doc.xref_set_key(cat, "Lang", f"({DOC_LANGUAGE})")


# ══════════════════════════════════════════════════════════════════════════════
# ARTICLE THREADS  (Surya reading order as a native PDF feature)
# ══════════════════════════════════════════════════════════════════════════════

def _article_bead_boxes(
    elements: List[Dict],
    layout_regions: List[Dict],
    img_size: Tuple[int, int],
) -> List[Tuple[float, float, float, float]]:
    """
    One bead rectangle per reading-order block, in final reading order, as
    fractions (0–1) of the rendered page.

    `elements` must already be in final reading order.  Elements are grouped
    by reading_position; a group whose position belongs to a layout region
    uses that region's box (Surya's own block outline), otherwise — a
    relocated column of recovered lines — the union of the group's line
    boxes.  Regions that yielded no text get no bead.  Fractions (rather than
    points) are stored because the conversion to PDF user space is done
    later against the page's CropBox and /Rotate (see _frac_to_pdf_rect).
    """
    iw, ih = img_size
    region_boxes: Dict[float, List[List[float]]] = {}
    for r in layout_regions:
        if r["label"] not in SKIP_LABELS:
            region_boxes.setdefault(r["position"], []).append(r["bbox"])

    groups: Dict[float, List[List[float]]] = {}      # insertion order = reading order
    for e in elements:
        groups.setdefault(e["reading_position"], []).append(e["bbox"])

    beads: List[Tuple[float, float, float, float]] = []
    for pos, elem_boxes in groups.items():
        boxes = region_boxes.get(pos) or elem_boxes
        x0 = max(0.0, min(b[0] for b in boxes)); y0 = max(0.0, min(b[1] for b in boxes))
        x1 = min(iw,  max(b[2] for b in boxes)); y1 = min(ih,  max(b[3] for b in boxes))
        if x1 - x0 < MIN_LINE_W or y1 - y0 < MIN_LINE_H:
            continue
        beads.append((x0 / iw, y0 / ih, x1 / iw, y1 / ih))
    return beads


def _frac_to_pdf_rect(
    frac: Tuple[float, float, float, float],
    cropbox: List[float],
    rotation: int,
) -> List[float]:
    """
    Convert a box given as fractions of the DISPLAYED (rotated, cropped) page —
    exactly what the 300 DPI render shows — into PDF user-space coordinates
    [x0, y0, x1, y1] (origin bottom-left), using the page's CropBox and
    /Rotate.  Written out explicitly because PyMuPDF's derotation /
    transformation matrices do not combine correctly for rotated pages whose
    CropBox is offset from the MediaBox origin (verified by rendering).
    """
    cx0, cy0, cx1, cy1 = cropbox
    W, H = cx1 - cx0, cy1 - cy0
    rot = rotation % 360
    DW, DH = (H, W) if rot in (90, 270) else (W, H)
    pts = []
    for fx, fy in ((frac[0], frac[1]), (frac[2], frac[3])):
        X, Y = fx * DW, fy * DH
        if   rot == 0:   u, v = X, Y
        elif rot == 90:  u, v = Y, H - X
        elif rot == 180: u, v = W - X, H - Y
        else:            u, v = W - Y, X
        pts.append((cx0 + u, cy1 - v))
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def _effective_cropbox(pg: "pikepdf.Page") -> List[float]:
    """The page's CropBox (falling back to MediaBox), clipped to the MediaBox, normalised."""
    def norm(box) -> List[float]:
        v = [float(x) for x in box]
        return [min(v[0], v[2]), min(v[1], v[3]), max(v[0], v[2]), max(v[1], v[3])]
    mb = norm(pg.mediabox)
    try:
        cb = norm(pg.cropbox)
    except Exception:
        cb = mb
    return [max(cb[0], mb[0]), max(cb[1], mb[1]), min(cb[2], mb[2]), min(cb[3], mb[3])]


def add_article_threads(pdf: "pikepdf.Pdf", thread_specs: Dict[int, Dict]) -> int:
    """
    Write one article thread per page into an open pikepdf document.

    thread_specs: {page_index: {"title": str, "rotation": int,
                                "boxes": [(fx0, fy0, fx1, fy1), ...]}}

    Structure (ISO 32000-1 §12.4.3):
      Catalog /Threads  → [thread, ...]
      thread            → /Type /Thread  /F first-bead  /I << /Title … >>
      bead              → /Type /Bead  /P page  /R rect  /N next  /V previous
                          (circular list; the first bead also carries /T thread)
      page /B           → [bead, ...] in reading order

    Any threads and page /B arrays already in the file are removed first so
    stale threads from an earlier OCR pass never survive.  Returns the number
    of threads written.
    """
    root = pdf.Root
    if "/Threads" in root:
        logger.info("  Replacing pre-existing article threads.")
        del root["/Threads"]
    for pg in pdf.pages:
        if "/B" in pg.obj:
            del pg.obj["/B"]

    threads = pikepdf.Array()
    for idx in sorted(thread_specs):
        spec  = thread_specs[idx]
        boxes = spec.get("boxes") or []
        if not boxes or idx >= len(pdf.pages):
            continue
        pg      = pdf.pages[idx]
        cropbox = _effective_cropbox(pg)

        thread = pdf.make_indirect(pikepdf.Dictionary({
            "/Type": pikepdf.Name("/Thread"),
            "/I":    pikepdf.Dictionary({"/Title": pikepdf.String(spec["title"])}),
        }))
        beads = [
            pdf.make_indirect(pikepdf.Dictionary({
                "/Type": pikepdf.Name("/Bead"),
                "/P":    pg.obj,
                "/R":    pikepdf.Array([round(v, 2) for v in
                                        _frac_to_pdf_rect(b, cropbox, spec.get("rotation", 0))]),
            }))
            for b in boxes
        ]
        n = len(beads)
        for k, bead in enumerate(beads):
            bead["/N"] = beads[(k + 1) % n]
            bead["/V"] = beads[k - 1]
        beads[0]["/T"] = thread
        thread["/F"]   = beads[0]
        pg.obj["/B"]   = pikepdf.Array(beads)
        threads.append(thread)

    if len(threads):
        root["/Threads"] = threads
    return len(threads)


def setup_pdfa_compliance(pdf_path: str, thread_specs: Optional[Dict[int, Dict]] = None) -> None:
    """
    Post-save pass with pikepdf.  MUST run after the file is on disk.

      - embeds the sRGB OutputIntent (skipped if a GTS_PDFA1 intent exists);
      - writes the article threads (ARTICLE_THREADS_ENABLED, see
        add_article_threads);
      - object streams are disabled on save (not permitted in PDF/A-1).
    """
    try:
        icc = Path(SRGB_ICC_PATH)
        icc_ok = icc.exists() and _valid_icc(icc)
        if not icc_ok:
            logger.warning("  Valid sRGB ICC profile not found; PDF/A OutputIntent skipped.")
        with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
            if icc_ok:
                existing = pdf.Root.get("/OutputIntents")
                has_pdfa_intent = existing is not None and any(
                    str(oi.get("/S", "")) == "/GTS_PDFA1" for oi in existing
                )
                if has_pdfa_intent:
                    logger.info("  PDF/A OutputIntent already present — leaving as is.")
                else:
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
                    logger.info("  PDF/A OutputIntent embedded.")

            if ARTICLE_THREADS_ENABLED and thread_specs:
                try:
                    n_threads = add_article_threads(pdf, thread_specs)
                    n_beads   = sum(len(s.get("boxes") or []) for s in thread_specs.values())
                    logger.info(f"  Article threads: {n_threads} thread(s), {n_beads} bead(s).")
                except Exception as exc:
                    logger.error(f"  Article threads skipped: {exc}")

            pdf.save(pdf_path, object_stream_mode=pikepdf.ObjectStreamMode.disable)
    except Exception as exc:
        logger.error(f"  Failed to set up PDF/A compliance: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _select_device() -> "torch.device":
    """(h) TROCR_DEVICE, or for "auto": CUDA → Apple MPS → CPU."""
    want = str(TROCR_DEVICE).lower()
    mps_ok = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    if want == "cuda" or (want == "auto" and torch.cuda.is_available()):
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("  TROCR_DEVICE='cuda' but CUDA is unavailable.")
    if want == "mps" or (want == "auto" and mps_ok):
        if mps_ok:
            return torch.device("mps")
        logger.warning("  TROCR_DEVICE='mps' but MPS is unavailable.")
    return torch.device("cpu")


def _device_smoke_test(processor: "TrOCRProcessor", model: "VisionEncoderDecoderModel") -> bool:
    """
    (h) Run one greedy and one beam decode on a blank image on the model's
    current device.  _trocr_read swallows exceptions (returning an empty
    read), so without this a device that can't run generate() would silently
    produce an empty text layer instead of an error.
    """
    try:
        pv = processor(Image.new("RGB", (384, 64), "white"),
                       return_tensors="pt").pixel_values.to(next(model.parameters()).device)
        _decode(pv, processor, model, num_beams=1)
        if BEAM_RETRY_ENABLED and BEAM_NUM_BEAMS > 1:
            _decode(pv, processor, model, num_beams=BEAM_NUM_BEAMS)
        return True
    except Exception as exc:
        logger.warning(f"  Device smoke test failed: {exc}")
        return False


def load_models():
    """
    Load all models:

    Surya  — LayoutPredictor (layout backbone) + DetectionPredictor (line segmentation)
    TrOCR  — TrOCRProcessor + VisionEncoderDecoderModel (text recognition)

    RecognitionPredictor is NOT loaded in this edition; TrOCR replaces it
    entirely for the character-recognition step.

    Layout path (per Surya docs):
        LayoutPredictor(FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT))

    Environment variables tuned here for historic newsprint:
        LAYOUT_BATCH_SIZE    — controls GPU/CPU memory per batch item (~220 MB VRAM each).
                               Affects memory and throughput, not recall.
        DETECTOR_BATCH_SIZE  — same principle for the line-detection pass.
        RECOGNITION_BATCH_SIZE — not used (TrOCR handles recognition), but set conservatively
                               to avoid surprise if Surya initialises an internal recogniser.
    """
    logger.info("=" * 62)
    logger.info("  LOADING MODELS  (Surya Layout + TrOCR Recognition)")
    logger.info("=" * 62)

    # ── Surya performance / memory tuning ─────────────────────────────────────
    # These are set with setdefault so a caller can still override via env.
    os.environ.setdefault("LAYOUT_BATCH_SIZE",      "4")   # was 32 default
    os.environ.setdefault("DETECTOR_BATCH_SIZE",    "4")   # was 36 default
    os.environ.setdefault("RECOGNITION_BATCH_SIZE", "8")   # unused but safe

    # Inform Surya settings object of the updated values so anything that reads
    # settings.LAYOUT_BATCH_SIZE at import time picks up the override.
    try:
        if hasattr(settings, "LAYOUT_BATCH_SIZE"):
            settings.LAYOUT_BATCH_SIZE   = int(os.environ["LAYOUT_BATCH_SIZE"])
        if hasattr(settings, "DETECTOR_BATCH_SIZE"):
            settings.DETECTOR_BATCH_SIZE = int(os.environ["DETECTOR_BATCH_SIZE"])
    except Exception as exc:
        logger.warning(f"  Could not patch Surya settings object: {exc}")

    if not setup_pdfa_resources():
        logger.warning("  PDF/A resources setup incomplete.")

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
    device          = _select_device()
    trocr_model.to(device)
    trocr_model.eval()
    if device.type != "cpu" and not _device_smoke_test(trocr_processor, trocr_model):
        logger.warning(f"  TrOCR generation failed on {device}; falling back to CPU.")
        device = torch.device("cpu")
        trocr_model.to(device)
    logger.info(f"  TrOCR device: {device}")

    logger.info("  All models ready.\n")
    return det_predictor, layout_predictor, trocr_processor, trocr_model


# ══════════════════════════════════════════════════════════════════════════════
# LABEL NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

# The Surya docs specify hyphenated label strings ("Section-header", etc.).
# Some installed versions emit camelCase ("SectionHeader") or other variants.
# This map converts every known variant to the canonical documented form.
_LABEL_ALIAS: Dict[str, str] = {
    # camelCase variants emitted by some Surya builds
    "SectionHeader":     "Section-header",
    "PageHeader":        "Page-header",
    "PageFooter":        "Page-footer",
    "ListItem":          "List-item",
    "TableOfContents":   "Table-of-contents",
    "InlineMath":        "Text-inline-math",
    "TextInlineMath":    "Text-inline-math",
    # legacy / alias variants
    "Header":            "Page-header",
    "Footer":            "Page-footer",
    "Heading":           "Section-header",
    "Title":             "Section-header",
    "Handwritten":       "Handwriting",
    # snake_case variants observed in Surya ≥0.6 on document-heavy checkpoints
    "section_header":    "Section-header",
    "page_header":       "Page-header",
    "page_footer":       "Page-footer",
    "list_item":         "List-item",
    "table_of_contents": "Table-of-contents",
    "inline_math":       "Text-inline-math",
    "text_inline_math":  "Text-inline-math",
    "handwriting":       "Handwriting",
    # caption aliases from fine-tuned checkpoints
    "figure_caption":    "Caption",
    "FigureCaption":     "Caption",
    "table_caption":     "Caption",
    "TableCaption":      "Caption",
    # body-text synonyms
    "paragraph":         "Text",
    "Paragraph":         "Text",
    "body":              "Text",
    "Body":              "Text",
}

# Case- and underscore-insensitive lookup onto the canonical labels, so
# "Page-Header", "SECTION-HEADER" and "Table_Of_Contents" all resolve.
_CANON_LABELS: Dict[str, str] = {
    c.lower().replace("_", "-"): c for c in (OCR_LABELS | SKIP_LABELS)
}


def _normalise_label(raw: str) -> str:
    if raw in _LABEL_ALIAS:
        return _LABEL_ALIAS[raw]
    return _CANON_LABELS.get(str(raw).lower().replace("_", "-"), raw)


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _iou(a: List[float], b: List[float]) -> float:
    """Intersection-over-Union for two [x1, y1, x2, y2] boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _nms_regions(regions: List[Dict], iou_threshold: float = 0.45) -> List[Dict]:
    """
    Non-maximum suppression over layout regions.

    Removes duplicate / heavily overlapping boxes that can appear when the
    layout model fires twice on the same column strip (common on wide
    historic broadsheets where the model sees partial columns near tile
    boundaries in its internal attention window).

    Regions are sorted by area descending so the largest (most complete)
    box wins when two boxes of the SAME LABEL substantially overlap.
    Boxes with different labels are never suppressed against each other:
    a Picture box must not be able to delete an overlapping Text box.
    """
    if not regions:
        return regions
    sorted_r = sorted(
        regions,
        key=lambda r: (r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1]),
        reverse=True,
    )
    kept: List[Dict] = []
    suppressed: set  = set()
    for i, r in enumerate(sorted_r):
        if i in suppressed:
            continue
        kept.append(r)
        for j in range(i + 1, len(sorted_r)):
            if j in suppressed:
                continue
            if (sorted_r[j]["label"] == r["label"]
                    and _iou(r["bbox"], sorted_r[j]["bbox"]) > iou_threshold):
                suppressed.add(j)
    return kept


def _band_merge_test(lo: float, hi: float, glo: float, ghi: float,
                      overlap_threshold: float, min_gap: float) -> bool:
    """
    True if the interval [lo,hi] belongs in the same band as [glo,ghi]:
    either they overlap by more than `overlap_threshold` of the narrower
    interval's length, or they don't overlap but the gap between them is
    smaller than `min_gap` (too small to trust as a real separation — a
    couple of pixels of detector jitter, not a genuine masthead/body gap).
    """
    overlap_val = min(hi, ghi) - max(lo, glo)
    if overlap_val > 0:
        narrower = min(hi - lo, ghi - glo)
        return (overlap_val / narrower) > overlap_threshold if narrower > 0 else False
    return (-overlap_val) < min_gap


def _group_by_y_band(
    regions: List[Dict],
    overlap_threshold: float = BANNER_BAND_OVERLAP_THRESHOLD,
    min_gap: float = BANNER_BAND_MIN_GAP_PX,
) -> List[Tuple[float, float, List[Dict]]]:
    """
    Single-linkage clustering of `regions` into horizontal y-bands.

    Two regions land in the same band if their y-ranges overlap by more than
    `overlap_threshold` of the narrower one's height, or are separated by
    less than `min_gap` px. A genuine masthead/body gap runs to hundreds of
    px on a real scan -- far larger than either threshold -- so this
    reliably separates a shared top band (a masthead, spanning many
    x-positions) from the tall band of column-body content below it.

    Returns [(band_y0, band_y1, [region, ...]), ...] sorted top to bottom.
    """
    items = sorted(regions, key=lambda r: r["bbox"][1])
    groups: List[List] = []
    for r in items:
        y0, y1 = r["bbox"][1], r["bbox"][3]
        placed = False
        for g in groups:
            if _band_merge_test(y0, y1, g[0], g[1], overlap_threshold, min_gap):
                g[0] = min(g[0], y0); g[1] = max(g[1], y1); g[2].append(r)
                placed = True
                break
        if not placed:
            groups.append([y0, y1, [r]])

    # Cleanup pass: a region processed out of y0 order can retroactively
    # bridge two groups that were formed before it was seen. Repeat until
    # no further merges happen.
    changed = True
    while changed and len(groups) > 1:
        changed = False
        groups.sort(key=lambda g: g[0])
        merged = [groups[0]]
        for g in groups[1:]:
            last = merged[-1]
            if _band_merge_test(g[0], g[1], last[0], last[1], overlap_threshold, min_gap):
                last[0] = min(last[0], g[0]); last[1] = max(last[1], g[1]); last[2].extend(g[2])
                changed = True
            else:
                merged.append(g)
        groups = merged

    groups.sort(key=lambda g: g[0])
    return [(g[0], g[1], g[2]) for g in groups]


def _reorder_banner_regions(layout_regions: List[Dict]) -> None:
    """
    Correct Surya's own `position` for masthead/banner regions it
    occasionally mis-scores (see the BANNER_BAND_* constants above).

    Finds the page's shared top masthead band via _group_by_y_band() (the
    band sitting entirely above the widest band, the main body of column
    content), then re-anchors only the masthead-band members whose position
    doesn't already sort before the body. Every other region -- including
    every column's own Section-header -- is left exactly as Surya gave it.
    Mutates `position` in place on the affected region dicts.
    """
    if len(layout_regions) < 2:
        return

    bands = _group_by_y_band(layout_regions)
    if len(bands) < 2:
        return   # everything is one block -- nothing to separate out

    body_y0, body_y1, body_members = max(bands, key=lambda b: len(b[2]))

    banner_members: List[Dict] = []
    for y0, y1, members in bands:
        if members is body_members:
            continue
        # A band counts as masthead only if it sits entirely above the body
        # band AND contains at least one header-type region -- any(), not
        # all(), because Surya's label for a borderline region can flip
        # between runs, and the geometric isolation test is already the
        # load-bearing signal.
        if y1 <= body_y0 and any(r["label"] in HEADER_LABELS for r in members):
            banner_members.extend(members)

    if not banner_members:
        return

    floor = min(r["position"] for r in body_members)
    correctly_placed = [r for r in banner_members if r["position"] < floor]
    misplaced        = [r for r in banner_members if r["position"] >= floor]
    if not misplaced:
        return   # Surya already placed every masthead region before the body

    if not correctly_placed:
        # No already-correct masthead region to anchor against -- order the
        # whole masthead band by x0 instead, still strictly before the body.
        for i, r in enumerate(sorted(banner_members, key=lambda r: r["bbox"][0])):
            old = r["position"]
            r["position"] = floor - len(banner_members) + i
            logger.info(
                f"  [ORDER] moved {r['label']} bbox=({r['bbox'][0]:.0f},{r['bbox'][1]:.0f}"
                f"→{r['bbox'][2]:.0f},{r['bbox'][3]:.0f}) from position {old} to "
                f"{r['position']} (top masthead band, x0 order)."
            )
        return

    for r in misplaced:
        cx = (r["bbox"][0] + r["bbox"][2]) / 2
        nearest = min(
            correctly_placed,
            key=lambda c: abs((c["bbox"][0] + c["bbox"][2]) / 2 - cx),
        )
        old = r["position"]
        r["position"] = nearest["position"] + 0.5
        logger.info(
            f"  [ORDER] moved {r['label']} bbox=({r['bbox'][0]:.0f},{r['bbox'][1]:.0f}"
            f"→{r['bbox'][2]:.0f},{r['bbox'][3]:.0f}) from position {old} to "
            f"{r['position']} -- belongs in the top masthead band beside the "
            f"nearest masthead piece (pos={nearest['position']}), not the column flow."
        )


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_layout_result(
    result,
    orig_image_size: Optional[Tuple[int, int]] = None,
) -> Tuple[List[Dict], Optional[List[float]]]:
    """
    Normalise one LayoutPredictor page result.

    Surya LayoutPredictor output per page (docs):
        result.bboxes     — list of bbox objects
        result.image_bbox — [x1, y1, x2, y2] coordinate space of the model's
                            internal representation of the page image.
                            IMPORTANT: When Surya internally resizes a high-DPI
                            scan the returned bbox coordinates are in the
                            *resized* model space, NOT in original pixel space.
                            We must scale them back using image_bbox vs
                            orig_image_size.
        result.page       — 0-based page number

    Each bbox object:
        .bbox     [x1, y1, x2, y2]
        .polygon  [(x1,y1)…(x4,y4)] clockwise from top-left
        .position int   reading-order index (column-aware, model-native)
        .label    str   e.g. "Text", "SectionHeader", "Section-header" …
                  NOTE: may be camelCase; normalised via _normalise_label()
        .top_k    dict  {label: confidence} for OTHER candidate labels only
                  (the primary label is NOT in top_k — it is the argmax)

    Args:
        result:           One element of the list returned by layout_predictor([img]).
        orig_image_size:  (width, height) of the PIL image passed to the predictor.
                          Required for accurate coordinate rescaling on high-DPI input.

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

    # ── Coordinate-space rescaling ────────────────────────────────────────────
    # Surya may internally resize the input image before running the layout
    # backbone (e.g. to a fixed 1024-px long edge).  When it does, the returned
    # bboxes are in the *model's* pixel space.  image_bbox tells us how large
    # that space is; orig_image_size tells us the actual dimensions.
    #
    # If image_bbox == [0, 0, orig_w, orig_h] the two spaces match and sx=sy=1.
    # If they differ (common at 300 DPI on large broadsheet pages) we rescale.
    sx = sy = 1.0
    if image_bbox and orig_image_size:
        ib_w = image_bbox[2] - image_bbox[0]
        ib_h = image_bbox[3] - image_bbox[1]
        if ib_w > 0 and ib_h > 0:
            sx = orig_image_size[0] / ib_w
            sy = orig_image_size[1] / ib_h
            if abs(sx - 1.0) > 0.01 or abs(sy - 1.0) > 0.01:
                logger.debug(
                    f"  [LAYOUT] coordinate rescale  "
                    f"image_bbox={[round(v) for v in image_bbox]}  "
                    f"orig={orig_image_size}  sx={sx:.3f} sy={sy:.3f}"
                )

    def _scale_bbox(b: List[float]) -> List[float]:
        return [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]

    def _scale_polygon(poly) -> Optional[List[Tuple[float, float]]]:
        if not poly:
            return None
        return [(float(p[0]) * sx, float(p[1]) * sy) for p in poly]

    # ── Parse each bbox object ────────────────────────────────────────────────
    for box in result.bboxes:
        if hasattr(box, "bbox") and box.bbox:
            bbox = _scale_bbox([float(v) for v in box.bbox])
        elif hasattr(box, "polygon") and len(box.polygon) >= 4:
            xs   = [float(p[0]) for p in box.polygon]
            ys   = [float(p[1]) for p in box.polygon]
            bbox = _scale_bbox([min(xs), min(ys), max(xs), max(ys)])
        else:
            continue

        # Sanity-check: skip degenerate boxes (can appear on aged torn edges)
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        if bw < MIN_REGION_W or bh < MIN_REGION_H:
            continue

        polygon   = _scale_polygon(getattr(box, "polygon", None))
        raw_label = getattr(box, "label", "Text")
        label     = _normalise_label(raw_label)
        top_k     = {
            _normalise_label(k): v
            for k, v in (getattr(box, "top_k", {}) or {}).items()
        }
        position  = int(getattr(box, "position", 0))

        regions.append({
            "bbox":     bbox,
            "polygon":  polygon,
            "label":    label,
            "position": position,
            "top_k":    top_k,
        })

    # ── Deduplicate overlapping same-label detections ─────────────────────────
    before  = len(regions)
    regions = _nms_regions(regions)
    if len(regions) < before:
        logger.debug(
            f"  [LAYOUT] NMS removed {before - len(regions)} duplicate region(s)."
        )

    return regions, image_bbox


# ══════════════════════════════════════════════════════════════════════════════
# TROCR RECOGNITION  +  NOISE FILTER
# ══════════════════════════════════════════════════════════════════════════════

def _eos_ids(processor: TrOCRProcessor, model: VisionEncoderDecoderModel) -> set:
    """All token ids that end a generated sequence, from whichever config carries them."""
    ids: set = set()
    for src in (getattr(model, "generation_config", None),
                getattr(model, "config", None),
                getattr(getattr(model, "config", None), "decoder", None),
                getattr(processor, "tokenizer", None)):
        v = getattr(src, "eos_token_id", None) if src is not None else None
        if isinstance(v, int):
            ids.add(v)
        elif isinstance(v, (list, tuple)):
            ids.update(int(x) for x in v)
    return ids


def _sequence_confidence(out, model: VisionEncoderDecoderModel,
                         eos_ids: set, num_beams: int) -> float:
    """
    Mean per-token probability of the RETURNED sequence (0–1).

    Uses model.compute_transition_scores, which follows the chosen path
    through beam search as well as greedy decoding, so the value means the
    same thing for both.  For greedy output it is numerically identical to
    the previous metric (mean max-softmax per step), so CONFIDENCE_THRESHOLD
    and the other noise-filter thresholds keep their meaning.

    Only the tokens actually in the returned sequence are averaged: the
    decoder-start token is skipped, and everything after the first EOS
    (beam padding) is ignored.
    """
    if not getattr(out, "scores", None):
        return 0.0
    gen = out.sequences[0, 1:].tolist()                  # drop decoder-start token
    n   = next((i + 1 for i, t in enumerate(gen) if t in eos_ids), len(gen))
    n   = max(1, min(n, len(out.scores)))
    try:
        ts = model.compute_transition_scores(
            out.sequences,
            out.scores,
            getattr(out, "beam_indices", None) if num_beams > 1 else None,
            normalize_logits=(num_beams == 1),   # greedy scores are raw logits;
        )                                        # beam scores are already log-probs
        vals = ts[0, :min(n, ts.shape[1])].float()
        return torch.exp(vals).mean().item() if vals.numel() else 0.0
    except Exception as exc:
        logger.debug(f"    transition-score confidence unavailable ({exc}); using fallback")
        if num_beams == 1:
            probs = [torch.softmax(s[0], dim=-1).max().item() for s in out.scores[:n]]
            return sum(probs) / len(probs)
        seq = getattr(out, "sequences_scores", None)        # length-normalised log-prob
        return float(torch.exp(seq[0]).item()) if seq is not None else 0.0


def _decode(
    pixel_values: "torch.Tensor",
    processor: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
    num_beams: int,
) -> Tuple[str, float]:
    """One TrOCR generate() call at `num_beams`; returns (text, confidence)."""
    kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=num_beams,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    )
    if num_beams > 1:
        kwargs.update(early_stopping=True, length_penalty=BEAM_LENGTH_PENALTY)
    with torch.no_grad():
        out = model.generate(pixel_values, **kwargs)
    text = processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
    conf = _sequence_confidence(out, model, _eos_ids(processor, model), num_beams)
    return text, conf


_PUNCT_BEFORE = re.compile(r"\s+([.,;:!?)\]])")   # space before closing punctuation
_PUNCT_AFTER  = re.compile(r"([(\[])\s+")          # space after an opening bracket


def _normalize_punct_spacing(text: str) -> str:
    """(h) 'Moscow , Wednesday .' → 'Moscow, Wednesday.'; '( alas' → '(alas'.
    Quote marks are left alone: whether a quote opens or closes can't be told
    reliably from one line."""
    if not NORMALIZE_PUNCT_SPACING or not text:
        return text
    return _PUNCT_AFTER.sub(r"\1", _PUNCT_BEFORE.sub(r"\1", text)).strip()


def _content_key(text: str) -> str:
    """(h) Text reduced to lower-case letters and digits, for spotting cosmetic-only changes."""
    return re.sub(r"[\W_]+", "", text).lower()


def _trocr_read(
    image: Image.Image,
    processor: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
    min_conf: float = CONFIDENCE_THRESHOLD,
) -> Tuple[str, float]:
    """
    Run TrOCR on a single image crop, with selective beam search.

    Greedy decoding first.  A beam retry is attempted only when (h):
      - max(BEAM_RETRY_ABOVE, min_conf) <= greedy confidence < BEAM_RETRY_BELOW
        (`min_conf` is the noise floor the caller will apply, so a read that
        would be discarded anyway is never retried — and beam can never lift
        it over the filter);
      - the greedy text has >= BEAM_MIN_CHARS non-space characters.
    The beam result is adopted only if (h):
      - it is non-empty,
      - its confidence is >= the greedy confidence, and
      - it differs from the greedy text in more than case, spacing or
        punctuation (beam's trailing-period and lower-casing habits otherwise
        win on the per-token mean without improving anything).

    The returned text has punctuation spacing normalised
    (NORMALIZE_PUNCT_SPACING).  Returns (text, confidence), confidence being
    the mean per-token probability of the returned sequence (0–1).
    """
    try:
        pixel_values = processor(image.convert("RGB"), return_tensors="pt").pixel_values
        device       = next(model.parameters()).device
        pixel_values = pixel_values.to(device)

        text, confidence = _decode(pixel_values, processor, model, num_beams=1)
        _READ_STATS["reads"] += 1

        floor = max(BEAM_RETRY_ABOVE, min_conf)
        if (BEAM_RETRY_ENABLED and BEAM_NUM_BEAMS > 1
                and floor <= confidence < BEAM_RETRY_BELOW
                and len(re.sub(r"\s+", "", text)) >= BEAM_MIN_CHARS):
            b_text, b_conf = _decode(pixel_values, processor, model, num_beams=BEAM_NUM_BEAMS)
            _READ_STATS["beam_retries"] += 1
            if not b_text or b_text == text:
                _READ_STATS["beam_same"] += 1
            elif _content_key(b_text) == _content_key(text):
                _READ_STATS["beam_cosmetic"] += 1
                logger.debug(f"      [BEAM-COSMETIC] kept greedy {text[:40]!r} (beam {b_text[:40]!r})")
            elif b_conf < confidence:
                _READ_STATS["beam_lower"] += 1
                logger.debug(
                    f"      [BEAM-REJECT] {confidence:.2f}>{b_conf:.2f}  kept {text[:40]!r} "
                    f"(beam {b_text[:40]!r})"
                )
            else:
                _READ_STATS["beam_changed"] += 1
                logger.debug(
                    f"      [BEAM] {confidence:.2f}→{b_conf:.2f}  "
                    f"{text[:40]!r} → {b_text[:40]!r}"
                )
                text, confidence = b_text, b_conf
        return _normalize_punct_spacing(text), confidence
    except Exception as exc:
        logger.debug(f"    TrOCR error: {exc}")
        return "", 0.0


def _join_segments(texts: List[str]) -> str:
    """
    Join OCR'd segment texts left-to-right with a space, dropping a duplicated
    word at each boundary — the last word of one segment re-reading as the
    first word of the next, a side effect of the deliberate overlap between
    segments in _trocr_read_wide_crop().  Comparison ignores case and
    surrounding punctuation.
    """
    out: List[str] = []
    for t in texts:
        words = t.split()
        if out and words:
            prev = out[-1].strip(".,;:!?\"'").lower()
            cur  = words[0].strip(".,;:!?\"'").lower()
            if prev and prev == cur:
                words = words[1:]
        out.extend(words)
    return " ".join(out)


def _trocr_read_wide_crop(
    crop: Image.Image,
    processor: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
    max_ar: float = MAX_HEADER_AR,
    min_conf: float = CONFIDENCE_THRESHOLD,
) -> Tuple[str, float]:
    """
    OCR a crop that may be much wider than it is tall (a masthead or banner
    headline spanning most of the page width).

    TrOCRProcessor resizes every crop to a fixed square input regardless of
    its original aspect ratio.  A crop many times wider than tall gets every
    letterform compressed horizontally by roughly that same factor, which is
    a significant, avoidable source of misreads on wide single-line headers.

    Crops within `max_ar` are read as a single call (unchanged behaviour, no
    effect on normal headers).  Wider crops are split into overlapping
    horizontal segments — each close to `max_ar`, and each therefore closer
    to the aspect ratio ordinary OCR training data uses — read independently,
    and joined with _join_segments().  Confidence is the mean across segments
    that returned non-empty text.

    (h) Crops shorter than HEADER_SPLIT_MIN_H are always read whole: on thin
    single-line strips the fixed-width cuts fall mid-word and the joined read
    is worse than TrOCR's reading of the whole line.
    """
    w, h = crop.size
    if h <= 0 or w / h <= max_ar or h < HEADER_SPLIT_MIN_H:
        return _trocr_read(crop, processor, model, min_conf=min_conf)

    seg_w   = max(1, int(h * max_ar))
    overlap = int(seg_w * HEADER_SEGMENT_OVERLAP)
    stride  = max(1, seg_w - overlap)

    texts:  List[str]   = []
    confs:  List[float] = []
    x = 0
    while True:
        x1  = min(x + seg_w, w)
        seg = crop.crop((x, 0, x1, h))
        text, conf = _trocr_read(seg, processor, model, min_conf=min_conf)
        text = text.strip()
        if text:
            texts.append(text)
            confs.append(conf)
        if x1 >= w:
            break
        x += stride

    joined     = _join_segments(texts)
    confidence = sum(confs) / len(confs) if confs else 0.0
    logger.debug(
        f"      [WIDE-HEADER] AR={w/h:.1f} split into {len(texts)}/{-(-w // stride) if stride else 1} "
        f"segment(s) of ~{seg_w}px → {len(joined)} char(s)"
    )
    return joined, confidence


def _is_noise(
    text: str,
    confidence: float,
    h: int,
    w: int,
    min_conf: float = CONFIDENCE_THRESHOLD,
) -> bool:
    """
    Heuristic noise filter for TrOCR output.

    Returns True if the recognised text is likely artefact, ruling line,
    punctuation noise, or a confidence-floor rejection.

    min_conf lets callers apply a stricter floor (the recall sweep uses
    SWEEP_MIN_CONFIDENCE because it also sees photographs and halftone).
    """
    if not text:
        return True
    if h < MIN_LINE_H or w < MIN_LINE_W:
        return True
    ar = w / h
    if ar < 0.1 or ar > 400:            # was 100: rejected wide banner lines
        return True
    tc = text.strip()
    tl = len(tc)
    if tl == 1:
        return confidence < SINGLE_CHAR_CONFIDENCE_THRESHOLD
    if confidence < min_conf:
        return True
    if len(set(tc)) == 1 and tl > 2:    # repeated single character
        return True
    # (The old numeric-only pattern was removed: prices, dates and phone
    #  numbers are real newspaper content.)
    noise_pats = [r"^[oOlI\.\|]+$", r"^[^a-zA-Z0-9\s]+$"]
    for pat in noise_pats:
        if re.match(pat, tc) and confidence < SINGLE_CHAR_CONFIDENCE_THRESHOLD:
            return True
    # Vowel-less test applies only to purely alphabetic strings; strings with
    # digits or punctuation ("$125", "555-1234") are not judged by it.
    if (tl > 3 and tc.isalpha()
            and not any(c in "aeiouy" for c in tc.lower())
            and confidence < 0.7):
        return True
    # A box far wider than its own recognized text would need at that
    # height is usually a mostly-blank or non-textual strip that TrOCR has
    # filled in with a short, plausible-looking guess rather than genuinely
    # read. Only checked for tl >= 4: shorter strings make this ratio too
    # noisy to trust (a single wide digit or punctuation mark is normal).
    if tl >= 4 and (w / (h * tl)) > SPARSE_LINE_WIDTH_RATIO:
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

def _body_recognition_source(
    page_image: Image.Image,
    raw_image: Image.Image,
) -> Tuple[Image.Image, object]:
    """(image to cut body-text recognition crops from, preprocessing for them)."""
    if RECOGNIZE_FROM_RAW:
        return raw_image, preprocess_line_crop
    return page_image, _passthrough_crop


def ocr_region(
    page_image: Image.Image,
    raw_image: Image.Image,
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

    When both passes yield results, the one with more accepted CHARACTERS wins
    (line count is a poor proxy: Pass 2 always yields a single "line").

    Detection vs recognition images (revision g)
    ─────────────────────────────────────────────
    Line DETECTION runs on the preprocessed page (`page_image`), or for
    HEADER_LABELS on a header-preprocessed crop of the raw render, exactly as
    before.  RECOGNITION crops are cut separately via _recognition_crop():
      - from `raw_image` (headers always; body text when RECOGNIZE_FROM_RAW),
        each line autocontrasted on its own;
      - padded by LINE_PAD_X / LINE_PAD_Y into the surrounding page.
    Element boxes stay the unpadded detector boxes.

    HEADER_LABELS reads also go through _trocr_read_wide_crop, which splits
    a crop much wider than it is tall (a masthead or banner headline) into
    segments before recognition.  A no-op for normal-width crops.

    All returned bbox coordinates are in full-page (absolute) pixel space.
    """
    x0, y0, x1, y1 = [int(c) for c in region["bbox"]]
    iw, ih          = page_image.size
    label           = region["label"]
    is_header       = label in HEADER_LABELS

    x0 = max(0, x0);  y0 = max(0, y0)
    x1 = min(iw, x1); y1 = min(ih, y1)

    rw, rh = x1 - x0, y1 - y0
    if rw < MIN_REGION_W or rh < MIN_REGION_H:
        return []

    if is_header:
        det_crop           = preprocess_header_crop(raw_image.crop((x0, y0, x1, y1)))
        rec_src, rec_prep  = raw_image, preprocess_header_crop
        read = lambda img: _trocr_read_wide_crop(img, trocr_processor, trocr_model)
    else:
        det_crop           = page_image.crop((x0, y0, x1, y1))
        rec_src, rec_prep  = _body_recognition_source(page_image, raw_image)
        read = lambda img: _trocr_read(img, trocr_processor, trocr_model)

    # ── Pass 1: line detection → TrOCR per line ───────────────────────────────
    line_bboxes = _surya_line_bboxes(det_crop, det_predictor)
    pass1_elems: List[Dict] = []

    for lb in line_bboxes:
        lx0, ly0, lx1, ly1 = [int(v) for v in lb]
        lh, lw = ly1 - ly0, lx1 - lx0
        if lh < MIN_LINE_H or lw < MIN_LINE_W:
            continue
        # Absolute page coordinates (unpadded — used for the text layer)
        abs_bbox  = [lx0 + x0, ly0 + y0, lx1 + x0, ly1 + y0]
        line_crop = _recognition_crop(rec_src, abs_bbox, rec_prep)
        text, confidence = read(line_crop)
        if _is_noise(text, confidence, lh, lw):
            logger.debug(
                f"      [NOISE] conf={confidence:.2f} "
                f"{lw}×{lh}px | {text[:40]}"
            )
            continue
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
    whole_crop = _recognition_crop(rec_src, [x0, y0, x1, y1], rec_prep)
    text_wb, conf_wb = read(whole_crop)
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
        chars1 = sum(len(e["text"]) for e in pass1_elems)
        chars2 = sum(len(e["text"]) for e in pass2_elems)
        logger.debug(
            f"      Pass 2 (whole-crop TrOCR): {chars2} char(s)  "
            f"[Pass 1 had {n_pass1} line(s), {chars1} char(s)]"
        )
        # Prefer whichever pass recovered more text
        if chars2 >= chars1:
            return pass2_elems

    if n_pass1 > 0:
        logger.debug(f"      Kept Pass 1 ({n_pass1} lines) over Pass 2 ({n_pass2})")
        return pass1_elems

    logger.debug(f"      Both passes empty for label={label}")
    return []


# ══════════════════════════════════════════════════════════════════════════════
# RECALL SWEEP  –  recover text no layout region claimed
# ══════════════════════════════════════════════════════════════════════════════

def _coverage(box: List[float], elements: List[Dict], pad: float = 0.0) -> float:
    """
    Fraction of `box` already lying inside existing elements (0–1).

    `pad` grows each existing element's box by this many px on every side
    before checking overlap, absorbing the kind of few-pixel boundary
    disagreement two independent detector passes over different crops of
    the same physical line can produce, without it being (mis)read as
    genuinely new content.
    """
    area  = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    total = 0.0
    for e in elements:
        b  = e["bbox"]
        iw = min(box[2], b[2] + pad) - max(box[0], b[0] - pad)
        ih = min(box[3], b[3] + pad) - max(box[1], b[1] - pad)
        if iw > 0 and ih > 0:
            total += iw * ih
    return min(1.0, total / area)


def _tile_origins(length: int, tile: int, overlap: int) -> List[int]:
    """Evenly spaced tile start offsets covering `length` with >= `overlap` px overlap."""
    if length <= tile:
        return [0]
    stride = tile - overlap
    n      = -(-(length - tile) // stride) + 1          # ceil division
    return [round(i * (length - tile) / (n - 1)) for i in range(n)]


def _nearest_position(bbox: List[float], regions: List[Dict]) -> int:
    """Reading-order position of the layout region closest to `bbox`."""
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    best, best_d = 0, float("inf")
    for r in regions:
        x0, y0, x1, y1 = r["bbox"]
        dx = max(x0 - cx, 0, cx - x1)
        dy = max(y0 - cy, 0, cy - y1)
        d  = dx * dx + dy * dy
        if d < best_d:
            best, best_d = r["position"], d
    return best


def _column_bands(layout_regions: List[Dict], page_w: float) -> List[List[float]]:
    """
    x-ranges occupied by column-body layout regions, merged into bands.

    Returns [[x0, x1, min_position, max_position], ...] sorted left→right.
    Only nearly page-wide regions (mastheads, banner headlines) are excluded
    from defining a band — EVERY other region, however short, claims its own
    x-range.  A short region (a two-line brief, a subhead) is just as real a
    piece of Surya's own segmentation as a tall one; a column built entirely
    from several short regions must still register as claimed, or any stray
    content the sweep later finds there gets mistaken for a whole column the
    layout model skipped and relocated to a wrong, distant reading position.
    A genuinely missing column has NO regions of any height in its x-range,
    so it still shows up as a gap between bands.

    Neighbouring regions whose boxes touch or overlap in x merge into one
    band; a column the layout model skipped entirely leaves a wide gap.
    """
    ivals = [
        [r["bbox"][0], r["bbox"][2], r["position"], r["position"]]
        for r in layout_regions
        if (r["bbox"][2] - r["bbox"][0]) <= COLUMN_REGION_MAX_W_FRAC * page_w
    ]
    ivals.sort(key=lambda v: v[0])
    bands: List[List[float]] = []
    for x0, x1, pmin, pmax in ivals:
        if bands and x0 - bands[-1][1] < COLUMN_GAP_MIN:
            b = bands[-1]
            b[1], b[2], b[3] = max(b[1], x1), min(b[2], pmin), max(b[3], pmax)
        else:
            bands.append([x0, x1, pmin, pmax])
    return bands


def _assign_recovered_positions(
    recovered: List[Dict],
    layout_regions: List[Dict],
    page_w: float,
    page_h: float,
) -> int:
    """
    Give recovered lines that sit in an UNCLAIMED column a proper reading position.

    Lines centred inside an existing column band keep the nearest-region
    position assigned by the sweep.  Lines centred in an x-gap between (or
    beside) bands are grouped into columns by x-overlap; a group is only
    relocated in reading order if it clears BOTH MIN_UNCLAIMED_COLUMN_LINES
    and MIN_UNCLAIMED_COLUMN_HEIGHT_FRAC — i.e. it plausibly IS a whole
    column the layout model missed, not a handful of stray/duplicate lines a
    coverage false-negative swept up next to a real, already-correctly-
    positioned region.  A cluster that doesn't clear both keeps its original
    nearest-neighbour position: at worst a harmless local duplicate, never a
    relocation to an unrelated part of the page.

    A relocated group is placed just after the last region of the bands to
    its left, so a whole missing column reads top-to-bottom between its
    neighbours instead of being interleaved line-by-line with them.
    Positions may therefore be fractional (e.g. 16.5); sorting and reports
    handle that.

    Returns the number of lines actually re-positioned.
    """
    bands = _column_bands(layout_regions, page_w)
    if not bands or not recovered:
        return 0

    unclaimed = [
        e for e in recovered
        if not any(b[0] <= (e["bbox"][0] + e["bbox"][2]) / 2 <= b[1] for b in bands)
    ]
    if not unclaimed:
        return 0

    # Group unclaimed lines into columns by x-overlap
    clusters: List[List] = []                     # [x0, x1, [elements]]
    for e in sorted(unclaimed, key=lambda e: e["bbox"][0]):
        x0, x1 = e["bbox"][0], e["bbox"][2]
        for c in clusters:
            if min(x1, c[1]) - max(x0, c[0]) >= 0.5 * (x1 - x0):
                c[0], c[1] = min(c[0], x0), max(c[1], x1)
                c[2].append(e)
                break
        else:
            clusters.append([x0, x1, [e]])
    clusters.sort(key=lambda c: c[0])

    moved = 0
    for k, (cx0, cx1, elems) in enumerate(clusters):
        if len(elems) < MIN_UNCLAIMED_COLUMN_LINES:
            logger.debug(
                f"      [SWEEP] {len(elems)} stray line(s) at x={cx0:.0f}-{cx1:.0f}px "
                f"— too few to be a missed column, kept at nearest-neighbour position."
            )
            continue
        y_span = max(e["bbox"][3] for e in elems) - min(e["bbox"][1] for e in elems)
        if y_span < MIN_UNCLAIMED_COLUMN_HEIGHT_FRAC * page_h:
            logger.debug(
                f"      [SWEEP] {len(elems)} line(s) at x={cx0:.0f}-{cx1:.0f}px span only "
                f"{y_span:.0f}px — too short to be a missed column, kept at nearest-neighbour position."
            )
            continue
        centre = (cx0 + cx1) / 2
        left   = [b[3] for b in bands if b[1] <= centre]      # max position of bands left of it
        base   = max(left) if left else min(b[2] for b in bands) - 1
        pos    = base + 0.5 + 0.001 * k
        for e in elems:
            e["reading_position"] = pos
        moved += len(elems)
        logger.debug(
            f"      [SWEEP] unclaimed column x={cx0:.0f}–{cx1:.0f}px: "
            f"{len(elems)} line(s) → reading position {pos:.3f}"
        )
    return moved


def _assign_visual_rows(elements: List[Dict]) -> None:
    """
    Give every element a "_row" index (mutates in place) for the final sort.

    Elements sharing a reading position are grouped into visual rows by
    comparing each line only to the immediately preceding one (sorted
    top-to-bottom), never an accumulated row boundary -- that would let a
    chain of small overlaps merge an entire column into one row. Two lines
    count as the same row only if they overlap substantially in y (more
    than ROW_OVERLAP_FRACTION of the smaller line's height) AND sit mostly
    side by side rather than stacked (x-overlap under ROW_MAX_X_OVERLAP).
    The x condition is what separates genuine same-line fragments (two
    headline words, occupying different x-ranges) from stacked lines with
    messy, overlapping detection boxes (near-identical x-range, moderate
    y-overlap) that would otherwise get sorted left-to-right instead of
    top-to-bottom.
    """
    groups: Dict[float, List[Dict]] = {}
    for e in elements:
        groups.setdefault(e["reading_position"], []).append(e)
    for group in groups.values():
        group.sort(key=lambda e: e["bbox"][1])
        row = 0
        prev_box: Optional[List[float]] = None
        for e in group:
            x0, y0, x1, y1 = e["bbox"]
            if prev_box is not None:
                px0, py0, px1, py1 = prev_box
                y_overlap = min(y1, py1) - max(y0, py0)
                smaller_h = min(y1 - y0, py1 - py0)
                y_same = smaller_h > 0 and (y_overlap / smaller_h) > ROW_OVERLAP_FRACTION
                x_overlap = min(x1, px1) - max(x0, px0)
                smaller_w = min(x1 - x0, px1 - px0)
                x_side_by_side = smaller_w <= 0 or (x_overlap / smaller_w) < ROW_MAX_X_OVERLAP
                if not (y_same and x_side_by_side):
                    row += 1
            prev_box = [x0, y0, x1, y1]
            e["_row"] = row


def _overlap_area(a: List[float], b: List[float]) -> float:
    """Raw intersection area of two [x1, y1, x2, y2] boxes."""
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, ix) * max(0.0, iy)


def dedupe_overlapping_elements(elements: List[Dict]) -> List[Dict]:
    """
    Remove an element that is a duplicate re-read of another element's
    physical area.

    This happens when Surya's own layout or line-detection stage emits two
    overlapping boxes for the same content -- e.g. two overlapping regions
    under different labels (same-label duplicates are already merged by
    _nms_regions before OCR), or two overlapping lines detected within one
    region's multi-line pass. Both get OCR'd, so the same text is inserted
    twice.

    A pair counts as a duplicate if one box is contained in the other by
    more than DEDUPE_CONTAINMENT of its own area, and the two areas are
    comparable (DEDUPE_MAX_AREA_RATIO) -- a tiny stray fragment can sit
    almost entirely inside a large legitimate line's box without being a
    second read of it, but two genuine duplicate reads are close in size.
    Containment is checked regardless of reading position: genuinely
    different sequential lines never reach this containment level in
    practice (a descender's overlap is a small fraction of a line's area),
    so position is not a useful signal here. Of a genuinely duplicate pair:
    if exactly one element's label is a "page furniture" type
    (FURNITURE_LABELS), it is kept and the other dropped — (h) unless the
    other read's confidence is higher by more than DEDUPE_FURNITURE_MARGIN,
    in which case the furniture read is dropped; otherwise the
    lower-confidence read is dropped.
    """
    if len(elements) < 2:
        return elements
    order   = sorted(range(len(elements)), key=lambda i: elements[i]["bbox"][1])
    dropped = [False] * len(elements)
    n_dropped = 0

    for oi, i in enumerate(order):
        if dropped[i]:
            continue
        a   = elements[i]
        ay1 = a["bbox"][3]
        area_a = max(1.0, (a["bbox"][2] - a["bbox"][0]) * (ay1 - a["bbox"][1]))
        for j in order[oi + 1:]:
            if dropped[j]:
                continue
            b = elements[j]
            if b["bbox"][1] > ay1:      # sorted by y0 -- nothing further below can still overlap 'a'
                break
            area_b  = max(1.0, (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]))
            overlap = _overlap_area(a["bbox"], b["bbox"])
            if overlap / min(area_a, area_b) < DEDUPE_CONTAINMENT:
                continue
            if max(area_a, area_b) / min(area_a, area_b) > DEDUPE_MAX_AREA_RATIO:
                continue   # a tiny fragment inside a big line's box, not a duplicate read of it

            a_furn = a.get("source_label") in FURNITURE_LABELS
            b_furn = b.get("source_label") in FURNITURE_LABELS
            if a_furn != b_furn:
                # (h) furniture wins unless the other read is clearly more confident
                furn_k, other_k = (i, j) if a_furn else (j, i)
                if (elements[other_k]["confidence"] - elements[furn_k]["confidence"]
                        > DEDUPE_FURNITURE_MARGIN):
                    loser_key = furn_k
                else:
                    loser_key = other_k
            else:
                loser_key = i if a["confidence"] <= b["confidence"] else j

            dropped[loser_key] = True
            n_dropped += 1
            loser = elements[loser_key]
            logger.debug(
                f"      [DEDUPE] dropped pos={loser['reading_position']} "
                f"label={loser.get('source_label')} conf={loser['confidence']:.2f} | "
                f"{loser['text'][:40]!r}"
            )
            if loser_key == i:
                break   # 'a' is gone -- stop comparing it further

    if n_dropped:
        logger.info(
            f"  [DEDUPE] removed {n_dropped} duplicate element(s) "
            f"(same content recognised twice)."
        )
    return [e for k, e in enumerate(elements) if not dropped[k]]


def _split_wide_sweep_line(box: List[float], bands: List[List[float]]) -> List[List[float]]:
    """
    If `box` [x0,y0,x1,y1] straddles a known column-band edge, split it into
    per-band pieces at that edge. Returns [box] unchanged when no band edge
    falls strictly inside it -- true for every ordinary single-column line
    -- so this is safe to call unconditionally.

    Why this exists: sweep_uncovered_text()'s DetectionPredictor pass runs
    on a raw, unconfined tile crop -- nothing tells it where one column ends
    and the next begins, unlike the per-region OCR pass. On degraded
    newsprint it can bridge a narrow or layout-missed gutter and return one
    "line" spanning two adjacent columns' text, producing a single element
    whose text fuses two unrelated sentences with no boundary between them.
    `bands` only carries the column edges _column_bands() established, so
    this can't split a line inside a run of columns _column_bands already
    merged into one band (e.g. when several real gutters are narrower than
    COLUMN_GAP_MIN).
    """
    x0, y0, x1, y1 = box
    cuts = sorted(set(
        round(c) for c in
        [b[0] for b in bands if x0 < b[0] < x1] +
        [b[1] for b in bands if x0 < b[1] < x1]
    ))
    if not cuts:
        return [box]
    edges = [x0] + cuts + [x1]
    return [
        [edges[i], y0, edges[i + 1], y1]
        for i in range(len(edges) - 1)
        if edges[i + 1] - edges[i] >= MIN_LINE_W
    ]


def sweep_uncovered_text(
    page_image: Image.Image,
    raw_image: Image.Image,
    elements: List[Dict],
    layout_regions: List[Dict],
    det_predictor: DetectionPredictor,
    trocr_processor: TrOCRProcessor,
    trocr_model: VisionEncoderDecoderModel,
) -> List[Dict]:
    """
    Recall safety-net.

    The layout model runs at a much lower internal resolution than the page
    render, so on dense broadsheets it can miss small blocks entirely.  This
    pass tiles the whole page (overlapping tiles), detects text lines at near
    native resolution, and OCRs every line that is not already covered by an
    existing element.

    - Boxes cut by an interior tile edge are ignored; the overlapping
      neighbour tile sees them whole (needs SWEEP_OVERLAP > line length).
    - `known` grows as lines are recovered, so overlapping tiles never
      re-read the same line.  Coverage is checked with SWEEP_COVERAGE_PAD of
      slack, absorbing minor boundary disagreement between this pass's
      detector run and the per-region pass's, so an already-read line isn't
      mistaken for new content over a few pixels of jitter.
    - Recovered lines use SWEEP_MIN_CONFIDENCE (stricter than region OCR)
      because the sweep also sees photographs and halftone.
    - Recovered lines inherit the reading position of the nearest layout
      region and are labelled "Recovered"; _assign_recovered_positions()
      then relocates only the clusters large enough to plausibly be a whole
      column the layout model missed (see MIN_UNCLAIMED_COLUMN_LINES /
      MIN_UNCLAIMED_COLUMN_HEIGHT_FRAC) — everything else keeps this
      nearest-neighbour position, never landing far from where it visually
      sits on the page.
    - A detected line that straddles a known column-band edge is split
      there (_split_wide_sweep_line) before coverage-checking or OCR, each
      piece read independently -- otherwise the raw tile detector can fuse
      two columns' text into one box with no boundary between them.
    - Detection runs on `page_image` (preprocessed); recognition crops are
      cut like body text in ocr_region (raw render when RECOGNIZE_FROM_RAW,
      padded by LINE_PAD_X / LINE_PAD_Y).

    Coordinates are in the same space as `page_image` (full-page pixels).
    """
    iw, ih = page_image.size
    m      = SWEEP_EDGE_MARGIN
    recovered: List[Dict] = []
    known: List[Dict]     = list(elements)
    column_bands = _column_bands(layout_regions, iw)
    rec_src, rec_prep = _body_recognition_source(page_image, raw_image)

    for ty in _tile_origins(ih, SWEEP_TILE, SWEEP_OVERLAP):
        for tx in _tile_origins(iw, SWEEP_TILE, SWEEP_OVERLAP):
            tx1, ty1 = min(tx + SWEEP_TILE, iw), min(ty + SWEEP_TILE, ih)
            tile = page_image.crop((tx, ty, tx1, ty1))
            for lx0, ly0, lx1, ly1 in _surya_line_bboxes(tile, det_predictor):
                # Skip boxes cut by an interior tile edge (page borders are fine)
                if ((tx > 0   and lx0 < m) or (tx1 < iw and lx1 > (tx1 - tx) - m) or
                        (ty > 0 and ly0 < m) or (ty1 < ih and ly1 > (ty1 - ty) - m)):
                    continue
                lh, lw = ly1 - ly0, lx1 - lx0
                if lh < MIN_LINE_H or lw < MIN_LINE_W:
                    continue
                raw_box = [lx0 + tx, ly0 + ty, lx1 + tx, ly1 + ty]
                for box in _split_wide_sweep_line(raw_box, column_bands):
                    bw, bh = box[2] - box[0], box[3] - box[1]
                    if bw < MIN_LINE_W:
                        continue
                    if _coverage(box, known, pad=SWEEP_COVERAGE_PAD) >= SWEEP_COVERED_FRAC:
                        continue
                    line = _recognition_crop(rec_src, box, rec_prep)
                    text, conf = _trocr_read(line, trocr_processor, trocr_model,
                                             min_conf=SWEEP_MIN_CONFIDENCE)
                    if _is_noise(text, conf, int(bh), int(bw), min_conf=SWEEP_MIN_CONFIDENCE):
                        logger.debug(
                            f"      [SWEEP-NOISE] conf={conf:.2f} "
                            f"{int(bw)}×{int(bh)}px | {text[:40]}"
                        )
                        continue
                    elem = {
                        "text":             text,
                        "bbox":             box,
                        "confidence":       conf,
                        "font_size":        max(6.0, min(bh * 0.85, 72.0)),
                        "source_label":     "Recovered",
                        "reading_position": _nearest_position(box, layout_regions),
                    }
                    recovered.append(elem)
                    known.append(elem)

    moved = _assign_recovered_positions(recovered, layout_regions, iw, ih)
    if moved:
        logger.info(f"  [SWEEP] {moved} recovered line(s) lie in a column the layout stage skipped.")
    return recovered


# ══════════════════════════════════════════════════════════════════════════════
# DEBUG VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _debug_path(stem: str, page_num: int, tag: str, ext: str) -> Path:
    """
    Path for one debug output file (a JPEG or a .txt report).

    DEBUG_OVERWRITE = True  → fixed names ("latest_<tag>.<ext>"): every page
        overwrites the previous page's file of that kind, so the debug
        folder holds at most one page's worth of debug output however large
        the batch is.  Applies equally to the JPEGs and the layout/OCR .txt
        reports — a long batch run no longer accumulates one pair of reports
        per page.
    DEBUG_OVERWRITE = False → per-page names (original behaviour).
    """
    if DEBUG_OVERWRITE:
        return DEBUG_PATH / f"latest_{tag}.{ext}"
    return DEBUG_PATH / f"{stem}_p{page_num:03d}_{tag}.{ext}"


def _debug_img_path(stem: str, page_num: int, tag: str) -> Path:
    """Path for a debug JPEG.  See _debug_path()."""
    return _debug_path(stem, page_num, tag, "jpg")


def clear_debug_files() -> None:
    """Delete debug JPEGs and .txt reports left over from earlier runs (overwrite mode only)."""
    if not DEBUG_OVERWRITE:
        return
    removed = 0
    for pattern in ("*.jpg", "*.txt"):
        for f in DEBUG_PATH.glob(pattern):
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
    if removed:
        logger.info(f"  Cleared {removed} stale debug file(s).")


def audit_coverage(
    page_image: Image.Image,
    elements: List[Dict],
    layout_regions: List[Dict],
    stem: str,
    page_num: int,
) -> float:
    """
    Diagnostic: how much printed ink still lies outside every OCR'd box?

    Works at 1/AUDIT_SCALE size.  "Ink" = dark blocks.  "Covered" = the boxes
    of all OCR elements plus Picture/Figure regions (intentionally not read).
    Logs the uncovered share, lists any wide vertical bands that are mostly
    uncovered (a whole missing column shows up as one), and saves
    latest_03_uncovered.jpg with the uncovered ink painted red.

    Margin speckle and library stamps count as uncovered ink; the band test is
    the more telling signal.  Returns the uncovered fraction (0–1).
    """
    s      = AUDIT_SCALE
    small  = page_image.convert("L").reduce(s)
    sw, sh = small.size
    ink    = small.point(lambda v: 255 if v < AUDIT_INK_LEVEL else 0)
    ink    = ink.filter(ImageFilter.MaxFilter(3))          # bridge the gaps between text lines

    covered = Image.new("L", (sw, sh), 0)
    cdraw   = ImageDraw.Draw(covered)
    pad     = 6
    boxes   = [e["bbox"] for e in elements]
    boxes  += [r["bbox"] for r in layout_regions if r["label"] in SKIP_LABELS]
    for x0, y0, x1, y1 in boxes:
        cdraw.rectangle([(x0 - pad) / s, (y0 - pad) / s, (x1 + pad) / s, (y1 + pad) / s], fill=255)

    uncovered = ImageChops.subtract(ink, covered)
    total = ink.histogram()[255]
    miss  = uncovered.histogram()[255]
    frac  = miss / total if total else 0.0

    # Vertical bands that are mostly uncovered
    box_filter = getattr(Image, "Resampling", Image).BOX
    col_ink  = list(ink.resize((sw, 1), box_filter).tobytes())
    col_miss = list(uncovered.resize((sw, 1), box_filter).tobytes())
    bands, start = [], None
    for i in range(sw + 1):
        bad = i < sw and col_ink[i] >= 8 and col_miss[i] >= 0.5 * col_ink[i]
        if bad and start is None:
            start = i
        elif not bad and start is not None:
            if (i - start) * s >= AUDIT_MIN_BAND_PX:
                bands.append((start * s, i * s))
            start = None

    msg = f"  [COVERAGE] {frac * 100:.1f}% of page ink lies outside every OCR'd box"
    if bands:
        msg += "; mostly-uncovered band(s) at x=" + ", ".join(f"{a}–{b}px" for a, b in bands)
    (logger.warning if (frac > AUDIT_WARN_FRAC or bands) else logger.info)(msg)

    vis = Image.merge("RGB", (small, small, small))
    vis.paste(Image.new("RGB", (sw, sh), (255, 0, 0)), mask=uncovered)
    out = _debug_img_path(stem, page_num, "03_uncovered")
    vis.save(str(out), "JPEG", quality=85)
    logger.info(f"    [DEBUG] Uncovered-ink map → {out.name}")
    return frac


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
) -> Tuple[List[Dict], List[Tuple[float, float, float, float]]]:
    """
    Full pipeline for a single newspaper page.

    1.  Preprocess  — tiled CLAHE-approx + unsharp mask (layout + detection input)
    2.  Layout      — LayoutPredictor → semantic regions + reading order positions
    2b. Reorder     — correct Surya's own position for masthead/banner regions
                      it occasionally mis-scores (see _reorder_banner_regions)
    3.  OCR         — DetectionPredictor (line segmentation) + TrOCR per line,
                      for every region not in SKIP_LABELS; recognition crops
                      come from the raw render, padded, with selective beam
                      search (see ocr_region / _trocr_read)
    3b. Sweep       — tile the page, OCR any detected line no region claimed;
                      lines in a column the layout model skipped are slotted
                      into reading order between the neighbouring columns
    3c. Dedupe      — drop an element that is a duplicate read of another
                      element's physical area at a different reading position
                      (a cross-label duplicate Surya's own layout stage emitted)
    4.  Sort        — by Surya layout position, then visual row (y-overlap
                      clustering), then x within a row (left-to-right)
    4b. Beads       — one article-thread bead per reading-order block
    5.  Audit       — report any printed ink still outside every OCR'd box

    Returns (elements in reading order, article bead boxes as page fractions).
    """
    logger.info("")
    logger.info("─" * 62)
    logger.info(f"  PAGE {page_num}  [{pil_image.width}×{pil_image.height}]  {filename}")
    logger.info("─" * 62)
    _READ_STATS.clear()

    # ── Stage 1: Preprocess ───────────────────────────────────────────────────
    processed = preprocess_newspaper(pil_image)
    processed.save(str(_debug_img_path(stem, page_num, "00_preprocessed")),
                   "JPEG", quality=85)

    # ── Stage 2: Surya layout detection ──────────────────────────────────────
    logger.info("  [LAYOUT] Running LayoutPredictor …")
    layout_regions: List[Dict]         = []
    image_bbox: Optional[List[float]]  = None

    try:
        layout_out = layout_predictor([processed])
        if layout_out:
            # Pass processed.size so parse_layout_result can rescale bbox
            # coordinates from Surya's internal model space back to our
            # full 300-DPI pixel space.
            layout_regions, image_bbox = parse_layout_result(
                layout_out[0], processed.size
            )
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
                          _debug_img_path(stem, page_num, "01_layout"))
        save_layout_report(layout_regions, image_bbox,
                           _debug_path(stem, page_num, "01_layout_report", "txt"),
                           filename, page_num)

        # Stage 2b: correct Surya's position for misplaced masthead/banner
        # regions before anything downstream trusts it (see
        # _reorder_banner_regions). Debug artefacts above show Surya's raw
        # positions; [ORDER] log lines report what this step changed.
        _reorder_banner_regions(layout_regions)

    # ── Stage 3: Per-region DetectionPredictor + TrOCR ────────────────────────
    # Default-to-OCR: anything not explicitly in SKIP_LABELS is read, so a label
    # the alias map has never seen can no longer silently drop content.
    skip_regions    = [r for r in layout_regions if r["label"] in SKIP_LABELS]
    text_regions    = [r for r in layout_regions if r["label"] not in SKIP_LABELS]
    unknown_regions = [r for r in text_regions   if r["label"] not in OCR_LABELS]
    if unknown_regions:
        unk_lbls = sorted({r["label"] for r in unknown_regions})
        logger.warning(
            f"  [LAYOUT] {len(unknown_regions)} region(s) with unrecognised "
            f"label(s) {unk_lbls} — OCR'ing anyway."
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
            processed, pil_image, region,
            det_predictor, trocr_processor, trocr_model,
        )
        logger.debug(f"      → {len(elems)} element(s) accepted.")
        all_elements.extend(elems)

    # ── Stage 3b: Recall sweep ────────────────────────────────────────────────
    if SWEEP_ENABLED:
        recovered = sweep_uncovered_text(
            processed, pil_image, all_elements, layout_regions,
            det_predictor, trocr_processor, trocr_model,
        )
        logger.info(
            f"  [SWEEP] recovered {len(recovered)} line(s) the layout stage missed."
        )
        all_elements.extend(recovered)

    # ── Stage 3c: De-duplicate cross-label region overlaps ────────────────────
    before_dedupe = len(all_elements)
    all_elements  = dedupe_overlapping_elements(all_elements)
    if len(all_elements) < before_dedupe:
        logger.info(
            f"  [DEDUPE] {before_dedupe - len(all_elements)} duplicate element(s) removed."
        )

    # ── Stage 4: Final reading order sort ────────────────────────────────────
    # Same-position elements are grouped into visual rows by y-overlap (not a
    # raw y0 compare — see _assign_visual_rows), then rows sorted top-to-
    # bottom and elements within a row left-to-right by x.
    _assign_visual_rows(all_elements)
    all_elements.sort(key=lambda e: (e["reading_position"], e["_row"], e["bbox"][0]))
    for e in all_elements:
        del e["_row"]

    logger.info(f"  [RESULT] {len(all_elements)} OCR element(s) on page {page_num}.")
    logger.info(
        f"  [TrOCR] {_READ_STATS['reads']} read(s); beam retries {_READ_STATS['beam_retries']}: "
        f"{_READ_STATS['beam_changed']} adopted, "
        f"{_READ_STATS['beam_lower']} rejected (lower conf), "
        f"{_READ_STATS['beam_cosmetic']} cosmetic-only, "
        f"{_READ_STATS['beam_same']} unchanged."
    )

    # ── Stage 4b: Article-thread beads (reading order as PDF structure) ──────
    beads = _article_bead_boxes(all_elements, layout_regions, pil_image.size)

    # ── Stage 5: Coverage audit (diagnostic only) ─────────────────────────────
    if AUDIT_ENABLED:
        try:
            audit_coverage(processed, all_elements, layout_regions, stem, page_num)
        except Exception as exc:
            logger.warning(f"  [COVERAGE] audit failed: {exc}")

    if all_elements:
        save_ocr_debug(pil_image, all_elements,
                       _debug_img_path(stem, page_num, "02_ocr_overlay"))
        save_ocr_report(all_elements,
                        _debug_path(stem, page_num, "02_ocr_report", "txt"),
                        filename, page_num)

    return all_elements, beads


# ══════════════════════════════════════════════════════════════════════════════
# INVISIBLE TEXT LAYER
# ══════════════════════════════════════════════════════════════════════════════

def insert_text_layer(
    page: "fitz.Page",
    elements: List[Dict],
    img_size: Tuple[int, int],
    font: "fitz.Font",
) -> int:
    """
    Insert `elements` into `page` as invisible text (render mode 3), each line
    stretched to exactly fill its printed box.

    Pixel coordinates (from the OCR render) are scaled to PDF points.  For
    every line:
      - font size = the line's height-derived size (unchanged from before);
      - baseline  = box bottom raised by the font's descender, so the glyph
                    box sits on the printed line rather than below it;
      - a horizontal morph matrix scales the string so it spans exactly
        x0 → x1 of the box (clipped to the page), whatever the OCR text's
        natural width.  This is the same approach as Tesseract/OCRmyPDF's
        text layers (their Tz operator): viewers that reconstruct lines and
        columns from glyph geometry rather than content-stream order
        (e.g. Nitro PDF Pro) then see text exactly where the ink is.

    Each line is written with its own TextWriter.write_text() call, in the
    given (reading) order, so the content stream still follows Surya's
    reading order.  PyMuPDF reuses one embedded font object across the calls,
    and save(clean=True) merges the per-line content snippets into a single
    stream.

    ELEMENT_SEPARATOR (a space) is still appended to every line as a guard
    against word fusion between lines/columns.  The stretch is computed on
    the text WITHOUT it, so the space falls just past the line's right edge,
    which is where a reader expects a word break.

    Returns the number of elements inserted.
    """
    iw, ih = img_size
    pw, ph = page.rect.width, page.rect.height
    sx, sy = pw / iw, ph / ih

    logger.debug(f"    {iw}×{ih}px → {pw:.1f}×{ph:.1f}pt  (sx={sx:.4f}, sy={sy:.4f})")

    inserted = 0
    stretches: List[float] = []

    for elem in elements:
        bx0, by0, bx1, by1 = elem["bbox"]
        try:
            body = elem["text"].strip()
            if not body:
                continue
            x0_pt    = max(0.0, bx0 * sx)
            x1_pt    = min(pw,  bx1 * sx)
            target_w = x1_pt - x0_pt
            if target_w <= 0:
                continue
            fontsize  = max(MIN_FONT_PT, elem["font_size"] * sy)
            natural_w = font.text_length(body, fontsize=fontsize)
            if natural_w <= 0:
                continue
            stretch = min(MAX_TEXT_STRETCH, max(MIN_TEXT_STRETCH, target_w / natural_w))
            origin  = fitz.Point(x0_pt, by1 * sy + font.descender * fontsize)

            writer = fitz.TextWriter(page.rect)
            writer.append(origin, body + ELEMENT_SEPARATOR, font=font, fontsize=fontsize)
            writer.write_text(
                page,
                overlay=True,
                render_mode=3,
                color=(0, 0, 0),
                morph=(origin, fitz.Matrix(stretch, 1)),
            )
            inserted += 1
            stretches.append(stretch)
        except Exception as exc:
            logger.debug(f"    Text insert skipped: {exc}")

    if stretches:
        s = sorted(stretches)
        logger.debug(
            f"    Horizontal stretch: min {s[0]:.2f}  median {s[len(s) // 2]:.2f}  max {s[-1]:.2f}"
        )
    return inserted


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
    """
    Add an invisible OCR text layer to `input_path` and write `output_path`.

    Workflow (no flatten step — original image streams are never re-encoded):
      1. Open the ORIGINAL PDF.
      2. For each page: render (OCR copy only) → OCR → strip any pre-existing
         text layer → insert the new invisible text layer; collect the
         page's article-thread beads.
      3. Write Info-dict + XMP metadata and accessibility catalog entries.
      4. Subset the embedded font (if fonttools is installed) and save once
         with deflate.
      5. pikepdf pass: PDF/A sRGB OutputIntent + article threads; then verify
         extractable text.

    Returns True if the output PDF contains extractable text.
    """
    filename = os.path.basename(input_path)
    stem     = Path(input_path).stem
    thread_specs: Dict[int, Dict] = {}

    logger.info(f"\n{'━'*62}")
    logger.info(f"  PROCESSING: {filename}")
    logger.info(f"{'━'*62}")

    try:
        with fitz.open(input_path) as doc:
            n_pages = len(doc)
            logger.info(f"  {n_pages} page(s) — rendering at {DPI} DPI for OCR …")

            font, using_freesans = load_text_font()
            logger.info(
                f"  Text-layer font: "
                f"{'FreeSans (embedded, subset if fonttools is installed)' if using_freesans else 'built-in substitute (embedded, unsubset)'}"
            )

            for idx in range(n_pages):
                page_num = idx + 1
                page     = doc[idx]

                # ── Render → OCR (one page at a time; image freed afterwards) ─
                pil_img  = page_to_pil(page, dpi=DPI)
                img_size = pil_img.size
                try:
                    elements, beads = process_page(
                        pil_img, page_num, filename, stem,
                        det_predictor, layout_predictor,
                        trocr_processor, trocr_model,
                    )
                except Exception as exc:
                    logger.error(f"  OCR failed for page {page_num}: {exc}")
                    import traceback; traceback.print_exc()
                    elements, beads = [], []
                del pil_img

                # ── Strip any pre-existing text layer ─────────────────────────
                # Previously OCR'd / partially searchable scans would otherwise
                # end up with a doubled text layer.  A page-sized redaction is
                # applied with PDF_REDACT_IMAGE_NONE so image streams are
                # untouched — only text operators are removed.
                # (h) Done on EVERY page, before the no-elements check, so a
                # page where OCR finds nothing ends up with no text rather than
                # silently keeping the input's original (non-TrOCR) text layer.
                # All text in the output therefore comes from TrOCR.
                existing_text = page.get_text().strip()
                if existing_text:
                    logger.info(
                        f"  Page {page_num}: removing existing text layer "
                        f"({len(existing_text)} chars) before inserting OCR."
                    )
                    page.add_redact_annot(page.rect)
                    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

                if not elements:
                    logger.info(f"  Page {page_num}: no elements (page left without text).")
                    continue

                inserted = insert_text_layer(page, elements, img_size, font)
                logger.info(
                    f"  Page {page_num}: inserted {inserted}/{len(elements)} element(s)."
                )

                if beads:
                    thread_specs[idx] = {
                        "title":    f"{stem} — page {page_num}",
                        "rotation": page.rotation,
                        "boxes":    beads,
                    }

            # ── Metadata + accessibility entries ──────────────────────────────
            apply_document_metadata(doc, filename)

            # ── Shrink the embedded font to the glyphs actually used ──────────
            if using_freesans:
                try:
                    doc.subset_fonts()
                    logger.info("  Embedded font subset to the glyphs used.")
                except Exception as exc:
                    logger.warning(
                        f"  Font subsetting skipped ({exc}); the full font stays "
                        f"embedded.  `pip install fonttools` enables subsetting."
                    )

            # ── Save once; deflate streams, never re-encode images ────────────
            doc.save(
                output_path,
                deflate=True,          # compress streams (text, metadata, etc.)
                garbage=4,             # remove unused / merge duplicate objects
                clean=True,            # also merges the per-line text snippets
                deflate_images=False,  # leave original image streams untouched
                encryption=fitz.PDF_ENCRYPT_KEEP,
            )
            logger.info(f"  Saved: {output_path}")

        # ── PDF/A OutputIntent + article threads — MUST run after the save ────
        setup_pdfa_compliance(output_path, thread_specs)

        # ── Verify OCR layer ──────────────────────────────────────────────────
        with fitz.open(output_path) as chk:
            total_chars = 0
            for i, pg in enumerate(chk):
                n = len(pg.get_text().strip())
                total_chars += n
                logger.info(f"  Final PDF page {i+1}: {n} extractable characters")
        if total_chars > 0:
            logger.info(f"  SUCCESS: {total_chars} characters of searchable text.")
        else:
            logger.error("  PROBLEM: final PDF has no extractable text!")
        return total_chars > 0

    except Exception as exc:
        logger.error(f"  process_pdf failed: {exc}")
        import traceback; traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# COMPRESSION  (size targeting)
# ══════════════════════════════════════════════════════════════════════════════

def compress_to_target_size(input_pdf: Path, output_pdf: Path, original_size: int) -> Path:
    """
    Try to keep the output within 15 % of the original size using PDF-native
    deflate only.  Image streams are never re-encoded (no generation loss,
    archival quality preserved).  A result is accepted only if it fits the
    budget AND still contains extractable text.  (Article threads are
    reachable from the catalog and survive these re-saves.)
    """
    max_target = int(original_size * 1.15)
    current    = input_pdf.stat().st_size
    logger.info(
        f"  Target ≤ {max_target // 1024} KB (original {original_size // 1024} KB + 15%); "
        f"OCR file is {current // 1024} KB."
    )

    if current <= max_target:
        shutil.copy2(input_pdf, output_pdf)
        logger.info("  Within budget — no compression needed.")
        return output_pdf

    options = [
        {"deflate": True, "garbage": 4, "clean": True, "deflate_images": False},
        {"deflate": True, "garbage": 3, "clean": True, "deflate_images": False},
        {"deflate": True, "garbage": 2, "clean": True, "deflate_images": False},
    ]
    for i, opts in enumerate(options):
        tmp = output_pdf.with_suffix(f".temp_{i}.pdf")
        try:
            with fitz.open(str(input_pdf)) as d:
                d.save(str(tmp), **opts, encryption=fitz.PDF_ENCRYPT_KEEP)

            size = tmp.stat().st_size
            pct  = (size - original_size) / original_size * 100
            logger.info(f"  Option {i+1}: {size // 1024} KB ({pct:+.1f}% from original)")

            if size <= max_target:
                try:
                    with fitz.open(str(tmp)) as chk:
                        chars = sum(len(p.get_text().strip()) for p in chk)
                except Exception:
                    chars = -1
                if chars > 0:
                    shutil.move(str(tmp), str(output_pdf))
                    logger.info(f"  Option {i+1} accepted; OCR preserved ({chars} chars).")
                    return output_pdf
                logger.error("  OCR lost after compression — using the uncompressed OCR file.")
                tmp.unlink(missing_ok=True)
                shutil.copy2(input_pdf, output_pdf)
                return output_pdf
            tmp.unlink(missing_ok=True)
        except Exception as exc:
            logger.error(f"  Compression option {i+1} failed: {exc}")
            tmp.unlink(missing_ok=True)

    logger.warning(
        "  All deflate options exceeded the 15% budget; returning the OCR file "
        "as-is (images untouched)."
    )
    shutil.copy2(input_pdf, output_pdf)
    return output_pdf


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║  OPTICOLUMNS 2026  –  Surya Layout / TrOCR Recognition       ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")
    logger.info(f"  Input    : {INPUT_DIR}")
    logger.info(f"  Output   : {OUTPUT_DIR}")
    logger.info(f"  Debug    : {DEBUG_PATH.resolve()}")
    logger.info(f"  DPI      : {DPI}")
    logger.info(f"  TrOCR    : {TROCR_MODEL_NAME}")
    logger.info(
        f"  Recog.   : {'raw render' if RECOGNIZE_FROM_RAW else 'preprocessed page'}, "
        f"pad {LINE_PAD_X}×{LINE_PAD_Y}px ({LINE_PAD_X_MODE}), "
        f"punct spacing {'normalised' if NORMALIZE_PUNCT_SPACING else 'raw'}"
    )
    logger.info(
        f"  Beams    : "
        + (f"{BEAM_NUM_BEAMS} when greedy conf in [max({BEAM_RETRY_ABOVE:.2f}, floor), "
           f"{BEAM_RETRY_BELOW:.2f}) and ≥{BEAM_MIN_CHARS} chars; adopt only if conf ≥ greedy"
           if BEAM_RETRY_ENABLED else "off (greedy only)")
    )
    logger.info(f"  Headers  : split only when ≥{HEADER_SPLIT_MIN_H}px tall and AR > {MAX_HEADER_AR}")
    logger.info(f"  Sweep    : {'on' if SWEEP_ENABLED else 'off'}")
    logger.info(f"  Threads  : {'on' if ARTICLE_THREADS_ENABLED else 'off'}")
    logger.info(
        f"  Debug out: {'overwrite (latest_*.jpg / .txt)' if DEBUG_OVERWRITE else 'per page'}"
    )
    logger.info("")

    input_folder  = Path(INPUT_DIR)
    output_folder = Path(OUTPUT_DIR)

    if not input_folder.exists():
        logger.error(f"Input folder '{INPUT_DIR}' not found.")
        sys.exit(1)
    output_folder.mkdir(exist_ok=True)
    clear_debug_files()

    pdf_files = sorted(input_folder.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files in '{INPUT_DIR}'.")
        sys.exit(1)

    # ── Identify already-processed files and skip them ────────────────────────
    pending: List[Path] = []
    summary: List[Tuple[str, str, int]] = []
    for pdf_path in pdf_files:
        final_path = output_folder / f"{pdf_path.stem}.pdf"
        if SKIP_EXISTING_OUTPUT and final_path.exists():
            summary.append((pdf_path.name, "SKIPPED", final_path.stat().st_size))
        else:
            pending.append(pdf_path)

    skipped = [n for n, s, _ in summary if s == "SKIPPED"]
    if skipped:
        logger.info(f"  Skipping {len(skipped)} already-processed file(s): {', '.join(skipped)}")
    if not pending:
        logger.info("  All files have already been processed. Nothing to do.")
        return

    logger.info(f"  Processing {len(pending)} file(s).  Target: size ≤ original + 15% "
                f"(OCR text layer only; images untouched)\n")

    # Load all models once (only when there is work to do); reuse for every file
    det_predictor, layout_predictor, trocr_processor, trocr_model = load_models()

    for pdf_path in pending:
        orig_size = pdf_path.stat().st_size
        logger.info(f"\n{'━'*62}")
        logger.info(f"  FILE: {pdf_path.name}  ({orig_size // 1024} KB)")

        tmp_path   = output_folder / f"{pdf_path.stem}_ocr_temp.pdf"
        final_path = output_folder / f"{pdf_path.stem}.pdf"

        ok = process_pdf(
            str(pdf_path), str(tmp_path),
            det_predictor, layout_predictor,
            trocr_processor, trocr_model,
        )

        if not ok:
            logger.error("  Skipping — OCR stage failed.")
            summary.append((pdf_path.name, "FAILED", 0))
            tmp_path.unlink(missing_ok=True)
            continue

        result = compress_to_target_size(tmp_path, final_path, orig_size)
        fsz    = result.stat().st_size
        delta  = (fsz - orig_size) / orig_size * 100.0
        logger.info(f"\n  ✓ {result.name}  {fsz // 1024} KB ({delta:+.1f}% from original)")
        summary.append((pdf_path.name, "OK", fsz))

        try:
            tmp_path.unlink()
        except Exception as exc:
            logger.warning(f"  Could not delete temp file: {exc}")

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
    logger.info(f"\nAll done! Output files in '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()