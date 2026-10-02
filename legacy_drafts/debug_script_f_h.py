#!/usr/bin/env python3
"""
Opticolumns  –  debug_script_f_h.py
======================================================================
Base: debug_script_f_g.py.  Surya layout + TrOCR recognition.
"""

import sys
import os
import re
import csv
import math
import time
import bisect
import difflib
import heapq
import datetime
import shutil
import platform
import logging
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple
from xml.sax.saxutils import escape as xml_escape

import numpy as np
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

SKIP_EXISTING_OUTPUT = True
DPI = 300

# ── Branding / document metadata ──────────────────────────────────────────────
APP_NAME      = "Opticolumns"
APP_VERSION   = "2026"
APP_CREATOR   = f"{APP_NAME} {APP_VERSION}"
DOC_SUBJECT   = "OCR-processed historic newspaper"
DOC_LANGUAGE  = "en-US"
OPT_NAMESPACE = "https://github.com/Scholarly-Projects/opticolumns/ns/"

# ── Page preparation ──────────────────────────────────────────────────────────
LAYOUT_AUTOCONTRAST_CUTOFF = 0.5  # % clipped each end for the global normalisation
BG_REDUCE      = 8                # background estimate computed at 1/8 size
BG_MAX_KERNEL  = 9                # max-filter (odd) at 1/8 size ≈ 72 px: removes text, keeps paper
BG_BLUR        = 6
# Image Surya layout sees: "norm" = globally normalised render (F-F behaviour);
# "flat" = background-flattened (helps yellowed / vignetted microfilm, but can
# wash out very heavy masthead strokes wider than ~70 px).
LAYOUT_SOURCE  = "norm"

# Rule (printed line) removal – applied to the detection/OCR image only.
RULE_REMOVAL_ENABLED   = True
RULE_INK_LEVEL         = 140
RULE_MIN_LEN_FRAC_V    = 0.06
RULE_MIN_LEN_FRAC_H    = 0.06
RULE_MAX_THICK_PX      = max(4, round(DPI * 0.03))
RULE_CHUNK             = 512
RULE_SPLIT_LABELS      = {"Text", "Section-header", "Caption", "List-item", "Footnote"}
RULE_EDGE_MARGIN       = 40
RULE_SPLIT_MIN_YFRAC   = 0.5

# ── Layout completeness (problem 1) ───────────────────────────────────────────
SURYA_LAYOUT_MAX_SIDE   = 1024    # surya 0.17 layout task input size (foundation img_size); for logging
LAYOUT_TILING           = True
LAYOUT_TILE_COLS        = 3       # tiles across the page …
LAYOUT_TILE_ROWS        = None    # … and down (None = follow the page aspect ratio)
LAYOUT_TILE_OVERLAP     = 0.20    # fraction of a tile shared with its neighbour
LAYOUT_TILE_EDGE_MARGIN = 24      # px; a box this close to an interior tile edge is a fragment
LAYOUT_TILE_ALIGN       = 0.60    # fragments join when they share ≥ 60 % of the narrower extent
LAYOUT_TILE_ADD_BELOW   = 0.50    # tile text region added where page regions cover < 50 % of it
PICTURE_TILE_TEXT_COVER = 0.40    # page "Picture" ≥ 40 % covered by tile text → replaced by that text

PICTURE_RESCUE          = True
PICTURE_MIN_LINES       = 3       # detected text lines inside a Picture before it is re-examined
PICTURE_TEXT_LINE_COVER = 0.20    # line boxes cover ≥ 20 % of the picture's area
PICTURE_MAX_MIDTONE     = 0.30    # photos / halftones have more mid-grey pixels than this
PICTURE_FORCE_COVER     = 0.40    # convert even when Surya's re-layout still says Picture …
PICTURE_MAX_HEIGHT_CV   = 0.45    # … if the lines are this regular (std / mean of heights) …
PICTURE_TOPK_TEXT       = 0.15    # … or Surya's own top-k gave text labels this much weight

# ── Column gutters measured from the page (F-H) ──────────────────────────────
GUTTERS_ENABLED        = True
GUTTER_SCALE           = 4        # gutter map works on 4×4 px cells
GUTTER_INK_LEVEL       = 160      # pixel darker than this (flattened, rule-free page) is ink
GUTTER_CELL_MIN_DARK   = 4        # a cell is ink if ≥ 4 of its 16 pixels are dark
GUTTER_MIN_W_PX        = max(8, round(DPI * 0.035))   # narrowest gutter (≈10 px at 300 DPI)
GUTTER_RULE_MIN_LINES  = 1.5      # thin vertical strokes longer than this (× line height) are rule dashes
GUTTER_MIN_TALL_LINES  = 12       # gap must stay blank ≥ 12 line-heights (word gaps and rivers don't)
GUTTER_LOOSE_TALL_LINES = 3       # looser test used only for the COLUMN_BLEED check …
GUTTER_BLEED_NEAR_BOUND = 0.10    # … and only within ± 10 % of a column width of a column boundary
GUTTER_REGION_BLANK    = 0.92     # a box is split where a gap is blank over ≥ 92 % of its height
GUTTER_SPLIT_LABELS    = {"Text", "List-item", "Caption", "Footnote"}   # headlines/decks never split
GUTTER_SPLIT_MIN_LINES = 3        # only boxes at least 3 lines tall are split
GUTTER_MIN_PIECE_FRAC  = 0.30     # each piece of a split box must be ≥ 30 % of a column wide
GUTTER_SNAP            = True     # move text-box edges onto the measured gutter
GUTTER_SNAP_FRAC       = 0.12     # search ± 12 % of a column width around each edge
GUTTER_SNAP_PAD        = 6        # px left between the text and the snapped edge

# ── Short subheads the sweep used to drop (F-H) ───────────────────────────────
SUBHEAD_RESCUE         = True
SUBHEAD_MAX_LINES      = 2
SUBHEAD_MIN_WIDTH_FRAC = 0.15     # × column width
SUBHEAD_NEIGHBOUR_GAP  = 3.0      # text block above and below within 3 line-heights …
SUBHEAD_X_INSIDE       = 0.80     # … whose x-span contains ≥ 80 % of the subhead

# ── TrOCR ─────────────────────────────────────────────────────────────────────
TROCR_MODELS = {
    "handwritten":       "microsoft/trocr-base-handwritten",
    "printed":           "microsoft/trocr-base-printed",
    "large_handwritten": "microsoft/trocr-large-handwritten",
    "large_printed":     "microsoft/trocr-large-printed",
}
TROCR_MODEL_NAME          = TROCR_MODELS["large_handwritten"]
TROCR_FALLBACK_MODEL_NAME = TROCR_MODELS["large_printed"]   # also the legibility gate's second reader
FALLBACK_BELOW_CONF       = 0.80
FALLBACK_SWITCH_MARGIN    = 0.03

NUM_BEAMS          = 3
TROCR_BATCH_SIZE   = 8
MAX_NEW_TOKENS     = 192
USE_MPS            = True

# ── Noise filter ──────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD             = 0.25
SINGLE_CHAR_CONFIDENCE_THRESHOLD = 0.50
MIN_LINE_H                       = 5
MIN_LINE_W                       = 10
SPARSE_LINE_WIDTH_RATIO          = 2.0
HALLUCINATION_MAX_CONF           = 0.60
HALLUCINATION_PATTERNS = [
    r"^(1[89]\d\d|20\d\d)(\s+\S{1,4})*$",
    r"^[0-9.,\s]*0[.,]0{3}[0-9.,\s]*$",
    r"^[\W\d_]+$",
]

# ── Legibility gate for low-confidence pages (problem 2) ─────────────────────
STRICT_ENABLED           = True
LOW_CONF_LINE            = 0.60   # a line read below this counts as low-confidence
STRICT_PAGE_MEAN_CONF    = 0.85   # page is "low-confidence" if its raw mean line conf is below this …
STRICT_PAGE_LOW_FRAC     = 0.20   # … or more than 20 % of its body lines are low-confidence
STRICT_REGION_TRIGGER    = 0.70   # on any page: region whose median line conf is below this is gated
STRICT_REGION_MIN_LINES  = 3      # (only regions with at least this many lines)
STRICT_LINE_CONF         = 0.80   # gated line must be read at least this confidently …
STRICT_MIN_AGREEMENT     = 0.75   # … both TrOCR models must read it alike (character similarity) …
STRICT_MIN_LEXICAL       = 0.60   # … and this share of its words must be real words
STRICT_LEXICAL_MIN_WORDS = 2      # lexical test applies to lines of ≥ 2 words
STRICT_REGION_MIN_KEEP   = 0.50   # gated region where fewer lines pass is dropped entirely
# Word lists (first ones found are merged).  Put local names, places and
# period vocabulary one per line in lexicon.txt to help them pass.
LEXICON_PATHS    = ["lexicon.txt", "/usr/share/dict/words", "/usr/share/dict/web2"]
LEXICON_MIN_SIZE = 5000           # below this the lexical test is switched off

# ── Line crops ────────────────────────────────────────────────────────────────
LINE_PAD_X = 0.30
LINE_PAD_Y = 0.10
CROP_BORDER_FRAC = 0.10
MAX_HEADER_AR   = 10.0
HEADER_SEGMENT_OVERLAP = 0.20
EDGE_STROKE_FRAC = 0.80

# ── Line detection / ownership ────────────────────────────────────────────────
DET_PAD             = 12
LINE_MIN_IN_REGION  = 0.50
LINE_DUP_CONTAIN    = 0.60
LINE_CLIP_PAD       = 6

# ── Page-wide line inventory (finds text no region claimed) ──────────────────
INVENTORY_TILE     = 1920
INVENTORY_OVERLAP  = 960
INVENTORY_EDGE_MARGIN = 6
ORPHAN_REGION_PAD     = 8
ORPHAN_MAX_COVERED    = 0.30
ORPHAN_LINE_GAP       = 2.5
ORPHAN_MIN_WIDTH_FRAC = 0.45
ORPHAN_EDGE_FRAC      = 0.02
ORPHAN_EDGE_MAX_WIDTH_FRAC = 0.50
ORPHAN_MIN_LINES      = 2
RELAYOUT_MIN_LINES    = 3
RELAYOUT_PAD          = 30

# ── Region housekeeping ───────────────────────────────────────────────────────
MIN_REGION_W = 40
MIN_REGION_H = 15
GUTTER_MAX_OVERLAP_FRAC = 0.25
WHOLE_CROP_MAX_LINES    = 3.0

# ── Reading order ─────────────────────────────────────────────────────────────
READING_ORDER_MODE    = "columns"
COL_LABELS            = {"Text", "List-item", "Caption", "Footnote", "Section-header",
                         "Handwriting", "Text-inline-math"}
COL_MAX_WIDTH_FACTOR  = 1.40
COL_CLUSTER_GAP       = 0.40
COL_MIN_SUPPORT_FRAC  = 0.05
COL_SPAN_MIN_OVERLAP  = 0.40
HBREAK_TOL_FRAC       = 0.006
BODY_MIN_H_FRAC       = 0.03
XY_SHRINK_FRAC        = 0.03
XY_SHRINK_MAX         = 12
XY_MIN_GAP            = 2
XY_VERTICAL_TOL_FRAC  = 0.008
XY_SPAN_FACTOR        = 1.40
LINE_SUBCOL_MIN_LINES = 3
LINE_SUBCOL_MIN_SPAN  = 0.40
ROW_OVERLAP_FRACTION  = 0.5
ROW_MAX_X_OVERLAP     = 0.5

# ── Safety-net dedupe ─────────────────────────────────────────────────────────
DEDUPE_CONTAINMENT     = 0.5
DEDUPE_MAX_AREA_RATIO  = 3.0
DEDUPE_MIN_TEXT_SIM    = 0.60

# ── QA thresholds (page flagged in qa_report.csv) ────────────────────────────
QA_MIN_ORDER_AGREEMENT = 0.85
QA_MAX_UNCOVERED       = 0.12
QA_MIN_MEAN_CONF       = 0.70
QA_MAX_BLEED_LINES     = 2        # COLUMN_BLEED flag when more final lines than this still cross a gutter

# ── Coverage audit ────────────────────────────────────────────────────────────
AUDIT_ENABLED     = True
AUDIT_SCALE       = 8
AUDIT_INK_LEVEL   = 215
AUDIT_MIN_BAND_PX = 150
AUDIT_TEXT_AREA_ONLY = True       # ignore ink outside the page's text area (facing page, borders)
AUDIT_TEXT_AREA_PAD  = 0.01       # × page width, padding around that area

DEBUG_OVERWRITE = True

# ── Layout label taxonomy ─────────────────────────────────────────────────────
OCR_LABELS = {
    "Text", "Section-header", "Caption", "Footnote", "List-item", "Page-footer",
    "Page-header", "Table-of-contents", "Handwriting", "Text-inline-math",
    "Formula", "Table", "Form",
}
SKIP_LABELS = {"Picture", "Figure"}
HEADER_LABELS = {"Section-header", "Page-header"}
ROW_ORDER_LABELS = {"Table", "Form", "Table-of-contents"}

# ── Text layer ────────────────────────────────────────────────────────────────
EMBED_FONT      = True
TEXT_LAYER_MODE = "shape"
TEXT_FONT_TAG   = "OcrTxt"
TEXT_HEIGHT_FACTOR = 0.80
TEXT_BASELINE_FRAC = 0.20
HSCALE_MIN, HSCALE_MAX = 0.20, 4.0
MIN_FONT_PT         = 1.5
MAX_FONT_PT         = 72.0
ELEMENT_SEPARATOR   = " "
FONT_NAME     = "helv"
FONT_PATH     = "fonts/FreeSans.ttf"
FONT_URL      = ("https://github.com/opensourcedesign/fonts/raw/master/"
                 "gnu-freefont_freesans/FreeSans.ttf")
SRGB_ICC_PATH = "srgb.icc"

LABEL_COLOURS: Dict[str, str] = {
    "Page-header": "#1565C0", "Section-header": "#C62828", "Text": "#2E7D32",
    "Caption": "#6A1B9A", "Footnote": "#4E342E", "Page-footer": "#37474F",
    "Table": "#E65100", "Picture": "#00838F", "Figure": "#00695C",
    "List-item": "#558B2F", "Handwriting": "#AD1457", "Form": "#FF6F00",
    "Table-of-contents": "#0277BD", "Recovered": "#FFD600",
}
DEFAULT_COLOUR = "#9E9E9E"


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

DEBUG_PATH = Path(DEBUG_DIR)
DEBUG_PATH.mkdir(exist_ok=True)
QA_REPORT_PATH = DEBUG_PATH / "qa_report.csv"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(DEBUG_PATH / "run.log"), mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)
for _noisy in ("urllib3", "PIL", "filelock", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

_RESAMPLE = getattr(Image, "Resampling", Image)


# ══════════════════════════════════════════════════════════════════════════════
# DATA TYPES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Region:
    rid: int
    bbox: List[float]
    label: str
    position: Optional[float]          # Surya reading-order position (None = not from Surya page layout)
    top_k: Dict[str, float] = field(default_factory=dict)
    source: str = "layout"             # "layout" | "tile" | "picture-text" | "relayout" | "orphan" | "fallback"

    @property
    def w(self) -> float: return self.bbox[2] - self.bbox[0]
    @property
    def h(self) -> float: return self.bbox[3] - self.bbox[1]
    @property
    def is_text(self) -> bool: return self.label not in SKIP_LABELS


@dataclass
class PageImages:
    raw_rgb: Image.Image
    norm_gray: Image.Image       # global autocontrast of the render (header crops)
    layout_rgb: Image.Image      # what Surya layout sees
    clean_gray: Image.Image      # background-flattened, rules removed (detection + body OCR)
    vrules: List[List[float]]
    rule_frac: float


# ══════════════════════════════════════════════════════════════════════════════
# SMALL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _hex_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _label_hex(label: str) -> str:
    return LABEL_COLOURS.get(label, DEFAULT_COLOUR)


def _pil_font(size: int = 16):
    try:
        return ImageFont.truetype(FONT_PATH, size=size)
    except Exception:
        return ImageFont.load_default()


def _area(b: Sequence[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _overlap_area(a: Sequence[float], b: Sequence[float]) -> float:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, ix) * max(0.0, iy)


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    inter = _overlap_area(a, b)
    if inter <= 0:
        return 0.0
    return inter / (_area(a) + _area(b) - inter)


def _frac_inside(a: Sequence[float], b: Sequence[float]) -> float:
    """Fraction of box a's area that lies inside box b."""
    return _overlap_area(a, b) / max(1.0, _area(a))


def _grow(b: Sequence[float], px: float) -> List[float]:
    return [b[0] - px, b[1] - px, b[2] + px, b[3] + px]


def _clip_box(b: Sequence[float], w: float, h: float) -> List[float]:
    return [max(0.0, b[0]), max(0.0, b[1]), min(float(w), b[2]), min(float(h), b[3])]


def _intersect(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]


def _center(b: Sequence[float]) -> Tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _median(vals: Sequence[float], default: float) -> float:
    vals = [v for v in vals if v > 0]
    return statistics.median(vals) if vals else default


def _coverage(box: Sequence[float], covers: List[List[float]]) -> float:
    return min(1.0, sum(_overlap_area(box, c) for c in covers) / max(1.0, _area(box)))


def _is_text_label(label: str) -> bool:
    return label not in SKIP_LABELS


# ══════════════════════════════════════════════════════════════════════════════
# PAGE PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def page_to_pil(page: "fitz.Page", dpi: int = DPI) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def _flatten_background(gray: Image.Image) -> Image.Image:
    """
    Divide out uneven paper tone (yellowing, microfilm vignetting) using a
    low-frequency background estimate.  Blank gutters stay blank.
    """
    small = gray.reduce(BG_REDUCE)
    bg = small.filter(ImageFilter.MaxFilter(BG_MAX_KERNEL)).filter(
        ImageFilter.GaussianBlur(BG_BLUR))
    bg = bg.resize(gray.size, _RESAMPLE.BILINEAR)
    g = np.asarray(gray, dtype=np.float32)
    b = np.maximum(np.asarray(bg, dtype=np.float32), 1.0)
    out = np.clip(g / b * 255.0, 0, 255).astype(np.uint8)
    return ImageOps.autocontrast(Image.fromarray(out, "L"), cutoff=0.5)


def _long_runs(mask: np.ndarray, length: int, axis: int) -> np.ndarray:
    """Pixels of `mask` belonging to an unbroken run of ≥ `length` along `axis`."""
    if axis == 1:
        return np.ascontiguousarray(_long_runs(np.ascontiguousarray(mask.T), length, 0).T)
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    if length <= 1:
        return mask.astype(bool).copy()
    if length > h:
        return out
    n = h - length + 1
    y = np.arange(h)
    hi = np.minimum(y + 1, n)
    lo = np.maximum(y - length + 1, 0)
    for c0 in range(0, w, RULE_CHUNK):
        m = mask[:, c0:c0 + RULE_CHUNK]
        cs = np.zeros((h + 1, m.shape[1]), dtype=np.int32)
        cs[1:] = np.cumsum(m, axis=0, dtype=np.int32)
        full = (cs[length:] - cs[:-length]) == length
        fs = np.zeros((n + 1, m.shape[1]), dtype=np.int32)
        fs[1:] = np.cumsum(full, axis=0, dtype=np.int32)
        out[:, c0:c0 + RULE_CHUNK] = (fs[hi] - fs[lo]) > 0
    return out


def _dilate1(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _runs_1d(flags: np.ndarray, max_gap: int = 0) -> List[Tuple[int, int]]:
    """[start, end) runs of True in a 1-D bool array, bridging gaps ≤ max_gap."""
    idx = np.flatnonzero(flags)
    if idx.size == 0:
        return []
    runs = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev > max_gap + 1:
            runs.append((start, prev + 1))
            start = i
        prev = i
    runs.append((start, prev + 1))
    return runs


def _extract_vertical_rules(vmask: np.ndarray, min_len: int) -> List[List[float]]:
    rules: List[List[float]] = []
    cols = vmask.any(axis=0)
    for x0, x1 in _runs_1d(cols, max_gap=2):
        rows = vmask[:, x0:x1].any(axis=1)
        for y0, y1 in _runs_1d(rows, max_gap=25):
            if y1 - y0 >= min_len:
                rules.append([float(x0), float(y0), float(x1), float(y1)])
    return rules


def remove_rules(arr: np.ndarray) -> Tuple[np.ndarray, List[List[float]], float]:
    """
    Detect thin, long printed rules and whiten them.  Returns (cleaned array,
    vertical rules, masked fraction).  Rules are removed only from the
    detection/OCR image; their positions are kept as hard column separators.
    """
    h, w = arr.shape
    dark = arr < RULE_INK_LEVEL
    lv = max(60, int(h * RULE_MIN_LEN_FRAC_V))
    lh = max(60, int(w * RULE_MIN_LEN_FRAC_H))
    t = RULE_MAX_THICK_PX
    v = _long_runs(dark, lv, 0)
    v &= ~_long_runs(v, t + 1, 1)
    hz = _long_runs(dark, lh, 1)
    hz &= ~_long_runs(hz, t + 1, 0)
    mask = _dilate1(v | hz)
    cleaned = arr.copy()
    cleaned[mask] = 255
    return cleaned, _extract_vertical_rules(v, lv), float(mask.mean())


def prepare_page(pil: Image.Image) -> PageImages:
    gray = pil.convert("L")
    norm = ImageOps.autocontrast(gray, cutoff=LAYOUT_AUTOCONTRAST_CUTOFF)
    flat = _flatten_background(norm)
    arr = np.asarray(flat, dtype=np.uint8)
    vrules: List[List[float]] = []
    frac = 0.0
    if RULE_REMOVAL_ENABLED:
        try:
            arr, vrules, frac = remove_rules(arr)
        except MemoryError:
            logger.warning("  Rule removal skipped (memory).")
    layout_src = flat if LAYOUT_SOURCE == "flat" else norm
    return PageImages(
        raw_rgb=pil, norm_gray=norm, layout_rgb=layout_src.convert("RGB"),
        clean_gray=Image.fromarray(arr, "L"), vrules=vrules, rule_frac=frac,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PDF/A  +  ACCESSIBILITY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _now() -> datetime.datetime:
    return datetime.datetime.now().astimezone()


def get_pdf_date_string(dt: Optional[datetime.datetime] = None) -> str:
    dt = dt or _now()
    off = dt.strftime("%z") or "+0000"
    return dt.strftime("D:%Y%m%d%H%M%S") + f"{off[0]}{off[1:3]}'{off[3:5]}'"


def get_xmp_date_string(dt: Optional[datetime.datetime] = None) -> str:
    dt = dt or _now()
    return dt.isoformat(timespec="seconds")


def _valid_icc(path: Path) -> bool:
    try:
        data = path.read_bytes()
        return len(data) > 128 and data[36:40] == b"acsp" and data[16:20] == b"RGB "
    except Exception:
        return False


def setup_pdfa_resources() -> bool:
    ok = True
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
                "Windows": [os.path.join(os.environ.get("WINDIR", r"C:\Windows"),
                            "System32", "spool", "drivers", "color",
                            "sRGB Color Space Profile.icm")],
            }.get(platform.system(), [])
            source = next((c for c in candidates
                           if Path(c).exists() and _valid_icc(Path(c))), None)
            if source:
                shutil.copy2(source, icc)
                logger.info(f"  sRGB profile copied from {source}")
            else:
                icc.write_bytes(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
                logger.info("  sRGB profile generated locally (Pillow/LittleCMS).")
    except Exception as exc:
        logger.warning(f"  Could not obtain sRGB ICC profile: {exc}")
        ok = False
    return ok


def load_text_font() -> Tuple["fitz.Font", bool]:
    if EMBED_FONT and Path(FONT_PATH).exists():
        try:
            return fitz.Font(fontfile=FONT_PATH), True
        except Exception as exc:
            logger.warning(f"  Could not load {FONT_PATH}: {exc}")
    logger.warning("  FreeSans unavailable — embedding PyMuPDF's built-in substitute font.")
    return fitz.Font(FONT_NAME), False


def create_xmp_metadata(title, author, subject, creator, producer,
                        creation_date, modify_date, language=DOC_LANGUAGE) -> Optional[str]:
    try:
        t, a, s, c, p, lang = (xml_escape(str(v)) for v in
                               (title, author, subject, creator, producer, language))
        app, ver = xml_escape(APP_NAME), xml_escape(APP_VERSION)
        ns = xml_escape(OPT_NAMESPACE)
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
    now = _now()
    pdf_date, xmp_date = get_pdf_date_string(now), get_xmp_date_string(now)
    producer = f"PyMuPDF {fitz.VersionBind}"
    doc.set_metadata({
        "title": filename, "author": APP_NAME, "subject": DOC_SUBJECT,
        "creator": APP_CREATOR, "producer": producer,
        "creationDate": pdf_date, "modDate": pdf_date,
    })
    xmp = create_xmp_metadata(filename, APP_NAME, DOC_SUBJECT, APP_CREATOR, producer,
                              xmp_date, xmp_date, DOC_LANGUAGE)
    if xmp:
        doc.set_xml_metadata(xmp)
    cat = doc.pdf_catalog()
    doc.xref_set_key(cat, "ViewerPreferences", "<</DisplayDocTitle true>>")
    doc.xref_set_key(cat, "Lang", f"({DOC_LANGUAGE})")


def setup_pdfa_compliance(pdf_path: str) -> None:
    try:
        icc = Path(SRGB_ICC_PATH)
        if not icc.exists() or not _valid_icc(icc):
            logger.warning("  Valid sRGB ICC profile not found; PDF/A OutputIntent skipped.")
            return
        with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
            existing = pdf.Root.get("/OutputIntents")
            has = existing is not None and any(
                str(oi.get("/S", "")) == "/GTS_PDFA1" for oi in existing)
            if not has:
                if "/OutputIntents" not in pdf.Root:
                    pdf.Root["/OutputIntents"] = pikepdf.Array()
                stream = pdf.make_stream(icc.read_bytes())
                stream.stream_dict["/N"] = pikepdf.Integer(3)
                stream.stream_dict["/Alternate"] = pikepdf.Name("/DeviceRGB")
                pdf.Root["/OutputIntents"].append(pdf.make_indirect(pikepdf.Dictionary({
                    "/Type": pikepdf.Name("/OutputIntent"),
                    "/S": pikepdf.Name("/GTS_PDFA1"),
                    "/Info": pikepdf.String("sRGB IEC61966-2.1"),
                    "/OutputConditionIdentifier": pikepdf.String("sRGB"),
                    "/DestOutputProfile": pdf.make_indirect(stream),
                })))
                logger.info("  PDF/A OutputIntent embedded.")
            pdf.save(pdf_path, object_stream_mode=pikepdf.ObjectStreamMode.disable)
    except Exception as exc:
        logger.error(f"  Failed to set up PDF/A compliance: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# TROCR RECOGNISER  (batched, beam search, calibrated confidence)
# ══════════════════════════════════════════════════════════════════════════════

def _pick_device() -> "torch.device":
    if torch.cuda.is_available():
        return torch.device("cuda")
    if USE_MPS and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Recognizer:
    def __init__(self, name: str):
        self.name = name
        self.processor = TrOCRProcessor.from_pretrained(name)
        self.model = VisionEncoderDecoderModel.from_pretrained(name)
        self.model.eval()
        self.device = _pick_device()
        self.model.to(self.device)
        gc = getattr(self.model, "generation_config", None)
        self.pad_id = (getattr(gc, "pad_token_id", None)
                       if gc is not None else None)
        if self.pad_id is None:
            self.pad_id = self.processor.tokenizer.pad_token_id
        logger.info(f"  TrOCR {name} on {self.device}")

    def _to_cpu(self) -> None:
        logger.warning(f"  {self.name}: falling back to CPU.")
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def _confidences(self, out, n: int) -> List[float]:
        """Mean per-token probability of the emitted sequence (padding after EOS excluded)."""
        try:
            if NUM_BEAMS > 1:
                ts = self.model.compute_transition_scores(
                    out.sequences, out.scores, out.beam_indices, normalize_logits=False)
            else:
                ts = self.model.compute_transition_scores(
                    out.sequences, out.scores, normalize_logits=True)
            t = ts.shape[1]
            toks = out.sequences[:, 1:1 + t]
            if toks.shape[1] < t:
                ts = ts[:, :toks.shape[1]]
            mask = (toks != self.pad_id) & torch.isfinite(ts)
            probs = torch.where(mask, torch.exp(ts), torch.zeros_like(ts))
            denom = mask.sum(dim=1).clamp(min=1)
            return (probs.sum(dim=1) / denom).float().cpu().tolist()
        except Exception as exc:
            logger.debug(f"    confidence fallback ({exc})")
            if getattr(out, "sequences_scores", None) is not None:
                return torch.exp(out.sequences_scores).float().cpu().tolist()
            return [0.5] * n

    def _read_batch(self, batch: List[Image.Image]) -> List[Tuple[str, float]]:
        pv = self.processor(images=[im.convert("RGB") for im in batch],
                            return_tensors="pt").pixel_values.to(self.device)
        kw = dict(max_new_tokens=MAX_NEW_TOKENS, num_beams=NUM_BEAMS,
                  output_scores=True, return_dict_in_generate=True)
        if NUM_BEAMS > 1:
            kw["early_stopping"] = True
        with torch.inference_mode():
            out = self.model.generate(pv, **kw)
        texts = [t.strip() for t in
                 self.processor.batch_decode(out.sequences, skip_special_tokens=True)]
        return list(zip(texts, self._confidences(out, len(batch))))

    def read(self, images: List[Image.Image]) -> List[Tuple[str, float]]:
        results: List[Tuple[str, float]] = []
        for i in range(0, len(images), TROCR_BATCH_SIZE):
            batch = images[i:i + TROCR_BATCH_SIZE]
            try:
                results.extend(self._read_batch(batch))
            except Exception as exc:
                if self.device.type != "cpu":
                    logger.warning(f"  TrOCR batch failed on {self.device}: {exc}")
                    self._to_cpu()
                    try:
                        results.extend(self._read_batch(batch))
                        continue
                    except Exception as exc2:
                        exc = exc2
                logger.error(f"  TrOCR batch failed: {exc}")
                results.extend([("", 0.0)] * len(batch))
        return results


# ══════════════════════════════════════════════════════════════════════════════
# LEXICON  (is this line made of real words?)
# ══════════════════════════════════════════════════════════════════════════════

class Lexicon:
    """
    English word list for the legibility gate.  Merges the files in
    LEXICON_PATHS (macOS and most Linux systems ship /usr/share/dict/words)
    with whole-word entries of the TrOCR tokenizer vocabulary (≥ 4 letters).
    Inflections are accepted by suffix stripping.
    """
    _COMMON = set((
        "a i an as at be by do go he if in is it me my no of on or so to up us we am "
        "the and for are but not you all any can had her was one our out day get has him "
        "his how man new now old see two way who boy did its let put say she too use yes "
        "yet mr mrs dr st co inc jr sr vs ave"
    ).split())
    _SUFFIXES = ("'s", "s", "es", "ed", "d", "ing", "ly", "er", "ers", "est", "ment", "ness", "ies")

    def __init__(self, tokenizer=None):
        words: Set[str] = set()
        sources = []
        for p in LEXICON_PATHS:
            path = Path(p)
            try:
                if not path.is_file():
                    continue
                n0 = len(words)
                with open(path, encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        w = line.strip().lower()
                        if w.isalpha():
                            words.add(w)
                sources.append(f"{p} (+{len(words) - n0})")
            except OSError:
                continue
        if tokenizer is not None:
            try:
                n0 = len(words)
                for tok in tokenizer.get_vocab():
                    if tok[:1] in ("Ġ", "▁"):
                        w = tok[1:]
                        if len(w) >= 4 and w.isalpha():
                            words.add(w.lower())
                sources.append(f"TrOCR vocabulary (+{len(words) - n0})")
            except Exception as exc:
                logger.debug(f"  Tokenizer vocabulary unavailable for lexicon: {exc}")
        words |= self._COMMON
        self.words = words
        self.available = len(words) >= LEXICON_MIN_SIZE
        if self.available:
            logger.info(f"  Lexicon: {len(words):,} words from " + ", ".join(sources))
        else:
            logger.warning(f"  Lexicon too small ({len(words)} words) — the legibility gate "
                           f"will not test for real words.  Add a word list to LEXICON_PATHS.")

    def known(self, w: str) -> bool:
        w = w.lower().strip("'")
        if not w:
            return False
        if w in self.words:
            return True
        for suf in self._SUFFIXES:
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                stem = w[:-len(suf)]
                if stem in self.words:
                    return True
                if suf in ("ing", "ed", "er", "est") and stem + "e" in self.words:
                    return True                                     # "making", "raised"
                if suf == "ies" and stem + "y" in self.words:
                    return True                                     # "cities"
                if (suf in ("ing", "ed", "er") and len(stem) >= 4 and stem[-1] == stem[-2]
                        and stem[:-1] in self.words):
                    return True                                     # "running", "planned"
        return False

    def score(self, text: str) -> Tuple[float, int]:
        """(share of words that are known, number of words)."""
        toks = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
        words = [t for t in toks if len(t) >= 2 or t.lower() in ("a", "i")]
        if text.rstrip().endswith("-") and words:
            words = words[:-1]                    # hyphenated fragment continued on the next line
        if not words:
            return 1.0, 0
        return sum(1 for w in words if self.known(w)) / len(words), len(words)


@dataclass
class Models:
    det: DetectionPredictor
    layout: LayoutPredictor
    primary: Recognizer
    fallback: Optional[Recognizer]
    lexicon: Lexicon


def load_models() -> Models:
    logger.info("=" * 62)
    logger.info("  LOADING MODELS  (Surya Layout/Detection + TrOCR)")
    logger.info("=" * 62)
    os.environ.setdefault("LAYOUT_BATCH_SIZE", "4")
    os.environ.setdefault("DETECTOR_BATCH_SIZE", "4")
    os.environ.setdefault("RECOGNITION_BATCH_SIZE", "8")
    try:
        if hasattr(settings, "LAYOUT_BATCH_SIZE"):
            settings.LAYOUT_BATCH_SIZE = int(os.environ["LAYOUT_BATCH_SIZE"])
        if hasattr(settings, "DETECTOR_BATCH_SIZE"):
            settings.DETECTOR_BATCH_SIZE = int(os.environ["DETECTOR_BATCH_SIZE"])
    except Exception as exc:
        logger.warning(f"  Could not patch Surya settings object: {exc}")
    if not setup_pdfa_resources():
        logger.warning("  PDF/A resources setup incomplete.")

    logger.info("  DetectionPredictor …")
    det = DetectionPredictor()
    logger.info(f"  LayoutPredictor ({settings.LAYOUT_MODEL_CHECKPOINT}); layout input is scaled "
                f"to fit {SURYA_LAYOUT_MAX_SIDE}px …")
    layout = LayoutPredictor(FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT))
    primary = Recognizer(TROCR_MODEL_NAME)
    fallback = None
    if TROCR_FALLBACK_MODEL_NAME and TROCR_FALLBACK_MODEL_NAME != TROCR_MODEL_NAME:
        try:
            fallback = Recognizer(TROCR_FALLBACK_MODEL_NAME)
        except Exception as exc:
            logger.warning(f"  Fallback TrOCR not loaded: {exc}")
    if fallback is None and STRICT_ENABLED:
        logger.warning("  No second TrOCR model — the legibility gate cannot test model agreement.")
    lexicon = Lexicon(getattr(primary.processor, "tokenizer", None))
    logger.info("  All models ready.\n")
    return Models(det=det, layout=layout, primary=primary, fallback=fallback, lexicon=lexicon)


# ══════════════════════════════════════════════════════════════════════════════
# SURYA LAYOUT  –  parsing + region housekeeping
# ══════════════════════════════════════════════════════════════════════════════

_LABEL_ALIAS: Dict[str, str] = {
    "SectionHeader": "Section-header", "PageHeader": "Page-header",
    "PageFooter": "Page-footer", "ListItem": "List-item",
    "TableOfContents": "Table-of-contents", "InlineMath": "Text-inline-math",
    "TextInlineMath": "Text-inline-math", "Header": "Page-header",
    "Footer": "Page-footer", "Heading": "Section-header", "Title": "Section-header",
    "Handwritten": "Handwriting", "section_header": "Section-header",
    "page_header": "Page-header", "page_footer": "Page-footer",
    "list_item": "List-item", "table_of_contents": "Table-of-contents",
    "inline_math": "Text-inline-math", "text_inline_math": "Text-inline-math",
    "handwriting": "Handwriting", "figure_caption": "Caption",
    "FigureCaption": "Caption", "table_caption": "Caption", "TableCaption": "Caption",
    "paragraph": "Text", "Paragraph": "Text", "body": "Text", "Body": "Text",
}
_CANON_LABELS: Dict[str, str] = {c.lower().replace("_", "-"): c for c in (OCR_LABELS | SKIP_LABELS)}


def _normalise_label(raw: str) -> str:
    if raw in _LABEL_ALIAS:
        return _LABEL_ALIAS[raw]
    return _CANON_LABELS.get(str(raw).lower().replace("_", "-"), raw)


def _box_of(obj) -> Optional[List[float]]:
    if hasattr(obj, "bbox") and obj.bbox:
        return [float(v) for v in obj.bbox]
    if hasattr(obj, "polygon") and obj.polygon and len(obj.polygon) >= 4:
        xs = [float(p[0]) for p in obj.polygon]
        ys = [float(p[1]) for p in obj.polygon]
        return [min(xs), min(ys), max(xs), max(ys)]
    return None


def _numeric_topk(top_k) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in (top_k or {}).items():
        try:
            out[_normalise_label(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def parse_layout_result(result, orig_size: Tuple[int, int]) -> List[dict]:
    """Normalise one LayoutPredictor result into dicts in original-pixel space."""
    out: List[dict] = []
    if result is None or not hasattr(result, "bboxes"):
        return out
    sx = sy = 1.0
    ib = getattr(result, "image_bbox", None)
    if ib:
        ib_w, ib_h = ib[2] - ib[0], ib[3] - ib[1]
        if ib_w > 0 and ib_h > 0:
            sx, sy = orig_size[0] / ib_w, orig_size[1] / ib_h
    for box in result.bboxes:
        b = _box_of(box)
        if b is None:
            continue
        b = [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
        if b[2] - b[0] < MIN_REGION_W or b[3] - b[1] < MIN_REGION_H:
            continue
        out.append({
            "bbox": b,
            "label": _normalise_label(getattr(box, "label", "Text")),
            "position": float(getattr(box, "position", 0)),
            "top_k": _numeric_topk(getattr(box, "top_k", {})),
        })
    return out


def _nms_same_label(regions: List[Region], iou_threshold: float = 0.45) -> List[Region]:
    ordered = sorted(regions, key=lambda r: _area(r.bbox), reverse=True)
    kept: List[Region] = []
    for r in ordered:
        if any(k.label == r.label and _iou(k.bbox, r.bbox) > iou_threshold for k in kept):
            continue
        kept.append(r)
    return kept


def _split_regions_at_rules(regions: List[Region], rules: List[List[float]],
                            next_id: List[int]) -> List[Region]:
    """A text region that a printed vertical rule runs through spans two columns: split it there."""
    if not rules:
        return regions
    out: List[Region] = []
    for r in regions:
        if r.label not in RULE_SPLIT_LABELS:
            out.append(r)
            continue
        cuts = []
        for rx0, ry0, rx1, ry1 in rules:
            cx = (rx0 + rx1) / 2
            if r.bbox[0] + RULE_EDGE_MARGIN < cx < r.bbox[2] - RULE_EDGE_MARGIN:
                yov = min(r.bbox[3], ry1) - max(r.bbox[1], ry0)
                if yov >= RULE_SPLIT_MIN_YFRAC * r.h:
                    cuts.append(cx)
        if not cuts:
            out.append(r)
            continue
        edges = [r.bbox[0]] + sorted(cuts) + [r.bbox[2]]
        pieces = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)
                  if edges[i + 1] - edges[i] >= MIN_REGION_W]
        for k, (a, b) in enumerate(pieces):
            pos = None if r.position is None else r.position + 0.001 * k
            rid = r.rid if k == 0 else next_id[0]
            if k:
                next_id[0] += 1
            out.append(Region(rid, [a, r.bbox[1], b, r.bbox[3]], r.label, pos, r.top_k, r.source))
        logger.info(f"  [RULES] split {r.label} ({r.bbox[0]:.0f}–{r.bbox[2]:.0f}px) "
                    f"into {len(pieces)} column(s) at printed rule(s).")
    return out


def _normalise_gutters(regions: List[Region]) -> None:
    """Trim side-by-side text regions that bleed a little into each other back to the gutter mid-line."""
    texts = sorted([r for r in regions if r.is_text], key=lambda r: r.bbox[0])
    for i, a in enumerate(texts):
        for b in texts[i + 1:]:
            if b.bbox[0] >= a.bbox[2]:
                continue
            yov = min(a.bbox[3], b.bbox[3]) - max(a.bbox[1], b.bbox[1])
            if yov <= 0.5 * min(a.h, b.h):
                continue
            if b.bbox[2] <= a.bbox[2] or a.bbox[0] >= b.bbox[0]:
                continue
            xov = a.bbox[2] - b.bbox[0]
            if xov <= 0 or xov > GUTTER_MAX_OVERLAP_FRAC * min(a.w, b.w):
                continue
            mid = (a.bbox[2] + b.bbox[0]) / 2
            a.bbox[2] = mid
            b.bbox[0] = mid


def run_layout(images: List[Image.Image], layout_predictor: LayoutPredictor,
               log_scale: bool = False) -> List[List[dict]]:
    try:
        results = layout_predictor(images)
    except Exception as exc:
        logger.error(f"  LayoutPredictor failed: {exc}")
        return [[] for _ in images]
    if log_scale and images:
        iw, ih = images[0].size
        k = max(1.0, max(iw, ih) / SURYA_LAYOUT_MAX_SIDE)
        logger.info(f"  [LAYOUT] {iw}×{ih}px image → layout model sees ≈{iw / k:.0f}×{ih / k:.0f}px "
                    f"({k:.1f}× smaller; a {DPI // 30}px-tall body line becomes ≈{DPI / 30 / k:.1f}px)")
    return [parse_layout_result(res, im.size) for res, im in zip(results, images)]


# ══════════════════════════════════════════════════════════════════════════════
# TILED LAYOUT  (problem 1: columns Surya misses at full-page scale)
# ══════════════════════════════════════════════════════════════════════════════

def _tile_origins(length: int, tile: int, overlap: int) -> List[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    n = -(-(length - tile) // stride) + 1
    return [round(i * (length - tile) / (n - 1)) for i in range(n)]


def _layout_tile_grid(iw: int, ih: int) -> List[Tuple[int, int, int, int]]:
    cols = max(1, int(LAYOUT_TILE_COLS))
    rows = int(LAYOUT_TILE_ROWS) if LAYOUT_TILE_ROWS else max(1, round(cols * ih / max(1, iw)))
    ov = min(max(LAYOUT_TILE_OVERLAP, 0.0), 0.5)
    tw = int(math.ceil(iw / (cols - (cols - 1) * ov)))
    th = int(math.ceil(ih / (rows - (rows - 1) * ov)))
    xs = _tile_origins(iw, tw, int(tw * ov))
    ys = _tile_origins(ih, th, int(th * ov))
    return [(x, y, min(x + tw, iw), min(y + th, ih)) for y in ys for x in xs]


def _merge_tile_fragments(frags: List[dict]) -> List[dict]:
    """
    Join region fragments cut by interior tile edges (a column running from one
    tile into the next) and drop copies of the same region seen whole by two
    overlapping tiles.
    """
    n = len(frags)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        a = frags[i]
        for j in range(i + 1, n):
            b = frags[j]
            if a["tile"] == b["tile"] or not (a["cut"] or b["cut"]):
                continue
            if _is_text_label(a["label"]) != _is_text_label(b["label"]):
                continue
            A, B = a["bbox"], b["bbox"]
            ix = min(A[2], B[2]) - max(A[0], B[0])
            iy = min(A[3], B[3]) - max(A[1], B[1])
            if ix <= 0 or iy <= 0:
                continue
            wmin = min(A[2] - A[0], B[2] - B[0])
            hmin = min(A[3] - A[1], B[3] - B[1])
            if ix >= LAYOUT_TILE_ALIGN * wmin or iy >= LAYOUT_TILE_ALIGN * hmin:
                parent[find(i)] = find(j)

    groups: Dict[int, List[dict]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(frags[i])
    merged = []
    for g in groups.values():
        big = max(g, key=lambda f: _area(f["bbox"]))
        label_area: Counter = Counter()
        for f in g:
            label_area[f["label"]] += _area(f["bbox"])
        merged.append({
            "bbox": [min(f["bbox"][0] for f in g), min(f["bbox"][1] for f in g),
                     max(f["bbox"][2] for f in g), max(f["bbox"][3] for f in g)],
            "label": label_area.most_common(1)[0][0], "top_k": big["top_k"],
        })
    kept: List[dict] = []
    for m in sorted(merged, key=lambda d: _area(d["bbox"]), reverse=True):
        if any(_is_text_label(k["label"]) == _is_text_label(m["label"]) and
               _frac_inside(m["bbox"], k["bbox"]) >= 0.8 for k in kept):
            continue
        kept.append(m)
    return kept


def tiled_layout(imgs: PageImages, layout_predictor: LayoutPredictor) -> List[dict]:
    """Surya layout on an overlapping tile grid; regions returned in page coordinates."""
    iw, ih = imgs.layout_rgb.size
    frames = _layout_tile_grid(iw, ih)
    if len(frames) <= 1:
        return []
    logger.info(f"  [TILES] Surya layout on {len(frames)} overlapping tiles "
                f"(~{frames[0][2] - frames[0][0]}×{frames[0][3] - frames[0][1]}px each) …")
    parsed = run_layout([imgs.layout_rgb.crop(f) for f in frames], layout_predictor, log_scale=True)
    m = LAYOUT_TILE_EDGE_MARGIN
    frags = []
    for ti, ((fx0, fy0, fx1, fy1), boxes) in enumerate(zip(frames, parsed)):
        for d in boxes:
            x0, y0, x1, y1 = d["bbox"]
            cut = ((fx0 > 0 and x0 < m) or (fx1 < iw and x1 > (fx1 - fx0) - m) or
                   (fy0 > 0 and y0 < m) or (fy1 < ih and y1 > (fy1 - fy0) - m))
            frags.append({"bbox": [x0 + fx0, y0 + fy0, x1 + fx0, y1 + fy0], "label": d["label"],
                          "top_k": d["top_k"], "tile": ti, "cut": cut})
    return _merge_tile_fragments(frags)


def _uncovered_pieces(box: Sequence[float], covers: List[List[float]]) -> List[List[float]]:
    """
    The vertical stretches of a tile region that page regions do not already
    cover (a column whose top half Surya found at page scale but whose lower
    half it missed yields just the lower half).  Only covers spanning most of
    the box's width count, so a narrow neighbour does not chop a column.
    """
    x0, y0, x1, y1 = box
    w = x1 - x0
    spans = []
    for c in covers:
        if min(x1, c[2]) - max(x0, c[0]) >= 0.6 * w and c[3] > y0 and c[1] < y1:
            spans.append((max(y0, c[1]), min(y1, c[3])))
    pieces, y = [], y0
    for a, b in sorted(spans):
        if a > y:
            pieces.append([x0, y, x1, a])
        y = max(y, b)
    if y < y1:
        pieces.append([x0, y, x1, y1])
    return [p for p in pieces if p[3] - p[1] >= 3 * MIN_REGION_H]


def merge_tile_layout(regions: List[Region], tiles: List[dict], next_id: List[int],
                      stats: Counter) -> List[Region]:
    """
    Page regions stay primary (they group articles best and carry Surya's
    reading position).  Tile regions are used where the page pass failed:
      * a page "Picture" that the tiles mostly read as text is replaced by
        that text (Surya's clean_boxes had deleted the text boxes inside it);
      * tile text that page text regions do not cover is added.
    """
    if not tiles:
        return regions
    page_text = [r for r in regions if r.is_text]
    page_pics = [r for r in regions if not r.is_text]
    tile_text = [t for t in tiles if _is_text_label(t["label"])]
    used: Set[int] = set()
    out: List[Region] = list(page_text)
    kept_pics: List[Region] = []

    for p in page_pics:
        zone = _grow(p.bbox, ORPHAN_REGION_PAD)
        inside = [k for k, t in enumerate(tile_text) if _frac_inside(t["bbox"], zone) >= 0.6]
        cov = _coverage(p.bbox, [tile_text[k]["bbox"] for k in inside])
        if inside and cov >= PICTURE_TILE_TEXT_COVER:
            for k in inside:
                t = tile_text[k]
                b = _intersect(t["bbox"], zone)
                if b[2] - b[0] >= MIN_REGION_W and b[3] - b[1] >= MIN_REGION_H:
                    out.append(Region(next_id[0], b, t["label"], None, t["top_k"], "picture-text"))
                    next_id[0] += 1
                used.add(k)
            stats["pictures_to_text"] += 1
            logger.info(f"  [TILES] {p.label} at ({p.bbox[0]:.0f},{p.bbox[1]:.0f}→{p.bbox[2]:.0f},"
                        f"{p.bbox[3]:.0f}) is {cov:.0%} text at tile resolution → "
                        f"{len(inside)} text region(s)")
        else:
            kept_pics.append(p)
    out.extend(kept_pics)

    covers = [_grow(r.bbox, ORPHAN_REGION_PAD) for r in page_text]
    pic_boxes = [p.bbox for p in kept_pics]
    added = 0
    for k, t in enumerate(tile_text):
        if k in used:
            continue
        if _coverage(t["bbox"], covers) >= LAYOUT_TILE_ADD_BELOW:
            continue
        if pic_boxes and _coverage(t["bbox"], pic_boxes) >= 0.5:
            continue            # lettering inside a real picture
        for piece in _uncovered_pieces(t["bbox"], covers):
            out.append(Region(next_id[0], piece, t["label"], None, t["top_k"], "tile"))
            next_id[0] += 1
            added += 1
            logger.debug(f"      [TILES] + {t['label']} ({piece[0]:.0f},{piece[1]:.0f}→"
                         f"{piece[2]:.0f},{piece[3]:.0f})")
    stats["tile_added"] += added
    logger.info(f"  [TILES] {len(tiles)} tile region(s); {added} text region(s) the full-page "
                f"pass missed were added, {stats['pictures_to_text']} picture(s) re-read as text.")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# SURYA DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_lines(images: List[Image.Image], det: DetectionPredictor) -> List[List[List[float]]]:
    """Batched Surya line detection. Returns crop-relative boxes per image."""
    if not images:
        return []
    try:
        results = det([im.convert("RGB") for im in images])
    except Exception as exc:
        logger.debug(f"    batched detection failed ({exc}); retrying one by one")
        results = []
        for im in images:
            try:
                results.append(det([im.convert("RGB")])[0])
            except Exception as exc2:
                logger.debug(f"    DetectionPredictor error: {exc2}")
                results.append(None)
    out = []
    for res in results:
        boxes = []
        if res is not None and hasattr(res, "bboxes"):
            for box in res.bboxes:
                b = _box_of(box)
                if b and b[2] - b[0] >= MIN_LINE_W and b[3] - b[1] >= MIN_LINE_H:
                    boxes.append(b)
        out.append(boxes)
    return out


def _line_nms(boxes: List[List[float]], contain: float = 0.7) -> List[List[float]]:
    kept: List[List[float]] = []
    for b in sorted(boxes, key=_area, reverse=True):
        if any(_frac_inside(b, k) >= contain for k in kept):
            continue
        kept.append(b)
    return kept


def line_inventory(clean: Image.Image, det: DetectionPredictor) -> List[List[float]]:
    """Every text line on the page, detected on overlapping tiles at near-native resolution."""
    iw, ih = clean.size
    m = INVENTORY_EDGE_MARGIN
    origins = [(tx, ty) for ty in _tile_origins(ih, INVENTORY_TILE, INVENTORY_OVERLAP)
               for tx in _tile_origins(iw, INVENTORY_TILE, INVENTORY_OVERLAP)]
    tiles, frames = [], []
    for tx, ty in origins:
        tx1, ty1 = min(tx + INVENTORY_TILE, iw), min(ty + INVENTORY_TILE, ih)
        tiles.append(clean.crop((tx, ty, tx1, ty1)))
        frames.append((tx, ty, tx1, ty1))
    boxes: List[List[float]] = []
    for (tx, ty, tx1, ty1), lines in zip(frames, detect_lines(tiles, det)):
        for lx0, ly0, lx1, ly1 in lines:
            if ((tx > 0 and lx0 < m) or (tx1 < iw and lx1 > (tx1 - tx) - m) or
                    (ty > 0 and ly0 < m) or (ty1 < ih and ly1 > (ty1 - ty) - m)):
                continue
            boxes.append([lx0 + tx, ly0 + ty, lx1 + tx, ly1 + ty])
    return _line_nms(boxes)


# ══════════════════════════════════════════════════════════════════════════════
# COLUMN GUTTERS  (F-H: the whitespace between columns, measured on the page)
# ══════════════════════════════════════════════════════════════════════════════

class GutterMap:
    """
    Where the page is blank in tall, narrow vertical strips.

    Built once per page from the flattened, rule-free image at 1/GUTTER_SCALE
    resolution.  A cell counts as gutter when it belongs to a blank vertical
    run at least GUTTER_MIN_TALL_LINES line-heights long AND a blank
    horizontal run at least GUTTER_MIN_W_PX wide.  The space between two words
    is blank only for one line (the lines above and below have ink there), so
    it never qualifies; the gap between two columns does, whether or not a
    printed rule sits in it (rules were already erased from this image).
    """

    def __init__(self, clean: Image.Image, med_h: float):
        s = GUTTER_SCALE
        a = np.asarray(clean, dtype=np.uint8) < GUTTER_INK_LEVEL
        # Broken / dashed column rules survive remove_rules (it needs one unbroken run of
        # 6 % of the page).  Thin vertical strokes longer than a letter stem sit IN the
        # gutter, so they must not count as ink here.
        try:
            v = _long_runs(a, max(20, int(GUTTER_RULE_MIN_LINES * med_h)), 0)
            v &= ~_long_runs(v, RULE_MAX_THICK_PX + 1, 1)
            a = a & ~_dilate1(v)
        except MemoryError:
            logger.warning("  [GUTTER] dashed-rule filter skipped (memory).")
        H, W = a.shape[0] // s, a.shape[1] // s
        cnt = a[:H * s, :W * s].reshape(H, s, W, s).sum(axis=(1, 3))
        ink = cnt >= GUTTER_CELL_MIN_DARK
        # Remove isolated specks (≤ 1 inked neighbour) so scan noise does not break a gutter.
        p = np.pad(ink, 1).astype(np.int16)
        nb = sum(p[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx]
                 for dy in (-1, 0, 1) for dx in (-1, 0, 1)) - ink.astype(np.int16)
        ink &= nb >= 2
        self.s, self.H, self.W, self.med_h = s, H, W, med_h
        self.min_w = max(2, int(round(GUTTER_MIN_W_PX / s)))
        blank = ~ink
        self.mask = self._build(blank, GUTTER_MIN_TALL_LINES)
        self.loose = self._build(blank, GUTTER_LOOSE_TALL_LINES)

    def _build(self, blank: np.ndarray, tall_lines: float) -> np.ndarray:
        tall = max(3, int(round((tall_lines + 1) * self.med_h / self.s)))
        return _long_runs(_long_runs(blank, tall, 0), self.min_w, 1)

    def _cols(self, x0: float, x1: float) -> Tuple[int, int]:
        return max(0, int(x0 // self.s)), min(self.W, int(math.ceil(x1 / self.s)))

    def _rows(self, y0: float, y1: float) -> Tuple[int, int]:
        return max(0, int(y0 // self.s)), min(self.H, int(math.ceil(y1 / self.s)))

    def line_cuts(self, b: Sequence[float], loose: bool = False) -> List[float]:
        """x positions where a text line crosses a column gutter."""
        m = self.loose if loose else self.mask
        r0, r1 = self._rows(b[1], b[3])
        c0, c1 = self._cols(b[0], b[2])
        if r1 <= r0 or c1 - c0 < self.min_w + 2:
            return []
        ok = m[r0:r1, c0:c1].all(axis=0)
        return [(c0 + (a + e) / 2) * self.s for a, e in _runs_1d(ok)
                if a > 0 and e < len(ok) and e - a >= self.min_w]

    def split_line(self, b: List[float]) -> List[List[float]]:
        cuts = self.line_cuts(b)
        if not cuts:
            return [b]
        edges = [b[0]] + cuts + [b[2]]
        return [[edges[i], b[1], edges[i + 1], b[3]] for i in range(len(edges) - 1)
                if edges[i + 1] - edges[i] >= MIN_LINE_W]

    def _blank_frac(self, b: Sequence[float], c0: int, c1: int) -> Optional[np.ndarray]:
        r0, r1 = self._rows(b[1], b[3])
        if r1 <= r0 or c1 <= c0:
            return None
        return self.mask[r0:r1, c0:c1].mean(axis=0)

    def box_cuts(self, b: Sequence[float], min_piece: float) -> List[float]:
        """x positions of gutters running (almost) the full height of a box."""
        c0, c1 = self._cols(b[0], b[2])
        fr = self._blank_frac(b, c0, c1)
        if fr is None:
            return []
        cuts, last = [], b[0]
        for a, e in _runs_1d(fr >= GUTTER_REGION_BLANK):
            if a == 0 or e == len(fr) or e - a < self.min_w:
                continue
            x = (c0 + (a + e) / 2) * self.s
            if x - last >= min_piece and b[2] - x >= min_piece:
                cuts.append(x)
                last = x
        return cuts

    def snap_edges(self, b: Sequence[float], col_w: float) -> List[float]:
        """
        Move a box's left/right edge onto the nearest gutter within ± GUTTER_SNAP_FRAC
        of a column: an edge inside the neighbouring column is pulled back to the
        gutter; an edge that cuts off its own text is pushed out to it.
        """
        win = max(2 * self.min_w * self.s, GUTTER_SNAP_FRAC * col_w)
        nb = list(b)
        for side in (0, 2):
            x = b[side]
            c0, c1 = self._cols(x - win, x + win)
            fr = self._blank_frac(b, c0, c1)
            if fr is None:
                continue
            best, best_d = None, None
            for a, e in _runs_1d(fr >= GUTTER_REGION_BLANK):
                if e - a < self.min_w:
                    continue
                if side == 2 and a == 0:            # left end of run unknown → no text end to snap to
                    continue
                if side == 0 and e == len(fr):
                    continue
                ra, re_ = (c0 + a) * self.s, (c0 + e) * self.s
                d = max(ra - x, x - re_, 0.0)
                if best_d is None or d < best_d:
                    best, best_d = (ra, re_), d
            if best is None:
                continue
            nb[side] = best[1] - GUTTER_SNAP_PAD if side == 0 else best[0] + GUTTER_SNAP_PAD
        if nb[2] - nb[0] < MIN_REGION_W:
            return list(b)
        return nb

    def save_debug(self, clean: Image.Image, path: Path) -> None:
        small = clean.reduce(self.s).convert("RGB")
        m = Image.fromarray((self.mask * 255).astype(np.uint8), "L").resize(small.size, _RESAMPLE.NEAREST)
        small.paste(Image.new("RGB", small.size, (0, 140, 255)), mask=m.point(lambda v: 90 if v else 0))
        small.save(str(path), "JPEG", quality=80)


def split_regions_at_gutters(regions: List[Region], gmap: GutterMap, med_h: float, col_w: float,
                             next_id: List[int], stats: Counter) -> List[Region]:
    """A body-text box that a full-height gutter runs through spans two columns: split it there."""
    out: List[Region] = []
    for r in regions:
        if r.label not in GUTTER_SPLIT_LABELS or r.h < GUTTER_SPLIT_MIN_LINES * med_h:
            out.append(r)
            continue
        cuts = gmap.box_cuts(r.bbox, GUTTER_MIN_PIECE_FRAC * col_w)
        if not cuts:
            out.append(r)
            continue
        edges = [r.bbox[0]] + cuts + [r.bbox[2]]
        for k in range(len(edges) - 1):
            pos = None if r.position is None else r.position + 0.001 * k
            rid = r.rid if k == 0 else next_id[0]
            if k:
                next_id[0] += 1
            out.append(Region(rid, [edges[k], r.bbox[1], edges[k + 1], r.bbox[3]],
                              r.label, pos, r.top_k, r.source))
        stats["regions_split_at_gutters"] += 1
        logger.info(f"  [GUTTER] split {r.label} ({r.source}) at x=" +
                    ", ".join(f"{c:.0f}" for c in cuts) + f"px ({r.bbox[0]:.0f}–{r.bbox[2]:.0f}px wide box)")
    return out


def split_tile_boxes_at_gutters(tiles: List[dict], gmap: GutterMap, med_h: float,
                                col_w: float, stats: Counter) -> List[dict]:
    """Same as split_regions_at_gutters, for tile-pass regions (so coverage is judged column by column)."""
    out = []
    for t in tiles:
        b = t["bbox"]
        if t["label"] not in GUTTER_SPLIT_LABELS or b[3] - b[1] < GUTTER_SPLIT_MIN_LINES * med_h:
            out.append(t)
            continue
        cuts = gmap.box_cuts(b, GUTTER_MIN_PIECE_FRAC * col_w)
        if not cuts:
            out.append(t)
            continue
        edges = [b[0]] + cuts + [b[2]]
        for k in range(len(edges) - 1):
            out.append({**t, "bbox": [edges[k], b[1], edges[k + 1], b[3]]})
        stats["tile_boxes_split_at_gutters"] += 1
        logger.debug(f"      [GUTTER] tile {t['label']} ({b[0]:.0f}–{b[2]:.0f}px) split at x=" +
                     ", ".join(f"{c:.0f}" for c in cuts))
    return out


def snap_regions_to_gutters(regions: List[Region], gmap: GutterMap, med_h: float,
                            col_w: float, stats: Counter) -> None:
    """Move the side edges of body-text boxes onto the measured gutters (stops bleed into the next column)."""
    if not GUTTER_SNAP:
        return
    for r in regions:
        if r.label not in (GUTTER_SPLIT_LABELS | RULE_SPLIT_LABELS) or r.h < GUTTER_SPLIT_MIN_LINES * med_h:
            continue
        nb = gmap.snap_edges(r.bbox, col_w)
        moved = [abs(nb[i] - r.bbox[i]) for i in (0, 2)]
        if max(moved) > 2:
            logger.debug(f"      [GUTTER] r{r.rid} {r.label} edges {r.bbox[0]:.0f}–{r.bbox[2]:.0f} → "
                         f"{nb[0]:.0f}–{nb[2]:.0f}px")
            r.bbox = nb
            stats["edges_snapped"] += sum(1 for m in moved if m > 2)


# ══════════════════════════════════════════════════════════════════════════════
# PICTURE RESCUE  (text Surya labelled as a figure)
# ══════════════════════════════════════════════════════════════════════════════

def _midtone_frac(gray: Image.Image, box: Sequence[float]) -> float:
    """Share of mid-grey pixels: high for photos / halftones, low for type and line art."""
    crop = gray.crop(tuple(int(round(v)) for v in box))
    if crop.width >= 4 and crop.height >= 4:
        crop = crop.reduce(2)
    a = np.asarray(crop, dtype=np.uint8)
    if a.size == 0:
        return 1.0
    return float(((a > 70) & (a < 185)).mean())


def rescue_pictures(regions: List[Region], inventory: List[List[float]], imgs: PageImages,
                    layout_predictor: LayoutPredictor, next_id: List[int],
                    stats: Counter) -> List[Region]:
    """
    Second look at every remaining Picture/Figure.  One that contains several
    detected text lines on black-and-white ink is laid out again on its own
    crop; Surya's text regions from that crop are used.  If Surya still calls
    it a picture but the line evidence is strong, the picture is converted to
    a text region.  Everything recovered here is read under the legibility gate.
    """
    if not PICTURE_RESCUE:
        return regions
    iw, ih = imgs.layout_rgb.size
    cands = []
    for r in regions:
        if r.is_text:
            continue
        inside = [l for l in inventory if _frac_inside(l, r.bbox) >= 0.7]
        if len(inside) < PICTURE_MIN_LINES:
            continue
        cover = min(1.0, sum(_area(l) for l in inside) / max(1.0, _area(r.bbox)))
        hs = [l[3] - l[1] for l in inside]
        cv = statistics.pstdev(hs) / max(1.0, statistics.mean(hs))
        mid = _midtone_frac(imgs.norm_gray, r.bbox)
        tk = sum(v for k, v in r.top_k.items() if _is_text_label(k))
        logger.debug(f"      [PICTURE] {r.label} ({r.bbox[0]:.0f},{r.bbox[1]:.0f}→{r.bbox[2]:.0f},"
                     f"{r.bbox[3]:.0f}): {len(inside)} lines, cover {cover:.2f}, height CV {cv:.2f}, "
                     f"mid-grey {mid:.2f}, top-k text {tk:.2f}")
        if cover < PICTURE_TEXT_LINE_COVER or mid > PICTURE_MAX_MIDTONE:
            continue
        cands.append((r, cover, cv, tk))
    if not cands:
        return regions

    frames = [_clip_box(_grow(c[0].bbox, RELAYOUT_PAD), iw, ih) for c in cands]
    parsed = run_layout([imgs.layout_rgb.crop(tuple(int(v) for v in f)) for f in frames],
                        layout_predictor)
    drop: Set[int] = set()
    new: List[Region] = []
    for (r, cover, cv, tk), frame, res in zip(cands, frames, parsed):
        zone = _grow(r.bbox, 6)
        got = []
        for d in res:
            if not _is_text_label(d["label"]):
                continue
            b = [d["bbox"][0] + frame[0], d["bbox"][1] + frame[1],
                 d["bbox"][2] + frame[0], d["bbox"][3] + frame[1]]
            if _frac_inside(b, zone) < 0.5:
                continue
            b = _intersect(b, zone)
            if b[2] - b[0] >= MIN_REGION_W and b[3] - b[1] >= MIN_REGION_H:
                got.append(Region(next_id[0], b, d["label"], None, d["top_k"], "picture-text"))
                next_id[0] += 1
        where = f"({r.bbox[0]:.0f},{r.bbox[1]:.0f}→{r.bbox[2]:.0f},{r.bbox[3]:.0f})"
        if got:
            new.extend(got)
            text_cov = _coverage(r.bbox, [g.bbox for g in got])
            if text_cov >= 0.5:
                drop.add(r.rid)
            stats["pictures_to_text"] += 1
            logger.info(f"  [PICTURE] {r.label} at {where}: re-layout found {len(got)} text region(s) "
                        f"({text_cov:.0%} of its area){'; picture removed' if r.rid in drop else ''}.")
        elif (cover >= PICTURE_FORCE_COVER and cv <= PICTURE_MAX_HEIGHT_CV) or tk >= PICTURE_TOPK_TEXT:
            drop.add(r.rid)
            new.append(Region(r.rid, list(r.bbox), "Text", r.position, r.top_k, "picture-text"))
            stats["pictures_to_text"] += 1
            logger.info(f"  [PICTURE] {r.label} at {where}: text lines cover {cover:.0%} "
                        f"(height CV {cv:.2f}, top-k text {tk:.2f}) → read as Text under the legibility gate.")
        else:
            logger.info(f"  [PICTURE] {r.label} at {where} kept as a picture "
                        f"(lines cover {cover:.0%}, height CV {cv:.2f}).")
    return [r for r in regions if r.rid not in drop] + new


# ══════════════════════════════════════════════════════════════════════════════
# ORPHAN TEXT  →  targeted Surya re-layout  (safety net)
# ══════════════════════════════════════════════════════════════════════════════

def _cluster_lines(lines: List[List[float]], med_h: float) -> List[List[List[float]]]:
    """Union lines into blocks: same x-range (≥ 50 % of narrower) and small vertical gap."""
    n = len(lines)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    order = sorted(range(n), key=lambda i: lines[i][1])
    for ai, i in enumerate(order):
        a = lines[i]
        for j in order[ai + 1:]:
            b = lines[j]
            if b[1] - a[3] > ORPHAN_LINE_GAP * med_h:
                break
            xov = min(a[2], b[2]) - max(a[0], b[0])
            if xov >= 0.5 * min(a[2] - a[0], b[2] - b[0]):
                parent[find(i)] = find(j)
    groups: Dict[int, List[List[float]]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(lines[i])
    return list(groups.values())


def _is_subhead(bx: Sequence[float], n_lines: int, texts: List[Region], med_h: float, col_w: float) -> bool:
    """
    A short line sitting inside a column between two claimed text blocks
    ("Work for Organ.", "Moe Possibly Out.") — an in-article subhead, not a speck.
    """
    if not SUBHEAD_RESCUE or n_lines > SUBHEAD_MAX_LINES:
        return False
    if bx[2] - bx[0] < SUBHEAD_MIN_WIDTH_FRAC * col_w:
        return False
    gap = SUBHEAD_NEIGHBOUR_GAP * med_h
    above = below = False
    for r in texts:
        xin = (min(bx[2], r.bbox[2]) - max(bx[0], r.bbox[0])) / max(1.0, bx[2] - bx[0])
        if xin < SUBHEAD_X_INSIDE:
            continue
        if 0 <= bx[1] - r.bbox[3] <= gap or (r.bbox[1] < bx[1] < r.bbox[3] <= bx[1] + gap):
            above = True
        if 0 <= r.bbox[1] - bx[3] <= gap or (bx[3] - gap <= r.bbox[1] < bx[3] < r.bbox[3]):
            below = True
    return above and below


def find_orphan_blocks(inventory: List[List[float]], regions: List[Region],
                       page_w: int, page_h: int, col_w: float, med_h: float
                       ) -> Tuple[List[List[float]], List[List[float]], int]:
    """
    Blocks of detected text lines that no layout region covers.  Returns
    (blocks, subheads, number dropped).  Inventory lines arrive already split
    at column gutters, so a line that ran across into an unclaimed column is
    judged on its own column's part.  Picture boxes only claim a line that
    lies mostly inside them (a signature overlapping the last line of a quote
    no longer hides that line).
    """
    texts = [r for r in regions if r.is_text]
    covers = [_grow(r.bbox, ORPHAN_REGION_PAD) for r in texts]
    pics = [r.bbox for r in regions if not r.is_text]
    orphans = [b for b in inventory if _coverage(b, covers) < ORPHAN_MAX_COVERED
               and not any(_frac_inside(b, p) >= 0.6 for p in pics)]
    if not orphans:
        return [], [], 0
    blocks, subheads, dropped = [], [], 0
    edge = ORPHAN_EDGE_FRAC * page_w
    for group in _cluster_lines(orphans, med_h):
        bx = [min(b[0] for b in group), min(b[1] for b in group),
              max(b[2] for b in group), max(b[3] for b in group)]
        bw = bx[2] - bx[0]
        touches_edge = bx[0] <= edge or bx[2] >= page_w - edge
        reason = None
        if touches_edge and bw < ORPHAN_EDGE_MAX_WIDTH_FRAC * col_w:
            reason = "page-edge sliver (facing page / scan border)"
        elif bw < ORPHAN_MIN_WIDTH_FRAC * col_w:
            reason = "too narrow for a column"
        elif len(group) < ORPHAN_MIN_LINES and bw < 0.6 * col_w:
            reason = "single short line"
        if reason and not touches_edge and _is_subhead(bx, len(group), texts, med_h, col_w):
            subheads.append(bx)
            logger.info(f"  [SUBHEAD] {len(group)} short line(s) at ({bx[0]:.0f},{bx[1]:.0f}→"
                        f"{bx[2]:.0f},{bx[3]:.0f}) between two text blocks — kept as a subhead.")
            continue
        if reason:
            dropped += 1
            logger.debug(f"      [ORPHAN] dropped {len(group)} line(s) at "
                         f"({bx[0]:.0f},{bx[1]:.0f}→{bx[2]:.0f},{bx[3]:.0f}): {reason}")
            continue
        blocks.append(bx)
        logger.info(f"  [ORPHAN] {len(group)} unclaimed line(s) at "
                    f"({bx[0]:.0f},{bx[1]:.0f}→{bx[2]:.0f},{bx[3]:.0f}) — layout missed this block.")
    return blocks, subheads, dropped


def relayout_blocks(blocks: List[List[float]], imgs: PageImages, layout_predictor: LayoutPredictor,
                    inventory: List[List[float]], next_id: List[int]) -> List[Region]:
    """Ask Surya to lay out each missed block on its own crop; fall back to a synthetic Text region."""
    if not blocks:
        return []
    iw, ih = imgs.layout_rgb.size
    frames = [_clip_box(_grow(b, RELAYOUT_PAD), iw, ih) for b in blocks]
    crops = [imgs.layout_rgb.crop(tuple(int(v) for v in f)) for f in frames]
    n_lines = [sum(1 for l in inventory if _frac_inside(l, b) > 0.5) for b in blocks]
    todo = [i for i, n in enumerate(n_lines) if n >= RELAYOUT_MIN_LINES]
    parsed: Dict[int, List[dict]] = {}
    if todo:
        for i, res in zip(todo, run_layout([crops[i] for i in todo], layout_predictor)):
            parsed[i] = res
    new: List[Region] = []
    for i, (block, frame) in enumerate(zip(blocks, frames)):
        got = []
        for d in parsed.get(i, []):
            b = [d["bbox"][0] + frame[0], d["bbox"][1] + frame[1],
                 d["bbox"][2] + frame[0], d["bbox"][3] + frame[1]]
            if _frac_inside(b, _grow(block, RELAYOUT_PAD)) >= 0.5:
                got.append(Region(next_id[0], b, d["label"], None, d["top_k"], "relayout"))
                next_id[0] += 1
        if got:
            logger.info(f"  [RELAYOUT] block at x={block[0]:.0f}–{block[2]:.0f}px → "
                        f"{len(got)} region(s): " + ", ".join(r.label for r in got))
            new.extend(got)
        else:
            new.append(Region(next_id[0], _grow(block, 6), "Text", None, {}, "orphan"))
            next_id[0] += 1
    return new


# ══════════════════════════════════════════════════════════════════════════════
# PHASE A + B  –  detect lines per region, then give every line ONE owner
# ══════════════════════════════════════════════════════════════════════════════

def _split_line_at_rules(b: List[float], rules: List[List[float]]) -> List[List[float]]:
    cy = (b[1] + b[3]) / 2
    cuts = sorted((r[0] + r[2]) / 2 for r in rules
                  if b[0] + 10 < (r[0] + r[2]) / 2 < b[2] - 10 and r[1] <= cy <= r[3])
    if not cuts:
        return [b]
    edges = [b[0]] + cuts + [b[2]]
    return [[edges[i], b[1], edges[i + 1], b[3]] for i in range(len(edges) - 1)
            if edges[i + 1] - edges[i] >= MIN_LINE_W]


def collect_region_lines(regions: List[Region], imgs: PageImages, det: DetectionPredictor,
                         gmap: Optional[GutterMap] = None, stats: Optional[Counter] = None,
                         col_w: float = 600.0) -> Tuple[Dict[int, List[List[float]]], Dict[int, bool]]:
    """
    A: Surya line detection on each text region's crop, batched.
    B: global ownership — a physical line belongs to exactly one region.
    Every detected line is cut at printed rules AND at measured gutters; a
    piece that lies in the neighbouring column is not this region's text and
    is dropped (that piece is what used to bleed into the OCR).
    Returns (lines per region id, "detector found anything" per region id).
    """
    iw, ih = imgs.clean_gray.size
    texts = [r for r in regions if r.is_text]
    frames = [_clip_box(_grow(r.bbox, DET_PAD), iw, ih) for r in texts]
    crops = [imgs.clean_gray.crop(tuple(int(v) for v in f)) for f in frames]
    detected = detect_lines(crops, det)

    cands = []
    found: Dict[int, bool] = {}
    for r, f, lines in zip(texts, frames, detected):
        found[r.rid] = bool(lines)
        for lx0, ly0, lx1, ly1 in lines:
            b = [lx0 + f[0], ly0 + f[1], lx1 + f[0], ly1 + f[1]]
            pieces = []
            for p in _split_line_at_rules(b, imgs.vrules):
                pieces.extend(gmap.split_line(p) if gmap is not None else [p])
            if len(pieces) > 1:
                if stats is not None:
                    stats["gutter_line_splits"] += 1
                # Keep the piece that is this region's own text, plus any other piece that is a
                # full column's worth inside the region; the rest is the neighbour column bleeding in.
                main = max(pieces, key=lambda p: _overlap_area(p, r.bbox))
                kept_p = [p for p in pieces if p is main or
                          (_frac_inside(p, r.bbox) >= LINE_MIN_IN_REGION and
                           p[2] - p[0] >= GUTTER_MIN_PIECE_FRAC * col_w)]
                if stats is not None:
                    stats["bleed_pieces_dropped"] += len(pieces) - len(kept_p)
                pieces = kept_p
            for piece in pieces:
                frac = _frac_inside(piece, r.bbox)
                if frac >= LINE_MIN_IN_REGION:
                    cands.append((piece, r.rid, frac))

    cands.sort(key=lambda c: (_area(c[0]), c[2]), reverse=True)
    kept: List[list] = []
    for b, rid, frac in cands:
        dup_idx = None
        for k, (kb, _krid, kfrac) in enumerate(kept):
            if _overlap_area(b, kb) <= 0:
                continue
            if _frac_inside(b, kb) >= LINE_DUP_CONTAIN:
                dup_idx = k
                break
        if dup_idx is None:
            kept.append([b, rid, frac])
            continue
        kb, krid, kfrac = kept[dup_idx]
        mutual = _frac_inside(kb, b) >= LINE_DUP_CONTAIN
        if mutual and frac > kfrac + 0.05:
            kept[dup_idx] = [b, rid, frac]

    region_by_id = {r.rid: r for r in texts}
    per_region: Dict[int, List[List[float]]] = {r.rid: [] for r in texts}
    for b, rid, _ in kept:
        rb = region_by_id[rid].bbox
        clipped = [max(b[0], rb[0] - LINE_CLIP_PAD), b[1], min(b[2], rb[2] + LINE_CLIP_PAD), b[3]]
        if clipped[2] - clipped[0] >= MIN_LINE_W:
            per_region[rid].append(clipped)
    return per_region, found


# ══════════════════════════════════════════════════════════════════════════════
# PHASE C  –  recognition helpers
# ══════════════════════════════════════════════════════════════════════════════

def _edge_strokes(crop: Image.Image) -> Tuple[bool, bool]:
    """Residual rule at the left / right edge of a line crop (a tall thin dark stroke)."""
    a = np.asarray(crop, dtype=np.uint8)
    h, w = a.shape
    if h < 6 or w < 12:
        return False, False
    strip = max(3, int(0.06 * w))
    dark = a < 128
    frac = dark.mean(axis=0)
    return bool((frac[:strip] >= EDGE_STROKE_FRAC).any()), bool((frac[-strip:] >= EDGE_STROKE_FRAC).any())


def _prepare_crop(src: Image.Image, bbox: Sequence[float], clip: Sequence[float]) -> Image.Image:
    x0, y0, x1, y1 = bbox
    h = y1 - y0
    box = [max(clip[0], x0 - LINE_PAD_X * h), max(clip[1], y0 - LINE_PAD_Y * h),
           min(clip[2], x1 + LINE_PAD_X * h), min(clip[3], y1 + LINE_PAD_Y * h)]
    crop = src.crop(tuple(int(round(v)) for v in box))
    return ImageOps.autocontrast(crop, cutoff=1)


def _bordered(crop: Image.Image) -> Image.Image:
    return ImageOps.expand(crop, border=max(2, int(CROP_BORDER_FRAC * crop.height)), fill=255).convert("RGB")


def _wide_segments(crop: Image.Image, max_ar: float = MAX_HEADER_AR) -> List[Tuple[int, int, bool]]:
    """Split a very wide header crop into pieces of aspect ratio ≈ max_ar, cutting in inter-word gaps."""
    w, h = crop.size
    if h <= 0 or w / h <= max_ar:
        return [(0, w, False)]
    n_seg = math.ceil((w / h) / max_ar)
    seg_w = w / n_seg
    ink = (np.asarray(crop, dtype=np.uint8) < 128).sum(axis=0)
    blank = ink <= max(1, int(0.01 * h))
    gaps = [((a + b) / 2) for a, b in _runs_1d(blank) if b - a >= max(3, int(0.15 * h)) and 0 < a and b < w]
    segs: List[Tuple[int, int, bool]] = []
    start, overlap_prev = 0, False
    for k in range(1, n_seg):
        ideal = k * seg_w
        near = [g for g in gaps if abs(g - ideal) <= 0.35 * seg_w and g > start + 0.3 * seg_w]
        if near:
            cut = int(min(near, key=lambda g: abs(g - ideal)))
            segs.append((start, cut, overlap_prev))
            start, overlap_prev = cut, False
        else:
            ov = int(seg_w * HEADER_SEGMENT_OVERLAP)
            cut = int(ideal)
            segs.append((start, min(w, cut + ov // 2), overlap_prev))
            start, overlap_prev = max(0, cut - ov // 2), True
    segs.append((start, w, overlap_prev))
    return [s for s in segs if s[1] - s[0] >= 4]


def _alnum_norm(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())


def _merge_overlapping(a: str, b: str) -> str:
    """Join two segment reads that share an overlapping strip, removing the repeated text."""
    if not a:
        return b
    if not b:
        return a
    na, nb = _alnum_norm(a), _alnum_norm(b)
    best = 0
    for k in range(min(len(na), len(nb), 24), 2, -1):
        if na[-k:] == nb[:k]:
            best = k
            break
    if best:
        seen, cut = 0, len(b)
        for i, ch in enumerate(b):
            if ch.isalnum():
                seen += 1
                if seen == best:
                    cut = i + 1
                    break
        rest = b[cut:]
        if rest and rest[0].isalnum() and a[-1].isalnum():
            return a + rest
        return (a + rest) if rest.startswith(" ") else (a + " " + rest.lstrip()).rstrip()
    wa, wb = a.split(), b.split()
    if wa and wb and wa[-1].strip(".,;:!?\"'").lower() == wb[0].strip(".,;:!?\"'").lower():
        wb = wb[1:]
    return " ".join(wa + wb)


_LEADING_JUNK = {")", "!", "|", "]", "}", "¦", "/", "\\"}
_TRAILING_JUNK = {"(", "|", "[", "{", "¦", "/", "\\"}
_STROKE_TOKENS = {"I", "l", "1", "|", "!", "(", ")", "[", "]", "0", "0.", "o", "'", "i", ":", ";"}


def clean_text(text: str, left_stroke: bool, right_stroke: bool) -> str:
    """Remove TrOCR's '#' fill tokens and rule/gutter artefacts at line ends."""
    toks = [t for t in text.split() if t.strip("#") != ""]
    toks = [t.replace("#", "") if len(t.strip("#")) >= 2 else t for t in toks]
    changed = True
    while toks and changed:
        changed = False
        if toks and toks[0] in _LEADING_JUNK:
            toks.pop(0); changed = True
        if toks and toks[-1] in _TRAILING_JUNK:
            toks.pop(); changed = True
        if toks and toks[0] == "(" and ")" not in "".join(toks[1:]):
            toks.pop(0); changed = True
        if toks and toks[-1] == ")" and "(" not in "".join(toks[:-1]):
            toks.pop(); changed = True
        if left_stroke and toks and toks[0] in _STROKE_TOKENS:
            toks.pop(0); changed = True
        if right_stroke and toks and toks[-1] in _STROKE_TOKENS:
            toks.pop(); changed = True
    return " ".join(toks).strip()


_HALLUCINATION_RE = [re.compile(p) for p in HALLUCINATION_PATTERNS]


def _is_noise(text: str, confidence: float, h: float, w: float,
              min_conf: float = CONFIDENCE_THRESHOLD) -> bool:
    if not text:
        return True
    if h < MIN_LINE_H or w < MIN_LINE_W:
        return True
    ar = w / max(h, 1.0)
    if ar < 0.1 or ar > 400:
        return True
    tc = text.strip()
    tl = len(tc)
    if tl == 1:
        return confidence < SINGLE_CHAR_CONFIDENCE_THRESHOLD
    if confidence < min_conf:
        return True
    if len(set(tc)) == 1 and tl > 2:
        return True
    if tc.count("#") >= 2 and confidence < 0.70:
        return True
    if confidence < HALLUCINATION_MAX_CONF and any(p.match(tc) for p in _HALLUCINATION_RE):
        return True
    if re.match(r"^[oOlI\.\|]+$", tc) and confidence < SINGLE_CHAR_CONFIDENCE_THRESHOLD:
        return True
    if tl > 3 and tc.isalpha() and not any(c in "aeiouy" for c in tc.lower()) and confidence < 0.7:
        return True
    if tl >= 4 and (w / (h * tl)) > SPARSE_LINE_WIDTH_RATIO:
        return True
    return False


@dataclass
class ReadJob:
    rid: int
    bbox: List[float]
    crop: Image.Image
    header: bool
    label: str
    whole_region: bool = False


@dataclass
class ReadResult:
    text: str                                  # best reading (what goes into the PDF)
    conf: float
    primary: Tuple[str, float]                 # primary model's reading
    alt: Optional[Tuple[str, float]]           # second model's reading (None = not read)


class LineReader:
    """
    Batched recognition of every line on a page.  Header crops are split at
    word gaps.  The second model reads low-confidence lines straight away, and
    any further lines on demand (the legibility gate asks for all lines of a
    gated region).
    """

    def __init__(self, jobs: List[ReadJob], models: Models, stats: Counter):
        self.models, self.stats = models, stats
        self.flat: List[Image.Image] = []
        self.layout: List[List[Tuple[int, bool]]] = []
        for job in jobs:
            segs = _wide_segments(job.crop) if job.header else [(0, job.crop.width, False)]
            idxs = []
            for x0, x1, ov in segs:
                idxs.append((len(self.flat), ov))
                self.flat.append(_bordered(job.crop.crop((x0, 0, x1, job.crop.height))))
            self.layout.append(idxs)
        self.primary = models.primary.read(self.flat) if self.flat else []
        self.alt: Dict[int, Tuple[str, float]] = {}
        low = [i for i, (_t, c) in enumerate(self.primary) if c < FALLBACK_BELOW_CONF]
        stats["fallback_reads"] += self._read_alt(low)

    def _read_alt(self, idxs: List[int]) -> int:
        if self.models.fallback is None:
            return 0
        todo = [i for i in dict.fromkeys(idxs) if i not in self.alt]
        if not todo:
            return 0
        for i, r in zip(todo, self.models.fallback.read([self.flat[i] for i in todo])):
            self.alt[i] = r
        return len(todo)

    def ensure_second_opinion(self, job_ids: List[int]) -> int:
        return self._read_alt([fi for j in job_ids for fi, _ in self.layout[j]])

    def _best(self, fi: int) -> Tuple[str, float]:
        p = self.primary[fi]
        a = self.alt.get(fi)
        if a is not None and a[0].strip() and a[1] > p[1] + FALLBACK_SWITCH_MARGIN:
            return a
        return p

    def _assemble(self, j: int, pick: Callable) -> Optional[Tuple[str, float]]:
        text, confs = "", []
        for fi, ov in self.layout[j]:
            r = pick(fi)
            if r is None:
                return None
            t = r[0].strip()
            if t:
                text = _merge_overlapping(text, t) if ov else f"{text} {t}".strip()
                confs.append(r[1])
        return text, (sum(confs) / len(confs) if confs else 0.0)

    def result(self, j: int) -> ReadResult:
        best = self._assemble(j, self._best)
        prim = self._assemble(j, lambda fi: self.primary[fi])
        alt = self._assemble(j, self.alt.get)
        return ReadResult(best[0], best[1], prim, alt)


# ══════════════════════════════════════════════════════════════════════════════
# LEGIBILITY GATE  (problem 2)
# ══════════════════════════════════════════════════════════════════════════════

def _agreement(a: str, b: str) -> float:
    na, nb = _alnum_norm(a), _alnum_norm(b)
    if not na and not nb:
        return 1.0
    return difflib.SequenceMatcher(None, na, nb, autojunk=False).ratio()


def assess_page(jobs: List[ReadJob], results: List[ReadResult], regions: List[Region]
                ) -> Tuple[bool, float, float, Set[int]]:
    """
    Is this a low-confidence page?  Uses the raw reads of body lines (before
    any filtering, so a page that only looks confident because its bad lines
    were thrown away is still caught).  Returns (strict page, raw mean conf,
    raw low-conf fraction, ids of the regions to gate).
    """
    body = [r.conf for job, r in zip(jobs, results) if r.text and job.label not in HEADER_LABELS]
    raw_mean = sum(body) / len(body) if body else 0.0
    raw_low = sum(1 for c in body if c < LOW_CONF_LINE) / len(body) if body else 0.0
    if not STRICT_ENABLED:
        return False, raw_mean, raw_low, set()
    strict_page = bool(body) and (raw_mean < STRICT_PAGE_MEAN_CONF or raw_low > STRICT_PAGE_LOW_FRAC)
    per_region: Dict[int, List[float]] = {}
    for job, r in zip(jobs, results):
        if r.text:
            per_region.setdefault(job.rid, []).append(r.conf)
    gated: Set[int] = set()
    for reg in regions:
        if not reg.is_text:
            continue
        confs = per_region.get(reg.rid, [])
        if strict_page or reg.source == "picture-text":
            gated.add(reg.rid)
        elif len(confs) >= STRICT_REGION_MIN_LINES and statistics.median(confs) < STRICT_REGION_TRIGGER:
            gated.add(reg.rid)
    return strict_page, raw_mean, raw_low, gated


def judge_line(res: ReadResult, lexicon: Lexicon) -> dict:
    """A gated line is kept only if it is confident, both models agree, and it is made of words."""
    reasons = []
    agree = _agreement(res.primary[0], res.alt[0]) if res.alt is not None else None
    lex, nw = lexicon.score(res.text) if lexicon.available else (1.0, 0)
    if res.conf < STRICT_LINE_CONF:
        reasons.append(f"conf {res.conf:.2f}")
    if agree is not None and agree < STRICT_MIN_AGREEMENT:
        reasons.append(f"models disagree {agree:.2f}")
    if lexicon.available and nw >= STRICT_LEXICAL_MIN_WORDS and lex < STRICT_MIN_LEXICAL:
        reasons.append(f"{lex:.0%} real words")
    if lexicon.available and nw == 0 and not re.search(r"\d", res.text):
        reasons.append("no words")
    return {"ok": not reasons, "reason": "; ".join(reasons), "agree": agree, "lex": lex, "words": nw}


def legibility_gate(jobs: List[ReadJob], results: List[ReadResult], gated: Set[int],
                    lexicon: Lexicon) -> Tuple[Set[int], List[dict], Set[int]]:
    """
    Returns (job indices to drop, log of dropped lines, region ids dropped whole).
    A region in which fewer than STRICT_REGION_MIN_KEEP of the lines pass is
    dropped completely: in an illegible column the few lines that pass are
    usually confident hallucinations, not real text.
    """
    by_region: Dict[int, List[int]] = {}
    for j, (job, r) in enumerate(zip(jobs, results)):
        if job.rid in gated and r.text:
            by_region.setdefault(job.rid, []).append(j)
    drop: Set[int] = set()
    log: List[dict] = []
    dropped_regions: Set[int] = set()
    for rid, idxs in by_region.items():
        verdicts = [judge_line(results[j], lexicon) for j in idxs]
        passed = sum(1 for v in verdicts if v["ok"])
        region_ok = passed / len(idxs) >= STRICT_REGION_MIN_KEEP
        if not region_ok:
            dropped_regions.add(rid)
        for j, v in zip(idxs, verdicts):
            if v["ok"] and region_ok:
                continue
            drop.add(j)
            reason = v["reason"] if not v["ok"] else f"region illegible ({passed}/{len(idxs)} lines pass)"
            if not v["ok"] and not region_ok:
                reason += f"; region illegible ({passed}/{len(idxs)} pass)"
            log.append({"rid": rid, "bbox": list(jobs[j].bbox), "text": results[j].text,
                        "conf": results[j].conf, "agree": v["agree"], "lex": v["lex"],
                        "reason": reason, "label": jobs[j].label})
    return drop, log, dropped_regions


@dataclass
class RecognitionOutcome:
    per_region: Dict[int, List[dict]]
    illegible: List[dict]
    dropped_regions: Set[int]
    strict_page: bool
    raw_mean: float
    raw_low: float
    gated: Set[int]


def recognise_page(regions: List[Region], lines: Dict[int, List[List[float]]],
                   found: Dict[int, bool], imgs: PageImages, models: Models,
                   med_h: float, stats: Counter) -> RecognitionOutcome:
    iw, ih = imgs.clean_gray.size
    jobs: List[ReadJob] = []
    for r in regions:
        if not r.is_text:
            continue
        header = r.label in HEADER_LABELS
        src = imgs.norm_gray if header else imgs.clean_gray
        clip = _clip_box(_grow(r.bbox, 4), iw, ih)
        rl = lines.get(r.rid, [])
        for b in rl:
            jobs.append(ReadJob(r.rid, b, _prepare_crop(src, b, clip), header, r.label))
        # Whole-region read only when the detector saw nothing in it at all.
        if not rl and not found.get(r.rid, False) and (header or r.h <= WHOLE_CROP_MAX_LINES * med_h):
            jobs.append(ReadJob(r.rid, list(r.bbox),
                                ImageOps.autocontrast(src.crop(tuple(int(v) for v in clip)), cutoff=1),
                                header, r.label, whole_region=True))

    reader = LineReader(jobs, models, stats)
    results = [reader.result(j) for j in range(len(jobs))]

    strict_page, raw_mean, raw_low, gated = assess_page(jobs, results, regions)
    drop: Set[int] = set()
    illegible: List[dict] = []
    dropped_regions: Set[int] = set()
    if gated:
        if strict_page:
            logger.warning(f"  [LEGIBILITY] low-confidence page (raw mean {raw_mean:.2f}, "
                           f"{raw_low:.0%} of body lines < {LOW_CONF_LINE}) — every region is gated.")
        else:
            logger.info(f"  [LEGIBILITY] {len(gated)} low-confidence / recovered region(s) gated.")
        gated_jobs = [j for j, job in enumerate(jobs) if job.rid in gated]
        n2 = reader.ensure_second_opinion(gated_jobs)
        stats["gate_second_reads"] += n2
        logger.info(f"  [LEGIBILITY] {n2} additional second-model read(s) for {len(gated_jobs)} gated line(s).")
        results = [reader.result(j) for j in range(len(jobs))]
        drop, illegible, dropped_regions = legibility_gate(jobs, results, gated, models.lexicon)
        logger.info(f"  [LEGIBILITY] dropped {len(drop)} illegible line(s), "
                    f"{len(dropped_regions)} whole region(s); kept "
                    f"{len(gated_jobs) - len([j for j in drop if j in set(gated_jobs)])} gated line(s).")

    per_region: Dict[int, List[dict]] = {r.rid: [] for r in regions if r.is_text}
    for j, (job, res) in enumerate(zip(jobs, results)):
        if res.alt is not None and res.text == res.alt[0] and res.text != res.primary[0]:
            stats["fallback_switches"] += 1
        if j in drop:
            stats["illegible_lines"] += 1
            continue
        text, conf = res.text, res.conf
        bw, bh = job.bbox[2] - job.bbox[0], job.bbox[3] - job.bbox[1]
        if _is_noise(text, conf, bh, bw):
            stats["noise_rejected"] += 1
            logger.debug(f"      [NOISE] conf={conf:.2f} {bw:.0f}×{bh:.0f}px | {text[:40]}")
            continue
        ls, rs = _edge_strokes(job.crop)
        cleaned = clean_text(text, ls, rs)
        if not cleaned or _is_noise(cleaned, conf, bh, bw):
            stats["noise_rejected"] += 1
            continue
        if cleaned != text:
            stats["lines_cleaned"] += 1
        per_region[job.rid].append({
            "text": cleaned, "bbox": list(job.bbox), "confidence": conf,
            "source_label": job.label, "region_id": job.rid,
        })
    return RecognitionOutcome(per_region, illegible, dropped_regions, strict_page,
                              raw_mean, raw_low, gated)


# ══════════════════════════════════════════════════════════════════════════════
# READING ORDER  –  column model + pairwise precedence (XY-cut kept as fallback)
# ══════════════════════════════════════════════════════════════════════════════

def _shrunk(b: Sequence[float]) -> Tuple[float, float, float, float]:
    mx = min((b[2] - b[0]) * XY_SHRINK_FRAC, XY_SHRINK_MAX)
    my = min((b[3] - b[1]) * XY_SHRINK_FRAC, XY_SHRINK_MAX)
    return (b[0] + mx, b[1] + my, b[2] - mx, b[3] - my)


def _free_intervals(spans: List[Tuple[float, float, float]], lo: float, hi: float,
                    tol: float) -> List[Tuple[float, float]]:
    """Interior sub-intervals of (lo, hi) covered by total weight ≤ tol."""
    pts = sorted({lo, hi} | {min(hi, max(lo, a)) for a, _, _ in spans}
                 | {min(hi, max(lo, b)) for _, b, _ in spans})
    free: List[List[float]] = []
    for p, q in zip(pts, pts[1:]):
        if q - p <= 0:
            continue
        m = (p + q) / 2
        if sum(wt for a, b, wt in spans if a < m < b) <= tol:
            if free and abs(free[-1][1] - p) < 1e-6:
                free[-1][1] = q
            else:
                free.append([p, q])
    return [(a, b) for a, b in free if a > lo + 1e-6 and b < hi - 1e-6 and b - a >= XY_MIN_GAP]


def _group_by_cuts(items: list, cuts: List[float], key: Callable) -> List[list]:
    cuts = sorted(cuts)
    groups: List[list] = [[] for _ in range(len(cuts) + 1)]
    for it in items:
        groups[bisect.bisect_left(cuts, key(it))].append(it)
    return [g for g in groups if g]


def _split_vertical(items: list, bbox_of: Callable, tol_frac: float) -> Optional[List[list]]:
    bs = [_shrunk(bbox_of(i)) for i in items]
    ylo, yhi = min(b[1] for b in bs), max(b[3] for b in bs)
    spans = [(b[0], b[2], b[3] - b[1]) for b in bs]
    free = _free_intervals(spans, min(b[0] for b in bs), max(b[2] for b in bs),
                           tol_frac * (yhi - ylo))
    if not free:
        return None
    groups = _group_by_cuts(items, [(a + b) / 2 for a, b in free],
                            lambda it: _center(bbox_of(it))[0])
    return groups if len(groups) > 1 else None


def _split_horizontal(items: list, bbox_of: Callable, only_around_spanning: bool) -> Optional[List[list]]:
    bs = [_shrunk(bbox_of(i)) for i in items]
    spans = [(b[1], b[3], 1.0) for b in bs]
    free = _free_intervals(spans, min(b[1] for b in bs), max(b[3] for b in bs), 0.0)
    if not free:
        return None
    if only_around_spanning:
        widths = [bbox_of(i)[2] - bbox_of(i)[0] for i in items]
        med = _median(widths, 0)
        spanning = [b for b, wd in zip(bs, widths) if med and wd >= XY_SPAN_FACTOR * med]
        if not spanning or len(spanning) == len(items):
            return None
        chosen = set()
        for sb in spanning:
            above = [g for g in free if g[1] <= sb[1] + 1]
            below = [g for g in free if g[0] >= sb[3] - 1]
            if above:
                chosen.add(max(above, key=lambda g: g[1]))
            if below:
                chosen.add(min(below, key=lambda g: g[0]))
        free = sorted(chosen)
        if not free:
            return None
    groups = _group_by_cuts(items, [(a + b) / 2 for a, b in free],
                            lambda it: _center(bbox_of(it))[1])
    return groups if len(groups) > 1 else None


def xy_cut_order(items: list, bbox_of: Callable, pos_of: Callable = lambda it: None,
                 depth: int = 0) -> list:
    """Newspaper XY-cut reading order (fallback when no column model can be built)."""
    if len(items) <= 1 or depth > 40:
        return list(items)
    for splitter in (lambda: _split_vertical(items, bbox_of, XY_VERTICAL_TOL_FRAC),
                     lambda: _split_horizontal(items, bbox_of, True),
                     lambda: _split_horizontal(items, bbox_of, False)):
        groups = splitter()
        if groups:
            out = []
            for g in groups:
                out.extend(xy_cut_order(g, bbox_of, pos_of, depth + 1))
            return out
    if all(pos_of(i) is not None for i in items):
        return sorted(items, key=lambda i: (pos_of(i), _center(bbox_of(i))[1]))
    return sorted(items, key=lambda i: (_center(bbox_of(i))[1], bbox_of(i)[0]))


def _rows(lines: List[dict]) -> List[dict]:
    """Top-to-bottom visual rows, left-to-right within a row (headline fragments)."""
    lines = sorted(lines, key=lambda e: e["bbox"][1])
    rows: List[List[dict]] = []
    for e in lines:
        x0, y0, x1, y1 = e["bbox"]
        if rows:
            p = rows[-1][-1]["bbox"]
            y_ov = min(y1, p[3]) - max(y0, p[1])
            sh = min(y1 - y0, p[3] - p[1])
            x_ov = min(x1, p[2]) - max(x0, p[0])
            sw = min(x1 - x0, p[2] - p[0])
            if sh > 0 and y_ov / sh > ROW_OVERLAP_FRACTION and (sw <= 0 or x_ov / sw < ROW_MAX_X_OVERLAP):
                rows[-1].append(e)
                continue
        rows.append([e])
    out = []
    for row in rows:
        out.extend(sorted(row, key=lambda e: e["bbox"][0]))
    return out


def order_lines(lines: List[dict], label: str) -> List[dict]:
    """Line order inside one region; a region that really holds two columns is read column by column."""
    if len(lines) <= 2 or label in ROW_ORDER_LABELS:
        return _rows(lines)
    groups = _split_vertical(lines, lambda e: e["bbox"], 0.0)
    if groups:
        ylo = min(e["bbox"][1] for e in lines)
        yhi = max(e["bbox"][3] for e in lines)
        span = max(1.0, yhi - ylo)
        ok = all(len(g) >= LINE_SUBCOL_MIN_LINES and
                 (max(e["bbox"][3] for e in g) - min(e["bbox"][1] for e in g)) >= LINE_SUBCOL_MIN_SPAN * span
                 for g in groups)
        if ok:
            out = []
            for g in groups:
                out.extend(order_lines(g, label))
            return out
    return _rows(lines)


@dataclass
class ColumnModel:
    centers: List[float]
    bounds: List[float]
    width: float
    head_ids: set = field(default_factory=set)

    @property
    def n(self) -> int:
        return len(self.centers)

    def span(self, b: Sequence[float]) -> Tuple[int, int]:
        """Column range [c0, c1] a box occupies: its home column plus any neighbour it covers substantially."""
        edges = [-math.inf] + self.bounds + [math.inf]
        x0, x1 = b[0], b[2]
        home = bisect.bisect_right(self.bounds, (x0 + x1) / 2)
        need = COL_SPAN_MIN_OVERLAP * self.width

        def cover(j: int) -> float:
            return min(x1, edges[j + 1]) - max(x0, edges[j])

        c0 = c1 = home
        while c0 > 0 and cover(c0 - 1) >= need:
            c0 -= 1
        while c1 < self.n - 1 and cover(c1 + 1) >= need:
            c1 += 1
        return c0, c1


def detect_columns(regions: List[Region], page_w: float, page_h: float) -> Optional[ColumnModel]:
    """Find the page's text columns by clustering the horizontal centres of column-width text regions."""
    cands = [r for r in regions if r.label in COL_LABELS and 0.04 * page_w <= r.w <= 0.6 * page_w]
    if len(cands) < 2:
        return None
    body = [r.w for r in cands if r.label in ("Text", "List-item")] or [r.w for r in cands]
    width = _median(body, page_w)
    pts = sorted(((r.bbox[0] + r.bbox[2]) / 2, r.h) for r in cands if r.w <= COL_MAX_WIDTH_FACTOR * width)
    clusters: List[List[Tuple[float, float]]] = []
    for x, h in pts:
        if clusters and x - clusters[-1][-1][0] <= COL_CLUSTER_GAP * width:
            clusters[-1].append((x, h))
        else:
            clusters.append([(x, h)])
    centers = [sum(x * h for x, h in c) / max(1.0, sum(h for _, h in c))
               for c in clusters if sum(h for _, h in c) >= COL_MIN_SUPPORT_FRAC * page_h]
    merged: List[float] = []
    for c in centers:
        if merged and c - merged[-1] < 0.6 * width:
            merged[-1] = (merged[-1] + c) / 2
        else:
            merged.append(c)
    if len(merged) < 2:
        return None
    bounds = [(a + b) / 2 for a, b in zip(merged, merged[1:])]
    pitch = _median([b - a for a, b in zip(merged, merged[1:])], width)
    return ColumnModel(merged, bounds, min(width, pitch))


def _precedence_graph(items: list, bbox_of: Callable, cols: ColumnModel):
    """Edges a→b ("a is read before b") from rules R1/R2 of precedence_order."""
    n = len(items)
    B = np.array([bbox_of(i) for i in items], dtype=float)
    S = np.array([cols.span(b) for b in B], dtype=int)
    c0, c1 = S[:, 0], S[:, 1]
    cy = (B[:, 1] + B[:, 3]) / 2
    h = np.maximum(B[:, 3] - B[:, 1], 1.0)
    w = np.maximum(B[:, 2] - B[:, 0], 1.0)
    succ: List[set] = [set() for _ in range(n)]
    indeg = [0] * n

    def edge(a: int, b: int) -> None:
        if b not in succ[a]:
            succ[a].add(b)
            indeg[b] += 1

    for a in range(n):
        for b in range(a + 1, n):
            if c0[a] <= c1[b] and c0[b] <= c1[a]:                           # R1
                y_ov = min(B[a, 3], B[b, 3]) - max(B[a, 1], B[b, 1])
                x_ov = min(B[a, 2], B[b, 2]) - max(B[a, 0], B[b, 0])
                if y_ov > 0.5 * min(h[a], h[b]) and x_ov < 0.2 * min(w[a], w[b]):
                    first = a if B[a, 0] < B[b, 0] else b
                else:
                    first = a if cy[a] < cy[b] else b
                edge(first, b if first == a else a)
            else:                                                           # R2
                l, r = (a, b) if c1[a] < c0[b] else (b, a)
                lo, hi = (cy[l], cy[r]) if cy[l] < cy[r] else (cy[r], cy[l])
                blocker = ((cy > lo) & (cy < hi) &
                           (c0 <= c1[l]) & (c1 >= c0[l]) &
                           (c0 <= c1[r]) & (c1 >= c0[r]))
                if not blocker.any():
                    edge(l, r)

    return succ, indeg, c0, B


def precedence_order(items: list, bbox_of: Callable, cols: ColumnModel,
                     stats: Optional[Counter] = None) -> list:
    """
    Newspaper reading order from pairwise precedence rules on column indices
    (after Breuel 2003): R1 same column → upper first; R2 left column first
    unless a spanning block sits between them.  Topologically sorted.
    """
    n = len(items)
    if n <= 1:
        return list(items)
    succ, indeg, c0, B = _precedence_graph(items, bbox_of, cols)
    key = lambda i: (int(c0[i]), float(B[i, 1]), float(B[i, 0]))
    heap = [(key(i), i) for i in range(n) if indeg[i] == 0]
    heapq.heapify(heap)
    placed = [False] * n
    out = []
    while len(out) < n:
        if not heap:
            i = min((k for k in range(n) if not placed[k]), key=key)
            heapq.heappush(heap, (key(i), i))
            if stats is not None:
                stats["order_cycles"] += 1
        _, i = heapq.heappop(heap)
        if placed[i]:
            continue
        placed[i] = True
        out.append(items[i])
        for j in succ[i]:
            indeg[j] -= 1
            if indeg[j] == 0 and not placed[j]:
                heapq.heappush(heap, (key(j), j))
    return out


def page_reading_order(regions: List[Region], page_w: float, page_h: float,
                       stats: Optional[Counter] = None) -> Tuple[List[Region], Optional[ColumnModel]]:
    """Masthead zone first, then the body — both by column precedence."""
    cols = detect_columns(regions, page_w, page_h)
    if cols is None:
        logger.info("  [ORDER] fewer than 2 columns found — using XY-cut.")
        return xy_cut_order(regions, lambda r: r.bbox, lambda r: r.position), None
    bb = lambda r: r.bbox
    body_tops = [r.bbox[1] for r in regions if r.label == "Text" and r.h >= BODY_MIN_H_FRAC * page_h]
    head: List[Region] = []
    if body_tops:
        y_body = min(body_tops)
        tol = HBREAK_TOL_FRAC * page_h
        wide = [r for r in regions if r.bbox[3] <= y_body + tol and
                (lambda s: s[1] - s[0] + 1)(cols.span(r.bbox)) >= max(2, cols.n / 2)]
        if wide:
            y_band = max(r.bbox[3] for r in wide)
            head = [r for r in regions if r.bbox[3] <= y_band + 3 * tol and r.bbox[3] <= y_body + tol]
    rest = [r for r in regions if r not in head]
    cols.head_ids = {r.rid for r in head}
    ordered = (precedence_order(head, bb, cols, stats) if head else []) + \
              precedence_order(rest, bb, cols, stats)
    logger.info(f"  [ORDER] {cols.n} column(s) at x≈" + ", ".join(f"{c:.0f}" for c in cols.centers) +
                f"; masthead zone {len(head)} region(s).")
    return ordered, cols


def order_violations(ordered: List[Region], cols: Optional[ColumnModel]) -> int:
    """Number of reading-order rules the final order breaks (0 for a clean page)."""
    body = [r for r in ordered if r.rid not in cols.head_ids] if cols is not None else []
    if len(body) < 2:
        return 0
    succ, _, _, _ = _precedence_graph(body, lambda r: r.bbox, cols)
    return sum(1 for a, nxt in enumerate(succ) for b in nxt if b < a)


def order_page(regions: List[Region], per_region: Dict[int, List[dict]],
               page_w: float, page_h: float, stats: Optional[Counter] = None
               ) -> Tuple[List[dict], List[Region], Optional[ColumnModel]]:
    cols = None
    if READING_ORDER_MODE == "surya":
        ordered = sorted(regions, key=lambda r: (r.position if r.position is not None else 1e9, r.bbox[1]))
    elif READING_ORDER_MODE == "xycut":
        ordered = xy_cut_order(regions, lambda r: r.bbox, lambda r: r.position)
    else:
        ordered, cols = page_reading_order(regions, page_w, page_h, stats)
    elements: List[dict] = []
    for r in ordered:
        if not r.is_text:
            continue
        for e in order_lines(per_region.get(r.rid, []), r.label):
            e["reading_position"] = len(elements)
            elements.append(e)
    return elements, ordered, cols


def order_agreement(ordered: List[Region]) -> float:
    """Kendall-style concordance between our order and Surya's position (1.0 = identical)."""
    pos = [r.position for r in ordered if r.is_text and r.source == "layout" and r.position is not None]
    conc = disc = 0
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            if pos[i] < pos[j]:
                conc += 1
            elif pos[i] > pos[j]:
                disc += 1
    return conc / (conc + disc) if (conc + disc) else 1.0


def dedupe_safety_net(elements: List[dict]) -> Tuple[List[dict], int]:
    """Last resort: drop a geometric duplicate only if it also READS like the other copy."""
    if len(elements) < 2:
        return elements, 0
    order = sorted(range(len(elements)), key=lambda i: elements[i]["bbox"][1])
    dropped = [False] * len(elements)
    for oi, i in enumerate(order):
        if dropped[i]:
            continue
        a = elements[i]
        area_a = max(1.0, _area(a["bbox"]))
        for j in order[oi + 1:]:
            if dropped[j]:
                continue
            b = elements[j]
            if b["bbox"][1] > a["bbox"][3]:
                break
            area_b = max(1.0, _area(b["bbox"]))
            ov = _overlap_area(a["bbox"], b["bbox"])
            if ov / min(area_a, area_b) < DEDUPE_CONTAINMENT:
                continue
            if max(area_a, area_b) / min(area_a, area_b) > DEDUPE_MAX_AREA_RATIO:
                continue
            sim = difflib.SequenceMatcher(None, a["text"].lower(), b["text"].lower()).ratio()
            if sim < DEDUPE_MIN_TEXT_SIM:
                continue
            loser = i if a["confidence"] <= b["confidence"] else j
            dropped[loser] = True
            logger.debug(f"      [DEDUPE] dropped '{elements[loser]['text'][:40]}' (sim={sim:.2f})")
            if loser == i:
                break
    kept = [e for k, e in enumerate(elements) if not dropped[k]]
    for k, e in enumerate(kept):
        e["reading_position"] = k
    return kept, sum(dropped)


# ══════════════════════════════════════════════════════════════════════════════
# DEBUG VISUALISATION + REPORTS
# ══════════════════════════════════════════════════════════════════════════════

def _debug_path(stem: str, page_num: int, tag: str, ext: str) -> Path:
    if DEBUG_OVERWRITE:
        return DEBUG_PATH / f"latest_{tag}.{ext}"
    return DEBUG_PATH / f"{stem}_p{page_num:03d}_{tag}.{ext}"


def clear_debug_files() -> None:
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


def text_area(regions: List[Region], inventory: List[List[float]], col_w: float,
              page_w: int, page_h: int) -> List[float]:
    """Bounding box of the page's text (regions + column-width detected lines), padded."""
    boxes = [r.bbox for r in regions if r.source != "fallback"]
    boxes += [l for l in inventory if l[2] - l[0] >= ORPHAN_MIN_WIDTH_FRAC * col_w]
    if not boxes:
        return [0.0, 0.0, float(page_w), float(page_h)]
    pad = AUDIT_TEXT_AREA_PAD * page_w
    return _clip_box([min(b[0] for b in boxes) - pad, min(b[1] for b in boxes) - pad,
                      max(b[2] for b in boxes) + pad, max(b[3] for b in boxes) + pad], page_w, page_h)


def audit_coverage(page_gray: Image.Image, elements: List[dict], regions: List[Region],
                   stem: str, page_num: int, skipped: Sequence[Sequence[float]] = (),
                   area: Optional[Sequence[float]] = None) -> Tuple[float, List[Tuple[int, int]], float]:
    """
    Share of page ink that lies outside every OCR'd box.  Ink the legibility
    gate chose to skip is counted separately (orange in the debug image), and
    ink outside the page's text area (facing-page slivers, scan borders) is
    ignored.  Returns (uncovered fraction, uncovered bands, skipped fraction).
    """
    s = AUDIT_SCALE
    small = page_gray.reduce(s)
    sw, sh = small.size
    ink = small.point(lambda v: 255 if v < AUDIT_INK_LEVEL else 0).filter(ImageFilter.MaxFilter(3))
    if area is not None:
        inside = Image.new("L", (sw, sh), 0)
        ImageDraw.Draw(inside).rectangle([area[0] / s, area[1] / s, area[2] / s, area[3] / s], fill=255)
        ink = ImageChops.multiply(ink, inside)
    pad = 6

    def mask_of(boxes) -> Image.Image:
        m = Image.new("L", (sw, sh), 0)
        d = ImageDraw.Draw(m)
        for x0, y0, x1, y1 in boxes:
            d.rectangle([(x0 - pad) / s, (y0 - pad) / s, (x1 + pad) / s, (y1 + pad) / s], fill=255)
        return m

    covered = mask_of([e["bbox"] for e in elements] + [r.bbox for r in regions if not r.is_text])
    skip_m = mask_of(skipped)
    skipped_ink = ImageChops.subtract(ImageChops.multiply(ink, skip_m), covered)
    uncovered = ImageChops.subtract(ImageChops.subtract(ink, covered), skip_m)
    total = ink.histogram()[255]
    miss, skip_n = uncovered.histogram()[255], skipped_ink.histogram()[255]
    frac = miss / total if total else 0.0
    skip_frac = skip_n / total if total else 0.0
    col_ink = list(ink.resize((sw, 1), _RESAMPLE.BOX).tobytes())
    col_miss = list(uncovered.resize((sw, 1), _RESAMPLE.BOX).tobytes())
    bands, start = [], None
    for i in range(sw + 1):
        bad = i < sw and col_ink[i] >= 8 and col_miss[i] >= 0.5 * col_ink[i]
        if bad and start is None:
            start = i
        elif not bad and start is not None:
            if (i - start) * s >= AUDIT_MIN_BAND_PX:
                bands.append((start * s, i * s))
            start = None
    msg = f"  [COVERAGE] {frac * 100:.1f}% of text-area ink lies outside every OCR'd box"
    if skip_frac:
        msg += f" ({skip_frac * 100:.1f}% more skipped as illegible)"
    if bands:
        msg += "; mostly-uncovered band(s) at x=" + ", ".join(f"{a}–{b}px" for a, b in bands)
    (logger.warning if frac > QA_MAX_UNCOVERED else logger.info)(msg)
    vis = Image.merge("RGB", (small, small, small))
    vis.paste(Image.new("RGB", (sw, sh), (255, 0, 0)), mask=uncovered)
    vis.paste(Image.new("RGB", (sw, sh), (255, 150, 0)), mask=skipped_ink)
    if area is not None:
        ImageDraw.Draw(vis).rectangle([area[0] / s, area[1] / s, area[2] / s, area[3] / s],
                                      outline=(0, 120, 255), width=1)
    vis.save(str(_debug_path(stem, page_num, "03_uncovered", "jpg")), "JPEG", quality=85)
    return frac, bands, skip_frac


def save_layout_debug(image: Image.Image, regions: List[Region], path: Path) -> None:
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    f = _pil_font(15)
    for r in regions:
        x0, y0, x1, y1 = r.bbox
        rgb = _hex_rgb(_label_hex(r.label))
        pos = "·" if r.position is None else f"{r.position:g}"
        tag = f"[{pos}] {r.label}{'' if r.source == 'layout' else ' (' + r.source + ')'}"
        draw.rectangle([x0, y0, x1, y1], outline=rgb + (210,), fill=rgb + (22,),
                       width=4 if r.source != "layout" else 2)
        draw.rectangle([x0, y0, x0 + len(tag) * 9 + 6, y0 + 20], fill=rgb + (175,))
        draw.text((x0 + 3, y0 + 2), tag, fill=(255, 255, 255, 255), font=f)
    img.save(str(path), "JPEG", quality=85)


def save_order_debug(image: Image.Image, ordered: List[Region], rules: List[List[float]], path: Path,
                     cols: Optional["ColumnModel"] = None) -> None:
    """Numbered regions joined by the reading path; dashed purple lines are column boundaries."""
    s = 3
    img = image.convert("RGB").reduce(s)
    draw = ImageDraw.Draw(img, "RGBA")
    f = _pil_font(18)
    if cols is not None:
        for x in cols.bounds:
            for y in range(0, img.height, 24):
                draw.line([(x / s, y), (x / s, y + 12)], fill=(140, 0, 200, 200), width=2)
    for rx0, ry0, rx1, ry1 in rules:
        draw.rectangle([rx0 / s - 1, ry0 / s, rx1 / s + 1, ry1 / s], fill=(0, 160, 255, 160))
    pts = []
    k = 0
    for r in ordered:
        if not r.is_text:
            continue
        x0, y0, x1, y1 = [v / s for v in r.bbox]
        draw.rectangle([x0, y0, x1, y1], outline=_hex_rgb(_label_hex(r.label)) + (200,), width=2)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        pts.append((cx, cy))
        draw.text((cx - 8, cy - 9), str(k), fill=(200, 0, 0, 255), font=f)
        k += 1
    if len(pts) > 1:
        draw.line(pts, fill=(220, 0, 0, 120), width=2)
    img.save(str(path), "JPEG", quality=85)


def save_ocr_debug(image: Image.Image, elements: List[dict], path: Path,
                   illegible: Sequence[dict] = ()) -> None:
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    f = _pil_font(12)
    for e in illegible:
        x0, y0, x1, y1 = e["bbox"]
        draw.rectangle([x0, y0, x1, y1], outline=(255, 140, 0, 200), fill=(255, 140, 0, 40), width=1)
    for e in elements:
        x0, y0, x1, y1 = e["bbox"]
        rgb = _hex_rgb(_label_hex(e.get("source_label", "Text")))
        draw.rectangle([x0, y0, x1, y1], outline=rgb + (180,), width=1)
        draw.text((x0 + 1, y0), e["text"][:55], fill=rgb + (220,), font=f)
    img.save(str(path), "JPEG", quality=85)


def save_layout_report(regions: List[Region], path: Path, filename: str, page_num: int) -> None:
    lines = [f"FILE: {filename}   PAGE: {page_num}", f"Regions: {len(regions)}", "=" * 86,
             f"{'ID':>4} {'POS':>7}  {'LABEL':<20} {'SOURCE':<13} {'X0':>6} {'Y0':>6} {'X1':>6} {'Y1':>6}",
             "-" * 86]
    for r in regions:
        pos = "—" if r.position is None else f"{r.position:g}"
        x0, y0, x1, y1 = r.bbox
        lines.append(f"{r.rid:>4} {pos:>7}  {r.label:<20} {r.source:<13} "
                     f"{x0:>6.0f} {y0:>6.0f} {x1:>6.0f} {y1:>6.0f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_ocr_report(elements: List[dict], path: Path, filename: str, page_num: int) -> None:
    lines = [f"FILE: {filename}   PAGE: {page_num}", f"OCR elements: {len(elements)}", "=" * 80]
    for e in elements:
        x0, y0, x1, y1 = e["bbox"]
        lines.append(f"[{e['reading_position']:>4}] r{e['region_id']:<4} {e['source_label']:<15} "
                     f"conf={e['confidence']:.2f} ({x0:.0f},{y0:.0f}→{x1:.0f},{y1:.0f}) | {e['text']}")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_illegible_report(outcome: RecognitionOutcome, path: Path, filename: str, page_num: int) -> None:
    lines = [f"FILE: {filename}   PAGE: {page_num}",
             f"Low-confidence page: {'YES' if outcome.strict_page else 'no'}   "
             f"raw mean conf {outcome.raw_mean:.2f}, {outcome.raw_low:.0%} body lines < {LOW_CONF_LINE}",
             f"Gated regions: {len(outcome.gated)}   dropped whole: {len(outcome.dropped_regions)}   "
             f"dropped lines: {len(outcome.illegible)}",
             "Thresholds: conf ≥ {:.2f}, model agreement ≥ {:.2f}, real words ≥ {:.0%}, region keep ≥ {:.0%}"
             .format(STRICT_LINE_CONF, STRICT_MIN_AGREEMENT, STRICT_MIN_LEXICAL, STRICT_REGION_MIN_KEEP),
             "=" * 100]
    for e in sorted(outcome.illegible, key=lambda d: (d["rid"], d["bbox"][1])):
        ag = "—" if e["agree"] is None else f"{e['agree']:.2f}"
        lines.append(f"r{e['rid']:<4} conf={e['conf']:.2f} agree={ag} words={e['lex']:.0%} "
                     f"[{e['reason']}] | {e['text']}")
    path.write_text("\n".join(lines), encoding="utf-8")


QA_FIELDS = ["file", "page", "regions", "tile_regions_added", "pictures_to_text",
             "regions_split_at_rules", "regions_split_at_gutters", "edges_snapped",
             "gutter_line_splits", "bleed_pieces_dropped", "column_bleed_lines",
             "orphan_blocks", "orphan_blocks_dropped", "subheads_rescued", "elements",
             "raw_mean_conf", "raw_low_conf_frac", "low_confidence_page", "gated_regions",
             "illegible_lines", "illegible_regions", "mean_conf", "low_conf_frac",
             "uncovered_ink", "skipped_ink", "order_agreement", "columns", "order_violations",
             "order_cycles", "fallback_switches", "noise_rejected", "seconds", "flags"]


def init_qa_report() -> None:
    with open(QA_REPORT_PATH, "w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=QA_FIELDS).writeheader()


def append_qa_row(row: dict) -> None:
    with open(QA_REPORT_PATH, "a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=QA_FIELDS).writerow(row)


def column_bleed(elements: List[dict], gmap: Optional[GutterMap], cols: Optional["ColumnModel"],
                 col_w: float) -> List[dict]:
    """
    Final body lines that still cross a column boundary through a blank gap
    (looser than the splitting test, so it catches short gutters the splitter
    had to leave alone; only gaps near the page's column boundaries count,
    so rivers of word spaces inside a column do not).
    """
    if gmap is None or cols is None:
        return []
    near = GUTTER_BLEED_NEAR_BOUND * col_w
    out = []
    for e in elements:
        if e.get("source_label") in HEADER_LABELS:
            continue
        x0, x1 = e["bbox"][0], e["bbox"][2]
        bounds = [b for b in cols.bounds if x0 + near < b < x1 - near]
        if not bounds:
            continue
        cuts = gmap.line_cuts(e["bbox"], loose=True)
        if any(abs(c - b) <= near for c in cuts for b in bounds):
            out.append(e)
            logger.debug(f"      [BLEED] r{e['region_id']} ({x0:.0f},{e['bbox'][1]:.0f}→{x1:.0f},"
                         f"{e['bbox'][3]:.0f}) | {e['text'][:60]}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# PAGE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def process_page(pil_image: Image.Image, page_num: int, filename: str, stem: str,
                 models: Models) -> List[dict]:
    """
    1  Prepare   normalised page for layout; flattened + rule-free page for detection/OCR
    2  Gutters   page-wide line inventory → line height → gutter map; inventory lines
                 cut at gutters
    3  Layout    Surya full page + tiles (both split at gutters first, so coverage is judged
                 column by column); split at rules; snap box edges to gutters
    4  Sweep     re-examine pictures → orphan blocks (+ subheads) → targeted re-layout
    5  Detect    lines per region, cut at rules and gutters     ┐ nothing is OCR'd
    6  Resolve   each physical line gets exactly one owner      ┘ until here
    7  Recognise batched TrOCR (+ second opinion) → legibility gate
    8  Order     column model + precedence
    9  QA        coverage audit, column-bleed check, per-page CSV row, debug images
    """
    t0 = time.time()
    stats: Counter = Counter()
    iw, ih = pil_image.size
    logger.info("")
    logger.info("─" * 62)
    logger.info(f"  PAGE {page_num}  [{iw}×{ih}]  {filename}")
    logger.info("─" * 62)

    # 1 ─ Prepare
    imgs = prepare_page(pil_image)
    imgs.clean_gray.save(str(_debug_path(stem, page_num, "00_clean", "jpg")), "JPEG", quality=80)
    if imgs.vrules:
        logger.info(f"  [RULES] {len(imgs.vrules)} vertical rule(s) found and removed from the OCR image.")

    # 2 ─ Line inventory + gutter map
    inventory = line_inventory(imgs.clean_gray, models.det)
    med_h = _median([b[3] - b[1] for b in inventory], 40.0)
    gmap: Optional[GutterMap] = None
    if GUTTERS_ENABLED:
        gmap = GutterMap(imgs.clean_gray, med_h)
        gmap.save_debug(imgs.clean_gray, _debug_path(stem, page_num, "00_gutters", "jpg"))
        split_inv: List[List[float]] = []
        for b in inventory:
            pieces = gmap.split_line(b)
            if len(pieces) > 1:
                stats["inventory_lines_split"] += 1
            split_inv.extend(pieces)
        inventory = split_inv
        logger.info(f"  [GUTTER] gutter map built (line height {med_h:.0f}px); "
                    f"{stats['inventory_lines_split']} detected line(s) ran across a gutter and were cut.")

    # 3 ─ Layout: full page, then tiles
    logger.info("  [LAYOUT] Running LayoutPredictor on the full page …")
    parsed = run_layout([imgs.layout_rgb], models.layout, log_scale=True)[0]
    next_id = [0]
    regions: List[Region] = []
    for d in parsed:
        regions.append(Region(next_id[0], d["bbox"], d["label"], d["position"], d["top_k"], "layout"))
        next_id[0] += 1
    n_page = len(regions)
    regions = _nms_same_label(regions)
    if len(regions) < n_page:
        logger.debug(f"  [LAYOUT] NMS removed {n_page - len(regions)} duplicate region(s).")
    logger.info(f"  [LAYOUT] full-page pass: {len(regions)} region(s).")
    col_w = _median([r.w for r in regions if r.label == "Text" and r.w < 0.4 * iw], iw / 5)
    if gmap is not None:
        regions = split_regions_at_gutters(regions, gmap, med_h, col_w, next_id, stats)
    if LAYOUT_TILING:
        tiles = tiled_layout(imgs, models.layout)
        if gmap is not None:
            tiles = split_tile_boxes_at_gutters(tiles, gmap, med_h, col_w, stats)
        regions = _nms_same_label(merge_tile_layout(regions, tiles, next_id, stats))
    if not regions:
        logger.warning("  No layout regions — full page treated as one Text region.")
        regions = [Region(next_id[0], [0.0, 0.0, float(iw), float(ih)], "Text", 0.0, {}, "fallback")]
        next_id[0] += 1
    n0 = len(regions)
    regions = _split_regions_at_rules(regions, imgs.vrules, next_id)
    stats["regions_split"] = len(regions) - n0
    if gmap is not None:
        snap_regions_to_gutters(regions, gmap, med_h, col_w, stats)
    _normalise_gutters(regions)
    counts = Counter(r.label for r in regions)
    logger.info(f"  [LAYOUT] {len(regions)} regions: " +
                "  ".join(f"{k}×{v}" for k, v in sorted(counts.items())))

    # 4 ─ Picture re-check → orphan blocks / subheads → targeted re-layout
    regions = rescue_pictures(regions, inventory, imgs, models.layout, next_id, stats)
    blocks, subheads, dropped = find_orphan_blocks(inventory, regions, iw, ih, col_w, med_h)
    stats["orphan_blocks"], stats["orphan_dropped"] = len(blocks), dropped
    stats["subheads"] = len(subheads)
    for sb in subheads:
        regions.append(Region(next_id[0], _grow(sb, 6), "Section-header", None, {}, "subhead"))
        next_id[0] += 1
    if blocks:
        added = relayout_blocks(blocks, imgs, models.layout, inventory, next_id)
        if gmap is not None:
            added = split_regions_at_gutters(added, gmap, med_h, col_w, next_id, stats)
            snap_regions_to_gutters(added, gmap, med_h, col_w, stats)
        regions.extend(added)
    _normalise_gutters(regions)
    if stats["edges_snapped"] or stats["regions_split_at_gutters"]:
        logger.info(f"  [GUTTER] {stats['regions_split_at_gutters']} box(es) split and "
                    f"{stats['edges_snapped']} box edge(s) moved onto measured gutters.")
    save_layout_debug(pil_image, regions, _debug_path(stem, page_num, "01_layout", "jpg"))
    save_layout_report(regions, _debug_path(stem, page_num, "01_layout_report", "txt"), filename, page_num)

    # 5 + 6 ─ Detect and resolve
    lines, found = collect_region_lines(regions, imgs, models.det, gmap, stats, col_w)
    logger.info(f"  [DETECT] {sum(len(v) for v in lines.values())} owned line(s) in "
                f"{sum(1 for r in regions if r.is_text)} text region(s); "
                f"{stats['gutter_line_splits']} cut at a gutter, "
                f"{stats['bleed_pieces_dropped']} piece(s) from a neighbouring column discarded.")

    # 7 ─ Recognise (+ legibility gate)
    outcome = recognise_page(regions, lines, found, imgs, models, med_h, stats)
    if outcome.gated:
        save_illegible_report(outcome, _debug_path(stem, page_num, "02_illegible_report", "txt"),
                              filename, page_num)

    # 8 ─ Order
    elements, ordered, cols = order_page(regions, outcome.per_region, iw, ih, stats)
    elements, n_dup = dedupe_safety_net(elements)
    if n_dup:
        logger.info(f"  [DEDUPE] safety net removed {n_dup} element(s).")
    agree = order_agreement(ordered)
    regress = order_violations(ordered, cols)
    logger.info(f"  [ORDER] {len(elements)} element(s); block order breaks {regress} column rule(s); "
                f"{stats['order_cycles']} tie-break(s) between overlapping boxes; "
                f"agreement with Surya order = {agree:.2f}")

    # 9 ─ QA
    bleed = column_bleed(elements, gmap, cols, col_w)
    if bleed:
        logger.info(f"  [BLEED] {len(bleed)} final line(s) still appear to cross a column gutter.")
    frac, bands, skip_frac = 0.0, [], 0.0
    if AUDIT_ENABLED:
        try:
            skipped = [e["bbox"] for e in outcome.illegible] + \
                      [r.bbox for r in regions if r.rid in outcome.dropped_regions]
            area = text_area(regions, inventory, col_w, iw, ih) if AUDIT_TEXT_AREA_ONLY else None
            frac, bands, skip_frac = audit_coverage(imgs.norm_gray, elements, regions, stem, page_num,
                                                    skipped, area)
        except Exception as exc:
            logger.warning(f"  [COVERAGE] audit failed: {exc}")
    save_order_debug(pil_image, ordered, imgs.vrules, _debug_path(stem, page_num, "02_order", "jpg"), cols)
    if elements or outcome.illegible:
        save_ocr_debug(pil_image, elements, _debug_path(stem, page_num, "02_ocr_overlay", "jpg"),
                       outcome.illegible)
    if elements:
        save_ocr_report(elements, _debug_path(stem, page_num, "02_ocr_report", "txt"), filename, page_num)

    confs = [e["confidence"] for e in elements]
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    low = sum(1 for c in confs if c < LOW_CONF_LINE) / len(confs) if confs else 1.0
    flags = []
    if agree < QA_MIN_ORDER_AGREEMENT:
        flags.append("ORDER_DISAGREES_WITH_SURYA")
    if regress > 0:
        flags.append("ORDER_RULE_BROKEN")
    if stats["order_cycles"] > 0:
        flags.append("ORDER_CYCLE")
    if len(bleed) > QA_MAX_BLEED_LINES:
        flags.append("COLUMN_BLEED")
    if frac > QA_MAX_UNCOVERED:
        flags.append("UNCOVERED_INK")
    if mean_conf < QA_MIN_MEAN_CONF:
        flags.append("LOW_CONFIDENCE")
    if outcome.strict_page:
        flags.append("LOW_LEGIBILITY_PAGE")
    if outcome.illegible:
        flags.append("ILLEGIBLE_TEXT_SKIPPED")
    if flags:
        logger.warning(f"  [QA] page {page_num} flagged for review: {', '.join(flags)}")
    append_qa_row({
        "file": filename, "page": page_num, "regions": len(regions),
        "tile_regions_added": stats["tile_added"], "pictures_to_text": stats["pictures_to_text"],
        "regions_split_at_rules": stats["regions_split"],
        "regions_split_at_gutters": stats["regions_split_at_gutters"] + stats["tile_boxes_split_at_gutters"],
        "edges_snapped": stats["edges_snapped"],
        "gutter_line_splits": stats["gutter_line_splits"] + stats["inventory_lines_split"],
        "bleed_pieces_dropped": stats["bleed_pieces_dropped"], "column_bleed_lines": len(bleed),
        "orphan_blocks": stats["orphan_blocks"], "orphan_blocks_dropped": stats["orphan_dropped"],
        "subheads_rescued": stats["subheads"], "elements": len(elements),
        "raw_mean_conf": f"{outcome.raw_mean:.3f}", "raw_low_conf_frac": f"{outcome.raw_low:.3f}",
        "low_confidence_page": int(outcome.strict_page), "gated_regions": len(outcome.gated),
        "illegible_lines": len(outcome.illegible), "illegible_regions": len(outcome.dropped_regions),
        "mean_conf": f"{mean_conf:.3f}", "low_conf_frac": f"{low:.3f}",
        "uncovered_ink": f"{frac:.3f}", "skipped_ink": f"{skip_frac:.3f}",
        "order_agreement": f"{agree:.3f}",
        "columns": cols.n if cols else 1, "order_violations": regress,
        "order_cycles": stats["order_cycles"],
        "fallback_switches": stats["fallback_switches"], "noise_rejected": stats["noise_rejected"],
        "seconds": f"{time.time() - t0:.1f}", "flags": ";".join(flags),
    })
    logger.info(f"  [RESULT] page {page_num}: {len(elements)} element(s), mean conf {mean_conf:.2f}, "
                f"{stats['fallback_switches']} second-opinion switch(es), "
                f"{len(outcome.illegible)} illegible line(s) skipped, {time.time() - t0:.0f}s")
    return elements


# ══════════════════════════════════════════════════════════════════════════════
# INVISIBLE TEXT LAYER
# ══════════════════════════════════════════════════════════════════════════════

def _insert_font(page: "fitz.Page", font: "fitz.Font", using_freesans: bool) -> None:
    if using_freesans:
        page.insert_font(fontname=TEXT_FONT_TAG, fontfile=FONT_PATH)
    else:
        page.insert_font(fontname=TEXT_FONT_TAG, fontbuffer=font.buffer)


def _insert_shape(page, elements, sx, sy, font) -> int:
    """Each line at a font size matching its printed height, horizontally scaled to its printed width."""
    pw = page.rect.width
    shape = page.new_shape()
    n = 0
    for e in elements:
        text = e["text"].replace("\n", " ").strip()
        if not text:
            continue
        x0, y0, x1, y1 = e["bbox"]
        box_w = (min(x1 * sx, pw) - x0 * sx)
        box_h = (y1 - y0) * sy
        if box_w <= 0 or box_h <= 0:
            continue
        fs = min(MAX_FONT_PT, max(MIN_FONT_PT, box_h * TEXT_HEIGHT_FACTOR))
        payload = text + ELEMENT_SEPARATOR
        tl = font.text_length(payload, fontsize=fs)
        if tl <= 0:
            continue
        hs = box_w / tl
        if hs < HSCALE_MIN:
            fs = max(MIN_FONT_PT, fs * hs / HSCALE_MIN)
            tl = font.text_length(payload, fontsize=fs)
            hs = box_w / tl if tl > 0 else HSCALE_MIN
        hs = min(HSCALE_MAX, max(HSCALE_MIN, hs))
        base = fitz.Point(x0 * sx, y1 * sy - box_h * TEXT_BASELINE_FRAC)
        shape.insert_text(base, payload, fontname=TEXT_FONT_TAG, fontsize=fs,
                          render_mode=3, morph=(base, fitz.Matrix(hs, 1)))
        n += 1
    shape.commit(overlay=True)
    return n


def _insert_textwriter(page, elements, sx, sy, font) -> int:
    """Script F's method (kept as a fallback)."""
    pw = page.rect.width
    writer = fitz.TextWriter(page.rect)
    n = 0
    for e in elements:
        x0, y0, x1, y1 = e["bbox"]
        text = e["text"] + ELEMENT_SEPARATOR
        fs = max(4.0, (y1 - y0) * 0.85 * sy)
        avail = min(x1 * sx, pw) - x0 * sx
        if avail > 0:
            tw = font.text_length(text, fontsize=fs)
            if tw > avail:
                fs = max(MIN_FONT_PT, fs * avail / tw)
        try:
            writer.append(fitz.Point(x0 * sx, y1 * sy), text, font=font, fontsize=fs)
            n += 1
        except Exception as exc:
            logger.debug(f"    Text insert skipped: {exc}")
    if n:
        writer.write_text(page, overlay=True, render_mode=3, color=(0, 0, 0))
    return n


def insert_text_layer(page, elements, img_size, font, using_freesans) -> int:
    iw, ih = img_size
    sx, sy = page.rect.width / iw, page.rect.height / ih
    if TEXT_LAYER_MODE == "shape":
        try:
            _insert_font(page, font, using_freesans)
            return _insert_shape(page, elements, sx, sy, font)
        except Exception as exc:
            logger.warning(f"  Scaled text layer failed ({exc}); using TextWriter layer.")
    return _insert_textwriter(page, elements, sx, sy, font)


# ══════════════════════════════════════════════════════════════════════════════
# PDF PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def process_pdf(input_path: str, output_path: str, models: Models) -> bool:
    filename = os.path.basename(input_path)
    stem = Path(input_path).stem
    logger.info(f"\n{'━' * 62}\n  PROCESSING: {filename}\n{'━' * 62}")
    try:
        with fitz.open(input_path) as doc:
            logger.info(f"  {len(doc)} page(s) — rendering at {DPI} DPI for OCR …")
            font, using_freesans = load_text_font()
            for idx in range(len(doc)):
                page_num, page = idx + 1, doc[idx]
                pil_img = page_to_pil(page, dpi=DPI)
                img_size = pil_img.size
                try:
                    elements = process_page(pil_img, page_num, filename, stem, models)
                except Exception as exc:
                    logger.error(f"  OCR failed for page {page_num}: {exc}")
                    import traceback; traceback.print_exc()
                    elements = []
                del pil_img
                if not elements:
                    logger.info(f"  Page {page_num}: no elements.")
                    continue
                existing = page.get_text().strip()
                if existing:
                    logger.info(f"  Page {page_num}: removing existing text layer ({len(existing)} chars).")
                    page.add_redact_annot(page.rect)
                    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
                n = insert_text_layer(page, elements, img_size, font, using_freesans)
                logger.info(f"  Page {page_num}: inserted {n}/{len(elements)} element(s).")
            apply_document_metadata(doc, filename)
            try:
                doc.subset_fonts()
                logger.info("  Embedded font subset to the glyphs used.")
            except Exception as exc:
                logger.warning(f"  Font subsetting skipped ({exc}). `pip install fonttools` enables it.")
            doc.save(output_path, deflate=True, garbage=4, clean=True,
                     deflate_images=False, encryption=fitz.PDF_ENCRYPT_KEEP)
            logger.info(f"  Saved: {output_path}")
        setup_pdfa_compliance(output_path)
        with fitz.open(output_path) as chk:
            total = 0
            for i, pg in enumerate(chk):
                n = len(pg.get_text().strip())
                total += n
                logger.info(f"  Final PDF page {i + 1}: {n} extractable characters")
        (logger.info if total else logger.error)(
            f"  {'SUCCESS' if total else 'PROBLEM'}: {total} characters of searchable text.")
        return total > 0
    except Exception as exc:
        logger.error(f"  process_pdf failed: {exc}")
        import traceback; traceback.print_exc()
        return False


def compress_to_target_size(input_pdf: Path, output_pdf: Path, original_size: int) -> Path:
    max_target = int(original_size * 1.15)
    current = input_pdf.stat().st_size
    logger.info(f"  Target ≤ {max_target // 1024} KB (original {original_size // 1024} KB + 15%); "
                f"OCR file is {current // 1024} KB.")
    if current <= max_target:
        shutil.copy2(input_pdf, output_pdf)
        logger.info("  Within budget — no compression needed.")
        return output_pdf
    for i, g in enumerate((4, 3, 2)):
        tmp = output_pdf.with_suffix(f".temp_{i}.pdf")
        try:
            with fitz.open(str(input_pdf)) as d:
                d.save(str(tmp), deflate=True, garbage=g, clean=True, deflate_images=False,
                       encryption=fitz.PDF_ENCRYPT_KEEP)
            size = tmp.stat().st_size
            logger.info(f"  Option {i + 1}: {size // 1024} KB")
            if size <= max_target:
                with fitz.open(str(tmp)) as chk:
                    chars = sum(len(p.get_text().strip()) for p in chk)
                if chars > 0:
                    shutil.move(str(tmp), str(output_pdf))
                    return output_pdf
            tmp.unlink(missing_ok=True)
        except Exception as exc:
            logger.error(f"  Compression option {i + 1} failed: {exc}")
            tmp.unlink(missing_ok=True)
    logger.warning("  All deflate options exceeded the 15% budget; returning the OCR file as-is.")
    shutil.copy2(input_pdf, output_pdf)
    return output_pdf


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║  OPTICOLUMNS 2026  –  script F-H  (gutters → tiles → gate)   ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")
    logger.info(f"  Input {INPUT_DIR}  Output {OUTPUT_DIR}  DPI {DPI}  "
                f"TrOCR {TROCR_MODEL_NAME} (+{TROCR_FALLBACK_MODEL_NAME}), beams {NUM_BEAMS}  "
                f"order {READING_ORDER_MODE}  layout tiles {'on' if LAYOUT_TILING else 'off'}  "
                f"legibility gate {'on' if STRICT_ENABLED else 'off'}  "
                f"gutters {'on' if GUTTERS_ENABLED else 'off'}")
    input_folder, output_folder = Path(INPUT_DIR), Path(OUTPUT_DIR)
    if not input_folder.exists():
        logger.error(f"Input folder '{INPUT_DIR}' not found.")
        sys.exit(1)
    output_folder.mkdir(exist_ok=True)
    clear_debug_files()
    init_qa_report()

    pdf_files = sorted(input_folder.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files in '{INPUT_DIR}'.")
        sys.exit(1)
    pending, summary = [], []
    for p in pdf_files:
        final = output_folder / f"{p.stem}.pdf"
        if SKIP_EXISTING_OUTPUT and final.exists():
            summary.append((p.name, "SKIPPED", final.stat().st_size))
        else:
            pending.append(p)
    if not pending:
        logger.info("  All files have already been processed. Nothing to do.")
        return

    models = load_models()
    for pdf_path in pending:
        orig = pdf_path.stat().st_size
        tmp = output_folder / f"{pdf_path.stem}_ocr_temp.pdf"
        final = output_folder / f"{pdf_path.stem}.pdf"
        if not process_pdf(str(pdf_path), str(tmp), models):
            summary.append((pdf_path.name, "FAILED", 0))
            tmp.unlink(missing_ok=True)
            continue
        result = compress_to_target_size(tmp, final, orig)
        summary.append((pdf_path.name, "OK", result.stat().st_size))
        tmp.unlink(missing_ok=True)

    logger.info(f"\n{'═' * 62}\n  SUMMARY\n{'═' * 62}")
    for name, status, sz in summary:
        logger.info(f"  {status:<8}  {name}  ({sz // 1024} KB)")
    logger.info(f"  QA report: {QA_REPORT_PATH.resolve()}  (pages with flags need a human look)")
    logger.info(f"\nAll done! Output files in '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()