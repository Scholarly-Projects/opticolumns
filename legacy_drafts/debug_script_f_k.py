#!/usr/bin/env python3
"""
Opticolumns  –  debug_script_f_k.py
======================================================================
Base: debug_script_f_j.py (tiled Surya layout, one TrOCR model, no sweep),
with the recognition lessons of the 11-page ground-truth survey of F-F,
F-I and F-J.
"""

import sys
import os
import re
import csv
import gc
import math
import time
import bisect
import difflib
import heapq
import datetime
import hashlib
import uuid
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

# ── Branding / attribution / document metadata ────────────────────────────────
# Written into every output PDF (document properties + XMP) and into the QA
# report, so a derivative can always be traced to the tool, its author, the
# run that produced it and the exact source file.
APP_NAME        = "Opticolumns"
APP_VERSION     = "2026"
APP_SCRIPT      = "debug_script_f_k.py"
APP_URL         = "https://github.com/Scholarly-Projects/opticolumns"
APP_AUTHOR      = "Andrew Weymouth"
APP_INSTITUTION = "University of Idaho"       # the author's affiliation
APP_LICENSE     = "MIT"
APP_CREATOR     = f"{APP_NAME} {APP_VERSION} ({APP_URL})"
APP_CITATION    = f"{APP_AUTHOR}. {APP_NAME} ({APP_VERSION}). {APP_INSTITUTION}. {APP_URL}"
DOC_SUBJECT     = "OCR-processed historic newspaper"
DOC_KEYWORDS    = "OCR; historic newspaper; searchable PDF; Opticolumns; Surya; TrOCR"
DOC_LANGUAGE    = "en-US"
OPT_NAMESPACE   = "https://github.com/Scholarly-Projects/opticolumns/ns/"
KEEP_SOURCE_TITLE_AUTHOR = True   # keep the input PDF's own Title/Author when it has them

# ── Page preparation ──────────────────────────────────────────────────────────
LAYOUT_AUTOCONTRAST_CUTOFF = 0.5  # % clipped each end for the global normalisation
BG_REDUCE      = 8                # background estimate computed at 1/8 size
BG_MAX_KERNEL  = 9                # max-filter (odd) at 1/8 size ≈ 72 px: removes text, keeps paper
BG_BLUR        = 6

# Rule (printed line) removal – applied to the detection/OCR image only.
# Layout sees the rules: they are real column cues for Surya.
RULE_REMOVAL_ENABLED   = True
RULE_INK_LEVEL         = 140      # grey below which a pixel can belong to a rule
RULE_MIN_LEN_FRAC_V    = 0.06     # vertical rule ≥ 6 % of page height (≈300–390 px)
RULE_MIN_LEN_FRAC_H    = 0.06     # horizontal rule ≥ 6 % of page width
RULE_MAX_THICK_PX      = max(4, round(DPI * 0.03))   # thicker = letter stroke / photo, not a rule
RULE_CHUNK             = 512      # column chunk for the run-length pass (memory bound)
RULE_SPLIT_LABELS      = {"Text", "Section-header", "Caption", "List-item", "Footnote"}
RULE_EDGE_MARGIN       = 40       # px; a rule this close to a region edge is its gutter, not inside it
RULE_SPLIT_MIN_YFRAC   = 0.5      # rule must run alongside ≥ 50 % of the region's height to split it

# ── Tiled layout (F-J) ────────────────────────────────────────────────────────
# Surya 0.17 shrinks every layout input to ≤ 1024×1024 px of area
# (LAYOUT_MODEL_PIXELS).  A tile of side T reaches the model at
# DPI × 1024 / T.  LAYOUT_TARGET_DPI = 100 → T ≈ 3070 px at 300 DPI, about
# four broadsheet columns; a 5 000 × 6 500 page becomes 3 × 3 tiles.
LAYOUT_MODEL_PIXELS    = 1024 * 1024
LAYOUT_TARGET_DPI      = 100      # resolution each tile reaches the model at (Surya's own default is 96)
LAYOUT_TILE_OVERLAP    = 1024     # px at render DPI; must exceed the widest column (≈ 650–1000 px)
LAYOUT_BATCH_SIZE      = 2        # tiles per LayoutPredictor call (MPS memory)
LAYOUT_EDGE_TOL        = 10       # px; a box this close to an interior tile edge was cut by it
LAYOUT_CUT_COVERED     = 0.70     # a cut box ≥ 70 % inside a box another tile saw whole is a duplicate
LAYOUT_DUP_IOU         = 0.50     # whole boxes from two tiles with IoU above this are one region
LAYOUT_DUP_CONTAIN     = 0.85     # … as is a whole box ≥ 85 % inside a larger one of the same kind
LAYOUT_STITCH_ALIGN    = 0.60     # cut pieces are stitched when ≥ 60 % aligned across the cut
LAYOUT_STITCH_GAP      = 24       # px; and no more than this far apart

# ── Line extent (F-K) ─────────────────────────────────────────────────────────
# Lines are detected and cropped inside an "allowed box": the region grown by
# up to this much, stopping half-way to any side-by-side region and at
# printed rules (see _allowed_boxes).
LINE_EXPAND_X_FRAC     = 0.35     # × typical column width, sideways (≈ 200–230 px at 300 DPI)
LINE_EXPAND_Y_PX       = max(24, round(DPI * 0.12))   # up/down (≈ one body line)
# Whitespace-gutter splitting only inside regions wider than this many
# columns (Surya merged two columns).  Never inside ordinary columns.
GUTTER_SPLIT_MIN_COLS  = 1.4
# Body lines wider than this aspect ratio are read in word-gap segments
# (TrOCR squeezes every crop into 384 × 384).  A normal column line is ≈ 13.
MAX_BODY_AR            = 18.0
REPEAT_MIN_LEN         = 6        # a line of ≥ 6 non-space chars drawn from ≤ 2 symbols is noise

# ── Memory (F-J) ──────────────────────────────────────────────────────────────
DETECTOR_BATCH_SIZE    = 2        # region crops per DetectionPredictor call

# ── TrOCR (single model) ──────────────────────────────────────────────────────
TROCR_MODELS = {
    "handwritten":       "microsoft/trocr-base-handwritten",
    "printed":           "microsoft/trocr-base-printed",       # fine-tuned on SROIE receipts
    "large_handwritten": "microsoft/trocr-large-handwritten",  # fine-tuned on IAM
    "large_printed":     "microsoft/trocr-large-printed",      # fine-tuned on SROIE receipts
    "large_stage1":      "microsoft/trocr-large-stage1",       # pre-trained on synthetic printed lines only
}
# The model that produced F-I's clean lower-case lines.  Worth benchmarking
# "large_stage1" against your ground truth too: it was never fine-tuned on
# receipts or handwriting.
TROCR_MODEL_NAME   = TROCR_MODELS["large_handwritten"]

NUM_BEAMS          = 3     # beam search; 1 = greedy
TROCR_BATCH_SIZE   = 8
MAX_NEW_TOKENS     = 192
USE_MPS            = True  # Apple-silicon GPU if available (falls back to CPU on error)

# ── Noise filter ──────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD             = 0.25
SINGLE_CHAR_CONFIDENCE_THRESHOLD = 0.50
MIN_LINE_H                       = 5
MIN_LINE_W                       = 10
SPARSE_LINE_WIDTH_RATIO          = 2.0
HALLUCINATION_MAX_CONF           = 0.60   # patterns below are rejected under this confidence
HALLUCINATION_PATTERNS = [
    r"^(1[89]\d\d|20\d\d)(\s+\S{1,4})*$",   # "1907", "1961 62", "1907 08 09"
    r"^[0-9.,\s]*0[.,]0{3}[0-9.,\s]*$",     # "0.000", "0,000 000"
    r"^[\W\d_]+$",                          # only punctuation / digits / symbols
]

# ── Line crops ────────────────────────────────────────────────────────────────
LINE_PAD_X = 0.30   # × line height, horizontal padding (keeps line-end hyphens)
LINE_PAD_Y = 0.10   # × line height, vertical padding
CROP_BORDER_FRAC = 0.10   # white border added around each crop before TrOCR
MAX_HEADER_AR   = 10.0    # header crops wider than this (w/h) are split at word gaps
HEADER_SEGMENT_OVERLAP = 0.20   # only used where no word gap is found
EDGE_STROKE_FRAC = 0.80   # a crop-edge column this dark over its height = residual rule

# ── Line detection / ownership ────────────────────────────────────────────────
DET_PAD             = 12     # px padding around region crops for detection
LINE_MIN_IN_REGION  = 0.50   # a detected line must lie ≥ 50 % inside its region
LINE_DUP_CONTAIN    = 0.60   # line ≥ 60 % inside a larger line = same ink, drop
LINE_CLIP_PAD       = 6      # px; lines are clipped to region box ± this (horizontally)
LARGE_TYPE_H        = 1.6    # a line ≥ 1.6 × body line height is headline / ad type

# ── Column gutters ────────────────────────────────────────────────────────────
# Whitespace between columns, measured on the flattened, rule-free page.
GUTTERS_ENABLED        = True
GUTTER_SCALE           = 4        # gutter map works on 4×4 px cells
GUTTER_INK_LEVEL       = 160      # pixel darker than this is ink
GUTTER_CELL_MIN_DARK   = 4        # a cell is ink if ≥ 4 of its 16 pixels are dark
GUTTER_MIN_W_PX        = max(8, round(DPI * 0.035))   # narrowest gutter (≈10 px at 300 DPI)
GUTTER_RULE_MIN_LINES  = 1.5      # thin vertical strokes longer than this (× line height) are rule dashes
GUTTER_MIN_TALL_LINES  = 12       # gap must stay blank ≥ 12 line-heights (word gaps / rivers never do)

# ── Region housekeeping ───────────────────────────────────────────────────────
MIN_REGION_W = 40
MIN_REGION_H = 15
GUTTER_MAX_OVERLAP_FRAC = 0.25   # side-by-side regions overlapping less than this are
                                 # trimmed back to the gutter mid-line
WHOLE_CROP_MAX_LINES    = 3.0    # whole-region fallback read only for regions ≤ 3 line-heights tall

# ── Reading order ─────────────────────────────────────────────────────────────
# "columns" (recommended): regions are assigned to page columns and ordered on
#   column INDICES — down the leftmost column, then the next column to the
#   right — with horizontal breaks only at elements that span columns
#   (mastheads, banners, wide photos).  Immune to scan skew / page warp.
# "xycut": pixel-gutter XY-cut (fails on skewed scans where no gutter is clean).
# "surya": Surya's own position.
READING_ORDER_MODE    = "columns"
COL_LABELS            = {"Text", "List-item", "Caption", "Footnote", "Section-header",
                         "Handwriting", "Text-inline-math"}
COL_MAX_WIDTH_FACTOR  = 1.40      # regions wider than 1.4 × typical column width don't define columns
COL_CLUSTER_GAP       = 0.40      # × column width: centre gap that starts a new column
COL_MIN_SUPPORT_FRAC  = 0.05      # a column needs regions totalling ≥ 5 % of page height
COL_SPAN_MIN_OVERLAP  = 0.40      # region also occupies a neighbouring column if it covers ≥ 40 % of it
HBREAK_TOL_FRAC       = 0.006     # × page height: tolerance when finding the masthead zone
BODY_MIN_H_FRAC       = 0.03      # "body text" region: ≥ 3 % of page height (defines where the masthead zone ends)
XY_SHRINK_FRAC        = 0.03      # boxes shrunk this much (per side) before gap search …
XY_SHRINK_MAX         = 12        # … capped at this many px (absorbs gutter bleed)
XY_MIN_GAP            = 2
XY_VERTICAL_TOL_FRAC  = 0.008     # vertical cut may cross items totalling ≤ 0.8 % of height
XY_SPAN_FACTOR        = 1.40      # item ≥ 1.4 × median width counts as spanning columns
LINE_SUBCOL_MIN_LINES = 3         # in-region sub-column needs ≥ 3 lines …
LINE_SUBCOL_MIN_SPAN  = 0.40      # … spanning ≥ 40 % of the region's height
ROW_OVERLAP_FRACTION  = 0.5
ROW_MAX_X_OVERLAP     = 0.5

# ── Safety-net dedupe (should rarely fire now) ───────────────────────────────
DEDUPE_CONTAINMENT     = 0.5
DEDUPE_MAX_AREA_RATIO  = 3.0
DEDUPE_MIN_TEXT_SIM    = 0.60     # geometric duplicates must also read alike

# ── Legibility gate — only on low-confidence pages ───────────────────────────
# "auto": probe reads decide whether a page is low-confidence; "always": treat
# every page as low-confidence; "off": noise filter only.
LEGIBILITY_GATE        = "auto"
PROBES_PER_REGION      = 3        # lines read first in every text region (kept, never re-read)
PROBES_EXTRA           = 3        # more probes for a region with some, but too few, passing probes
# Page is low-confidence when the body-text probes have …
LOWPAGE_MEDIAN_CONF    = 0.80     # … a median confidence below this, OR
LOWPAGE_WEAK_CONF      = 0.60     # … more than LOWPAGE_WEAK_FRAC of them below this
LOWPAGE_WEAK_FRAC      = 0.25
LOWPAGE_MIN_PROBES     = 8        # too few probes to judge → never low-confidence
# A probe PASSES (region test) when:
PROBE_MIN_CONF         = 0.70     # confidence ≥ this
PROBE_MIN_WORDS        = 0.50     # ≥ this share of its checkable words are real words
# A line in a legible region is KEPT when:
KEEP_MIN_CONF          = 0.55
KEEP_MIN_WORDS         = 0.34
NOWORD_MIN_CONF        = (0.85, 0.75)   # (probe, keep) conf needed when a line has no checkable word
# Character density = characters ÷ (box width ÷ box height).  Body type ≈ 1.7–2.6.
# TrOCR's guesses on blotched lines are much too short (or run-on).
DENSITY_MIN            = 0.90
DENSITY_MIN_HEADER     = 0.50     # headline / ad type can be wide or letter-spaced
DENSITY_MAX            = 4.50
REGION_MIN_PASS        = 0.60     # region legible if ≥ 60 % of its probes pass …
REGION_MIN_PASS_EXTRA  = 0.50     # … or ≥ 50 % after the extra probes
REGION_MIN_KEEP        = 0.50     # a legible region whose lines mostly fail is dropped after all
# Word list: system dictionary + the words in the TrOCR vocabularies + optional lexicon.txt
LEXICON_PATHS          = ["/usr/share/dict/words", "/usr/dict/words"]
USER_LEXICON_PATH      = "lexicon.txt"          # one word per line (local names, places …)
CAPITALISED_UNKNOWN_CREDIT = 0.5                # unknown Capitalised word (a name?) counts half

# ── QA thresholds (page flagged in qa_report.csv) ────────────────────────────
QA_MIN_ORDER_AGREEMENT = 0.85
QA_MAX_UNCOVERED       = 0.12
QA_MIN_MEAN_CONF       = 0.70

# ── Coverage audit ────────────────────────────────────────────────────────────
AUDIT_ENABLED     = True
AUDIT_SCALE       = 8
AUDIT_INK_LEVEL   = 215
AUDIT_MIN_BAND_PX = 150
AUDIT_TEXT_AREA_PAD = 40          # px; ink farther than this outside the text regions' extent is ignored

DEBUG_OVERWRITE = True

# ── Layout label taxonomy ─────────────────────────────────────────────────────
OCR_LABELS = {
    "Text", "Section-header", "Caption", "Footnote", "List-item", "Page-footer",
    "Page-header", "Table-of-contents", "Handwriting", "Text-inline-math",
    "Formula", "Table", "Form",
}
SKIP_LABELS = {"Picture", "Figure"}
HEADER_LABELS = {"Section-header", "Page-header"}
ROW_ORDER_LABELS = {"Table", "Form", "Table-of-contents"}   # never split into sub-columns

# ── Text layer ────────────────────────────────────────────────────────────────
EMBED_FONT      = True
TEXT_LAYER_MODE = "shape"       # "shape" (h-scaled, recommended) or "textwriter" (script F)
TEXT_FONT_TAG   = "OcrTxt"
TEXT_HEIGHT_FACTOR = 0.80       # font size = box height × this
TEXT_BASELINE_FRAC = 0.20       # baseline sits this fraction of box height above the bottom
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
ILLEGIBLE_RGB = (255, 143, 0)     # orange: text deliberately left out by the legibility gate
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
    position: Optional[float]          # Surya reading-order position within its tile
    top_k: Dict[str, float] = field(default_factory=dict)
    source: str = "layout"             # "layout" (whole in its tile) | "stitched" (joined across tiles)
    tile: int = -1                     # layout tile the region came from
    confidence: float = 1.0            # Surya layout confidence

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
    vrules: List[List[float]]    # vertical rules [x0, y0, x1, y1]
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


def _center(b: Sequence[float]) -> Tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _median(vals: Sequence[float], default: float) -> float:
    vals = [v for v in vals if v > 0]
    return statistics.median(vals) if vals else default


# ══════════════════════════════════════════════════════════════════════════════
# PAGE PREPARATION  –  clean inputs instead of tiled contrast stretching
# ══════════════════════════════════════════════════════════════════════════════

def page_to_pil(page: "fitz.Page", dpi: int = DPI) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def _flatten_background(gray: Image.Image) -> Image.Image:
    """
    Divide out uneven paper tone (yellowing, microfilm vignetting) using a
    low-frequency background estimate.  Unlike tiled autocontrast, blank
    gutters stay blank: noise is never stretched to full contrast.
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
        full = (cs[length:] - cs[:-length]) == length          # window [i, i+length) all set
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
    Detect thin, long printed rules (column rules, cut-off rules, box borders)
    and whiten them.  Returns (cleaned array, vertical rules, masked fraction).

    Rules are the source of the "I", "!", "( … )", "0 0" tokens at line ends in
    script F output, and of line boxes that fuse two columns across a narrow
    ruled gutter.  They are removed only from the detection/OCR image; their
    positions are kept and used as hard column separators.
    """
    h, w = arr.shape
    dark = arr < RULE_INK_LEVEL
    lv = max(60, int(h * RULE_MIN_LEN_FRAC_V))
    lh = max(60, int(w * RULE_MIN_LEN_FRAC_H))
    t = RULE_MAX_THICK_PX
    v = _long_runs(dark, lv, 0)
    v &= ~_long_runs(v, t + 1, 1)          # too thick horizontally → a stroke/photo
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
    return PageImages(
        raw_rgb=pil, norm_gray=norm, layout_rgb=norm.convert("RGB"),
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


# One id per batch run: every PDF and every QA row from the same run share it.
RUN_ID = str(uuid.uuid4())


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class DocProvenance:
    """Per-document facts gathered while a PDF is processed."""
    source_file: str
    source_sha256: str
    pages: int = 0
    pages_ocrd: int = 0
    confidences: List[float] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)

    @property
    def mean_conf(self) -> float:
        return sum(self.confidences) / len(self.confidences) if self.confidences else 0.0


# Custom Opticolumns properties: (name, PDF/A valueType, description).
# Every property written must be declared in the PDF/A extension schema, so
# both the values and the declarations are generated from this one list.
_OPT_PROPERTIES = [
    ("ToolName",          "Text",    "Name of the tool that produced the OCR text layer"),
    ("Version",           "Text",    "Version of the tool that produced the OCR text layer"),
    ("Script",            "Text",    "Script file that produced the OCR text layer"),
    ("ProjectURL",        "Text",    "Project site of the OCR tool"),
    ("ToolAuthor",        "Text",    "Author of the OCR tool"),
    ("ToolAffiliation",   "Text",    "Institutional affiliation of the tool author"),
    ("ToolLicense",       "Text",    "Licence of the OCR tool"),
    ("Citation",          "Text",    "Recommended citation for the OCR tool"),
    ("RunID",             "Text",    "Identifier of the processing run that produced this file"),
    ("SourceFile",        "Text",    "File name of the source PDF"),
    ("SourceSHA256",      "Text",    "SHA-256 checksum of the source PDF"),
    ("RenderDPI",         "Integer", "Resolution at which pages were rendered for OCR"),
    ("LayoutModel",       "Text",    "Layout analysis model and checkpoint"),
    ("LayoutTargetDPI",   "Integer", "Resolution at which layout tiles reached the layout model"),
    ("OCRModel",          "Text",    "Text recognition model"),
    ("PagesProcessed",    "Integer", "Pages given a new OCR text layer"),
    ("PagesTotal",        "Integer", "Pages in the document"),
    ("MeanConfidence",    "Real",    "Mean recognition confidence over all OCR lines (0-1)"),
    ("QAFlags",           "Text",    "Quality-assurance flags raised during processing"),
]


def _opt_values(prov: Optional[DocProvenance]) -> Dict[str, str]:
    v = {
        "ToolName": APP_NAME, "Version": APP_VERSION, "Script": APP_SCRIPT,
        "ProjectURL": APP_URL, "ToolAuthor": APP_AUTHOR, "ToolAffiliation": APP_INSTITUTION,
        "ToolLicense": APP_LICENSE, "Citation": APP_CITATION, "RunID": RUN_ID,
        "RenderDPI": str(DPI), "LayoutTargetDPI": str(LAYOUT_TARGET_DPI),
        "LayoutModel": f"Surya LayoutPredictor {settings.LAYOUT_MODEL_CHECKPOINT}",
        "OCRModel": TROCR_MODEL_NAME,
    }
    if prov is not None:
        v.update({
            "SourceFile": prov.source_file, "SourceSHA256": prov.source_sha256,
            "PagesProcessed": str(prov.pages_ocrd), "PagesTotal": str(prov.pages),
            "MeanConfidence": f"{prov.mean_conf:.3f}",
            "QAFlags": "; ".join(prov.flags) if prov.flags else "none",
        })
    return v


def create_xmp_metadata(title, author, subject, keywords, creator, producer,
                        creation_date, modify_date, language=DOC_LANGUAGE,
                        prov: Optional[DocProvenance] = None,
                        doc_id: str = "", instance_id: str = "") -> Optional[str]:
    """
    XMP packet for PDF/A-1b.  Standard schemas carry what any repository
    system reads (title, creator tool, keywords, source, identifiers, a
    processing-history event); the custom opt: schema carries attribution
    and run provenance, each property declared in the PDF/A extension
    schema.  Title/Author/Subject/Keywords/Creator/Producer mirror the
    document Info dictionary exactly, as PDF/A requires.
    """
    try:
        e = lambda v: xml_escape(str(v))
        ns = e(OPT_NAMESPACE)
        creator_xml = (f"\n      <dc:creator><rdf:Seq><rdf:li>{e(author)}</rdf:li></rdf:Seq></dc:creator>"
                       if author else "")
        source_xml = (f"\n      <dc:source>{e(prov.source_file)}</dc:source>" if prov is not None else "")
        params = (f"{APP_SCRIPT}; DPI {DPI}; layout tiles at {LAYOUT_TARGET_DPI} DPI; "
                  f"OCR {TROCR_MODEL_NAME}; run {RUN_ID}")
        values = _opt_values(prov)
        opt_xml = "\n".join(f"      <opt:{n}>{e(values[n])}</opt:{n}>"
                            for n, _t, _d in _OPT_PROPERTIES if n in values)
        decl_xml = "\n".join(f"""                <rdf:li rdf:parseType="Resource">
                  <pdfaProperty:name>{n}</pdfaProperty:name>
                  <pdfaProperty:valueType>{t}</pdfaProperty:valueType>
                  <pdfaProperty:category>internal</pdfaProperty:category>
                  <pdfaProperty:description>{e(d)}</pdfaProperty:description>
                </rdf:li>""" for n, t, d in _OPT_PROPERTIES)
        return f"""<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about="" xmlns:pdf="http://ns.adobe.com/pdf/1.3/">
      <pdf:Producer>{e(producer)}</pdf:Producer>
      <pdf:Keywords>{e(keywords)}</pdf:Keywords>
    </rdf:Description>
    <rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">
      <dc:format>application/pdf</dc:format>
      <dc:title><rdf:Alt><rdf:li xml:lang="x-default">{e(title)}</rdf:li></rdf:Alt></dc:title>{creator_xml}
      <dc:description><rdf:Alt><rdf:li xml:lang="x-default">{e(subject)}</rdf:li></rdf:Alt></dc:description>
      <dc:language><rdf:Bag><rdf:li>{e(language)}</rdf:li></rdf:Bag></dc:language>{source_xml}
    </rdf:Description>
    <rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/">
      <xmp:CreatorTool>{e(creator)}</xmp:CreatorTool>
      <xmp:CreateDate>{creation_date}</xmp:CreateDate>
      <xmp:ModifyDate>{modify_date}</xmp:ModifyDate>
      <xmp:MetadataDate>{modify_date}</xmp:MetadataDate>
    </rdf:Description>
    <rdf:Description rdf:about=""
        xmlns:xmpMM="http://ns.adobe.com/xap/1.0/mm/"
        xmlns:stEvt="http://ns.adobe.com/xap/1.0/sType/ResourceEvent#">
      <xmpMM:DocumentID>{e(doc_id)}</xmpMM:DocumentID>
      <xmpMM:InstanceID>{e(instance_id)}</xmpMM:InstanceID>
      <xmpMM:History>
        <rdf:Seq>
          <rdf:li rdf:parseType="Resource">
            <stEvt:action>converted</stEvt:action>
            <stEvt:instanceID>{e(instance_id)}</stEvt:instanceID>
            <stEvt:parameters>{e(params)}</stEvt:parameters>
            <stEvt:softwareAgent>{e(creator)}</stEvt:softwareAgent>
            <stEvt:when>{modify_date}</stEvt:when>
          </rdf:li>
        </rdf:Seq>
      </xmpMM:History>
    </rdf:Description>
    <rdf:Description rdf:about="" xmlns:pdfaid="http://www.aiim.org/pdfa/ns/id/">
      <pdfaid:part>1</pdfaid:part>
      <pdfaid:conformance>B</pdfaid:conformance>
    </rdf:Description>
    <rdf:Description rdf:about="" xmlns:opt="{ns}">
{opt_xml}
    </rdf:Description>
    <rdf:Description rdf:about=""
        xmlns:pdfaExtension="http://www.aiim.org/pdfa/ns/extension/"
        xmlns:pdfaSchema="http://www.aiim.org/pdfa/ns/schema#"
        xmlns:pdfaProperty="http://www.aiim.org/pdfa/ns/property#">
      <pdfaExtension:schemas>
        <rdf:Bag>
          <rdf:li rdf:parseType="Resource">
            <pdfaSchema:schema>{e(APP_NAME)} processing and attribution metadata</pdfaSchema:schema>
            <pdfaSchema:namespaceURI>{ns}</pdfaSchema:namespaceURI>
            <pdfaSchema:prefix>opt</pdfaSchema:prefix>
            <pdfaSchema:property>
              <rdf:Seq>
{decl_xml}
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


def apply_document_metadata(doc: "fitz.Document", filename: str,
                            prov: Optional[DocProvenance] = None) -> None:
    """
    Document Info dictionary + XMP.  Title and Author describe the newspaper,
    not the tool: the source PDF's own Title/Author are kept when present
    (KEEP_SOURCE_TITLE_AUTHOR), otherwise Title is the file name and Author
    is left empty.  The tool, its author and the run are credited in Creator,
    Keywords and the XMP opt: properties.
    """
    now = _now()
    pdf_date, xmp_date = get_pdf_date_string(now), get_xmp_date_string(now)
    producer = f"PyMuPDF {fitz.VersionBind}"
    src = doc.metadata or {}
    title = (src.get("title") or "").strip() if KEEP_SOURCE_TITLE_AUTHOR else ""
    author = (src.get("author") or "").strip() if KEEP_SOURCE_TITLE_AUTHOR else ""
    title = title or filename
    doc_id = f"uuid:{uuid.uuid4()}"
    instance_id = f"uuid:{uuid.uuid4()}"
    doc.set_metadata({
        "title": title, "author": author, "subject": DOC_SUBJECT, "keywords": DOC_KEYWORDS,
        "creator": APP_CREATOR, "producer": producer,
        "creationDate": pdf_date, "modDate": pdf_date,
    })
    xmp = create_xmp_metadata(title, author, DOC_SUBJECT, DOC_KEYWORDS, APP_CREATOR, producer,
                              xmp_date, xmp_date, DOC_LANGUAGE, prov, doc_id, instance_id)
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
        """
        Mean per-token probability of the emitted sequence (same scale as
        script F's mean-max-probability for greedy decoding, so the noise
        thresholds carry over).  Padding after EOS is excluded.
        """
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


class Lexicon:
    """
    Word list for the legibility gate.  Sources (all offline):
      * the system dictionary (/usr/share/dict/words — present on macOS);
      * whole words in the TrOCR tokenizers' vocabularies — these hold the
        common inflected forms (students, reported, taking) that Webster's
        word list lacks;
      * an optional user list (USER_LEXICON_PATH) for local names and places.
    Unknown words are also tried with common suffixes removed.
    """
    _STRIP = ".,;:!?\"'`()[]{}<>*_«»“”‘’„—–-…·/\\|"
    _SUFFIXES = (("'s", ""), ("’s", ""), ("ies", "y"), ("ied", "y"), ("iest", "y"), ("ier", "y"),
                 ("ing", ""), ("ing", "e"), ("ed", ""), ("ed", "e"), ("es", ""), ("s", ""),
                 ("ly", ""), ("er", ""), ("er", "e"), ("est", ""), ("ness", ""), ("ment", ""),
                 ("ers", ""), ("ings", ""))

    def __init__(self, tokenizers: Sequence = ()):
        words: Set[str] = set()
        n_dict = 0
        for p in LEXICON_PATHS:
            try:
                with open(p, encoding="utf-8", errors="ignore") as fh:
                    for ln in fh:
                        w = ln.strip().lower()
                        if w.isalpha() and len(w) >= 2:
                            words.add(w)
                            n_dict += 1
                break
            except OSError:
                continue
        n_tok = 0
        for tok in tokenizers:
            try:
                for t in tok.get_vocab():
                    if t[:1] in ("Ġ", "▁"):
                        w = t[1:].lower()
                        if w.isalpha() and len(w) >= 3 and w not in words:
                            words.add(w)
                            n_tok += 1
            except Exception:
                continue
        n_user = 0
        try:
            with open(USER_LEXICON_PATH, encoding="utf-8") as fh:
                for ln in fh:
                    w = ln.strip().lower()
                    if w:
                        words.add(w)
                        n_user += 1
        except OSError:
            pass
        self.words = words
        logger.info(f"  Lexicon: {len(words)} words (dictionary {n_dict}, TrOCR vocab +{n_tok}, "
                    f"{USER_LEXICON_PATH} +{n_user})"
                    + ("" if n_dict else " — no system dictionary found; word test is weaker"))

    def known(self, w: str) -> bool:
        w = w.lower()
        if w in self.words:
            return True
        for suf, rep in self._SUFFIXES:
            if w.endswith(suf) and len(w) - len(suf) >= 3 and (w[:-len(suf)] + rep) in self.words:
                return True
        return False

    def word_score(self, text: str) -> Tuple[float, int]:
        """(credit for real words, number of checkable words) for one line of OCR text."""
        toks = text.split()
        credit, n = 0.0, 0
        for i, raw in enumerate(toks):
            last = i == len(toks) - 1
            if last and raw.rstrip(".,;:!?\"'’”").endswith(("-", "¬")):
                continue                                   # hyphenated line end: a word fragment
            for part in raw.split("-"):
                w = part.strip(self._STRIP)
                if len(w) < 3 or any(ch.isdigit() for ch in w) or not w.replace("'", "").replace("’", "").isalpha():
                    continue                               # short words, numbers, codes: not checked
                if w.isupper() and len(w) <= 5:
                    continue                               # ASUI, KUOI, NCAA …
                n += 1
                if self.known(w):
                    credit += 1.0
                elif w[0].isupper() and w[1:].islower():
                    credit += CAPITALISED_UNKNOWN_CREDIT   # probably a name
        return credit, n


@dataclass
class Models:
    det: DetectionPredictor
    layout: LayoutPredictor
    reader: Recognizer
    lexicon: "Lexicon"


def free_device_memory() -> None:
    """Return cached accelerator memory between stages (MPS keeps freed blocks otherwise)."""
    gc.collect()
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def load_models() -> Models:
    logger.info("=" * 62)
    logger.info("  LOADING MODELS  (Surya Layout/Detection + TrOCR)")
    logger.info("=" * 62)
    # Batch sizes are passed on every call: Surya 0.17 predictors read
    # settings.*_BATCH_SIZE as class attributes at import time, so patching
    # the settings object here (as F-I did) had no effect.
    if not setup_pdfa_resources():
        logger.warning("  PDF/A resources setup incomplete.")

    logger.info("  DetectionPredictor …")
    det = DetectionPredictor()
    logger.info(f"  LayoutPredictor ({settings.LAYOUT_MODEL_CHECKPOINT}) …")
    layout = LayoutPredictor(FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT))
    reader = Recognizer(TROCR_MODEL_NAME)
    lexicon = Lexicon([reader.processor.tokenizer])
    logger.info("  All models ready.\n")
    return Models(det=det, layout=layout, reader=reader, lexicon=lexicon)


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
            "top_k": {_normalise_label(k): v for k, v in (getattr(box, "top_k", {}) or {}).items()},
            "confidence": float(getattr(box, "confidence", 1.0) or 1.0),
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
    """
    A text region that a printed vertical rule runs through (not along its
    edge) spans two columns: Surya merged them across a narrow ruled gutter.
    Split it at the rule, so each column is read on its own.
    """
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
                continue                        # nested, not side-by-side
            xov = a.bbox[2] - b.bbox[0]
            if xov <= 0 or xov > GUTTER_MAX_OVERLAP_FRAC * min(a.w, b.w):
                continue
            mid = (a.bbox[2] + b.bbox[0]) / 2
            a.bbox[2] = mid
            b.bbox[0] = mid


def run_layout(images: List[Image.Image], layout_predictor: LayoutPredictor,
               batch_size: int = LAYOUT_BATCH_SIZE) -> List[Optional[List[dict]]]:
    """
    LayoutPredictor on a list of images, `batch_size` at a time.  A batch that
    fails (typically MPS out-of-memory) is retried one image at a time after
    freeing the device cache.  An image whose layout still fails comes back as
    None — never as an empty list, which would read as "no text here".
    """
    out: List[Optional[List[dict]]] = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        try:
            res = layout_predictor(batch, batch_size=len(batch))
            out.extend(parse_layout_result(r, im.size) for r, im in zip(res, batch))
            continue
        except Exception as exc:
            logger.warning(f"  LayoutPredictor batch failed ({str(exc)[:120]}); retrying one tile at a time")
        free_device_memory()
        for im in batch:
            try:
                out.append(parse_layout_result(layout_predictor([im], batch_size=1)[0], im.size))
            except Exception as exc:
                logger.error(f"  LayoutPredictor failed on a {im.width}×{im.height} tile: {str(exc)[:160]}")
                out.append(None)
            free_device_memory()
    return out


def layout_tile_size(dpi: int = DPI) -> int:
    """Tile side (render px) that reaches Surya's 1024² layout budget at LAYOUT_TARGET_DPI."""
    return int(round(math.sqrt(LAYOUT_MODEL_PIXELS) * dpi / LAYOUT_TARGET_DPI))


def layout_tiles(w: int, h: int) -> List[Tuple[int, int, int, int]]:
    t = layout_tile_size()
    ov = min(LAYOUT_TILE_OVERLAP, t // 2)
    return [(x, y, min(x + t, w), min(y + t, h))
            for y in _tile_origins(h, t, ov) for x in _tile_origins(w, t, ov)]


@dataclass
class _TileBox:
    bbox: List[float]
    label: str
    position: float
    top_k: Dict[str, float]
    confidence: float
    tile: int
    cut: Set[str]                 # sides cut by an interior tile edge: "l", "t", "r", "b"
    centrality: float             # 0 at the tile centre … 1 at its edge (more context = better)

    @property
    def is_text(self) -> bool: return self.label not in SKIP_LABELS


def _cut_sides(b: Sequence[float], f: Sequence[int], w: int, h: int) -> Set[str]:
    tol = LAYOUT_EDGE_TOL
    s: Set[str] = set()
    if f[0] > 0 and b[0] - f[0] <= tol: s.add("l")
    if f[1] > 0 and b[1] - f[1] <= tol: s.add("t")
    if f[2] < w and f[2] - b[2] <= tol: s.add("r")
    if f[3] < h and f[3] - b[3] <= tol: s.add("b")
    return s


def _stitchable(a: _TileBox, b: _TileBox) -> bool:
    """Two pieces of one region cut apart by a tile edge (same kind, different tiles, aligned, touching)."""
    if a.tile == b.tile or a.is_text != b.is_text or not (a.cut or b.cut):
        return False
    A, B = a.bbox, b.bbox
    gap_y = max(A[1], B[1]) - min(A[3], B[3])              # < 0: vertical overlap
    gap_x = max(A[0], B[0]) - min(A[2], B[2])
    x_align = (min(A[2], B[2]) - max(A[0], B[0])) / max(1.0, min(A[2] - A[0], B[2] - B[0]))
    y_align = (min(A[3], B[3]) - max(A[1], B[1])) / max(1.0, min(A[3] - A[1], B[3] - B[1]))
    upper, lower = (a, b) if A[1] <= B[1] else (b, a)
    left, right = (a, b) if A[0] <= B[0] else (b, a)
    # Different labels (a subhead above body text) are joined only when BOTH
    # pieces were cut at the shared edge — otherwise they are two blocks that
    # merely happen to sit next to a tile edge.
    same = a.label == b.label
    vertical = (x_align >= LAYOUT_STITCH_ALIGN and gap_y <= LAYOUT_STITCH_GAP and
                (("b" in upper.cut and "t" in lower.cut) or
                 (same and ("b" in upper.cut or "t" in lower.cut))))
    horizontal = (y_align >= LAYOUT_STITCH_ALIGN and gap_x <= LAYOUT_STITCH_GAP and
                  (("r" in left.cut and "l" in right.cut) or
                   (same and ("r" in left.cut or "l" in right.cut))))
    return vertical or horizontal


def merge_tile_layouts(per_tile: List[Optional[List[dict]]], frames: List[Tuple[int, int, int, int]],
                       w: int, h: int, stats: Counter) -> List[Region]:
    """
    Map every tile's regions to page space and merge them into one layout.
      1  a box cut by an interior tile edge is dropped when another tile saw
         the same block whole (the cut box lies mostly inside a whole box);
      2  whole boxes seen by two tiles are de-duplicated (higher confidence,
         then the more central copy, wins; a box inside a larger box of the
         same kind is dropped, as Surya's own clean_boxes() does per tile);
      3  remaining cut pieces are stitched to their continuation across the
         tile edge (a column taller than a tile, a banner wider than one).
    """
    boxes: List[_TileBox] = []
    for k, (parsed, f) in enumerate(zip(per_tile, frames)):
        if not parsed:
            continue
        cx, cy = (f[0] + f[2]) / 2, (f[1] + f[3]) / 2
        hw, hh = max(1.0, (f[2] - f[0]) / 2), max(1.0, (f[3] - f[1]) / 2)
        for d in parsed:
            b = [d["bbox"][0] + f[0], d["bbox"][1] + f[1], d["bbox"][2] + f[0], d["bbox"][3] + f[1]]
            b = _clip_box(b, w, h)
            bc = _center(b)
            cen = max(abs(bc[0] - cx) / hw, abs(bc[1] - cy) / hh)
            boxes.append(_TileBox(b, d["label"], d["position"], d["top_k"], d.get("confidence", 1.0),
                                  k, _cut_sides(b, f, w, h), cen))
    stats["layout_raw_boxes"] = len(boxes)
    whole = [b for b in boxes if not b.cut]
    cut = [b for b in boxes if b.cut]

    # 1 ─ cut boxes that another tile saw whole
    cut = [c for c in cut if not any(o.tile != c.tile and o.is_text == c.is_text and
                                     _frac_inside(c.bbox, o.bbox) >= LAYOUT_CUT_COVERED for o in whole)]

    # 2 ─ whole duplicates across tiles
    whole.sort(key=lambda b: (-b.confidence, b.centrality, -_area(b.bbox)))
    kept: List[_TileBox] = []
    for b in whole:
        dup = False
        for k in kept:
            if k.tile == b.tile or k.is_text != b.is_text:
                continue
            if _iou(b.bbox, k.bbox) > LAYOUT_DUP_IOU:
                # Same block seen by two tiles: keep the better copy's label,
                # but the UNION of both extents — one tile's box often ends a
                # letter or two short of the type (F-K).
                k.bbox = [min(k.bbox[0], b.bbox[0]), min(k.bbox[1], b.bbox[1]),
                          max(k.bbox[2], b.bbox[2]), max(k.bbox[3], b.bbox[3])]
                dup = True
                break
            if _area(b.bbox) <= _area(k.bbox) and _frac_inside(b.bbox, k.bbox) >= LAYOUT_DUP_CONTAIN:
                dup = True
                break
        if not dup:
            kept.append(b)
    # a kept box that turns out to sit inside a larger kept box of the same kind from another tile
    kept = [b for b in kept if not any(o is not b and o.tile != b.tile and o.is_text == b.is_text and
                                       _area(o.bbox) > _area(b.bbox) and
                                       _frac_inside(b.bbox, o.bbox) >= LAYOUT_DUP_CONTAIN for o in kept)]

    # 3 ─ stitch cut pieces (to each other, or to a whole box they continue)
    pool = kept + cut
    parent = list(range(len(pool)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            if (pool[i].cut or pool[j].cut) and _stitchable(pool[i], pool[j]):
                parent[find(i)] = find(j)
    groups: Dict[int, List[_TileBox]] = {}
    for i, b in enumerate(pool):
        groups.setdefault(find(i), []).append(b)

    regions: List[Region] = []
    for g in sorted(groups.values(), key=lambda g: min((b.tile, b.position) for b in g)):
        main = max(g, key=lambda b: _area(b.bbox))
        if len(g) == 1:
            bb, src = list(main.bbox), "layout"
        else:
            bb = [min(b.bbox[0] for b in g), min(b.bbox[1] for b in g),
                  max(b.bbox[2] for b in g), max(b.bbox[3] for b in g)]
            src = "stitched"
            stats["layout_stitched"] += len(g) - 1
            label_area: Counter = Counter()
            for b in g:
                label_area[b.label] += _area(b.bbox)
            main = max(g, key=lambda b: (label_area[b.label], _area(b.bbox)))
        regions.append(Region(len(regions), bb, main.label, main.position, main.top_k,
                              src, main.tile, main.confidence))
    stats["layout_dropped_dups"] = len(boxes) - len(pool)
    return regions


def layout_page(imgs: "PageImages", layout_predictor: LayoutPredictor,
                stats: Counter) -> Tuple[Optional[List[Region]], List[Tuple[int, int, int, int]]]:
    """
    Surya layout at the resolution the model is built for: overlapping tiles,
    each reaching the model at ≈ LAYOUT_TARGET_DPI.  Returns (regions, tiles);
    regions is None when layout failed on any tile (the page is then not
    OCR'd rather than OCR'd with a hole in it).
    """
    w, h = imgs.layout_rgb.size
    frames = layout_tiles(w, h)
    crops = [imgs.layout_rgb.crop(f) for f in frames]
    t = layout_tile_size()
    logger.info(f"  [LAYOUT] {len(frames)} tile(s) of ≤{t}px (≈{LAYOUT_TARGET_DPI} DPI at the model; "
                f"a whole-page pass would be ≈{DPI * math.sqrt(LAYOUT_MODEL_PIXELS / (w * h)):.0f} DPI)")
    per_tile = run_layout(crops, layout_predictor)
    del crops
    failed = [k for k, r in enumerate(per_tile) if r is None]
    stats["layout_tiles"], stats["layout_tiles_failed"] = len(frames), len(failed)
    if failed:
        logger.error(f"  [LAYOUT] {len(failed)}/{len(frames)} tile(s) failed: "
                     + ", ".join(f"{frames[k]}" for k in failed))
        return None, frames
    regions = merge_tile_layouts(per_tile, frames, w, h, stats)
    logger.info(f"  [LAYOUT] {stats['layout_raw_boxes']} tile box(es) → {len(regions)} region(s) "
                f"({stats['layout_dropped_dups']} duplicate/cut copies dropped, "
                f"{stats['layout_stitched']} piece(s) stitched across tile edges)")
    return regions, frames


# ══════════════════════════════════════════════════════════════════════════════
# SURYA DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_lines(images: List[Image.Image], det: DetectionPredictor
                 ) -> Tuple[List[List[List[float]]], int]:
    """
    Surya line detection, DETECTOR_BATCH_SIZE crops per call.  A failed batch
    is retried crop by crop after freeing the device cache.  Returns
    (crop-relative boxes per image, number of crops whose detection failed).
    F-I logged these failures at DEBUG level only — a region that silently
    yields no lines is exactly how text goes missing — so they are now
    warnings and are counted in the QA report.
    """
    if not images:
        return [], 0
    results: List[object] = []
    for i in range(0, len(images), DETECTOR_BATCH_SIZE):
        batch = [im.convert("RGB") for im in images[i:i + DETECTOR_BATCH_SIZE]]
        try:
            results.extend(det(batch, batch_size=len(batch)))
            continue
        except Exception as exc:
            logger.warning(f"    batched detection failed ({str(exc)[:120]}); retrying one by one")
        free_device_memory()
        for im in batch:
            try:
                results.append(det([im], batch_size=1)[0])
            except Exception as exc2:
                logger.warning(f"    DetectionPredictor failed on a {im.width}×{im.height} crop: {str(exc2)[:120]}")
                results.append(None)
    out, failed = [], 0
    for res in results:
        boxes = []
        if res is None:
            failed += 1
        elif hasattr(res, "bboxes"):
            for box in res.bboxes:
                b = _box_of(box)
                if b and b[2] - b[0] >= MIN_LINE_W and b[3] - b[1] >= MIN_LINE_H:
                    boxes.append(b)
        out.append(boxes)
    return out, failed


def _tile_origins(length: int, tile: int, overlap: int) -> List[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    n = -(-(length - tile) // stride) + 1
    return [round(i * (length - tile) / (n - 1)) for i in range(n)]


# ══════════════════════════════════════════════════════════════════════════════
# COLUMN GUTTERS  –  the whitespace between columns, measured on the page
# ══════════════════════════════════════════════════════════════════════════════

class GutterMap:
    """
    Where the page is blank in tall, narrow vertical strips.

    Built once per page from the flattened, rule-free image at 1/GUTTER_SCALE
    resolution.  A cell is gutter when it belongs to a blank vertical run at
    least GUTTER_MIN_TALL_LINES line-heights long AND a blank horizontal run at
    least GUTTER_MIN_W_PX wide.  The space between two words is blank for one
    line only, so it never qualifies; the gap between two columns does, with or
    without a printed rule in it (rules are already erased from this image, and
    broken/dashed rules that survived are ignored here).
    """

    def __init__(self, clean: Image.Image, med_h: float):
        s = GUTTER_SCALE
        a = np.asarray(clean, dtype=np.uint8) < GUTTER_INK_LEVEL
        try:
            v = _long_runs(a, max(20, int(GUTTER_RULE_MIN_LINES * med_h)), 0)
            v &= ~_long_runs(v, RULE_MAX_THICK_PX + 1, 1)
            a = a & ~_dilate1(v)
        except MemoryError:
            logger.warning("  [GUTTER] dashed-rule filter skipped (memory).")
        H, W = a.shape[0] // s, a.shape[1] // s
        cnt = a[:H * s, :W * s].reshape(H, s, W, s).sum(axis=(1, 3))
        ink = cnt >= GUTTER_CELL_MIN_DARK
        p = np.pad(ink, 1).astype(np.int16)                 # drop isolated specks (scan noise)
        nb = sum(p[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx]
                 for dy in (-1, 0, 1) for dx in (-1, 0, 1)) - ink.astype(np.int16)
        ink &= nb >= 2
        self.s, self.H, self.W = s, H, W
        self.min_w = max(2, int(round(GUTTER_MIN_W_PX / s)))
        tall = max(3, int(round((GUTTER_MIN_TALL_LINES + 1) * med_h / s)))
        self.mask = _long_runs(_long_runs(~ink, tall, 0), self.min_w, 1)

    def line_cuts(self, b: Sequence[float]) -> List[float]:
        """x positions where a text line crosses a column gutter."""
        r0, r1 = max(0, int(b[1] // self.s)), min(self.H, int(math.ceil(b[3] / self.s)))
        c0, c1 = max(0, int(b[0] // self.s)), min(self.W, int(math.ceil(b[2] / self.s)))
        if r1 <= r0 or c1 - c0 < self.min_w + 2:
            return []
        ok = self.mask[r0:r1, c0:c1].all(axis=0)
        return [(c0 + (a + e) / 2) * self.s for a, e in _runs_1d(ok)
                if a > 0 and e < len(ok) and e - a >= self.min_w]

    def split_line(self, b: List[float]) -> List[List[float]]:
        cuts = self.line_cuts(b)
        if not cuts:
            return [b]
        edges = [b[0]] + cuts + [b[2]]
        return [[edges[i], b[1], edges[i + 1], b[3]] for i in range(len(edges) - 1)
                if edges[i + 1] - edges[i] >= MIN_LINE_W]

    def save_debug(self, clean: Image.Image, path: Path) -> None:
        small = clean.reduce(self.s).convert("RGB")
        m = Image.fromarray((self.mask * 255).astype(np.uint8), "L").resize(small.size, _RESAMPLE.NEAREST)
        small.paste(Image.new("RGB", small.size, (0, 140, 255)), mask=m.point(lambda v: 90 if v else 0))
        small.save(str(path), "JPEG", quality=80)


def split_at_separators(b: List[float], rules: List[List[float]],
                        gmap: Optional[GutterMap]) -> List[List[float]]:
    """Cut one detected line at printed column rules and at measured gutters."""
    pieces = _split_line_at_rules(b, rules)
    if gmap is None:
        return pieces
    return [q for p in pieces for q in gmap.split_line(p)]


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


def _allowed_boxes(regions: List[Region], rules: List[List[float]], iw: int, ih: int,
                   col_w: float) -> Dict[int, List[float]]:
    """
    How far each text region's lines may reach beyond the region box.

    Surya's boxes at ~100 DPI hug the ink closely, and a tile-merged box can
    end a letter or two short of the text (the survey's "OTESTANT",
    "nister", "Leonard Chi … ove").  F-J detected lines inside the region
    crop and clipped every line and every TrOCR crop to the region ± a few
    px, so those letters were cut off before TrOCR saw them.

    The allowed box grows the region by up to LINE_EXPAND_X_FRAC of a column
    sideways and LINE_EXPAND_Y_PX up/down, but never past:
      * the half-way line to a side-by-side region (text or picture), or
      * a printed vertical rule running alongside the region.
    So a line may reach into its own margin and gutter, never into the next
    column; region membership (and so reading order) is unchanged.
    """
    hx = LINE_EXPAND_X_FRAC * col_w
    vy = LINE_EXPAND_Y_PX
    out: Dict[int, List[float]] = {}
    for r in regions:
        if not r.is_text:
            continue
        x0, x1 = r.bbox[0] - hx, r.bbox[2] + hx
        rcx = (r.bbox[0] + r.bbox[2]) / 2
        for n in regions:
            if n is r:
                continue
            yov = min(r.bbox[3], n.bbox[3]) - max(r.bbox[1], n.bbox[1])
            if yov <= 0.25 * min(r.h, n.h):
                continue
            ncx = (n.bbox[0] + n.bbox[2]) / 2
            if n.bbox[2] <= rcx and ncx < rcx:                 # neighbour on the left
                x0 = max(x0, min(r.bbox[0], (n.bbox[2] + r.bbox[0]) / 2))
            elif n.bbox[0] >= rcx and ncx > rcx:               # neighbour on the right
                x1 = min(x1, max(r.bbox[2], (r.bbox[2] + n.bbox[0]) / 2))
        for rx0, ry0, rx1, ry1 in rules:
            if min(r.bbox[3], ry1) - max(r.bbox[1], ry0) < 0.3 * r.h:
                continue
            cx = (rx0 + rx1) / 2
            if x0 < cx <= r.bbox[0] + RULE_EDGE_MARGIN:
                x0 = max(x0, min(cx, r.bbox[0]))
            elif r.bbox[2] - RULE_EDGE_MARGIN <= cx < x1:
                x1 = min(x1, max(cx, r.bbox[2]))
        out[r.rid] = _clip_box([x0, r.bbox[1] - vy, x1, r.bbox[3] + vy], iw, ih)
    return out


@dataclass
class LineSet:
    lines: Dict[int, List[List[float]]]     # owned lines per region id
    found: Dict[int, bool]                  # detector found anything in the region
    allowed: Dict[int, List[float]]         # how far each region's lines and crops may reach
    med_h: float                            # median body line height (px)
    failed_crops: int
    extended: int                           # lines that reach beyond their region box


def collect_region_lines(regions: List[Region], imgs: PageImages, det: DetectionPredictor,
                         debug_stem: str = "", page_num: int = 0) -> LineSet:
    """
    PHASE A (detect) and PHASE B (resolve).

    A: Surya line detection on each text region's ALLOWED box (the region
       plus its own margin/gutter, see _allowed_boxes), batched — so a line
       Surya's region box cuts short is still detected at its full length.
    B: global ownership — a physical line belongs to exactly one region.
       * lines crossing a printed vertical rule are split there; lines of a
         region wider than GUTTER_SPLIT_MIN_COLS columns (Surya merged two
         columns) are also split at measured whitespace gutters.  Ordinary
         single-column regions are never gutter-split: in F-I/F-J that split
         tables of contents and leader dots ("Course in I mining") and left
         rule slivers at line starts ("I withdrew", "It year");
       * a line must lie ≥ LINE_MIN_IN_REGION inside its own region;
       * a line (mostly) inside a larger line — the same ink seen from a
         neighbouring crop, or a truncated copy — is dropped, so the
         full-length copy wins;
       * lines are clipped to the allowed box, not to the region box.
    """
    iw, ih = imgs.clean_gray.size
    texts = [r for r in regions if r.is_text]
    col_w = _median([r.w for r in texts if r.label == "Text" and r.w < 0.4 * iw], iw / 5)
    allowed = _allowed_boxes(regions, imgs.vrules, iw, ih, col_w)
    frames = [_clip_box(_grow(allowed[r.rid], DET_PAD), iw, ih) for r in texts]
    crops = [imgs.clean_gray.crop(tuple(int(v) for v in f)) for f in frames]
    detected, failed = detect_lines(crops, det)
    del crops
    free_device_memory()
    med_h = _median([b[3] - b[1] for lines in detected for b in lines], 40.0)

    wide = {r.rid for r in texts
            if r.label not in HEADER_LABELS and r.w > GUTTER_SPLIT_MIN_COLS * col_w}
    gmap: Optional[GutterMap] = None
    if GUTTERS_ENABLED and wide:
        try:
            gmap = GutterMap(imgs.clean_gray, med_h)
            if debug_stem:
                gmap.save_debug(imgs.clean_gray, _debug_path(debug_stem, page_num, "00_gutters", "jpg"))
            logger.info(f"  [GUTTER] {len(wide)} region(s) wider than {GUTTER_SPLIT_MIN_COLS:g} columns "
                        f"— their lines are split at whitespace gutters.")
        except MemoryError:
            logger.warning("  [GUTTER] gutter map skipped (memory).")

    cands = []            # (bbox, rid, in_region)
    found: Dict[int, bool] = {}
    for r, f, lines in zip(texts, frames, detected):
        found[r.rid] = bool(lines)
        g = gmap if r.rid in wide else None
        for lx0, ly0, lx1, ly1 in lines:
            b = [lx0 + f[0], ly0 + f[1], lx1 + f[0], ly1 + f[1]]
            for piece in split_at_separators(b, imgs.vrules, g):
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
            kept[dup_idx] = [b, rid, frac]          # same line; this region owns it better

    region_by_id = {r.rid: r for r in texts}
    per_region: Dict[int, List[List[float]]] = {r.rid: [] for r in texts}
    extended = 0
    for b, rid, _ in kept:
        ab = allowed[rid]
        clipped = [max(b[0], ab[0]), max(b[1], ab[1]), min(b[2], ab[2]), min(b[3], ab[3])]
        if clipped[2] - clipped[0] < MIN_LINE_W or clipped[3] - clipped[1] < MIN_LINE_H:
            continue
        rb = region_by_id[rid].bbox
        if clipped[0] < rb[0] - LINE_CLIP_PAD or clipped[2] > rb[2] + LINE_CLIP_PAD:
            extended += 1
        per_region[rid].append(clipped)
    return LineSet(per_region, found, allowed, med_h, failed, extended)


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
    """
    Split a very wide header crop into pieces of aspect ratio ≈ max_ar, cutting
    in blank inter-word gaps so no word is sliced (script F's fixed 20 %
    overlap re-read partial words: "Classes To Off offer", "Tailor . lor ?").
    Returns [(x0, x1, overlaps_previous)].
    """
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
    packed = tc.replace(" ", "")
    if len(packed) >= REPEAT_MIN_LEN and len(set(packed)) <= 2:
        return True                      # "000000 0000…", "X X X X", ". . . ." — leaders and rules
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
    bbox: List[float]           # page-space box of the line (or whole region)
    crop: Image.Image           # grayscale crop
    header: bool
    label: str
    whole_region: bool = False


def recognise_jobs(jobs: List[ReadJob], models: Models) -> List[Tuple[str, float]]:
    """Batched recognition with the single TrOCR model.  Header crops are split at word gaps."""
    flat: List[Image.Image] = []
    layout: List[List[Tuple[int, bool]]] = []
    for job in jobs:
        # TrOCR squeezes every crop into 384×384.  Very wide lines (captions,
        # two-column-wide type) are cut at word gaps first, as headers were.
        segs = _wide_segments(job.crop, MAX_HEADER_AR if job.header else MAX_BODY_AR)
        idxs = []
        for x0, x1, ov in segs:
            idxs.append((len(flat), ov))
            flat.append(_bordered(job.crop.crop((x0, 0, x1, job.crop.height))))
        layout.append(idxs)

    reads = models.reader.read(flat)
    out: List[Tuple[str, float]] = []
    for idxs in layout:
        text, confs = "", []
        for flat_i, ov in idxs:
            t, c = reads[flat_i]
            t = t.strip()
            if t:
                text = _merge_overlapping(text, t) if ov else (f"{text} {t}".strip())
                confs.append(c)
        out.append((text, sum(confs) / len(confs) if confs else 0.0))
    return out


def _build_jobs(regions: List[Region], ls: "LineSet", imgs: PageImages) -> Dict[int, List[ReadJob]]:
    iw, ih = imgs.clean_gray.size
    jobs: Dict[int, List[ReadJob]] = {}
    for r in regions:
        if not r.is_text:
            continue
        header = r.label in HEADER_LABELS
        src = imgs.norm_gray if header else imgs.clean_gray
        clip = _clip_box(_grow(r.bbox, 4), iw, ih)
        # Line crops (with their LINE_PAD_X margin) may reach anywhere in the
        # allowed box — F-J clipped them to the region box, which cut off the
        # first/last letters whenever Surya's box hugged the type.
        line_clip = ls.allowed.get(r.rid, clip)
        rl = sorted(ls.lines.get(r.rid, []), key=lambda b: (b[1], b[0]))
        rj = [ReadJob(r.rid, b, _prepare_crop(src, b, line_clip), header, r.label) for b in rl]
        # Whole-region read only when the detector saw nothing in it at all
        # (a single big headline with no inter-line whitespace).  Regions whose
        # lines were all claimed by other regions are duplicates — skip them.
        if not rl and not ls.found.get(r.rid, False) and (header or r.h <= WHOLE_CROP_MAX_LINES * ls.med_h):
            rj.append(ReadJob(r.rid, list(r.bbox),
                              ImageOps.autocontrast(src.crop(tuple(int(v) for v in clip)), cutoff=1),
                              header, r.label, whole_region=True))
        if rj:
            jobs[r.rid] = rj
    return jobs


def _probe_indices(rj: List[ReadJob], k: int, exclude: Set[int] = frozenset()) -> List[int]:
    """Up to k lines spread evenly through a region, preferring full-width lines
    (paragraph-end lines are short and say little about legibility)."""
    cand = [i for i in range(len(rj)) if i not in exclude]
    if len(cand) <= k:
        return cand
    wmax = max(rj[i].bbox[2] - rj[i].bbox[0] for i in cand)
    full = [i for i in cand if rj[i].bbox[2] - rj[i].bbox[0] >= 0.5 * wmax]
    pool = full if len(full) >= k else cand
    step = len(pool) / k
    return sorted({pool[min(len(pool) - 1, int(step * (j + 0.5)))] for j in range(k)})


def _finalise(job: ReadJob, text: str, conf: float, stats: Counter) -> Optional[dict]:
    """Noise filter + line-end cleanup.  Returns the element, or None."""
    bw, bh = job.bbox[2] - job.bbox[0], job.bbox[3] - job.bbox[1]
    if _is_noise(text, conf, bh, bw):
        stats["noise_rejected"] += 1
        logger.debug(f"      [NOISE] conf={conf:.2f} {bw:.0f}×{bh:.0f}px | {text[:40]}")
        return None
    ls, rs = _edge_strokes(job.crop)
    cleaned = clean_text(text, ls, rs)
    if not cleaned or _is_noise(cleaned, conf, bh, bw):
        stats["noise_rejected"] += 1
        return None
    if cleaned != text:
        stats["lines_cleaned"] += 1
    return {"text": cleaned, "bbox": list(job.bbox), "confidence": conf,
            "source_label": job.label, "region_id": job.rid}


def legibility(job: ReadJob, text: str, conf: float, lexicon: Lexicon, probe: bool,
               med_h: float = 0.0) -> Tuple[bool, str]:
    """
    Strict per-line test used only on low-confidence pages.
      probe=True : the (stricter) test a probe line must pass for its region to count as legible;
      probe=False: the test every line of a legible region must pass to be kept.
    Independent signs of a real reading: confidence, a plausible number of
    characters for the width of the line, and real words.
    """
    t = text.strip()
    if not t:
        return False, "empty"
    min_conf = PROBE_MIN_CONF if probe else KEEP_MIN_CONF
    min_words = PROBE_MIN_WORDS if probe else KEEP_MIN_WORDS
    noword_conf = NOWORD_MIN_CONF[0] if probe else NOWORD_MIN_CONF[1]
    if conf < min_conf:
        return False, f"conf {conf:.2f} < {min_conf:.2f}"
    if not job.whole_region:
        bw, bh = job.bbox[2] - job.bbox[0], max(1.0, job.bbox[3] - job.bbox[1])
        density = len(t) / max(1.0, bw / bh)
        large = med_h > 0 and bh >= LARGE_TYPE_H * med_h          # ad / headline type in a Text box
        lo = DENSITY_MIN_HEADER if (job.header or large) else DENSITY_MIN
        if density < lo:
            return False, f"too few characters for the line ({density:.2f} < {lo:.2f})"
        if density > DENSITY_MAX:
            return False, f"too many characters for the line ({density:.2f})"
    credit, n = lexicon.word_score(t)
    if n == 0:
        if conf < noword_conf:
            return False, f"no checkable word and conf {conf:.2f} < {noword_conf:.2f}"
        return True, "ok"
    if credit / n < min_words:
        return False, f"real words {credit:.1f}/{n}"
    return True, "ok"


@dataclass
class GateResult:
    low_page: bool = False
    probe_median: float = 1.0
    probe_weak_frac: float = 0.0
    n_probes: int = 0
    skipped_regions: List[Tuple[Region, str]] = field(default_factory=list)
    skipped_lines: List[Tuple[List[float], str, str]] = field(default_factory=list)   # (bbox, text, reason)
    lines_not_read: int = 0


def _is_low_page(probe_confs: List[float]) -> Tuple[bool, float, float]:
    if not probe_confs:
        return False, 1.0, 0.0
    med = statistics.median(probe_confs)
    weak = sum(1 for c in probe_confs if c < LOWPAGE_WEAK_CONF) / len(probe_confs)
    if LEGIBILITY_GATE == "always":
        return True, med, weak
    if LEGIBILITY_GATE != "auto" or len(probe_confs) < LOWPAGE_MIN_PROBES:
        return False, med, weak
    return (med < LOWPAGE_MEDIAN_CONF or weak > LOWPAGE_WEAK_FRAC), med, weak


def recognise_page(regions: List[Region], ls: "LineSet", imgs: PageImages, models: Models,
                   stats: Counter) -> Tuple[Dict[int, List[dict]], GateResult]:
    """
    Two-step recognition.
      1  Probes: up to PROBES_PER_REGION lines of every region are read
         first.  They decide whether the page is
         low-confidence, and are kept — nothing is ever read twice.
      2a Normal page: every other line is read; noise filter.
      2b Low-confidence page: each region is judged on its probes.  An
         illegible region's other lines are never read.  In a legible region
         every line must pass legibility(probe=False); if most fail, the
         region is dropped after all.
    """
    jobs = _build_jobs(regions, ls, imgs)
    med_h = ls.med_h
    reads: Dict[int, Dict[int, Tuple[str, float]]] = {rid: {} for rid in jobs}
    region_by_id = {r.rid: r for r in regions}

    def read(pairs: List[Tuple[int, int]]) -> None:
        if not pairs:
            return
        out = recognise_jobs([jobs[rid][i] for rid, i in pairs], models)
        for (rid, i), tc in zip(pairs, out):
            reads[rid][i] = tc
        stats["lines_read"] += len(pairs)

    # 1 ─ probes
    probes = {rid: _probe_indices(rj, PROBES_PER_REGION) for rid, rj in jobs.items()}
    read([(rid, i) for rid, idx in probes.items() for i in idx])
    body_confs = [reads[rid][i][1] for rid, idx in probes.items() for i in idx
                  if region_by_id[rid].label not in HEADER_LABELS]
    gate = GateResult()
    gate.low_page, gate.probe_median, gate.probe_weak_frac = _is_low_page(body_confs)
    gate.n_probes = len(body_confs)
    logger.info(f"  [LEGIBILITY] {gate.n_probes} body probe(s): median conf {gate.probe_median:.2f}, "
                f"{gate.probe_weak_frac * 100:.0f}% below {LOWPAGE_WEAK_CONF:.2f} → "
                + ("LOW-CONFIDENCE page: strict legibility gate ON." if gate.low_page
                   else "normal page (gate off)."))

    per_region: Dict[int, List[dict]] = {r.rid: [] for r in regions if r.is_text}

    # 2a ─ normal page
    if not gate.low_page:
        read([(rid, i) for rid, rj in jobs.items() for i in range(len(rj)) if i not in reads[rid]])
        for rid, rj in jobs.items():
            for i, job in enumerate(rj):
                el = _finalise(job, *reads[rid][i], stats)
                if el:
                    per_region[rid].append(el)
        return per_region, gate

    # 2b ─ low-confidence page: judge regions on their probes
    lex = models.lexicon

    def pass_rate(rid: int) -> Tuple[float, int]:
        rj, got = jobs[rid], reads[rid]
        ok = sum(1 for i, (t, c) in got.items() if legibility(rj[i], t, c, lex, True, med_h)[0])
        return ok / max(1, len(got)), ok

    verdict: Dict[int, bool] = {}
    borderline = []
    for rid in jobs:
        rate, ok = pass_rate(rid)
        if rate >= REGION_MIN_PASS:
            verdict[rid] = True
        elif ok >= 1 and len(reads[rid]) < len(jobs[rid]):
            borderline.append(rid)
        else:
            verdict[rid] = False
    extra = {rid: _probe_indices(jobs[rid], PROBES_EXTRA, set(reads[rid])) for rid in borderline}
    read([(rid, i) for rid, idx in extra.items() for i in idx])
    for rid in borderline:
        verdict[rid] = pass_rate(rid)[0] >= REGION_MIN_PASS_EXTRA

    # read the rest of the legible regions only
    read([(rid, i) for rid, rj in jobs.items() if verdict[rid]
          for i in range(len(rj)) if i not in reads[rid]])

    for rid, rj in jobs.items():
        r = region_by_id[rid]
        if not verdict[rid]:
            n_unread = len(rj) - len(reads[rid])
            gate.lines_not_read += n_unread
            probe_txt = " | ".join(f"{reads[rid][i][0][:30]} ({reads[rid][i][1]:.2f})" for i in sorted(reads[rid]))
            gate.skipped_regions.append((r, f"{len(reads[rid])} probe(s) failed; {n_unread} line(s) not read"
                                            f" — probes: {probe_txt}"))
            continue
        kept, failed = [], []
        for i, job in enumerate(rj):
            text, conf = reads[rid][i]
            el = _finalise(job, text, conf, stats)
            if el is None:
                continue
            ok, why = legibility(job, el["text"], conf, lex, False, med_h)
            (kept if ok else failed).append((el, why))
        total = len(kept) + len(failed)
        if total and len(kept) / total < REGION_MIN_KEEP:
            gate.skipped_regions.append((r, f"only {len(kept)}/{total} line(s) legible after full read"))
            for el, why in kept + failed:
                gate.skipped_lines.append((el["bbox"], el["text"], why if why != "ok" else "region dropped"))
            continue
        per_region[rid] = [el for el, _ in kept]
        for el, why in failed:
            gate.skipped_lines.append((el["bbox"], el["text"], why))
    stats["regions_illegible"] = len(gate.skipped_regions)
    stats["lines_illegible"] = len(gate.skipped_lines)
    stats["lines_not_read"] = gate.lines_not_read
    logger.info(f"  [LEGIBILITY] kept {sum(1 for v in verdict.values() if v)}/{len(verdict)} region(s); "
                f"left out {len(gate.skipped_regions)} region(s) and {len(gate.skipped_lines)} further line(s); "
                f"{gate.lines_not_read} line(s) in illegible regions were never read.")
    return per_region, gate


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
    """
    Newspaper reading order.  At each level:
      1. split into columns wherever a clean vertical gutter crosses the whole group;
      2. otherwise cut horizontally just above/below elements that span columns
         (banner headlines, wide photos, mastheads) — never between aligned
         paragraphs, which is what makes naive XY-cut read across columns;
      3. otherwise any clean horizontal gap;
      4. leaf: Surya's position if all items have one, else top-to-bottom.
    """
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


# ── Column model ──────────────────────────────────────────────────────────────

@dataclass
class ColumnModel:
    centers: List[float]
    bounds: List[float]      # len(centers) - 1 boundaries between adjacent columns
    width: float
    head_ids: set = field(default_factory=set)   # masthead-zone region ids (read first by design)

    @property
    def n(self) -> int:
        return len(self.centers)

    def span(self, b: Sequence[float]) -> Tuple[int, int]:
        """Column range [c0, c1] a box occupies: its home column (by centre) plus
        any neighbour it covers substantially.  Small bleed across a gutter, or
        a column edge drifting with skew, never adds a column."""
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
    """
    Find the page's text columns by clustering the horizontal centres of
    column-width text regions.  Centres of one column stay within a few dozen
    pixels of each other even when the scan is skewed or warped, while
    neighbouring columns are a full column width apart — so this is robust
    where pixel gutters are not.
    """
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
    for c in centers:                       # columns can't be closer than ~0.6 column widths
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
                    first = a if B[a, 0] < B[b, 0] else b                   # same row
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
    (after Breuel, "High Performance Document Layout Analysis", 2003):

      R1  Two blocks sharing a column: the upper one is read first
          (blocks side by side in one row: the left one first).
      R2  Block A lies wholly in columns left of block B: A is read first,
          UNLESS some block C sits vertically between them and spans columns
          of both — a banner or multi-column headline that B is above and A
          below.  Then R1 through C decides.

    The blocks are then topologically sorted, preferring the leftmost column
    and the highest block whenever several are free to go next.

    Unlike cutting the page into rectangles, a spanning element only affects
    the columns it actually covers.  (The rectangle cut used before sliced
    column 6 at the bottom of a photo spanning columns 4–5, because
    "1960 Predicted…" tied columns 5–6 into the same group — so column 6 was
    read between the photo and its caption.)
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
        if not heap:                     # cycle (skewed geometry): release the best remaining block
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

    # Masthead zone: everything above the first body paragraph that sits in a
    # band anchored by an element spanning at least half the columns.
    body_tops = [r.bbox[1] for r in regions if r.label == "Text" and r.h >= BODY_MIN_H_FRAC * page_h]
    head: List[Region] = []
    if body_tops:
        y_body = min(body_tops)
        tol = HBREAK_TOL_FRAC * page_h
        wide = [r for r in regions if r.bbox[3] <= y_body + tol and
                (lambda s: s[1] - s[0] + 1)(cols.span(r.bbox)) >= max(2, cols.n / 2)]
        if wide:
            y_band = max(r.bbox[3] for r in wide)
            # A single-column block just under the banner (a story's first
            # subhead) belongs to its column, not to the masthead zone; only
            # multi-column strips (datelines, ear boxes) get the extra tolerance.
            head = [r for r in regions if r.bbox[3] <= y_body + tol and
                    (r.bbox[3] <= y_band + tol or
                     (r.bbox[3] <= y_band + 3 * tol and
                      (lambda s: s[1] - s[0] + 1)(cols.span(r.bbox)) >= 2))]
    rest = [r for r in regions if r not in head]
    cols.head_ids = {r.rid for r in head}
    ordered = (precedence_order(head, bb, cols, stats) if head else []) + \
              precedence_order(rest, bb, cols, stats)
    logger.info(f"  [ORDER] {cols.n} column(s) at x≈" + ", ".join(f"{c:.0f}" for c in cols.centers) +
                f"; masthead zone {len(head)} region(s).")
    return ordered, cols


def order_violations(ordered: List[Region], cols: Optional[ColumnModel]) -> int:
    """
    Number of reading-order rules (R1 same column → upper first; R2 left
    column first unless a spanning block intervenes) that the final order
    breaks.  0 for a clean page.  Non-zero means the page was read across
    columns somewhere (a cycle had to be broken, or a non-"columns" order
    mode was used) and is flagged READ_ACROSS_COLUMNS.
    """
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
    """
    Kendall-style concordance between our order and Surya's reading order
    (1.0 = identical).  Surya's positions are only comparable within one
    layout tile, so only pairs of regions from the same tile are counted.
    """
    conc = disc = 0
    by_tile: Dict[int, List[float]] = {}
    for r in ordered:
        if r.is_text and r.source == "layout" and r.position is not None:
            by_tile.setdefault(r.tile, []).append(r.position)
    for pos in by_tile.values():
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


def _text_area(regions: List[Region], w: int, h: int) -> List[float]:
    """Extent of the page's text regions (+ pad).  Ink outside it — facing-page
    slivers, scan borders, the binding shadow — is not page content."""
    tb = [r.bbox for r in regions if r.is_text]
    if not tb:
        return [0.0, 0.0, float(w), float(h)]
    p = AUDIT_TEXT_AREA_PAD
    return _clip_box([min(b[0] for b in tb) - p, min(b[1] for b in tb) - p,
                      max(b[2] for b in tb) + p, max(b[3] for b in tb) + p], w, h)


def audit_coverage(page_gray: Image.Image, elements: List[dict], regions: List[Region],
                   stem: str, page_num: int, skipped: Sequence[Sequence[float]] = ()
                   ) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Share of the text area's ink that no OCR'd line, picture or deliberately
    skipped (illegible) box accounts for.  Red on latest_03_uncovered.jpg =
    missed ink; orange = left out by the legibility gate; grey = outside the
    text area (not counted).
    """
    s = AUDIT_SCALE
    small = page_gray.reduce(s)
    sw, sh = small.size
    ink = small.point(lambda v: 255 if v < AUDIT_INK_LEVEL else 0).filter(ImageFilter.MaxFilter(3))
    ta = _text_area(regions, page_gray.width, page_gray.height)
    area = Image.new("L", (sw, sh), 0)
    ImageDraw.Draw(area).rectangle([ta[0] / s, ta[1] / s, ta[2] / s, ta[3] / s], fill=255)
    ink = ImageChops.multiply(ink, area)
    covered = Image.new("L", (sw, sh), 0)
    cdraw = ImageDraw.Draw(covered)
    pad = 6
    boxes = [e["bbox"] for e in elements] + [r.bbox for r in regions if not r.is_text]
    for x0, y0, x1, y1 in boxes:
        cdraw.rectangle([(x0 - pad) / s, (y0 - pad) / s, (x1 + pad) / s, (y1 + pad) / s], fill=255)
    skip = Image.new("L", (sw, sh), 0)
    sdraw = ImageDraw.Draw(skip)
    for x0, y0, x1, y1 in skipped:
        sdraw.rectangle([(x0 - pad) / s, (y0 - pad) / s, (x1 + pad) / s, (y1 + pad) / s], fill=255)
    skipped_ink = ImageChops.subtract(ImageChops.multiply(ink, skip), covered)
    uncovered = ImageChops.subtract(ImageChops.subtract(ink, covered), skip)
    total, miss = ink.histogram()[255], uncovered.histogram()[255]
    frac = miss / total if total else 0.0
    skipped_frac = skipped_ink.histogram()[255] / total if total else 0.0
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
    if skipped:
        msg += f" ({skipped_frac * 100:.1f}% more left out as illegible)"
    if bands:
        msg += "; mostly-uncovered band(s) at x=" + ", ".join(f"{a}–{b}px" for a, b in bands)
    (logger.warning if frac > QA_MAX_UNCOVERED else logger.info)(msg)
    vis = Image.merge("RGB", (small, small, small))
    vis.paste(Image.new("RGB", (sw, sh), (200, 200, 200)), mask=ImageChops.invert(area).point(lambda v: 110 if v else 0))
    vis.paste(Image.new("RGB", (sw, sh), ILLEGIBLE_RGB), mask=skipped_ink)
    vis.paste(Image.new("RGB", (sw, sh), (255, 0, 0)), mask=uncovered)
    vis.save(str(_debug_path(stem, page_num, "03_uncovered", "jpg")), "JPEG", quality=85)
    return frac, bands


def save_illegible_report(gate: "GateResult", path: Path, filename: str, page_num: int) -> None:
    out = [f"FILE: {filename}   PAGE: {page_num}",
           f"Low-confidence page: {'YES' if gate.low_page else 'no'}   body probes {gate.n_probes}, "
           f"median conf {gate.probe_median:.2f}, {gate.probe_weak_frac * 100:.0f}% below {LOWPAGE_WEAK_CONF:.2f}",
           "=" * 80]
    if not gate.low_page:
        out.append("Gate off — nothing was left out for legibility.")
    else:
        out.append(f"REGIONS LEFT OUT ({len(gate.skipped_regions)}):")
        for r, why in gate.skipped_regions:
            x0, y0, x1, y1 = r.bbox
            out.append(f"  r{r.rid:<4} {r.label:<15} ({x0:.0f},{y0:.0f}→{x1:.0f},{y1:.0f})  {why}")
        out.append("")
        out.append(f"LINES LEFT OUT IN LEGIBLE REGIONS ({len(gate.skipped_lines)}):")
        for (x0, y0, x1, y1), text, why in gate.skipped_lines:
            out.append(f"  ({x0:.0f},{y0:.0f}→{x1:.0f},{y1:.0f})  [{why}]  {text}")
    path.write_text("\n".join(out), encoding="utf-8")


def save_layout_debug(image: Image.Image, regions: List[Region], path: Path,
                      tiles: Sequence[Sequence[int]] = ()) -> None:
    """Regions labelled [tile:position]; thick outline = stitched across tiles; dashed grey = layout tiles."""
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    f = _pil_font(15)
    for k, (tx0, ty0, tx1, ty1) in enumerate(tiles if len(tiles) > 1 else ()):
        for x in range(tx0, tx1, 40):
            draw.line([(x, ty0), (min(x + 20, tx1), ty0)], fill=(90, 90, 90, 150), width=3)
            draw.line([(x, ty1 - 2), (min(x + 20, tx1), ty1 - 2)], fill=(90, 90, 90, 150), width=3)
        for y in range(ty0, ty1, 40):
            draw.line([(tx0, y), (tx0, min(y + 20, ty1))], fill=(90, 90, 90, 150), width=3)
            draw.line([(tx1 - 2, y), (tx1 - 2, min(y + 20, ty1))], fill=(90, 90, 90, 150), width=3)
        draw.text((tx0 + 8, ty0 + 6), f"tile {k}", fill=(90, 90, 90, 220), font=_pil_font(28))
    for r in regions:
        x0, y0, x1, y1 = r.bbox
        rgb = _hex_rgb(_label_hex(r.label))
        pos = "·" if r.position is None else f"{r.position:g}"
        tag = f"[{r.tile}:{pos}] {r.label}{'' if r.source == 'layout' else ' (' + r.source + ')'}"
        draw.rectangle([x0, y0, x1, y1], outline=rgb + (210,), fill=rgb + (22,),
                       width=4 if r.source != "layout" else 2)
        draw.rectangle([x0, y0, x0 + len(tag) * 9 + 6, y0 + 20], fill=rgb + (175,))
        draw.text((x0 + 3, y0 + 2), tag, fill=(255, 255, 255, 255), font=f)
    img.save(str(path), "JPEG", quality=85)


def save_order_debug(image: Image.Image, ordered: List[Region], rules: List[List[float]], path: Path,
                     cols: Optional["ColumnModel"] = None) -> None:
    """Numbered regions joined by the reading path — a path zig-zagging across columns is an ordering error.
    Dashed purple lines are the detected column boundaries."""
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


def save_ocr_debug(image: Image.Image, elements: List[dict], path: Path) -> None:
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    f = _pil_font(12)
    for e in elements:
        x0, y0, x1, y1 = e["bbox"]
        rgb = _hex_rgb(_label_hex(e.get("source_label", "Text")))
        draw.rectangle([x0, y0, x1, y1], outline=rgb + (180,), width=1)
        draw.text((x0 + 1, y0), e["text"][:55], fill=rgb + (220,), font=f)
    img.save(str(path), "JPEG", quality=85)


def save_layout_report(regions: List[Region], path: Path, filename: str, page_num: int) -> None:
    lines = [f"FILE: {filename}   PAGE: {page_num}", f"Regions: {len(regions)}", "=" * 92,
             f"{'ID':>4} {'TILE':>4} {'POS':>7}  {'LABEL':<20} {'SOURCE':<9} {'CONF':>5} "
             f"{'X0':>6} {'Y0':>6} {'X1':>6} {'Y1':>6}", "-" * 92]
    for r in regions:
        pos = "—" if r.position is None else f"{r.position:g}"
        x0, y0, x1, y1 = r.bbox
        lines.append(f"{r.rid:>4} {r.tile:>4} {pos:>7}  {r.label:<20} {r.source:<9} {r.confidence:>5.2f} "
                     f"{x0:>6.0f} {y0:>6.0f} {x1:>6.0f} {y1:>6.0f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_ocr_report(elements: List[dict], path: Path, filename: str, page_num: int) -> None:
    lines = [f"FILE: {filename}   PAGE: {page_num}", f"OCR elements: {len(elements)}", "=" * 80]
    for e in elements:
        x0, y0, x1, y1 = e["bbox"]
        lines.append(f"[{e['reading_position']:>4}] r{e['region_id']:<4} {e['source_label']:<15} "
                     f"conf={e['confidence']:.2f} ({x0:.0f},{y0:.0f}→{x1:.0f},{y1:.0f}) | {e['text']}")
    path.write_text("\n".join(lines), encoding="utf-8")


QA_FIELDS = ["file", "page", "tool_version", "run_id", "source_sha256", "layout_tiles", "layout_tiles_failed", "layout_raw_boxes",
             "layout_dropped_dups", "layout_stitched", "regions", "regions_split_at_rules",
             "detect_failed_crops", "lines_extended", "elements", "mean_conf", "low_conf_frac",
             "low_confidence_page", "probe_median_conf", "regions_illegible", "lines_illegible",
             "lines_not_read", "lines_read", "uncovered_ink", "order_agreement", "columns",
             "order_violations", "order_cycles", "noise_rejected",
             "t_layout", "t_detect", "t_recognise", "seconds", "flags"]


def init_qa_report() -> None:
    with open(QA_REPORT_PATH, "w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=QA_FIELDS).writeheader()


# Run/document context merged into every QA row (set in process_pdf).
_QA_CONTEXT: Dict[str, str] = {}


def append_qa_row(row: dict) -> None:
    ctx = {"tool_version": f"{APP_NAME} {APP_VERSION} ({APP_SCRIPT})", "run_id": RUN_ID, **_QA_CONTEXT}
    row = {**ctx, **{k: v for k, v in row.items() if not (k in ctx and v in ("", None))}}
    with open(QA_REPORT_PATH, "a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=QA_FIELDS).writerow(row)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def _layout_failed_row(filename: str, page_num: int, stats: Counter, t0: float) -> None:
    row = {k: "" for k in QA_FIELDS}
    row.update({"file": filename, "page": page_num, "layout_tiles": stats["layout_tiles"],
                "layout_tiles_failed": stats["layout_tiles_failed"], "elements": 0,
                "seconds": f"{time.time() - t0:.1f}", "flags": "LAYOUT_FAILED"})
    append_qa_row(row)


def process_page(pil_image: Image.Image, page_num: int, filename: str, stem: str,
                 models: Models, prov: Optional["DocProvenance"] = None) -> List[dict]:
    """
    1  Prepare    normalised page for layout; flattened + rule-free page for detection/OCR
    2  Layout     Surya on overlapping tiles at ≈ LAYOUT_TARGET_DPI; tiles merged into one
                  page layout; same-label NMS; split at printed rules; trim gutter bleed
    3  Detect     lines per region (batched)                     ┐ nothing is OCR'd
    4  Resolve    each physical line gets exactly one owner      ┘ until here
    5  Recognise  one TrOCR model: probes → page legibility → rest
    6  Order      column model + precedence: left column first unless a spanning block intervenes
    7  QA         coverage audit (diagnostic only), Surya agreement, CSV row, debug images

    Every OCR'd line belongs to a region Surya produced.  If layout fails on
    any tile, the page is flagged LAYOUT_FAILED and left alone (its original
    text layer is kept) instead of being read as one page-sized region.
    """
    t0 = time.time()
    stats: Counter = Counter()
    tm: Dict[str, float] = {}
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

    # 2 ─ Layout (tiled)
    t = time.time()
    regions, tiles = layout_page(imgs, models.layout, stats)
    free_device_memory()
    tm["layout"] = time.time() - t
    if regions is None:
        logger.error(f"  [QA] page {page_num} flagged: LAYOUT_FAILED — page not OCR'd; "
                     f"its existing text layer (if any) is kept.")
        _layout_failed_row(filename, page_num, stats, t0)
        if prov is not None:
            prov.flags.append(f"p{page_num}:LAYOUT_FAILED")
        return []
    before = len(regions)
    regions = _nms_same_label(regions)
    if len(regions) < before:
        logger.debug(f"  [LAYOUT] NMS removed {before - len(regions)} duplicate region(s).")
    next_id = [max((r.rid for r in regions), default=-1) + 1]
    n0 = len(regions)
    regions = _split_regions_at_rules(regions, imgs.vrules, next_id)
    stats["regions_split"] = len(regions) - n0
    _normalise_gutters(regions)
    counts = Counter(r.label for r in regions)
    logger.info(f"  [LAYOUT] {len(regions)} regions: " +
                "  ".join(f"{k}×{v}" for k, v in sorted(counts.items())))
    save_layout_debug(pil_image, regions, _debug_path(stem, page_num, "01_layout", "jpg"), tiles)
    save_layout_report(regions, _debug_path(stem, page_num, "01_layout_report", "txt"), filename, page_num)

    # 3 + 4 ─ Detect and resolve
    t = time.time()
    ls = collect_region_lines(regions, imgs, models.det, stem, page_num)
    med_h = ls.med_h
    stats["detect_failed"] = ls.failed_crops
    stats["lines_extended"] = ls.extended
    logger.info(f"  [DETECT] {sum(len(v) for v in ls.lines.values())} owned line(s) in "
                f"{sum(1 for r in regions if r.is_text)} text region(s); median line height {med_h:.0f}px; "
                f"{ls.extended} line(s) reach past Surya's region box"
                + (f"; detection FAILED on {ls.failed_crops} region crop(s)" if ls.failed_crops else ""))
    tm["detect"] = time.time() - t

    # 5 ─ Recognise (probes → legibility → rest)
    t = time.time()
    per_region, gate = recognise_page(regions, ls, imgs, models, stats)
    free_device_memory()
    tm["recognise"] = time.time() - t
    save_illegible_report(gate, _debug_path(stem, page_num, "02_illegible_report", "txt"), filename, page_num)

    # 6 ─ Order
    elements, ordered, cols = order_page(regions, per_region, iw, ih, stats)
    elements, n_dup = dedupe_safety_net(elements)
    if n_dup:
        logger.info(f"  [DEDUPE] safety net removed {n_dup} element(s).")
    agree = order_agreement(ordered)
    regress = order_violations(ordered, cols)
    logger.info(f"  [ORDER] {len(elements)} element(s); {regress} reading-order rule violation(s), "
                f"{stats['order_cycles']} cycle(s) broken; agreement with Surya order (within tiles) = {agree:.2f}")

    # 7 ─ QA
    skipped_boxes = [r.bbox for r, _ in gate.skipped_regions] + [b for b, _, _ in gate.skipped_lines]
    frac, bands = (0.0, [])
    if AUDIT_ENABLED:
        try:
            frac, bands = audit_coverage(imgs.norm_gray, elements, regions, stem, page_num, skipped_boxes)
        except Exception as exc:
            logger.warning(f"  [COVERAGE] audit failed: {exc}")
    save_order_debug(pil_image, ordered, imgs.vrules, _debug_path(stem, page_num, "02_order", "jpg"), cols)
    if elements:
        save_ocr_debug(pil_image, elements, _debug_path(stem, page_num, "02_ocr_overlay", "jpg"))
        save_ocr_report(elements, _debug_path(stem, page_num, "02_ocr_report", "txt"), filename, page_num)

    confs = [e["confidence"] for e in elements]
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    low = sum(1 for c in confs if c < 0.6) / len(confs) if confs else 1.0
    flags = []
    if agree < QA_MIN_ORDER_AGREEMENT:
        flags.append("ORDER_DISAGREES_WITH_SURYA")
    if regress > 0:
        flags.append("READ_ACROSS_COLUMNS")
    if frac > QA_MAX_UNCOVERED:
        flags.append("UNCOVERED_INK")
    if mean_conf < QA_MIN_MEAN_CONF:
        flags.append("LOW_CONFIDENCE")
    if gate.low_page:
        flags.append("LOW_LEGIBILITY_PAGE")
    if ls.failed_crops:
        flags.append("DETECTION_FAILED")
    if flags:
        logger.warning(f"  [QA] page {page_num} flagged for review: {', '.join(flags)}")
        if prov is not None:
            prov.flags.extend(f"p{page_num}:{f}" for f in flags)
    logger.info("  [TIME] " + "  ".join(f"{k} {v:.0f}s" for k, v in tm.items()) +
                f"  (lines read {stats['lines_read']})")
    append_qa_row({
        "file": filename, "page": page_num,
        "layout_tiles": stats["layout_tiles"], "layout_tiles_failed": stats["layout_tiles_failed"],
        "layout_raw_boxes": stats["layout_raw_boxes"], "layout_dropped_dups": stats["layout_dropped_dups"],
        "layout_stitched": stats["layout_stitched"], "regions": len(regions),
        "regions_split_at_rules": stats["regions_split"], "detect_failed_crops": stats["detect_failed"],
        "lines_extended": stats["lines_extended"],
        "elements": len(elements), "mean_conf": f"{mean_conf:.3f}", "low_conf_frac": f"{low:.3f}",
        "low_confidence_page": int(gate.low_page), "probe_median_conf": f"{gate.probe_median:.3f}",
        "regions_illegible": stats["regions_illegible"], "lines_illegible": stats["lines_illegible"],
        "lines_not_read": stats["lines_not_read"], "lines_read": stats["lines_read"],
        "uncovered_ink": f"{frac:.3f}", "order_agreement": f"{agree:.3f}",
        "columns": cols.n if cols else 1, "order_violations": regress,
        "order_cycles": stats["order_cycles"], "noise_rejected": stats["noise_rejected"],
        **{f"t_{k}": f"{v:.1f}" for k, v in tm.items()},
        "seconds": f"{time.time() - t0:.1f}", "flags": ";".join(flags),
    })
    logger.info(f"  [RESULT] page {page_num}: {len(elements)} element(s), mean conf {mean_conf:.2f}, "
                f"{stats['lines_illegible'] + stats['lines_not_read']} illegible line(s) left out, "
                f"{time.time() - t0:.0f}s")
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
    """
    Each line is written at a font size matching its printed height and then
    horizontally scaled (morph) to exactly span its printed width.  Viewers
    therefore see consistent line heights and real inter-word geometry, so
    they group words into lines correctly and never run a line into the
    neighbouring column.  (Script F shrank the font to as little as 1.5 pt,
    which some viewers join to the next line without a space: "twozames".)
    """
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
        prov = DocProvenance(filename, _sha256(input_path))
        _QA_CONTEXT["source_sha256"] = prov.source_sha256
        logger.info(f"  Source SHA-256 {prov.source_sha256}")
        with fitz.open(input_path) as doc:
            prov.pages = len(doc)
            logger.info(f"  {len(doc)} page(s) — rendering at {DPI} DPI for OCR …")
            font, using_freesans = load_text_font()
            for idx in range(len(doc)):
                page_num, page = idx + 1, doc[idx]
                pil_img = page_to_pil(page, dpi=DPI)
                img_size = pil_img.size
                try:
                    elements = process_page(pil_img, page_num, filename, stem, models, prov)
                except Exception as exc:
                    logger.error(f"  OCR failed for page {page_num}: {exc}")
                    prov.flags.append(f"p{page_num}:OCR_FAILED")
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
                prov.pages_ocrd += 1
                prov.confidences.extend(e["confidence"] for e in elements)
            apply_document_metadata(doc, filename, prov)
            logger.info(f"  Metadata: {APP_CREATOR}, run {RUN_ID}, {prov.pages_ocrd}/{prov.pages} page(s) OCR'd, "
                        f"mean conf {prov.mean_conf:.3f}")
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
    logger.info("║  OPTICOLUMNS 2026  –  script F-K  (tile → detect → read)     ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")
    logger.info(f"  {APP_NAME} {APP_VERSION} ({APP_SCRIPT}) by {APP_AUTHOR}, {APP_INSTITUTION} — "
                f"{APP_LICENSE} licence — {APP_URL}")
    logger.info(f"  Run {RUN_ID}")
    logger.info(f"  Input {INPUT_DIR}  Output {OUTPUT_DIR}  DPI {DPI}  "
                f"TrOCR {TROCR_MODEL_NAME}, beams {NUM_BEAMS}  layout tiles ≤{layout_tile_size()}px "
                f"(≈{LAYOUT_TARGET_DPI} DPI)  "
                f"order {READING_ORDER_MODE}  legibility gate {LEGIBILITY_GATE}")
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