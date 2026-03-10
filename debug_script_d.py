#!/usr/bin/env python3
"""
Opticolumns
======================================================================
Revised: added auxiliary_layout_pass() after initial Surya layout
detection to recover columns that the model misses on historic
multi-column newsprint.
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

import numpy as np
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
CONFIDENCE_THRESHOLD             = 0.25
SINGLE_CHAR_CONFIDENCE_THRESHOLD = 0.50
MIN_LINE_H                       = 8
MIN_LINE_W                       = 15

# ── Layout label taxonomy ─────────────────────────────────────────────────────
OCR_LABELS = {
    "Text",
    "Section-header",
    "Caption",
    "Footnote",
    "List-item",
    "Page-footer",
    "Page-header",
    "Table-of-contents",
    "Handwriting",
    "Text-inline-math",
    "Formula",
}
SKIP_LABELS = {
    "Picture",
    "Figure",
    "Table",
    "Form",
}

SINGLE_BLOCK_LABELS = {
    "Section-header",
    "Page-header",
    "Caption",
    "Footnote",
}

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
DEFAULT_COLOUR   = "#9E9E9E"
AUXILIARY_COLOUR = "#E65C00"   # vivid amber — distinguishes auxiliary regions


# ══════════════════════════════════════════════════════════════════════════════
# AUXILIARY LAYOUT PASS — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Ink-detection threshold expressed as a fraction of (page_mean − sigma).
# Lower → more sensitive to faint/faded type; raise if backgrounds are noisy.
AUX_INK_ADAPTIVE_FACTOR = 0.5

# Minimum column width as a fraction of page width.
# At 300 DPI a typical historic newspaper column is ~5–10 % of page width.
AUX_MIN_COL_WIDTH_FRAC = 0.025

# Minimum column height as a fraction of page height for a block to be kept.
AUX_MIN_COL_HEIGHT_FRAC = 0.03

# Ink-density floor: fraction of column height × column width that must
# contain dark pixels for a band to be considered a text column at all.
# Applied as: (total ink in band) / (band_w × page_h) > threshold.
AUX_INK_DENSITY_THRESHOLD = 0.003   # 0.3 % of pixels must be dark

# Horizontal smoothing kernel expressed as fraction of page width.
# MUST be narrow enough to preserve inter-column gutters.
# For a 10-column broadsheet at 300 DPI (~8500px wide), gutters are ~60–100px.
# 1/400 ≈ 21px bridges intra-character gaps without bridging column gutters.
# (The previous value of 1/100 was ~85px — wide enough to merge all columns
# into one continuous band, causing a single whole-page auxiliary region.)
AUX_HSMOOTH_FRAC = 1 / 400

# Narrow gutter merge: column bands separated by a gap smaller than this
# (in px) are treated as one band.  Handles micro-fragments from decorative
# rules or noise without bridging genuine column gutters.
AUX_MIN_GUTTER_FRAC = 1 / 500      # ~17px at 8500px wide

# Maximum Surya region count whose x-centre falls inside a detected column
# band before that band is considered "adequately covered" and skipped.
# Analysis of the 1888–1930 batch shows covered columns have 8–37 region
# centres; the completely missed column at x≈6800–7200 has exactly 0.
# A threshold of 3 catches fully missed AND sparsely covered columns while
# leaving well-detected columns alone.
AUX_MAX_CENTRE_COUNT = 3

# Vertical smoothing kernel expressed as fraction of page height.
# Bridges inter-line whitespace within a single text block.
AUX_VSMOOTH_FRAC = 1 / 150

# Maximum gap (px) between two vertical text runs before they are split into
# separate regions.  Keep large so whole columns merge into one region — the
# OCR step's DetectionPredictor will find individual lines within each region.
AUX_LINE_GAP_FRAC = 1 / 60

# Page-edge exclusion fraction.  The outermost this fraction of page width
# on each side is excluded from auxiliary column search to avoid picking up
# decorative borders, torn edges, or scan artefacts on aged newsprint.
AUX_MARGIN_FRAC = 0.01


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


def _label_hex(label: str, auxiliary: bool = False) -> str:
    if auxiliary:
        return AUXILIARY_COLOUR
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
    Surya  — LayoutPredictor + DetectionPredictor
    TrOCR  — TrOCRProcessor + VisionEncoderDecoderModel
    """
    logger.info("=" * 62)
    logger.info("  LOADING MODELS  (Surya Layout + TrOCR Recognition)")
    logger.info("=" * 62)

    _setup_pdf_resources()

    logger.info("  DetectionPredictor (line segmentation) …")
    det_predictor = DetectionPredictor()

    logger.info(f"  FoundationPredictor (layout: {settings.LAYOUT_MODEL_CHECKPOINT}) …")
    foundation_lay   = FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    logger.info("  LayoutPredictor …")
    layout_predictor = LayoutPredictor(foundation_lay)

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
            "bbox":      bbox,
            "polygon":   polygon,
            "label":     label,
            "position":  position,
            "top_k":     top_k,
            "auxiliary": False,
        })

    return regions, image_bbox


# ══════════════════════════════════════════════════════════════════════════════
# AUXILIARY LAYOUT PASS
# ══════════════════════════════════════════════════════════════════════════════

def auxiliary_layout_pass(
    page_image: Image.Image,
    existing_regions: List[Dict],
    last_position: int,
) -> List[Dict]:
    """
    Recover text columns that Surya's LayoutPredictor missed.

    Historic multi-column newspapers (8–10 columns, aged newsprint) routinely
    cause Surya to miss 2–4 full columns per page because the model was trained
    predominantly on contemporary single- or double-column documents.

    Root-cause analysis of the v1 auxiliary pass (which returned a single whole-
    page region) identified two failure modes that are corrected here:

    Failure 1 — Smoothing kernel too wide
        The previous kernel (AUX_HSMOOTH_FRAC = 1/100, ~85 px at 8500 px page
        width) was wide enough to bridge genuine inter-column gutters (~60–100 px
        on 300 DPI broadsheets), merging ALL columns into one continuous band.
        Fix: use 1/400 (~21 px), which bridges intra-character gaps within a
        column without crossing column gutters.

    Failure 2 — Coverage mask obscures missed columns
        Surya's bboxes for the columns flanking a missed column typically extend
        100–200 px into that column's x-range.  The previous code built a 2-D
        pixel coverage mask and projected only "uncovered" ink, so the missed
        column's ink was invisible — its coverage appeared 100 % because the
        adjacent regions' bboxes overlapped it from both sides.
        Fix: project the FULL page ink to detect all column positions, then use
        Surya region-centre DENSITY (not bbox coverage) to decide whether each
        detected column has already been adequately identified.  A column whose
        x-range contains ≤ AUX_MAX_CENTRE_COUNT Surya region centres is treated
        as missed — this correctly flags the 1888 column at x ≈ 6 800–7 200
        (0 centres) while leaving columns with 8–37 centres untouched.

    Algorithm
    ─────────
    1.  Binarise with adaptive ink threshold tuned for aged newsprint.
    2.  Compute horizontal ink projection over the FULL page (sum dark pixels
        per column of pixels, axis=0), smoothed with a narrow kernel.
    3.  Exclude a thin page-edge margin (AUX_MARGIN_FRAC) to ignore torn
        borders and scan artefacts.
    4.  Collect contiguous x-runs above the ink floor as candidate column bands.
    5.  Merge bands separated by a very narrow gap (< AUX_MIN_GUTTER_FRAC × iw)
        to suppress micro-fragments from decorative rules.
    6.  For each band wider than AUX_MIN_COL_WIDTH_FRAC × iw:
        a. Count Surya region centres (x midpoints) that fall within the band.
        b. If count > AUX_MAX_CENTRE_COUNT → Surya covered it; skip.
        c. Verify minimum ink density (avoids spurious hits in margins).
        d. Compute per-row ink within the band; smooth and threshold to find
           vertical text extent.
        e. Merge close y-runs (gap ≤ AUX_LINE_GAP_FRAC × ih) into single blocks.
        f. Each block becomes one auxiliary "Text" region.

    All returned regions carry auxiliary=True.  They are rendered in
    AUXILIARY_COLOUR (amber) in the debug visualisation.

    Parameters
    ──────────
    page_image       : preprocessed RGB PIL Image
    existing_regions : Surya regions already detected
    last_position    : highest reading-order position from Surya; auxiliary
                       regions continue from last_position + 1

    Returns
    ───────
    List of new region dicts (may be empty).  Does NOT modify existing_regions.
    """
    iw, ih = page_image.size

    # ── 1. Ink mask ───────────────────────────────────────────────────────────
    gray_arr  = np.array(page_image.convert("L"), dtype=np.float32)
    mu        = gray_arr.mean()
    sigma     = gray_arr.std()
    threshold = float(np.clip(mu - AUX_INK_ADAPTIVE_FACTOR * sigma, 50.0, 200.0))
    ink       = (gray_arr < threshold).astype(np.float32)  # 1 = dark (text)

    # ── 2. Full-page horizontal ink projection ────────────────────────────────
    # Project ALL ink (not just uncovered) so adjacent Surya regions extending
    # into a missed column cannot hide that column's ink signal.
    vert_proj = ink.sum(axis=0)                            # shape (iw,)

    hk = max(3, int(iw * AUX_HSMOOTH_FRAC))
    hk = hk + (1 - hk % 2)                                # enforce odd length
    vert_smooth = np.convolve(vert_proj, np.ones(hk) / hk, mode="same")

    # ── 3. Exclude page-edge margin ───────────────────────────────────────────
    margin_px = max(5, int(iw * AUX_MARGIN_FRAC))
    vert_smooth[:margin_px]   = 0.0
    vert_smooth[iw-margin_px:] = 0.0

    # ── 4. Collect active x-runs (column candidates) ─────────────────────────
    ink_floor = ih * 0.005                                 # ≥ 0.5 % of height
    active_x  = vert_smooth > ink_floor

    x_runs: List[Tuple[int, int]] = []
    in_run = False; rx0 = 0
    for x in range(iw):
        if active_x[x] and not in_run:
            rx0, in_run = x, True
        elif not active_x[x] and in_run:
            x_runs.append((rx0, x))
            in_run = False
    if in_run:
        x_runs.append((rx0, iw))

    # ── 5. Merge narrow-gutter fragments ─────────────────────────────────────
    min_gutter = max(3, int(iw * AUX_MIN_GUTTER_FRAC))
    merged_runs: List[List[int]] = []
    for (rx0, rx1) in x_runs:
        if merged_runs and rx0 - merged_runs[-1][1] <= min_gutter:
            merged_runs[-1][1] = rx1
        else:
            merged_runs.append([rx0, rx1])

    min_col_w = max(MIN_REGION_W, int(iw * AUX_MIN_COL_WIDTH_FRAC))
    min_col_h = max(MIN_REGION_H, int(ih * AUX_MIN_COL_HEIGHT_FRAC))
    line_gap  = max(10, int(ih * AUX_LINE_GAP_FRAC))
    vk        = max(3, int(ih * AUX_VSMOOTH_FRAC))
    vk        = vk + (1 - vk % 2)

    new_regions: List[Dict] = []
    pos        = last_position + 1
    skipped_covered = 0
    skipped_noink   = 0

    # Pre-compute Surya region x-centres once
    surya_cx = [
        (r["bbox"][0] + r["bbox"][2]) / 2.0
        for r in existing_regions
    ]

    for run in merged_runs:
        bx0, bx1 = run[0], run[1]
        bw = bx1 - bx0
        if bw < min_col_w:
            continue

        # ── 6a. Count Surya region centres in this band ───────────────────────
        centre_count = sum(1 for cx in surya_cx if bx0 <= cx <= bx1)

        if centre_count > AUX_MAX_CENTRE_COUNT:
            skipped_covered += 1
            continue                              # Surya already covered this

        # ── 6b. Minimum ink density check ────────────────────────────────────
        band_ink_total = float(ink[:, bx0:bx1].sum())
        band_ink_density = band_ink_total / max(1, bw * ih)
        if band_ink_density < AUX_INK_DENSITY_THRESHOLD:
            skipped_noink += 1
            logger.debug(
                f"    [AUX] band x={bx0}-{bx1} skipped: ink_density="
                f"{band_ink_density:.4f} < {AUX_INK_DENSITY_THRESHOLD}"
            )
            continue

        # ── 6c–d. Vertical extent from per-row ink within band ────────────────
        col_ink    = ink[:, bx0:bx1].sum(axis=1)          # shape (ih,)
        col_smooth = np.convolve(col_ink, np.ones(vk) / vk, mode="same")

        # Row floor: relative to band width so faint columns still register
        row_floor = max(0.5, bw * 0.002)
        active_y  = col_smooth > row_floor

        y_runs: List[Tuple[int, int]] = []
        in_y = False; ry0 = 0
        for y in range(ih):
            if active_y[y] and not in_y:
                ry0, in_y = y, True
            elif not active_y[y] and in_y:
                y_runs.append((ry0, y))
                in_y = False
        if in_y:
            y_runs.append((ry0, ih))

        # ── 6e. Merge close y-runs ────────────────────────────────────────────
        merged_y: List[List[int]] = []
        for (ya, yb) in y_runs:
            if merged_y and ya - merged_y[-1][1] <= line_gap:
                merged_y[-1][1] = yb
            else:
                merged_y.append([ya, yb])

        # ── 6f. Emit auxiliary regions ────────────────────────────────────────
        for seg in merged_y:
            ya, yb = seg
            if yb - ya < min_col_h:
                continue
            new_regions.append({
                "bbox":      [float(bx0), float(ya), float(bx1), float(yb)],
                "polygon":   None,
                "label":     "Text",
                "position":  pos,
                "top_k":     {},
                "auxiliary": True,
            })
            logger.debug(
                f"    [AUX] +region: x={bx0}-{bx1} y={ya}-{yb}  "
                f"centres={centre_count}  ink_density={band_ink_density:.4f}"
            )
            pos += 1

    logger.info(
        f"  [AUX-LAYOUT] {len(new_regions)} auxiliary Text region(s) added.  "
        f"(ink_threshold={threshold:.0f}  hk={hk}px  "
        f"col_w≥{min_col_w}px  col_h≥{min_col_h}px  "
        f"max_centres≤{AUX_MAX_CENTRE_COUNT}  "
        f"skipped: covered={skipped_covered}  low-ink={skipped_noink})"
    )
    return new_regions


# ══════════════════════════════════════════════════════════════════════════════
# TROCR RECOGNITION  +  NOISE FILTER
# ══════════════════════════════════════════════════════════════════════════════

def _trocr_read(
    image: Image.Image,
    processor: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
) -> Tuple[str, float]:
    """Run TrOCR on a single image crop. Returns (text, confidence)."""
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
    if len(set(tc)) == 1 and tl > 2:
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
        bboxes.sort(key=lambda b: b[1])
        return bboxes
    except Exception as exc:
        logger.debug(f"    DetectionPredictor error: {exc}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# PER-REGION OCR
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
    Pass 2 — Whole-crop TrOCR fallback (always tried for SINGLE_BLOCK_LABELS,
              and for any region where Pass 1 returns zero accepted lines).

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

    # ── Pass 1 ────────────────────────────────────────────────────────────────
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
        abs_bbox = [lx0 + x0, ly0 + y0, lx1 + x0, ly1 + y0]
        pass1_elems.append({
            "text":             text,
            "bbox":             abs_bbox,
            "confidence":       confidence,
            "font_size":        max(6.0, min(lh * 0.85, 72.0)),
            "source_label":     label,
            "reading_position": region["position"],
            "auxiliary":        region.get("auxiliary", False),
        })

    n_pass1 = len(pass1_elems)

    if n_pass1 > 0 and label not in SINGLE_BLOCK_LABELS:
        logger.debug(f"      Pass 1 (det+TrOCR): {n_pass1} line(s)")
        return pass1_elems

    # ── Pass 2 ────────────────────────────────────────────────────────────────
    text_wb, conf_wb = _trocr_read(crop, trocr_processor, trocr_model)
    pass2_elems: List[Dict] = []

    if not _is_noise(text_wb, conf_wb, rh, rw):
        pass2_elems.append({
            "text":             text_wb,
            "bbox":             [float(x0), float(y0), float(x1), float(y1)],
            "confidence":       conf_wb,
            "font_size":        max(6.0, min(rh * 0.85, 72.0)),
            "source_label":     label,
            "reading_position": region["position"],
            "auxiliary":        region.get("auxiliary", False),
        })

    n_pass2 = len(pass2_elems)

    if n_pass2 > 0:
        logger.debug(
            f"      Pass 2 (whole-crop TrOCR): {n_pass2} line(s)  "
            f"[Pass 1 had {n_pass1}]"
        )
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

def save_layout_debug(
    image: Image.Image,
    regions: List[Dict],
    path: Path,
    title_suffix: str = "",
) -> None:
    """
    Render a layout debug image.

    Surya-detected regions are drawn in their label colour.
    Auxiliary regions are drawn in AUXILIARY_COLOUR (amber) with a dashed
    border style (alternating 6-px dash / 3-px gap approximated by offsetting
    a second thinner rectangle).
    """
    img      = image.copy().convert("RGB")
    draw     = ImageDraw.Draw(img, "RGBA")
    lbl_font = _pil_font(15)

    for region in sorted(regions, key=lambda r: r.get("position", 999)):
        x0, y0, x1, y1 = region["bbox"]
        label     = region.get("label", "?")
        pos       = region.get("position", "?")
        is_aux    = region.get("auxiliary", False)
        rgb       = _hex_rgb(_label_hex(label, auxiliary=is_aux))
        tag       = f"[{pos}] {'AUX:' if is_aux else ''}{label}"
        tag_w     = len(tag) * 9 + 6

        if is_aux:
            # Amber filled rectangle with a slightly thinner inner border to
            # visually suggest a dashed/provisional boundary.
            draw.rectangle(
                [x0, y0, x1, y1],
                outline=rgb + (230,),
                fill=rgb + (18,),
                width=3,
            )
            draw.rectangle(
                [x0 + 4, y0 + 4, x1 - 4, y1 - 4],
                outline=rgb + (90,),
                fill=None,
                width=1,
            )
        else:
            draw.rectangle(
                [x0, y0, x1, y1],
                outline=rgb + (210,),
                fill=rgb + (22,),
                width=2,
            )

        draw.rectangle([x0, y0, x0 + tag_w, y0 + 20], fill=rgb + (175,))
        draw.text((x0 + 3, y0 + 2), tag, fill=(255, 255, 255, 255), font=lbl_font)

    img.save(str(path), "JPEG", quality=90)
    logger.info(f"    [DEBUG] Layout map{title_suffix} → {path.name}")


def save_ocr_debug(image: Image.Image, elements: List[Dict], path: Path) -> None:
    img  = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    font = _pil_font(12)
    for elem in elements:
        x0, y0, x1, y1 = elem["bbox"]
        is_aux = elem.get("auxiliary", False)
        rgb    = _hex_rgb(_label_hex(elem.get("source_label", "Text"), auxiliary=is_aux))
        draw.rectangle([x0, y0, x1, y1], outline=rgb + (180,), width=1)
        draw.text((x0 + 1, y0), elem["text"][:55], fill=rgb + (220,), font=font)
    img.save(str(path), "JPEG", quality=88)
    logger.info(f"    [DEBUG] OCR overlay → {path.name}")


def save_layout_report(
    regions: List[Dict],
    image_bbox: Optional[List[float]],
    path: Path,
    filename: str,
    page_num: int,
    label: str = "LAYOUT",
) -> None:
    surya_n = sum(1 for r in regions if not r.get("auxiliary", False))
    aux_n   = sum(1 for r in regions if r.get("auxiliary", False))
    lines = [
        f"FILE: {filename}   PAGE: {page_num}  [{label}]",
        f"Regions detected: {len(regions)}  (Surya: {surya_n}  Auxiliary: {aux_n})",
        (f"image_bbox (coord space): {[round(v) for v in image_bbox]}"
         if image_bbox else "image_bbox: not reported"),
        "=" * 90,
        f"{'POS':>4}  {'SRC':<5}  {'LABEL':<22}  {'X0':>6} {'Y0':>6} {'X1':>6} {'Y1':>6}"
        f"  TOP-K ALTERNATIVES",
        "-" * 90,
    ]
    for r in sorted(regions, key=lambda x: x.get("position", 999)):
        x0, y0, x1, y1 = r["bbox"]
        top_k     = r.get("top_k", {})
        top_k_str = "  ".join(
            f"{lbl}:{conf:.2f}"
            for lbl, conf in sorted(top_k.items(), key=lambda kv: -kv[1])
        ) if top_k else "—"
        src = "AUX" if r.get("auxiliary", False) else "surya"
        lines.append(
            f"{r.get('position','?'):>4}  "
            f"{src:<5}  "
            f"{r.get('label','?'):<22}  "
            f"{x0:>6.0f} {y0:>6.0f} {x1:>6.0f} {y1:>6.0f}"
            f"  {top_k_str}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"    [DEBUG] Layout report → {path.name}")


def save_ocr_report(elements: List[Dict], path: Path,
                    filename: str, page_num: int) -> None:
    surya_n = sum(1 for e in elements if not e.get("auxiliary", False))
    aux_n   = sum(1 for e in elements if e.get("auxiliary", False))
    lines = [
        f"FILE: {filename}   PAGE: {page_num}",
        f"OCR elements: {len(elements)}  (from Surya regions: {surya_n}  "
        f"from auxiliary regions: {aux_n})",
        "=" * 90,
    ]
    for i, e in enumerate(elements):
        x0, y0, x1, y1 = e["bbox"]
        src = "AUX" if e.get("auxiliary", False) else "surya"
        lines.append(
            f"[{e.get('reading_position', i):>3}] "
            f"{src:<5}  "
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
    2. Layout      — LayoutPredictor → semantic regions
    3. Aux Layout  — auxiliary_layout_pass() → fills gaps in Surya output
    4. OCR         — DetectionPredictor (line segmentation) + TrOCR per line
    5. Sort        — by reading_position, then vertical baseline

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

    fallback_used = False
    if not layout_regions:
        logger.warning("  No layout regions — falling back to full-page single region.")
        fallback_used = True
        layout_regions = [{
            "bbox":      [0.0, 0.0, float(pil_image.width), float(pil_image.height)],
            "polygon":   None,
            "label":     "Text",
            "position":  0,
            "top_k":     {},
            "auxiliary": False,
        }]
    else:
        label_counts = Counter(r["label"] for r in layout_regions)
        logger.info(
            f"  [LAYOUT] {len(layout_regions)} Surya regions  image_bbox={image_bbox}: "
            + "  ".join(f"{lbl}×{n}" for lbl, n in sorted(label_counts.items()))
        )
        save_layout_debug(
            pil_image, layout_regions,
            Path(str(pfx) + "_01_layout.jpg"),
            title_suffix=" (Surya only)",
        )
        save_layout_report(
            layout_regions, image_bbox,
            Path(str(pfx) + "_01_layout_report.txt"),
            filename, page_num, label="SURYA",
        )

    # ── Stage 3: Auxiliary layout pass ───────────────────────────────────────
    # Skip the auxiliary pass if we are already using the whole-page fallback
    # (nothing meaningful to gap-fill in that case).
    aux_regions: List[Dict] = []
    if not fallback_used:
        logger.info("  [AUX-LAYOUT] Running auxiliary column-recovery pass …")
        try:
            last_pos    = max((r["position"] for r in layout_regions), default=0)
            aux_regions = auxiliary_layout_pass(processed, layout_regions, last_pos)
        except Exception as exc:
            logger.error(f"  auxiliary_layout_pass failed: {exc}")
            import traceback; traceback.print_exc()

        if aux_regions:
            combined_regions = layout_regions + aux_regions
            save_layout_debug(
                pil_image, combined_regions,
                Path(str(pfx) + "_01b_layout_combined.jpg"),
                title_suffix=" (Surya + Auxiliary)",
            )
            save_layout_report(
                combined_regions, image_bbox,
                Path(str(pfx) + "_01b_layout_combined_report.txt"),
                filename, page_num, label="COMBINED",
            )
        else:
            logger.info("  [AUX-LAYOUT] No additional regions found.")
            combined_regions = layout_regions
    else:
        combined_regions = layout_regions

    # ── Stage 4: Per-region OCR ───────────────────────────────────────────────
    text_regions    = [r for r in combined_regions if r["label"] in OCR_LABELS]
    skip_regions    = [r for r in combined_regions if r["label"] in SKIP_LABELS]
    unknown_regions = [
        r for r in combined_regions
        if r["label"] not in OCR_LABELS and r["label"] not in SKIP_LABELS
    ]
    if unknown_regions:
        unk_lbls = sorted({r["label"] for r in unknown_regions})
        logger.warning(
            f"  [LAYOUT] {len(unknown_regions)} region(s) with unrecognised "
            f"label(s) {unk_lbls} — skipping OCR for these."
        )

    surya_ocr_n = sum(1 for r in text_regions if not r.get("auxiliary", False))
    aux_ocr_n   = sum(1 for r in text_regions if r.get("auxiliary", False))
    logger.info(
        f"  [OCR] {len(text_regions)} regions to OCR "
        f"(Surya: {surya_ocr_n}  Auxiliary: {aux_ocr_n}  "
        f"TrOCR model: {TROCR_MODEL_NAME}), "
        f"{len(skip_regions)} region(s) skipped."
    )

    all_elements: List[Dict] = []

    for ri, region in enumerate(sorted(text_regions, key=lambda r: r["position"])):
        lbl  = region["label"]
        bbox = region["bbox"]
        src  = "AUX" if region.get("auxiliary", False) else "surya"
        logger.debug(
            f"    Region {ri+1}/{len(text_regions)}: [{src}] {lbl} "
            f"pos={region['position']}  "
            f"bbox=({bbox[0]:.0f},{bbox[1]:.0f}→{bbox[2]:.0f},{bbox[3]:.0f})"
        )
        elems = ocr_region(
            processed, region,
            det_predictor, trocr_processor, trocr_model,
        )
        logger.debug(f"      → {len(elems)} element(s) accepted.")
        all_elements.extend(elems)

    # ── Stage 5: Final reading order sort ────────────────────────────────────
    all_elements.sort(key=lambda e: (e["reading_position"], e["bbox"][1]))

    surya_elem_n = sum(1 for e in all_elements if not e.get("auxiliary", False))
    aux_elem_n   = sum(1 for e in all_elements if e.get("auxiliary", False))
    logger.info(
        f"  [RESULT] {len(all_elements)} OCR element(s) on page {page_num} "
        f"(Surya: {surya_elem_n}  Auxiliary: {aux_elem_n})."
    )

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
    logger.info("║  + Auxiliary Column-Recovery Pass for Historic Newsprint     ║")
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