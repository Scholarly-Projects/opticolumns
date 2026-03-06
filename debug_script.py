#!/usr/bin/env python3
"""
Opticolumn DEBUG VERSION - STRICT BOUNDARIES
============================================
CHANGES vs previous version
-----------------------------
1. order_lines_surya_debug - STRICT CLAMPING:
   - Text boxes are assigned to columns based on their center.
   - To guarantee page segments NEVER overlap, text boxes are physically clamped 
     (x0, x1) to the strict boundaries of their assigned column.
   - Text boxes are also clamped vertically (y0, y1) to prevent column segments 
     from overlapping with header segments.
"""

import sys
import os
import json
import textwrap
from pathlib import Path
import fitz
from PIL import Image, ImageFilter, ImageOps, ImageDraw, ImageFont
import logging
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import re
import platform
import datetime
import shutil
import pikepdf
from typing import List, Optional
from surya.foundation import FoundationPredictor
from surya.detection import DetectionPredictor

# ─────────────────────────── Configuration ───────────────────────────────────
INPUT_DIR  = "A"
OUTPUT_DIR = "B"
DEBUG_DIR  = "debug"
MODELS_DIR = "mlmodels"
DPI = 200
TROCR_MODELS = {
    "handwritten":       "microsoft/trocr-base-handwritten",
    "printed":           "microsoft/trocr-base-printed",
    "large_handwritten": "microsoft/trocr-large-handwritten",
    "large_printed":     "microsoft/trocr-large-printed",
}
TROCR_MODEL_NAME       = TROCR_MODELS["large_handwritten"]
ENABLE_PREPROCESSING             = True
CONFIDENCE_THRESHOLD             = 0.25
SINGLE_CHAR_CONFIDENCE_THRESHOLD = 0.5
MIN_SEGMENT_HEIGHT               = 10
FONT_NAME     = "helv"
FONT_PATH     = "fonts/FreeSans.ttf"
SRGB_ICC_PATH = "srgb.icc"

# ─────────────────────────── Debug Directory ─────────────────────────────────
DEBUG_PATH = Path(DEBUG_DIR)
DEBUG_PATH.mkdir(exist_ok=True)

# ─────────────────────────── Logging ─────────────────────────────────────────
log_file = DEBUG_PATH / "debug_run.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file), mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ─────────────────────────── Colour palette ──────────────────────────────────
SEGMENT_COLOURS = [
    "#E63946", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0",
    "#00BCD4", "#FF5722", "#8BC34A", "#3F51B5", "#F06292",
]

def seg_colour(idx: int) -> str:
    return SEGMENT_COLOURS[idx % len(SEGMENT_COLOURS)]

def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

# ─────────────────────────── Annotation helpers ──────────────────────────────
def _get_pil_font(size: int = 18):
    try:
        return ImageFont.truetype(FONT_PATH, size=size)
    except Exception:
        return ImageFont.load_default()

def annotate_raw_bboxes(pil_image: Image.Image, bboxes: List[List[float]],
                        save_path: Path):
    img  = pil_image.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    font = _get_pil_font(14)
    for i, (x0, y0, x1, y1) in enumerate(bboxes):
        draw.rectangle([x0, y0, x1, y1], outline=(30, 100, 220, 230), width=2)
        draw.text((x0 + 2, y0), str(i + 1), fill=(30, 100, 220, 230), font=font)
    img.save(str(save_path), "JPEG", quality=88)
    logger.info(f"  [DEBUG] Raw bbox image -> {save_path}")

def annotate_gutters(pil_image: Image.Image, gutters: List[float],
                     bboxes: List[List[float]], save_path: Path):
    img  = pil_image.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    for x0, y0, x1, y1 in bboxes:
        draw.rectangle([x0, y0, x1, y1], outline=(160, 160, 160, 120), width=1)
    for g in gutters:
        draw.line([(g, 0), (g, img.height)], fill=(30, 200, 60, 230), width=3)
    font = _get_pil_font(16)
    for ci, (lo, hi) in enumerate(zip([0.0] + gutters, gutters + [float(img.width)])):
        cx = (lo + hi) / 2.0
        draw.text((cx - 20, 10), f"Col {ci+1}", fill=(30, 200, 60, 230), font=font)
    img.save(str(save_path), "JPEG", quality=88)
    logger.info(f"  [DEBUG] Gutter image -> {save_path}")

def annotate_segments(pil_image: Image.Image,
                      segments: List[List[List[float]]],
                      save_path: Path,
                      scale: float = 1.0):
    img      = pil_image.copy().convert("RGB")
    draw     = ImageDraw.Draw(img, "RGBA")
    num_font  = _get_pil_font(28)
    line_font = _get_pil_font(12)

    for seg_idx, seg_bboxes in enumerate(segments):
        if not seg_bboxes:
            continue
        colour_hex  = seg_colour(seg_idx)
        colour_rgb  = hex_to_rgb(colour_hex)
        fill_rgba   = colour_rgb + (30,)
        border_rgba = colour_rgb + (220,)

        all_x0 = min(b[0] for b in seg_bboxes)
        all_y0 = min(b[1] for b in seg_bboxes)
        all_x1 = max(b[2] for b in seg_bboxes)
        all_y1 = max(b[3] for b in seg_bboxes)

        draw.rectangle([all_x0, all_y0, all_x1, all_y1],
                       outline=border_rgba, fill=fill_rgba, width=3)
        for b in seg_bboxes:
            draw.rectangle([b[0], b[1], b[2], b[3]],
                           outline=colour_rgb + (140,), width=1)
        draw.text((all_x0 + 4, all_y0 + 4), str(seg_idx + 1),
                  fill=(220, 20, 20, 255), font=num_font)
        draw.text((all_x0 + 4, all_y0 + 36),
                  f"{len(seg_bboxes)} lines",
                  fill=(80, 80, 80, 200), font=line_font)

    img.save(str(save_path), "JPEG", quality=90)
    logger.info(f"  [DEBUG] Segment image -> {save_path}")

# ─────────────────────────── Histogram dump ──────────────────────────────────
def dump_histogram(line_bboxes: List[List[float]], page_width: float,
                   gutters: List[float], save_path: Path):
    RESOLUTION          = 4
    MAX_LINE_WIDTH_FRAC = 0.28
    max_w  = page_width * MAX_LINE_WIDTH_FRAC
    narrow = [b for b in line_bboxes if (b[2] - b[0]) <= max_w]
    hist_w = int(page_width / RESOLUTION) + 2
    hist   = [0] * hist_w

    for x0, y0, x1, y1 in narrow:
        bi = int(((x0 + x1) / 2.0) / RESOLUTION)
        if 0 <= bi < hist_w:
            hist[bi] += 1

    lines_out = [
        f"Page width: {page_width:.1f}px  RESOLUTION: {RESOLUTION}px/bucket",
        f"Total bboxes: {len(line_bboxes)}  narrow (<={max_w:.0f}px): {len(narrow)}",
        f"Histogram buckets: {hist_w}",
        f"Detected gutters (px): {[round(g) for g in gutters]}",
        "",
        "Bucket | Centre_px | Count | Bar",
        "-" * 60,
    ]
    max_val = max(hist) if hist else 1
    for i, v in enumerate(hist):
        centre_px = (i + 0.5) * RESOLUTION
        bar       = chr(9608) * int(40 * v / max_val) if max_val else ""
        is_gutter = ""
        for g in gutters:
            if abs(centre_px - g) < RESOLUTION * 3:
                is_gutter = " <- GUTTER"
        lines_out.append(f"{i:5d} | {centre_px:8.1f} | {v:5d} | {bar}{is_gutter}")

    save_path.write_text("\n".join(lines_out), encoding="utf-8")
    logger.info(f"  [DEBUG] Histogram dump -> {save_path}")

# ─────────────────────────── Final order dump ────────────────────────────────
def dump_final_order(segments_with_text: List[List[dict]], save_path: Path,
                     page_num: int, filename: str):
    lines_out = [
        f"FILE: {filename}   PAGE: {page_num}",
        f"Segments: {len(segments_with_text)}",
        "=" * 70,
    ]
    for seg_idx, seg in enumerate(segments_with_text):
        lines_out.append(f"\n-- SEGMENT {seg_idx+1} ({len(seg)} lines) --")
        for li, elem in enumerate(seg):
            conf_info = (f"  [conf~{elem.get('confidence', '?'):.2f}]"
                         if 'confidence' in elem else "")
            lines_out.append(
                f"  Line {li+1:3d} | "
                f"x0={elem['x0']:6.1f}  ybase={elem['y_baseline']:6.1f}  "
                f"fs={elem['font_size']:5.1f}{conf_info} | {elem['text']}"
            )
    save_path.write_text("\n".join(lines_out), encoding="utf-8")
    logger.info(f"  [DEBUG] Final order dump -> {save_path}")

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
        font_dir  = Path("fonts")
        font_dir.mkdir(exist_ok=True)
        font_path = Path(FONT_PATH)
        if not font_path.exists():
            logger.info("Downloading FreeSans font...")
            import urllib.request
            urllib.request.urlretrieve(
                "https://github.com/opensourcedesign/fonts/raw/master/"
                "gnu-freefont_freesans/FreeSans.ttf",
                str(font_path),
            )
        srgb_path = Path(SRGB_ICC_PATH)
        if not srgb_path.exists():
            logger.info("Downloading sRGB ICC profile...")
            try:
                import urllib.request
                if platform.system() == "Darwin":
                    sp = "/System/Library/ColorSync/Profiles/sRGB Profile.icc"
                elif platform.system() == "Windows":
                    sp = os.path.join(
                        os.environ.get("WINDIR", "C:\\Windows"),
                        "System32", "spool", "drivers", "color",
                        "sRGB Color Space Profile.icm",
                    )
                else:
                    sp = "/usr/share/color/icc/sRGB.icc"
                if sp and Path(sp).exists():
                    shutil.copy2(sp, str(srgb_path))
                else:
                    urllib.request.urlretrieve(
                        "https://www.color.org/srgb.xalter", str(srgb_path)
                    )
            except Exception as e:
                logger.warning(f"Could not obtain sRGB ICC: {e}")
        return True
    except Exception as e:
        logger.error(f"PDF/A resource setup failed: {e}")
        return False

def create_xmp_metadata(title, author, subject, creator, producer,
                        creation_date, modify_date):
    try:
        return (
            '<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
            '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
            '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
            '<rdf:Description rdf:about="" xmlns:pdf="http://ns.adobe.com/pdf/1.3/">\n'
            f'<pdf:Producer>{producer}</pdf:Producer>\n'
            '</rdf:Description>\n'
            '<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
            f'<dc:title><rdf:Alt><rdf:li xml:lang="x-default">{title}</rdf:li></rdf:Alt></dc:title>\n'
            f'<dc:creator><rdf:Seq><rdf:li>{author}</rdf:li></rdf:Seq></dc:creator>\n'
            f'<dc:description><rdf:Alt><rdf:li xml:lang="x-default">{subject}</rdf:li></rdf:Alt></dc:description>\n'
            '</rdf:Description>\n'
            '<rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/">\n'
            f'<xmp:CreatorTool>{creator}</xmp:CreatorTool>\n'
            f'<xmp:CreateDate>{creation_date}</xmp:CreateDate>\n'
            f'<xmp:ModifyDate>{modify_date}</xmp:ModifyDate>\n'
            '</rdf:Description>\n'
            '<rdf:Description rdf:about="" '
            'xmlns:pdfaid="http://www.aiim.org/pdfa/ns/id/">\n'
            '<pdfaid:part>1</pdfaid:part>\n'
            '<pdfaid:conformance>B</pdfaid:conformance>\n'
            '</rdf:Description>\n'
            '</rdf:RDF>\n'
            '</x:xmpmeta>\n'
            '<?xpacket end="w"?>'
        )
    except Exception as e:
        logger.error(f"XMP metadata creation failed: {e}")
        return None

# ─────────────────────────── Model loading ───────────────────────────────────
def load_models():
    logger.info("=" * 60)
    logger.info("LOADING MODELS")
    logger.info("=" * 60)
    if not setup_pdfa_resources():
        logger.warning("PDF/A resources setup incomplete.")
    logger.info("Loading Surya FoundationPredictor...")
    foundation = FoundationPredictor()
    logger.info("Loading Surya DetectionPredictor...")
    detection_predictor = DetectionPredictor()
    logger.info(f"Loading TrOCR: {TROCR_MODEL_NAME}")
    processor   = TrOCRProcessor.from_pretrained(TROCR_MODEL_NAME)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trocr_model.to(device)
    logger.info(f"Device: {device}")
    return detection_predictor, processor, trocr_model

try:
    detection_predictor, processor, trocr_model = load_models()
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    sys.exit(1)

# ─────────────────────────── Image preprocessing ─────────────────────────────
def preprocess_for_ocr(pil_image: Image.Image) -> Image.Image:
    if not ENABLE_PREPROCESSING:
        return pil_image.copy()
    try:
        gray      = pil_image.convert("L")
        gray      = ImageOps.autocontrast(gray, cutoff=2)
        processed = gray.convert("RGB")
        processed = processed.filter(ImageFilter.SHARPEN)
        return processed
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return pil_image.copy()

def page_to_pil(page: fitz.Page, dpi: int = DPI) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

# ─────────────────────────── TrOCR recognition ───────────────────────────────
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
        logger.error(f"TrOCR error: {e}")
        return "", 0.0

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

# ─────────────────────────── Surya bbox helper ───────────────────────────────
def _bbox_from_surya_box(box) -> Optional[List[float]]:
    if hasattr(box, "bbox"):
        return list(box.bbox)
    if hasattr(box, "polygon") and len(box.polygon) >= 4:
        xs = [p[0] for p in box.polygon]
        ys = [p[1] for p in box.polygon]
        return [min(xs), min(ys), max(xs), max(ys)]
    return None

def get_surya_lines(image: Image.Image, debug_prefix: str = "") -> List[List[float]]:
    logger.debug(f"  [SURYA] Running DetectionPredictor on image {image.size}...")
    try:
        results = detection_predictor([image])
        if not results or not hasattr(results[0], "bboxes"):
            logger.warning("  [SURYA] No results.")
            return []
        bboxes    = []
        raw_boxes = results[0].bboxes
        logger.debug(f"  [SURYA] Raw box count: {len(raw_boxes)}")
        for box in raw_boxes:
            bbox = _bbox_from_surya_box(box)
            if bbox is not None:
                bboxes.append(bbox)
            else:
                logger.debug(f"    [SURYA] Skipped: {box}")
        if bboxes:
            widths  = [b[2] - b[0] for b in bboxes]
            heights = [b[3] - b[1] for b in bboxes]
            logger.info(
                f"  [SURYA] {len(bboxes)} bboxes  "
                f"W: min={min(widths):.0f} max={max(widths):.0f} avg={sum(widths)/len(widths):.0f}  "
                f"H: min={min(heights):.0f} max={max(heights):.0f} avg={sum(heights)/len(heights):.0f}"
            )
        return bboxes
    except Exception as e:
        logger.error(f"  [SURYA] Failed: {e}")
        return []

# ─────────────────────────── Gutter Detection ─────────────────────────
def _detect_column_gutters_debug(
    line_bboxes: List[List[float]],
    page_width: float,
    hist_dump_path: Path,
) -> List[float]:
    logger.info("  -- GUTTER DETECTION --")

    if not line_bboxes:
        return []

    MAX_LINE_WIDTH_FRAC = 0.28
    MIN_GAP_WIDTH_PX    = 30 
    MIN_GUTTER_MERGE_PX = 20
    RESOLUTION          = 4

    max_w  = page_width * MAX_LINE_WIDTH_FRAC
    narrow = [b for b in line_bboxes if (b[2] - b[0]) <= max_w]
    if not narrow:
        narrow = line_bboxes

    hist_w = int(page_width / RESOLUTION) + 2
    hist   = [0] * hist_w
    for x0, y0, x1, y1 in narrow:
        bi = int(((x0 + x1) / 2.0) / RESOLUTION)
        if 0 <= bi < hist_w:
            hist[bi] += 1

    NEAR_ZERO = 1
    gaps: List[float] = []
    in_gap    = False
    gap_start = 0
    for i, v in enumerate(hist):
        if v <= NEAR_ZERO:
            if not in_gap:
                in_gap    = True
                gap_start = i
        else:
            if in_gap:
                gap_w_px = (i - gap_start) * RESOLUTION
                mid_px   = ((gap_start + i - 1) / 2.0) * RESOLUTION
                if gap_w_px >= MIN_GAP_WIDTH_PX:
                    gaps.append(mid_px)
                in_gap = False
    if in_gap:
        gap_w_px = (hist_w - gap_start) * RESOLUTION
        mid_px   = ((gap_start + hist_w - 1) / 2.0) * RESOLUTION
        if gap_w_px >= MIN_GAP_WIDTH_PX:
            gaps.append(mid_px)

    merged: List[float] = []
    for g in sorted(gaps):
        if merged and g - merged[-1] < MIN_GUTTER_MERGE_PX:
            merged[-1] = (merged[-1] + g) / 2.0
        else:
            merged.append(g)

    MARGIN_FRAC = 0.08
    margin      = page_width * MARGIN_FRAC
    merged      = [g for g in merged if margin < g < page_width - margin]

    MIN_COLUMN_WIDTH_FRAC = 0.08 
    min_col_w = page_width * MIN_COLUMN_WIDTH_FRAC
    changed   = True
    while changed and merged:
        changed    = False
        edges      = [0.0] + merged + [page_width]
        col_widths = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
        min_w      = min(col_widths)
        if min_w < min_col_w:
            narrow_ci = col_widths.index(min_w)
            right_gi = narrow_ci
            left_gi  = narrow_ci - 1
            if right_gi < len(merged):
                merged.pop(right_gi)
            elif left_gi >= 0:
                merged.pop(left_gi)
            changed = True

    dump_histogram(line_bboxes, page_width, merged, hist_dump_path)
    return merged

# ─────────────────────────── REVISED: Strict Boundary Reading Order ──────────
def order_lines_surya_debug(
    line_bboxes: List[List[float]],
    image: Image.Image,
    debug_prefix: str,
) -> List[List[List[float]]]:
    """
    Enforces STRICT boundary constraints so segments NEVER overlap.
    """
    logger.info("  -- STRICT BORDER READING ORDER --")

    if not line_bboxes:
        return []

    page_width  = float(image.width)
    
    # Step 1: Detect gutters
    hist_path = Path(f"{debug_prefix}_01_gutter_hist.txt")
    gutters   = _detect_column_gutters_debug(line_bboxes, page_width, hist_path)

    annotate_gutters(image, gutters, line_bboxes,
                     Path(f"{debug_prefix}_02_gutters.jpg"))

    HEADER_WIDTH_FRAC = 0.65
    headers_raw = []
    columns_raw = []
    
    # Step 2: Separate headers from standard columns
    for box in line_bboxes:
        box_width = box[2] - box[0]
        if box_width / page_width > HEADER_WIDTH_FRAC:
            headers_raw.append(box)
        else:
            columns_raw.append(box)

    # Step 3: Define strict column boundaries from gutters
    sorted_gutters = sorted(gutters)
    col_bounds = []
    last_x = 0.0
    for g in sorted_gutters:
        col_bounds.append((last_x, g))
        last_x = g
    col_bounds.append((last_x, page_width))

    columns = {i: [] for i in range(len(col_bounds))}
    
    # Define vertical limits to prevent column boxes from overlapping headers
    header_y_ranges = [(h[1], h[3]) for h in headers_raw]

    PADDING = 2 # 2px buffer so segments literally never touch in visual output
    
    # Step 4: Assign and CLAMP column boxes
    for box in columns_raw:
        x1, y1, x2, y2 = box
        center_x = x1 + ((x2 - x1) / 2.0)
        center_y = y1 + ((y2 - y1) / 2.0)

        # Find which column the center falls into
        col_idx = 0
        for i, (cmin, cmax) in enumerate(col_bounds):
            if cmin <= center_x <= cmax:
                col_idx = i
                break

        # STRICT X CLAMPING
        cmin, cmax = col_bounds[col_idx]
        new_x1 = max(cmin + PADDING, x1)
        new_x2 = min(cmax - PADDING, x2)

        # STRICT Y CLAMPING
        new_y1, new_y2 = y1, y2
        for (hy1, hy2) in header_y_ranges:
            if center_y >= hy2:     # Box is below header
                new_y1 = max(new_y1, hy2 + PADDING)
            elif center_y <= hy1:   # Box is above header
                new_y2 = min(new_y2, hy1 - PADDING)

        # Ensure the clamped box hasn't been squashed into nothingness
        if new_x2 - new_x1 > 5 and new_y2 - new_y1 > 5:
            columns[col_idx].append([new_x1, new_y1, new_x2, new_y2])

    # Step 5: Final Assembly
    segments = []
    
    if headers_raw:
        headers_raw.sort(key=lambda b: b[1])
        segments.append(headers_raw)
        
    for col_idx in sorted(columns.keys()):
        if columns[col_idx]:
            columns[col_idx].sort(key=lambda b: b[1])
            segments.append(columns[col_idx])

    logger.info(
        f"  order_lines: {len(line_bboxes)} lines -> {len(segments)} strictly bounded segments"
    )
    _log_segment_table(segments, headers_raw)
    return segments

def _log_segment_table(segments: List[List[List[float]]],
                       header_lines: List[List[float]]) -> None:
    logger.info("")
    logger.info(f"  {'SEG':>4}  {'LINES':>5}  {'Y_TOP':>6}  {'Y_BOT':>6}  NOTE")
    logger.info(f"  {'---':>4}  {'-----':>5}  {'------':>6}  {'------':>6}  ----")
    for si, seg in enumerate(segments):
        if not seg:
            continue
        y_top = min(b[1] for b in seg)
        y_bot = max(b[3] for b in seg)
        note  = ("HEADER" if (si == 0 and header_lines)
                 else f"Col {si if header_lines else si + 1}")
        logger.info(
            f"  {si+1:>4}  {len(seg):>5}  {y_top:>6.0f}  {y_bot:>6.0f}  {note}"
        )
    logger.info("")

# ─────────────────────────── OCR element extraction ──────────────────────────
def create_ocr_text_elements_debug(
    pil_images: List[Image.Image],
    filename: str,
    stem: str,
) -> List[List[List[dict]]]:

    font_path = Path(FONT_PATH)
    if not font_path.exists():
        raise FileNotFoundError(f"Required font {font_path} is missing.")

    all_pages: List[List[List[dict]]] = []
    total_elements = 0

    for idx, pil_image in enumerate(pil_images):
        page_num    = idx + 1
        page_prefix = str(DEBUG_PATH / f"{stem}_p{page_num:03d}")
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"PAGE {page_num}/{len(pil_images)}  -  {filename}")

        page_segments: List[List[dict]] = []
        filtered            = 0
        page_elements_count = 0

        try:
            raw_bboxes = get_surya_lines(pil_image, debug_prefix=page_prefix)

            if not raw_bboxes:
                pil_image.save(f"{page_prefix}_00_EMPTY.jpg", "JPEG", quality=88)
                all_pages.append([])
                continue

            annotate_raw_bboxes(pil_image, raw_bboxes,
                                Path(f"{page_prefix}_00_raw_bboxes.jpg"))

            sorted_segments = order_lines_surya_debug(
                raw_bboxes, pil_image, page_prefix
            )

            annotate_segments(pil_image, sorted_segments,
                              Path(f"{page_prefix}_03_segments.jpg"))

            ocr_image = preprocess_for_ocr(pil_image)

            for seg_idx, segment_bboxes in enumerate(sorted_segments):
                segment_elements: List[dict] = []
                for i, bbox in enumerate(segment_bboxes):
                    try:
                        x0, y0, x1, y1 = bbox
                        sh, sw = y1 - y0, x1 - x0
                        if sh < 5 or sw < 5:
                            filtered += 1
                            continue
                        line_img  = ocr_image.crop((x0, y0, x1, y1))
                        text, confidence = recognize_text_with_trocr(
                            line_img, processor, trocr_model
                        )
                        if is_likely_noise(text, confidence, sh, sw):
                            filtered += 1
                            continue
                        segment_elements.append({
                            "x0":         x0,
                            "y_baseline": y1,
                            "font_size":  max(6, min(sh * 0.9, 72)),
                            "text":       text,
                            "confidence": confidence,
                        })
                    except Exception as e:
                        logger.error(f"  Error Seg{seg_idx+1}/Line{i+1}: {e}")

                if segment_elements:
                    page_segments.append(segment_elements)
                    page_elements_count += len(segment_elements)

            dump_final_order(
                page_segments,
                Path(f"{page_prefix}_04_final_order.txt"),
                page_num, filename,
            )
            total_elements += page_elements_count

        except Exception as e:
            logger.error(f"OCR failed for page {page_num}: {e}")

        all_pages.append(page_segments)

    return all_pages

# ─────────────────────────── PDF/A compliance ────────────────────────────────
def setup_pdfa_compliance(pdf_path: str):
    try:
        srgb_path = Path(SRGB_ICC_PATH)
        if not srgb_path.exists():
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
    except Exception as e:
        logger.error(f"PDF/A compliance failed: {e}")

# ─────────────────────────── PDF processing ──────────────────────────────────
def process_single_pdf_ocr(input_path: str, output_path: str) -> bool:
    filename = os.path.basename(input_path)
    stem     = Path(input_path).stem
    try:
        with fitz.open(input_path) as doc:
            pil_images: List[Image.Image] = []
            for page in doc:
                pil_images.append(page_to_pil(page, dpi=DPI))

            ocr_pages = create_ocr_text_elements_debug(pil_images, filename, stem)

            now           = datetime.datetime.now()
            creation_date = get_pdf_date_string(now)
            doc.set_metadata({
                "title":        filename, "author":       "Opticolumn",
                "subject":      "OCR processed document",
                "creator":      "Opticolumn 2026", "producer": "PyMuPDF",
                "creationDate": creation_date, "modDate": creation_date,
            })
            xmp = create_xmp_metadata(
                title=filename, author="Opticolumn",
                subject="OCR processed document", creator="Opticolumn 2026",
                producer="PyMuPDF",
                creation_date=get_xmp_date_string(now),
                modify_date=get_xmp_date_string(now),
            )
            if xmp:
                doc.set_xml_metadata(xmp)

            page_count = min(len(doc), len(ocr_pages))

            for page_num in range(page_count):
                page     = doc[page_num]
                segments = ocr_pages[page_num]
                pil_img  = pil_images[page_num]

                if page.get_text().strip():
                    page.add_redact_annot(page.rect)
                    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

                img_w, img_h   = pil_img.size
                page_w, page_h = page.rect.width, page.rect.height
                sx, sy         = page_w / img_w, page_h / img_h

                text_writer    = fitz.TextWriter(page.rect)
                total_inserted = 0

                for seg_idx, segment in enumerate(segments):
                    for elem in segment:
                        try:
                            text_writer.append(
                                fitz.Point(
                                    elem["x0"] * sx, elem["y_baseline"] * sy
                                ),
                                elem["text"],
                                font=fitz.Font(FONT_NAME),
                                fontsize=max(4, elem["font_size"] * sy),
                            )
                            total_inserted += 1
                        except Exception:
                            pass

                if total_inserted > 0:
                    text_writer.write_text(
                        page, overlay=True, render_mode=3, color=(0, 0, 0)
                    )

            doc.save(output_path, deflate=True, garbage=4, clean=True,
                     deflate_images=False, encryption=fitz.PDF_ENCRYPT_KEEP)

            if Path(SRGB_ICC_PATH).exists():
                setup_pdfa_compliance(output_path)

            return True
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        return False

# ─────────────────────────── Compression ─────────────────────────────────────
def compress_to_target_size(input_pdf: Path, output_pdf: Path,
                            original_size: int) -> Path:
    max_target   = int(original_size * 1.15)
    current_size = input_pdf.stat().st_size
    if current_size <= max_target:
        shutil.copy2(input_pdf, output_pdf)
        return output_pdf
    opts_list = [
        {"deflate": True, "garbage": 4, "clean": True, "deflate_images": False},
        {"deflate": True, "garbage": 3, "clean": True, "deflate_images": False},
        {"deflate": True, "garbage": 2, "clean": True, "deflate_images": False},
    ]
    for i, opts in enumerate(opts_list):
        temp_out = output_pdf.with_suffix(f".temp_{i}.pdf")
        try:
            with fitz.open(str(input_pdf)) as d:
                d.save(str(temp_out), **opts, encryption=fitz.PDF_ENCRYPT_KEEP)
            compressed_size = temp_out.stat().st_size
            if compressed_size <= max_target:
                try:
                    with fitz.open(str(temp_out)) as chk:
                        total_chars = sum(len(pg.get_text().strip()) for pg in chk)
                except Exception:
                    total_chars = -1
                if total_chars > 0:
                    shutil.move(str(temp_out), str(output_pdf))
                    return output_pdf
                else:
                    temp_out.unlink(missing_ok=True)
                    shutil.copy2(input_pdf, output_pdf)
                    return output_pdf
            temp_out.unlink(missing_ok=True)
        except Exception:
            temp_out.unlink(missing_ok=True)
    shutil.copy2(input_pdf, output_pdf)
    return output_pdf

# ─────────────────────────── Main ────────────────────────────────────────────
def main():
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║     OPTICOLUMN  -  STRICT BOUNDARY READING ORDER         ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info(f"Debug output : {DEBUG_PATH.resolve()}")

    input_folder  = Path(INPUT_DIR)
    output_folder = Path(OUTPUT_DIR)
    if not input_folder.exists():
        sys.exit(1)
    output_folder.mkdir(exist_ok=True)

    pdf_files = list(input_folder.glob("*.pdf"))
    if not pdf_files:
        sys.exit(1)

    for pdf_path in pdf_files:
        original_size = pdf_path.stat().st_size
        logger.info(f"\nFILE: {pdf_path.name}")

        ocr_temp_path = output_folder / f"{pdf_path.stem}_ocr_temp.pdf"
        if not process_single_pdf_ocr(str(pdf_path), str(ocr_temp_path)):
            continue

        final_path  = output_folder / f"{pdf_path.stem}_final.pdf"
        result_path = compress_to_target_size(ocr_temp_path, final_path, original_size)
        
        try:
            ocr_temp_path.unlink()
        except Exception:
            pass

    logger.info("\nAll done.")

if __name__ == "__main__":
    main()