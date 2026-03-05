#!/usr/bin/env python3
"""
Opticolumn DEBUG VERSION
========================
Adds full process-data logging for page segmentation and reading order,
plus annotated JPG visualisations saved to ./debug/

For every input PDF page this script produces:
  debug/<stem>_p<N>_00_raw_bboxes.jpg     – every Surya bbox (blue)
  debug/<stem>_p<N>_01_gutter_hist.txt    – centre-histogram raw data
  debug/<stem>_p<N>_02_gutters.jpg        – gutter lines overlaid (green)
  debug/<stem>_p<N>_03_segments.jpg       – all segments numbered in red
  debug/<stem>_p<N>_04_final_order.txt    – ordered text dump

A master console/log file is also written to debug/debug_run.log
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

# ─────────────────────────── Configuration (unchanged) ───────────────────────
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
TROCR_MODEL_NAME                 = TROCR_MODELS["large_handwritten"]
ENABLE_PREPROCESSING             = True
CONFIDENCE_THRESHOLD             = 0.25
SINGLE_CHAR_CONFIDENCE_THRESHOLD = 0.5
MIN_SEGMENT_HEIGHT               = 10
FONT_NAME     = "helv"
FONT_PATH     = "fonts/FreeSans.ttf"
SRGB_ICC_PATH = "srgb.icc"

# ─────────────────────────── Debug Directory Setup ───────────────────────────
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

# ─────────────────────────── Colour palette for segments ─────────────────────
# Each segment gets a distinct border colour so overlapping segments are visible
SEGMENT_COLOURS = [
    "#E63946",  # red
    "#2196F3",  # blue
    "#4CAF50",  # green
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
    "#FF5722",  # deep orange
    "#8BC34A",  # light green
    "#3F51B5",  # indigo
    "#F06292",  # pink
]

def seg_colour(idx: int) -> str:
    return SEGMENT_COLOURS[idx % len(SEGMENT_COLOURS)]

def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

# ─────────────────────────── Annotation helpers ──────────────────────────────
def _get_pil_font(size: int = 18):
    """Return a PIL font – falls back to default if FreeSans is missing."""
    try:
        return ImageFont.truetype(FONT_PATH, size=size)
    except Exception:
        return ImageFont.load_default()

def annotate_raw_bboxes(pil_image: Image.Image, bboxes: List[List[float]],
                        save_path: Path):
    """Step 0 – draw every raw Surya bbox in blue."""
    img = pil_image.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    font = _get_pil_font(14)
    for i, (x0, y0, x1, y1) in enumerate(bboxes):
        draw.rectangle([x0, y0, x1, y1], outline=(30, 100, 220, 230), width=2)
        draw.text((x0 + 2, y0), str(i + 1), fill=(30, 100, 220, 230), font=font)
    img.save(str(save_path), "JPEG", quality=88)
    logger.info(f"  [DEBUG] Raw bbox image → {save_path}")

def annotate_gutters(pil_image: Image.Image, gutters: List[float],
                     bboxes: List[List[float]], save_path: Path):
    """Step 2 – show gutter centre lines (green) over all bboxes (grey)."""
    img = pil_image.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    # faint bbox layer
    for x0, y0, x1, y1 in bboxes:
        draw.rectangle([x0, y0, x1, y1], outline=(160, 160, 160, 120), width=1)
    # gutter lines
    for g in gutters:
        draw.line([(g, 0), (g, img.height)], fill=(30, 200, 60, 230), width=3)
    # labels
    font = _get_pil_font(16)
    for ci, (lo, hi) in enumerate(zip([0.0] + gutters, gutters + [float(img.width)])):
        cx = (lo + hi) / 2.0
        draw.text((cx - 20, 10), f"Col {ci+1}", fill=(30, 200, 60, 230), font=font)
    img.save(str(save_path), "JPEG", quality=88)
    logger.info(f"  [DEBUG] Gutter image → {save_path}")

def annotate_segments(pil_image: Image.Image,
                      segments: List[List[List[float]]],
                      save_path: Path,
                      scale: float = 1.0):
    """
    Step 3 – draw every segment with:
      • a coloured bounding rectangle around all its lines
      • each line's own bbox (thin, same colour, semi-transparent)
      • a large red segment number at the top-left of the segment
    """
    img = pil_image.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    num_font  = _get_pil_font(28)
    line_font = _get_pil_font(12)

    for seg_idx, seg_bboxes in enumerate(segments):
        if not seg_bboxes:
            continue
        colour_hex = seg_colour(seg_idx)
        colour_rgb = hex_to_rgb(colour_hex)
        fill_rgba  = colour_rgb + (30,)   # very transparent fill
        border_rgba = colour_rgb + (220,)

        # Compute bounding box around entire segment
        all_x0 = min(b[0] for b in seg_bboxes)
        all_y0 = min(b[1] for b in seg_bboxes)
        all_x1 = max(b[2] for b in seg_bboxes)
        all_y1 = max(b[3] for b in seg_bboxes)

        # Draw segment bounding rect
        draw.rectangle([all_x0 - 4, all_y0 - 4, all_x1 + 4, all_y1 + 4],
                       outline=border_rgba, fill=fill_rgba, width=3)

        # Draw individual line bboxes
        for b in seg_bboxes:
            draw.rectangle([b[0], b[1], b[2], b[3]],
                           outline=colour_rgb + (140,), width=1)

        # Red segment number (reading order)
        label = str(seg_idx + 1)
        draw.text((all_x0 + 4, all_y0 + 4), label,
                  fill=(220, 20, 20, 255), font=num_font)

        # Line count in smaller text below number
        draw.text((all_x0 + 4, all_y0 + 36),
                  f"{len(seg_bboxes)} lines",
                  fill=(80, 80, 80, 200), font=line_font)

    img.save(str(save_path), "JPEG", quality=90)
    logger.info(f"  [DEBUG] Segment image → {save_path}")

# ─────────────────────────── Histogram text dump ─────────────────────────────
def dump_histogram(line_bboxes: List[List[float]], page_width: float,
                   gutters: List[float], save_path: Path):
    """Write the full centre-point histogram as a text file."""
    RESOLUTION = 4
    MAX_LINE_WIDTH_FRAC = 0.28
    max_w = page_width * MAX_LINE_WIDTH_FRAC
    narrow = [b for b in line_bboxes if (b[2] - b[0]) <= max_w]
    hist_w = int(page_width / RESOLUTION) + 2
    hist   = [0] * hist_w

    for x0, y0, x1, y1 in narrow:
        bi = int(((x0 + x1) / 2.0) / RESOLUTION)
        if 0 <= bi < hist_w:
            hist[bi] += 1

    lines_out = []
    lines_out.append(f"Page width: {page_width:.1f}px  RESOLUTION: {RESOLUTION}px/bucket")
    lines_out.append(f"Total bboxes: {len(line_bboxes)}  narrow (≤{max_w:.0f}px): {len(narrow)}")
    lines_out.append(f"Histogram buckets: {hist_w}")
    lines_out.append(f"Detected gutters (px): {[round(g) for g in gutters]}")
    lines_out.append("")
    lines_out.append("Bucket | Centre_px | Count | Bar")
    lines_out.append("-" * 60)
    max_val = max(hist) if hist else 1
    for i, v in enumerate(hist):
        centre_px = (i + 0.5) * RESOLUTION
        bar = "█" * int(40 * v / max_val) if max_val else ""
        is_gutter = ""
        for g in gutters:
            if abs(centre_px - g) < RESOLUTION * 3:
                is_gutter = " ← GUTTER"
        lines_out.append(f"{i:5d} | {centre_px:8.1f} | {v:5d} | {bar}{is_gutter}")

    save_path.write_text("\n".join(lines_out), encoding="utf-8")
    logger.info(f"  [DEBUG] Histogram dump → {save_path}")

# ─────────────────────────── Ordered text dump ───────────────────────────────
def dump_final_order(segments_with_text: List[List[dict]], save_path: Path,
                     page_num: int, filename: str):
    """Write segment-by-segment ordered text to a plain-text file."""
    lines_out = []
    lines_out.append(f"FILE: {filename}   PAGE: {page_num}")
    lines_out.append(f"Segments: {len(segments_with_text)}")
    lines_out.append("=" * 70)
    for seg_idx, seg in enumerate(segments_with_text):
        lines_out.append(f"\n── SEGMENT {seg_idx+1} ({len(seg)} lines) ──")
        for li, elem in enumerate(seg):
            conf_info = f"  [conf≈{elem.get('confidence', '?'):.2f}]" if 'confidence' in elem else ""
            lines_out.append(
                f"  Line {li+1:3d} | "
                f"x0={elem['x0']:6.1f}  ybase={elem['y_baseline']:6.1f}  "
                f"fs={elem['font_size']:5.1f}{conf_info} | {elem['text']}"
            )
    save_path.write_text("\n".join(lines_out), encoding="utf-8")
    logger.info(f"  [DEBUG] Final order dump → {save_path}")

# ─────────────────────────── Date helpers (unchanged) ────────────────────────
def get_pdf_date_string(dt=None):
    if dt is None:
        dt = datetime.datetime.now()
    return dt.strftime("D:%Y%m%d%H%M%S")

def get_xmp_date_string(dt=None):
    if dt is None:
        dt = datetime.datetime.now()
    return dt.strftime("%Y-%m-%dT%H:%M:%S")

# ─────────────────────────── Font and ICC Setup (unchanged) ──────────────────
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
                elif platform.system() == "Windows":
                    system_profile = os.path.join(
                        os.environ.get("WINDIR", "C:\\Windows"),
                        "System32", "spool", "drivers", "color",
                        "sRGB Color Space Profile.icm",
                    )
                elif platform.system() == "Linux":
                    system_profile = "/usr/share/color/icc/sRGB.icc"
                else:
                    system_profile = None
                if system_profile and Path(system_profile).exists():
                    shutil.copy2(system_profile, str(srgb_path))
                else:
                    urllib.request.urlretrieve("https://www.color.org/srgb.xalter", str(srgb_path))
            except Exception as e:
                logger.warning(f"Could not obtain sRGB ICC profile: {e}")
        return True
    except Exception as e:
        logger.error(f"Failed to setup PDF/A resources: {e}")
        return False

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
</rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""
    except Exception as e:
        logger.error(f"Failed to create XMP metadata: {e}")
        return None

# ─────────────────────────── Model Loading ───────────────────────────────────
def load_models():
    logger.info("=" * 60)
    logger.info("LOADING MODELS")
    logger.info("=" * 60)
    if not setup_pdfa_resources():
        logger.warning("PDF/A resources setup incomplete.")
    logger.info("Loading Surya FoundationPredictor (shared base model)...")
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

# ─────────────────────────── Image Preprocessing (unchanged) ─────────────────
def preprocess_for_ocr(pil_image: Image.Image) -> Image.Image:
    if not ENABLE_PREPROCESSING:
        return pil_image.copy()
    try:
        gray = pil_image.convert("L")
        gray = ImageOps.autocontrast(gray, cutoff=2)
        processed = gray.convert("RGB")
        processed = processed.filter(ImageFilter.SHARPEN)
        return processed
    except Exception as e:
        logger.error(f"Error preprocessing: {e}")
        return pil_image.copy()

def page_to_pil(page: fitz.Page, dpi: int = DPI) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

# ─────────────────────────── TrOCR Recognition (unchanged) ───────────────────
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
            logger.warning("  [SURYA] No results returned from DetectionPredictor.")
            return []
        bboxes = []
        raw_boxes = results[0].bboxes
        logger.debug(f"  [SURYA] Raw box count from model: {len(raw_boxes)}")
        for box in raw_boxes:
            bbox = _bbox_from_surya_box(box)
            if bbox is not None:
                bboxes.append(bbox)
            else:
                logger.debug(f"    [SURYA] Skipped box with unexpected structure: {box}")

        # ── Per-bbox stats ────────────────────────────────────────────────────
        if bboxes:
            widths  = [b[2] - b[0] for b in bboxes]
            heights = [b[3] - b[1] for b in bboxes]
            logger.info(
                f"  [SURYA] Accepted {len(bboxes)} bboxes  "
                f"W: min={min(widths):.0f} max={max(widths):.0f} avg={sum(widths)/len(widths):.0f}  "
                f"H: min={min(heights):.0f} max={max(heights):.0f} avg={sum(heights)/len(heights):.0f}"
            )
        return bboxes
    except Exception as e:
        logger.error(f"  [SURYA] DetectionPredictor failed: {e}")
        import traceback
        traceback.print_exc()
        return []

# ─────────────────────────── Gutter Detection (with debug dump) ──────────────
def _detect_column_gutters_debug(
    line_bboxes: List[List[float]],
    page_width: float,
    hist_dump_path: Path,
) -> List[float]:
    """Identical to original but emits detailed debug prints and histogram dump."""
    logger.info("  ── GUTTER DETECTION ──")
    logger.info(f"  Input: {len(line_bboxes)} bboxes  page_width={page_width:.1f}px")

    if not line_bboxes:
        logger.warning("  No bboxes – cannot detect gutters.")
        return []

    MAX_LINE_WIDTH_FRAC = 0.28
    MIN_GAP_WIDTH_PX    = 10
    MIN_GUTTER_MERGE_PX = 20
    RESOLUTION          = 4

    max_w  = page_width * MAX_LINE_WIDTH_FRAC
    narrow = [b for b in line_bboxes if (b[2] - b[0]) <= max_w]
    wide   = [b for b in line_bboxes if (b[2] - b[0]) >  max_w]
    logger.info(
        f"  Width filter (≤{max_w:.0f}px = {MAX_LINE_WIDTH_FRAC*100:.0f}% of page): "
        f"{len(narrow)} narrow / {len(wide)} wide"
    )
    if len(wide) > 0:
        logger.info(
            f"  Wide-line widths: {[round(b[2]-b[0]) for b in wide[:10]]}"
            + (" ..." if len(wide) > 10 else "")
        )

    if not narrow:
        logger.warning("  No narrow lines – using ALL lines for histogram.")
        narrow = line_bboxes

    hist_w = int(page_width / RESOLUTION) + 2
    hist   = [0] * hist_w
    for x0, y0, x1, y1 in narrow:
        bi = int(((x0 + x1) / 2.0) / RESOLUTION)
        if 0 <= bi < hist_w:
            hist[bi] += 1

    occupied   = sum(1 for v in hist if v > 0)
    peak_val   = max(hist)
    peak_idx   = hist.index(peak_val)
    peak_px    = (peak_idx + 0.5) * RESOLUTION
    logger.info(
        f"  Histogram: max={peak_val} at bucket {peak_idx} (~{peak_px:.0f}px)  "
        f"occupied={occupied}/{hist_w}"
    )

    # ── Zero-gap scanning ────────────────────────────────────────────────────
    NEAR_ZERO = 1
    gaps: List[float] = []
    in_gap = False
    gap_start = 0
    for i, v in enumerate(hist):
        if v <= NEAR_ZERO:
            if not in_gap:
                in_gap = True
                gap_start = i
        else:
            if in_gap:
                gap_w_px = (i - gap_start) * RESOLUTION
                mid_px   = ((gap_start + i - 1) / 2.0) * RESOLUTION
                logger.debug(f"    Gap candidate: bucket {gap_start}→{i}  width={gap_w_px:.0f}px  mid={mid_px:.0f}px")
                if gap_w_px >= MIN_GAP_WIDTH_PX:
                    gaps.append(mid_px)
                in_gap = False
    if in_gap:
        gap_w_px = (hist_w - gap_start) * RESOLUTION
        mid_px   = ((gap_start + hist_w - 1) / 2.0) * RESOLUTION
        if gap_w_px >= MIN_GAP_WIDTH_PX:
            gaps.append(mid_px)

    logger.info(f"  Raw gap candidates ({len(gaps)}): {[round(g) for g in gaps]}")

    # ── Merging ──────────────────────────────────────────────────────────────
    merged: List[float] = []
    for g in sorted(gaps):
        if merged and g - merged[-1] < MIN_GUTTER_MERGE_PX:
            before = merged[-1]
            merged[-1] = (merged[-1] + g) / 2.0
            logger.debug(f"    Merged {before:.0f}px + {g:.0f}px → {merged[-1]:.0f}px")
        else:
            merged.append(g)

    margin = page_width * 0.03
    before_margin = list(merged)
    merged = [g for g in merged if margin < g < page_width - margin]
    removed = set(round(g) for g in before_margin) - set(round(g) for g in merged)
    if removed:
        logger.info(f"  Margin filter removed: {sorted(removed)}")

    logger.info(
        f"  ✓ Final gutters ({len(merged)} → {len(merged)+1} columns): "
        f"{[round(g) for g in merged]}"
    )
    if not merged:
        logger.warning(
            "  ⚠ No gutters found. This may be correct for single-column pages, "
            "or MAX_LINE_WIDTH_FRAC may need raising above "
            f"{MAX_LINE_WIDTH_FRAC} if multi-column is expected."
        )

    # Write histogram dump
    dump_histogram(line_bboxes, page_width, merged, hist_dump_path)

    return merged

# ─────────────────────────── Reading-Order Sort (with debug) ─────────────────
def order_lines_surya_debug(
    line_bboxes: List[List[float]],
    image: Image.Image,
    debug_prefix: str,
) -> List[List[List[float]]]:
    """Full debug wrapper around order_lines_surya."""

    logger.info("  ── READING ORDER ──")

    if not line_bboxes:
        logger.warning("  No bboxes to sort.")
        return []

    page_width  = float(image.width)
    page_height = float(image.height)

    hist_path = Path(f"{debug_prefix}_01_gutter_hist.txt")
    gutters   = _detect_column_gutters_debug(line_bboxes, page_width, hist_path)

    # Gutter visual
    gutter_img_path = Path(f"{debug_prefix}_02_gutters.jpg")
    annotate_gutters(image, gutters, line_bboxes, gutter_img_path)

    if not gutters:
        logger.warning("  No gutters → single-segment top-to-bottom sort.")
        single = [sorted(line_bboxes, key=lambda b: b[1])]
        return [single]

    n_cols    = len(gutters) + 1
    col_edges = [0.0] + gutters + [page_width]
    intervals = [(col_edges[i], col_edges[i + 1]) for i in range(n_cols)]

    MAX_LINE_WIDTH_FRAC = 0.40
    wide_lines   = []
    narrow_lines = []
    for bbox in line_bboxes:
        x0, y0, x1, y1 = bbox
        if (x1 - x0) > page_width * MAX_LINE_WIDTH_FRAC:
            wide_lines.append(bbox)
        else:
            narrow_lines.append(bbox)

    logger.info(
        f"  Wide/narrow split (threshold={MAX_LINE_WIDTH_FRAC*100:.0f}% of {page_width:.0f}px): "
        f"{len(wide_lines)} wide, {len(narrow_lines)} narrow"
    )
    if wide_lines:
        logger.info(
            f"  Wide lines (top→bottom): "
            + str([(round(b[1]), round(b[3])) for b in sorted(wide_lines, key=lambda b: b[1])])
        )

    wide_lines.sort(key=lambda b: b[1])
    band_edges = [0.0] + [wl[1] for wl in wide_lines] + [page_height + 9999.0]

    logger.info(f"  Horizontal band edges: {[round(e) for e in band_edges]}")

    segments: List[List[List[float]]] = []

    for i in range(len(band_edges) - 1):
        band_top    = band_edges[i]
        band_bottom = band_edges[i + 1]
        band_lines  = [
            b for b in narrow_lines
            if band_top <= ((b[1] + b[3]) / 2.0) < band_bottom
        ]

        logger.info(
            f"  Band {i+1}: y=[{band_top:.0f}→{band_bottom:.0f})  "
            f"{len(band_lines)} narrow lines"
        )

        cols = [[] for _ in range(n_cols)]
        for b in band_lines:
            cx    = (b[0] + b[2]) / 2.0
            col_i = n_cols - 1
            for ci, (lo, hi) in enumerate(intervals):
                if lo <= cx < hi:
                    col_i = ci
                    break
            cols[col_i].append(b)

        for ci, col in enumerate(cols):
            col.sort(key=lambda b: b[1])
            logger.info(
                f"    Col {ci+1}: {len(col)} lines  "
                + (f"y=[{col[0][1]:.0f}→{col[-1][3]:.0f}]" if col else "empty")
            )
            if col:
                segments.append(col)

        if i < len(wide_lines):
            wl = wide_lines[i]
            logger.info(
                f"    → inserting wide separator line (y=[{wl[1]:.0f}→{wl[3]:.0f}], "
                f"w={wl[2]-wl[0]:.0f}px)"
            )
            segments.append([wl])

    logger.info(
        f"  ✓ order_lines_surya: {len(line_bboxes)} lines → "
        f"{n_cols} column(s), {len(wide_lines)} wide lines, "
        f"{len(segments)} segments total"
    )

    # Pretty-print segment summary table
    logger.info("")
    logger.info(f"  {'SEG':>4}  {'LINES':>5}  {'Y_TOP':>6}  {'Y_BOT':>6}  NOTE")
    logger.info(f"  {'---':>4}  {'-----':>5}  {'------':>6}  {'------':>6}  ----")
    for si, seg in enumerate(segments):
        if not seg:
            continue
        y_top = min(b[1] for b in seg)
        y_bot = max(b[3] for b in seg)
        note  = "WIDE" if len(seg) == 1 and (seg[0][2]-seg[0][0]) > page_width * MAX_LINE_WIDTH_FRAC else ""
        logger.info(f"  {si+1:>4}  {len(seg):>5}  {y_top:>6.0f}  {y_bot:>6.0f}  {note}")
    logger.info("")

    return segments

# ─────────────────────────── OCR Element Extraction (debug-aware) ─────────────
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
        page_num     = idx + 1
        page_prefix  = str(DEBUG_PATH / f"{stem}_p{page_num:03d}")
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"PAGE {page_num}/{len(pil_images)}  –  {filename}")
        logger.info(f"Image size: {pil_image.width}×{pil_image.height}px")
        logger.info("=" * 70)

        page_segments: List[List[dict]] = []
        filtered          = 0
        page_elements_count = 0

        try:
            # ── Step 0: Surya detection ───────────────────────────────────────
            raw_bboxes = get_surya_lines(pil_image, debug_prefix=page_prefix)
            logger.info(f"  Surya detected {len(raw_bboxes)} text lines on page {page_num}")

            if not raw_bboxes:
                logger.warning("  No text lines detected – saving original for inspection.")
                pil_image.save(f"{page_prefix}_00_EMPTY.jpg", "JPEG", quality=88)
                all_pages.append([])
                continue

            # Save raw bbox image
            annotate_raw_bboxes(pil_image, raw_bboxes,
                                Path(f"{page_prefix}_00_raw_bboxes.jpg"))

            # ── Step 1–2: Reading order + gutters ────────────────────────────
            sorted_segments = order_lines_surya_debug(
                raw_bboxes, pil_image, page_prefix
            )

            # Flatten segments for the visual (we need List[List[bbox]])
            annotate_segments(
                pil_image,
                sorted_segments,
                Path(f"{page_prefix}_03_segments.jpg"),
            )

            # ── Step 3: OCR ──────────────────────────────────────────────────
            ocr_image = preprocess_for_ocr(pil_image)
            logger.info(f"  Beginning TrOCR on {sum(len(s) for s in sorted_segments)} line images…")

            noise_reasons: List[str] = []

            for seg_idx, segment_bboxes in enumerate(sorted_segments):
                segment_elements: List[dict] = []
                for i, bbox in enumerate(segment_bboxes):
                    try:
                        x0, y0, x1, y1 = bbox
                        sh, sw = y1 - y0, x1 - x0
                        if sh < 5 or sw < 5:
                            filtered += 1
                            noise_reasons.append(
                                f"  Seg{seg_idx+1}/Line{i+1}: SKIP (too small {sw:.0f}×{sh:.0f})"
                            )
                            continue
                        line_img = ocr_image.crop((x0, y0, x1, y1))
                        text, confidence = recognize_text_with_trocr(
                            line_img, processor, trocr_model
                        )
                        noise = is_likely_noise(text, confidence, sh, sw)
                        if noise:
                            filtered += 1
                            noise_reasons.append(
                                f"  Seg{seg_idx+1}/Line{i+1}: NOISE "
                                f"conf={confidence:.2f} text='{text[:40]}'"
                            )
                            continue
                        segment_elements.append({
                            "x0":         x0,
                            "y_baseline": y1,
                            "font_size":  max(6, min(sh * 0.9, 72)),
                            "text":       text,
                            "confidence": confidence,
                        })
                        logger.debug(
                            f"    Seg{seg_idx+1}/Line{i+1}: "
                            f"conf={confidence:.2f}  '{text[:60]}'"
                        )
                    except Exception as e:
                        logger.error(f"  Error Seg{seg_idx+1}/Line{i+1}: {e}")

                if segment_elements:
                    page_segments.append(segment_elements)
                    page_elements_count += len(segment_elements)

            # Print filtered lines
            if noise_reasons:
                logger.info(f"  Filtered lines ({len(noise_reasons)} total):")
                for r in noise_reasons:
                    logger.info(r)

            logger.info(
                f"  Page {page_num} summary: "
                f"{page_elements_count} accepted | "
                f"{filtered} filtered | "
                f"{len(page_segments)} segments with text"
            )

            # Save ordered text dump
            dump_final_order(
                page_segments,
                Path(f"{page_prefix}_04_final_order.txt"),
                page_num,
                filename,
            )

            total_elements += page_elements_count
        except Exception as e:
            logger.error(f"OCR failed for page {page_num}: {e}")
            import traceback
            traceback.print_exc()

        all_pages.append(page_segments)

    logger.info("")
    logger.info("=" * 70)
    logger.info(
        f"OCR complete: {total_elements} total elements across "
        f"{len(pil_images)} pages"
    )
    logger.info("=" * 70)
    return all_pages

# ─────────────────────────── PDF/A Compliance (unchanged) ────────────────────
def setup_pdfa_compliance(pdf_path: str):
    try:
        srgb_path = Path(SRGB_ICC_PATH)
        if not srgb_path.exists():
            logger.error("sRGB ICC not found; skipping PDF/A OutputIntent.")
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
            logger.info("PDF/A OutputIntent embedded.")
    except Exception as e:
        logger.error(f"PDF/A compliance failed: {e}")

# ─────────────────────────── PDF Processing ──────────────────────────────────
def process_single_pdf_ocr(input_path: str, output_path: str) -> bool:
    filename = os.path.basename(input_path)
    stem     = Path(input_path).stem
    logger.info(f"\nStarting OCR for: {filename}")
    try:
        with fitz.open(input_path) as doc:
            logger.info(f"Rendering {len(doc)} pages at {DPI} DPI…")
            pil_images: List[Image.Image] = []
            for page in doc:
                pil_images.append(page_to_pil(page, dpi=DPI))

            ocr_pages = create_ocr_text_elements_debug(pil_images, filename, stem)

            now           = datetime.datetime.now()
            creation_date = get_pdf_date_string(now)
            doc.set_metadata({
                "title":        filename,
                "author":       "Opticolumn",
                "subject":      "OCR processed document",
                "creator":      "Opticolumn 2026",
                "producer":     "PyMuPDF",
                "creationDate": creation_date,
                "modDate":      creation_date,
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
            logger.info(f"Inserting text into {page_count} pages…")

            for page_num in range(page_count):
                page     = doc[page_num]
                segments = ocr_pages[page_num]
                pil_img  = pil_images[page_num]

                existing_text = page.get_text().strip()
                if existing_text:
                    logger.info(f"  Page {page_num+1}: removing existing text layer.")
                    page.add_redact_annot(page.rect)
                    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

                img_w, img_h = pil_img.size
                page_w, page_h = page.rect.width, page.rect.height
                sx, sy = page_w / img_w, page_h / img_h

                text_writer   = fitz.TextWriter(page.rect)
                total_inserted = 0

                for seg_idx, segment in enumerate(segments):
                    for elem in segment:
                        try:
                            text_writer.append(
                                fitz.Point(elem["x0"] * sx, elem["y_baseline"] * sy),
                                elem["text"],
                                font=fitz.Font(FONT_NAME),
                                fontsize=max(4, elem["font_size"] * sy),
                            )
                            total_inserted += 1
                        except Exception as e:
                            logger.error(f"  Text insert failed p{page_num+1}: {e}")

                if total_inserted > 0:
                    text_writer.write_text(
                        page, overlay=True,
                        render_mode=3, color=(0, 0, 0),
                    )
                logger.info(
                    f"  Page {page_num+1}: inserted {total_inserted} elements "
                    f"across {len(segments)} segments"
                )

            doc.save(output_path, deflate=True, garbage=4, clean=True,
                     deflate_images=False, encryption=fitz.PDF_ENCRYPT_KEEP)
            logger.info(f"Saved: {output_path}")

            if Path(SRGB_ICC_PATH).exists():
                setup_pdfa_compliance(output_path)

            # Verify
            with fitz.open(output_path) as final_pdf:
                total_chars = 0
                for i, pg in enumerate(final_pdf):
                    chars = len(pg.get_text().strip())
                    total_chars += chars
                    logger.info(f"  Final page {i+1}: {chars} extractable chars")
                status = "SUCCESS" if total_chars > 0 else "⚠ NO TEXT"
                logger.info(f"  {status}: {total_chars} total chars in final PDF")

            return True
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ─────────────────────────── Compression (unchanged) ─────────────────────────
def compress_to_target_size(input_pdf: Path, output_pdf: Path,
                            original_size: int) -> Path:
    max_target = int(original_size * 1.15)
    current_size = input_pdf.stat().st_size
    logger.info(
        f"Compression: {current_size//1024}KB → target ≤{max_target//1024}KB "
        f"(original {original_size//1024}KB)"
    )
    if current_size <= max_target:
        shutil.copy2(input_pdf, output_pdf)
        logger.info("  Already within target – no compression needed.")
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
            pct = (compressed_size - original_size) / original_size * 100
            logger.info(f"  Option {i+1}: {compressed_size//1024}KB ({pct:+.1f}%)")
            if compressed_size <= max_target:
                try:
                    with fitz.open(str(temp_out)) as chk:
                        total_chars = sum(len(pg.get_text().strip()) for pg in chk)
                except Exception:
                    total_chars = -1
                if total_chars > 0:
                    shutil.move(str(temp_out), str(output_pdf))
                    logger.info(f"  Option {i+1} accepted; {total_chars} chars preserved.")
                    return output_pdf
                else:
                    logger.error("  OCR lost after compression – keeping uncompressed.")
                    temp_out.unlink(missing_ok=True)
                    shutil.copy2(input_pdf, output_pdf)
                    return output_pdf
            else:
                temp_out.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"  Compression option {i+1} failed: {e}")
            temp_out.unlink(missing_ok=True)
    logger.warning("  All options exceeded budget – returning as-is.")
    shutil.copy2(input_pdf, output_pdf)
    return output_pdf

# ─────────────────────────── Main ────────────────────────────────────────────
def main():
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║              OPTICOLUMN  –  DEBUG MODE                  ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info(f"Debug output directory : {DEBUG_PATH.resolve()}")
    logger.info(f"Input directory        : {INPUT_DIR}")
    logger.info(f"Output directory       : {OUTPUT_DIR}")
    logger.info(f"TrOCR model            : {TROCR_MODEL_NAME}")
    logger.info(f"DPI                    : {DPI}")
    logger.info("")

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

    logger.info(f"Found {len(pdf_files)} PDF file(s) to process.")
    for pdf_path in pdf_files:
        original_size = pdf_path.stat().st_size
        logger.info(f"\n{'━'*60}")
        logger.info(f"FILE: {pdf_path.name}  ({original_size//1024} KB)")
        logger.info(f"{'━'*60}")

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
                f"\n✓ SUCCESS: {result_path.name} | "
                f"{final_size//1024} KB ({size_increase:+.1f}% from original)"
            )
        else:
            logger.error(f"Failed to generate final output for {pdf_path.name}")

        try:
            ocr_temp_path.unlink()
        except Exception as e:
            logger.warning(f"Could not delete temp file: {e}")

    logger.info(f"\nAll done. Debug artefacts in: {DEBUG_PATH.resolve()}")
    logger.info(f"Final PDFs in            : {output_folder.resolve()}")

    # ── Print index of all debug files created ────────────────────────────
    debug_files = sorted(DEBUG_PATH.iterdir())
    logger.info(f"\nDebug files written ({len(debug_files)}):")
    for f in debug_files:
        size_kb = f.stat().st_size // 1024
        logger.info(f"  {f.name:<55} {size_kb:>6} KB")

if __name__ == "__main__":
    main()