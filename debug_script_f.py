#!/usr/bin/env python3

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

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

try:
    import pymupdf as fitz      # current PyMuPDF (avoids the deprecated 'fitz' alias)
except ImportError:             # older PyMuPDF releases
    import fitz
import pikepdf
from PIL import Image, ImageCms, ImageDraw, ImageFilter, ImageFont, ImageOps

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
# TrOCR does not expose per-token probabilities in the same way as Surya;
# we approximate confidence from the softmax of the output scores.
CONFIDENCE_THRESHOLD             = 0.25   # minimum mean token confidence
SINGLE_CHAR_CONFIDENCE_THRESHOLD = 0.50   # tighter threshold for 1-char results
MIN_LINE_H                       = 8      # px — skip lines shorter than this
MIN_LINE_W                       = 15     # px — skip lines narrower than this

# TrOCR generation cap (tokens).  Without an explicit cap, long column lines can
# be truncated by the checkpoint's default generation length.
MAX_NEW_TOKENS = 192

# ── Recall sweep (safety-net for text the layout model never boxed) ──────────
# After region OCR, the whole page is tiled with overlap; every detected text
# line that is not already covered by an existing element is OCR'd and added.
SWEEP_ENABLED        = True
SWEEP_TILE           = 1920   # px per tile side (300 DPI ≈ 6.4 in)
SWEEP_OVERLAP        = 640    # px; must exceed the longest line the sweep should recover
SWEEP_EDGE_MARGIN    = 6      # px; boxes touching an interior tile edge are ignored
SWEEP_COVERED_FRAC   = 0.50   # candidate counts as "already read" above this overlap
SWEEP_MIN_CONFIDENCE = 0.40   # stricter than CONFIDENCE_THRESHOLD: sweep also sees photos

# ── Debug images ──────────────────────────────────────────────────────────────
# True  → one rolling set of debug JPEGs (latest_*.jpg), overwritten by every
#         page, so the debug folder does not grow with batch size.
# False → separate JPEGs for every file/page.
# Text reports (.txt) are always kept per page (they are tiny).
DEBUG_OVERWRITE_IMAGES = True

# ── Layout label taxonomy ─────────────────────────────────────────────────────
# OCR_LABELS  → segment with DetectionPredictor, recognise with TrOCR
# SKIP_LABELS → no region-level OCR.  Text embedded in these (e.g. an
#               advertisement boxed as a Picture) is picked up by the sweep.
# Any label in NEITHER set is OCR'd by default.
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
MIN_FONT_PT         = 4.0    # smallest initial font size for the hidden text
MIN_CLAMPED_FONT_PT = 1.5    # floor when shrinking a line to fit its segment; the
                             # text is invisible, so legibility is irrelevant —
                             # only alignment with the printed line matters
FONT_NAME     = "helv"                       # fallback (non-embedded) font only
FONT_PATH     = "fonts/FreeSans.ttf"
FONT_URL      = ("https://github.com/opensourcedesign/fonts/raw/master/"
                 "gnu-freefont_freesans/FreeSans.ttf")
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


def setup_pdfa_compliance(pdf_path: str) -> None:
    """
    Embed the sRGB OutputIntent into an already-saved PDF with pikepdf.
    MUST run after the file is on disk.

      - skipped if the PDF already has a GTS_PDFA1 OutputIntent;
      - object streams are disabled on save (not permitted in PDF/A-1).
    """
    try:
        icc = Path(SRGB_ICC_PATH)
        if not icc.exists() or not _valid_icc(icc):
            logger.warning("  Valid sRGB ICC profile not found; PDF/A OutputIntent skipped.")
            return
        with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
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
            pdf.save(pdf_path, object_stream_mode=pikepdf.ObjectStreamMode.disable)
    except Exception as exc:
        logger.error(f"  Failed to set up PDF/A compliance: {exc}")


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
    device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                max_new_tokens=MAX_NEW_TOKENS,
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

    When both passes yield results, the one with more accepted CHARACTERS wins
    (line count is a poor proxy: Pass 2 always yields a single "line").

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

def _coverage(box: List[float], elements: List[Dict]) -> float:
    """Fraction of `box` already lying inside existing elements (0–1)."""
    area  = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    total = 0.0
    for e in elements:
        b  = e["bbox"]
        iw = min(box[2], b[2]) - max(box[0], b[0])
        ih = min(box[3], b[3]) - max(box[1], b[1])
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


def sweep_uncovered_text(
    page_image: Image.Image,
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
      re-read the same line.
    - Recovered lines use SWEEP_MIN_CONFIDENCE (stricter than region OCR)
      because the sweep also sees photographs and halftone.
    - Recovered lines inherit the reading position of the nearest layout
      region and are labelled "Recovered".

    Coordinates are in the same space as `page_image` (full-page pixels).
    """
    iw, ih = page_image.size
    m      = SWEEP_EDGE_MARGIN
    recovered: List[Dict] = []
    known: List[Dict]     = list(elements)

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
                box = [lx0 + tx, ly0 + ty, lx1 + tx, ly1 + ty]
                if _coverage(box, known) >= SWEEP_COVERED_FRAC:
                    continue
                line = page_image.crop(tuple(int(v) for v in box))
                text, conf = _trocr_read(line, trocr_processor, trocr_model)
                if _is_noise(text, conf, int(lh), int(lw), min_conf=SWEEP_MIN_CONFIDENCE):
                    logger.debug(
                        f"      [SWEEP-NOISE] conf={conf:.2f} "
                        f"{int(lw)}×{int(lh)}px | {text[:40]}"
                    )
                    continue
                elem = {
                    "text":             text,
                    "bbox":             box,
                    "confidence":       conf,
                    "font_size":        max(6.0, min(lh * 0.85, 72.0)),
                    "source_label":     "Recovered",
                    "reading_position": _nearest_position(box, layout_regions),
                }
                recovered.append(elem)
                known.append(elem)
    return recovered


# ══════════════════════════════════════════════════════════════════════════════
# DEBUG VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _debug_img_path(stem: str, page_num: int, tag: str) -> Path:
    """
    Path for a debug JPEG.

    DEBUG_OVERWRITE_IMAGES = True  → fixed names ("latest_<tag>.jpg"): every
        page overwrites the previous page's images, so the debug folder holds
        at most one page's worth of JPEGs however large the batch is.
    DEBUG_OVERWRITE_IMAGES = False → per-page names.

    Text reports (.txt) are not affected; they stay per page.
    """
    if DEBUG_OVERWRITE_IMAGES:
        return DEBUG_PATH / f"latest_{tag}.jpg"
    return DEBUG_PATH / f"{stem}_p{page_num:03d}_{tag}.jpg"


def clear_debug_images() -> None:
    """Delete debug JPEGs left over from earlier runs (overwrite mode only)."""
    if not DEBUG_OVERWRITE_IMAGES:
        return
    removed = 0
    for f in DEBUG_PATH.glob("*.jpg"):
        try:
            f.unlink()
            removed += 1
        except OSError:
            pass
    if removed:
        logger.info(f"  Cleared {removed} stale debug image(s).")


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

    1.  Preprocess  — tiled CLAHE-approx + unsharp mask
    2.  Layout      — LayoutPredictor → semantic regions + reading order positions
    3.  OCR         — DetectionPredictor (line segmentation) + TrOCR per line,
                      for every region not in SKIP_LABELS
    3b. Sweep       — tile the page, OCR any detected line no region claimed
    4.  Sort        — by Surya layout position, then vertical baseline within region

    Returns a flat list of element dicts in reading order.
    """
    pfx = DEBUG_PATH / f"{stem}_p{page_num:03d}"      # used for the (small) .txt reports
    logger.info("")
    logger.info("─" * 62)
    logger.info(f"  PAGE {page_num}  [{pil_image.width}×{pil_image.height}]  {filename}")
    logger.info("─" * 62)

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
                           Path(str(pfx) + "_01_layout_report.txt"),
                           filename, page_num)

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
            processed, region,
            det_predictor, trocr_processor, trocr_model,
        )
        logger.debug(f"      → {len(elems)} element(s) accepted.")
        all_elements.extend(elems)

    # ── Stage 3b: Recall sweep ────────────────────────────────────────────────
    if SWEEP_ENABLED:
        recovered = sweep_uncovered_text(
            processed, all_elements, layout_regions,
            det_predictor, trocr_processor, trocr_model,
        )
        logger.info(
            f"  [SWEEP] recovered {len(recovered)} line(s) the layout stage missed."
        )
        all_elements.extend(recovered)

    # ── Stage 4: Final reading order sort ────────────────────────────────────
    all_elements.sort(key=lambda e: (e["reading_position"], e["bbox"][1]))

    logger.info(f"  [RESULT] {len(all_elements)} OCR element(s) on page {page_num}.")

    if all_elements:
        save_ocr_debug(pil_image, all_elements,
                       _debug_img_path(stem, page_num, "02_ocr_overlay"))
        save_ocr_report(all_elements,
                        Path(str(pfx) + "_02_ocr_report.txt"),
                        filename, page_num)

    return all_elements


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
    Insert `elements` into `page` as invisible text (render mode 3).

    Pixel coordinates (from the OCR render) are scaled to PDF points.  Each
    line's font size is clamped so the string never extends past the right
    edge of its own segment (or the page), keeping search hits and text
    selection aligned with the printed line.

    Returns the number of elements inserted.
    """
    iw, ih = img_size
    pw, ph = page.rect.width, page.rect.height
    sx, sy = pw / iw, ph / ih

    logger.debug(f"    {iw}×{ih}px → {pw:.1f}×{ph:.1f}pt  (sx={sx:.4f}, sy={sy:.4f})")

    writer   = fitz.TextWriter(page.rect)
    inserted = 0

    for elem in elements:
        bx0, by0, bx1, by1 = elem["bbox"]
        try:
            x0_pt       = bx0 * sx
            x1_pt       = bx1 * sx
            baseline_pt = by1 * sy
            fontsize    = max(MIN_FONT_PT, elem["font_size"] * sy)

            available_w = min(x1_pt, pw) - x0_pt
            if available_w > 0:
                text_w = font.text_length(elem["text"], fontsize=fontsize)
                if text_w > available_w:
                    fontsize = max(MIN_CLAMPED_FONT_PT, fontsize * available_w / text_w)

            writer.append(
                fitz.Point(x0_pt, baseline_pt),
                elem["text"],
                font=font,
                fontsize=fontsize,
            )
            inserted += 1
        except Exception as exc:
            logger.debug(f"    Text insert skipped: {exc}")

    if inserted:
        writer.write_text(page, overlay=True, render_mode=3, color=(0, 0, 0))
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
         text layer → insert the new invisible text layer.
      3. Write Info-dict + XMP metadata and accessibility catalog entries.
      4. Subset the embedded font (if fonttools is installed) and save once
         with deflate.
      5. Add the PDF/A sRGB OutputIntent (pikepdf) and verify extractable text.

    Returns True if the output PDF contains extractable text.
    """
    filename = os.path.basename(input_path)
    stem     = Path(input_path).stem

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
                    elements = process_page(
                        pil_img, page_num, filename, stem,
                        det_predictor, layout_predictor,
                        trocr_processor, trocr_model,
                    )
                except Exception as exc:
                    logger.error(f"  OCR failed for page {page_num}: {exc}")
                    import traceback; traceback.print_exc()
                    elements = []
                del pil_img

                if not elements:
                    logger.info(f"  Page {page_num}: no elements.")
                    continue

                # ── Strip any pre-existing text layer ─────────────────────────
                # Previously OCR'd / partially searchable scans would otherwise
                # end up with a doubled text layer.  A page-sized redaction is
                # applied with PDF_REDACT_IMAGE_NONE so image streams are
                # untouched — only text operators are removed.
                existing_text = page.get_text().strip()
                if existing_text:
                    logger.info(
                        f"  Page {page_num}: removing existing text layer "
                        f"({len(existing_text)} chars) before inserting OCR."
                    )
                    page.add_redact_annot(page.rect)
                    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

                inserted = insert_text_layer(page, elements, img_size, font)
                logger.info(
                    f"  Page {page_num}: inserted {inserted}/{len(elements)} element(s)."
                )

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
                garbage=4,             # remove unused objects
                clean=True,
                deflate_images=False,  # leave original image streams untouched
                encryption=fitz.PDF_ENCRYPT_KEEP,
            )
            logger.info(f"  Saved: {output_path}")

        # ── PDF/A OutputIntent — MUST run after the file is on disk ───────────
        setup_pdfa_compliance(output_path)

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
    budget AND still contains extractable text.
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
    logger.info(f"  Sweep    : {'on' if SWEEP_ENABLED else 'off'}")
    logger.info(
        f"  Debug img: {'overwrite (latest_*.jpg)' if DEBUG_OVERWRITE_IMAGES else 'per page'}"
    )
    logger.info("")

    input_folder  = Path(INPUT_DIR)
    output_folder = Path(OUTPUT_DIR)

    if not input_folder.exists():
        logger.error(f"Input folder '{INPUT_DIR}' not found.")
        sys.exit(1)
    output_folder.mkdir(exist_ok=True)
    clear_debug_images()

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