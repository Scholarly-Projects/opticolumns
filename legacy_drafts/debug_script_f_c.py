#!/usr/bin/env python3
"""
Opticolumns  –  debug_script_g.py
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
from typing import Callable, Dict, List, Optional, Tuple
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
OPT_NAMESPACE = "hhttps://github.com/Scholarly-Projects/opticolumns"

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

CONFIDENCE_THRESHOLD             = 0.25   # minimum mean token confidence
SINGLE_CHAR_CONFIDENCE_THRESHOLD = 0.50   # tighter threshold for 1-char results
MIN_LINE_H                       = 5      # px — skip lines shorter than this
MIN_LINE_W                       = 10     # px — skip lines narrower than this
SPARSE_LINE_WIDTH_RATIO          = 2.0    # width / (height * char count) above this

# Line-level plausibility (new).  These judge the OCR output against the
# physical size of its box, which TrOCR's language model cannot fake.
SHORT_TEXT_MAX_CHARS = 3      # outputs this short ...
SHORT_TEXT_MAX_AR    = 6.0    # ... on a box wider than this × its height are noise
                              # ("to", "0", "#" read off a 700-px smeared line)
MIN_CHAR_W_RATIO     = 0.18   # width / (height * chars) below this = more characters
                              # than can physically fit (hallucinated run-on text)
LINE_MAX_JUNK_RATE   = 0.50   # a line whose tokens are at least this share junk is noise

# TrOCR generation cap (tokens).  Without an explicit cap, long column lines can
# be truncated by the checkpoint's default generation length.
MAX_NEW_TOKENS = 192

# Pass 2 (whole-crop read) for non-single-block labels is only attempted when
# the detector found no lines at all AND the region is line-shaped (w/h at
# least this).  A tall multi-line body block read as one image only yields a
# hallucinated sentence.
PASS2_MIN_AR = 2.0

# ── Legibility vetting (second-stage quality gate, per region) ───────────────
VET_ENABLED  = True
VET_DRY_RUN  = False   # True → compute and report verdicts, but drop nothing.
                       # Run once like this and read latest_04_vetting_report.txt
                       # to calibrate the thresholds below on your own scans.

VET_STRIKES_TO_REJECT     = 2      # indicators that must fail before a region is dropped
VET_MIN_LINES             = 3      # line-based indicators need at least this many lines
VET_MAX_LINE_REJECT_RATE  = 0.50   # strike: more than this share of lines were rejected
VET_MIN_CHAR_YIELD        = 0.45   # strike: chars read / chars that fit below this
CHAR_W_FACTOR             = 0.50   # average character width as a fraction of line height
VET_MIN_TOKENS_FOR_JUNK   = 6      # junk indicator needs at least this many tokens
VET_MAX_JUNK_RATE         = 0.20   # strike: junk-token share above this
VET_MIN_WORDS             = 8      # lexical indicator needs at least this many words
VET_MIN_LEXICAL_RATE      = 0.65   # strike: dictionary-word share below this
VET_MAX_RUN_ANISOTROPY    = 1.30   # strike: horizontal/vertical ink-run ratio above this
ANISO_MIN_PIXELS          = 2000   # crops smaller than this are not measured
VET_EXEMPT_LABELS: set    = set()  # labels never dropped by vetting, e.g. {"Page-header"}

# Recall-sweep lines have no region context, so each is vetted on its own
# (a single failing indicator rejects it — the sweep is the riskiest source).
SWEEP_LEX_MIN_WORDS     = 4
SWEEP_MIN_LEXICAL_RATE  = 0.50

# Short strings repeated many times on one page are a hallucination signature.
REPEAT_SHORT_TEXT_MAX_LEN   = 12
REPEAT_SHORT_TEXT_MIN_COUNT = 3

# ── Lexicon for the dictionary-word indicator ────────────────────────────────
# Priority: LEXICON_PATH (one word per line) → `pip install wordfreq` →
# system word list.  If none is available the lexical indicator is skipped.
LEXICON_PATH: Optional[str] = None
LEXICON_SYSTEM_PATHS = ["/usr/share/dict/words", "/usr/dict/words"]
LEXICON_MIN_ZIPF     = 1.0    # wordfreq: minimum Zipf frequency counted as a real word

# ── Recall sweep (safety-net for text the layout model never boxed) ──────────
SWEEP_ENABLED        = True
SWEEP_TILE           = 1920   # px per tile side (300 DPI ≈ 6.4 in)
SWEEP_OVERLAP        = 960    # px; must exceed the longest line the sweep should recover
SWEEP_EDGE_MARGIN    = 6      # px; boxes touching an interior tile edge are ignored
SWEEP_COVERED_FRAC   = 0.50   # candidate counts as "already read" above this overlap
SWEEP_MIN_CONFIDENCE = 0.40   # stricter than CONFIDENCE_THRESHOLD: sweep also sees photos

# ── Column bands (give recovered lines a sensible reading position) ──────────
COLUMN_REGION_MAX_W_FRAC = 0.40
COLUMN_GAP_MIN           = 150

MIN_UNCLAIMED_COLUMN_LINES        = 5
MIN_UNCLAIMED_COLUMN_HEIGHT_FRAC  = 0.25

ROW_OVERLAP_FRACTION = 0.5
ROW_MAX_X_OVERLAP    = 0.5

# ── Banner/masthead reading-order correction ──────────────────────────────────

BANNER_BAND_OVERLAP_THRESHOLD = 0.18
BANNER_BAND_MIN_GAP_PX        = max(10, round(DPI * 0.06))

SWEEP_COVERAGE_PAD = 6

# ── Coverage audit (diagnostic; never changes the output) ────────────────────

AUDIT_ENABLED     = True
AUDIT_SCALE       = 8
AUDIT_INK_LEVEL   = 215
AUDIT_WARN_FRAC   = 0.08
AUDIT_MIN_BAND_PX = 150

# ── Debug output ──────────────────────────────────────────────────────────────
DEBUG_OVERWRITE = True

# ── Layout label taxonomy ─────────────────────────────────────────────────────

OCR_LABELS = {
    "Text", "Section-header", "Caption", "Footnote", "List-item",
    "Page-footer", "Page-header", "Table-of-contents", "Handwriting",
    "Text-inline-math", "Formula", "Table", "Form",
}
SKIP_LABELS = {
    "Picture",
    "Figure",
}

SINGLE_BLOCK_LABELS = {
    "Section-header",
    "Page-header",
    "Caption",
    "Footnote",
}

HEADER_LABELS = {
    "Section-header",
    "Page-header",
}

FURNITURE_LABELS = HEADER_LABELS | {"Page-footer", "Table-of-contents"}

DEDUPE_CONTAINMENT = 0.5

DEDUPE_MAX_AREA_RATIO = 3.0

HEADER_AUTOCONTRAST_CUTOFF = 1

MAX_HEADER_AR          = 6.0    # width / height threshold that triggers splitting
HEADER_SEGMENT_OVERLAP = 0.20   # overlap used only when no blank gap can be found
HEADER_CUT_SEARCH_FRAC = 0.35   # look for a blank gap in the last 35 % of each segment
HEADER_CUT_MIN_GAP_PX  = 3      # narrowest blank column run accepted as a cut point

ELEMENT_SEPARATOR = " "

MIN_REGION_W = 40
MIN_REGION_H = 15

# ── PDF/A font & colour-profile resources ─────────────────────────────────────
EMBED_FONT    = True
MIN_FONT_PT         = 4.0
MIN_CLAMPED_FONT_PT = 1.5
FONT_NAME     = "helv"
FONT_PATH     = "fonts/FreeSans.ttf"
FONT_URL      = ("https://github.com/opensourcedesign/fonts/raw/master/"
                 "gnu-freefont_freesans/FreeSans.ttf")
SRGB_ICC_PATH = "srgb.icc"

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
    "Recovered":         "#FFD600",
}
DEFAULT_COLOUR = "#9E9E9E"
REJECTED_COLOUR = (220, 0, 0)


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
    Light-touch preprocessing for a bold/stylized header crop: autocontrast
    only, taken from the un-sharpened render.
    """
    try:
        gray = crop.convert("L")
        gray = ImageOps.autocontrast(gray, cutoff=HEADER_AUTOCONTRAST_CUTOFF)
        return gray.convert("RGB")
    except Exception as exc:
        logger.warning(f"Header preprocessing fallback: {exc}")
        return crop.convert("RGB")


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
    Make sure the two PDF/A resources exist locally (FreeSans font, sRGB ICC).
    """
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
    """Return (font, using_freesans)."""
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
    """Build the XMP packet (Dublin Core, PDF/A-1b id, opt: extension schema)."""
    try:
        t, a, s, c, p, lang = (
            xml_escape(str(v)) for v in (title, author, subject, creator, producer, language)
        )
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
    """Write Info-dict metadata, XMP, /Lang and /DisplayDocTitle."""
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

    cat = doc.pdf_catalog()
    doc.xref_set_key(cat, "ViewerPreferences", "<</DisplayDocTitle true>>")
    doc.xref_set_key(cat, "Lang", f"({DOC_LANGUAGE})")


def setup_pdfa_compliance(pdf_path: str) -> None:
    """Embed the sRGB OutputIntent into an already-saved PDF with pikepdf."""
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
# LEXICON  (dictionary-word indicator for legibility vetting)
# ══════════════════════════════════════════════════════════════════════════════

_LEXICON: Optional[set] = None
_WORDFREQ: Optional[Callable] = None
_LEXICON_SOURCE = "none"

# Inflection fallbacks for plain word lists (e.g. macOS Webster's 2nd, which
# omits most plurals and verb forms).
_SUFFIXES = (("'s", ""), ("ies", "y"), ("es", ""), ("s", ""), ("ed", ""), ("ed", "e"),
             ("ing", ""), ("ing", "e"), ("ly", ""), ("er", ""), ("est", ""))


def load_lexicon() -> None:
    """Load the word source used by _is_word(); logs which one was found."""
    global _LEXICON, _WORDFREQ, _LEXICON_SOURCE

    def _read(path: str) -> Optional[set]:
        try:
            words = {w.strip().lower() for w in
                     Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
                     if w.strip()}
            return words if len(words) > 1000 else None
        except Exception:
            return None

    if LEXICON_PATH and Path(LEXICON_PATH).exists():
        lex = _read(LEXICON_PATH)
        if lex:
            _LEXICON, _LEXICON_SOURCE = lex, f"{LEXICON_PATH} ({len(lex):,} words)"
    if _LEXICON is None:
        try:
            from wordfreq import zipf_frequency
            _WORDFREQ, _LEXICON_SOURCE = zipf_frequency, "wordfreq"
        except ImportError:
            for p in LEXICON_SYSTEM_PATHS:
                if Path(p).exists():
                    lex = _read(p)
                    if lex:
                        _LEXICON, _LEXICON_SOURCE = lex, f"{p} ({len(lex):,} words)"
                        break
    if _lexicon_available():
        logger.info(f"  Lexicon: {_LEXICON_SOURCE}")
    else:
        logger.warning(
            "  No lexicon found — the dictionary-word vetting indicator is disabled. "
            "`pip install wordfreq` or set LEXICON_PATH to enable it."
        )


def _lexicon_available() -> bool:
    return _WORDFREQ is not None or _LEXICON is not None


def _is_word(w: str) -> bool:
    w = w.lower()
    if _WORDFREQ is not None:
        try:
            return _WORDFREQ(w, "en") >= LEXICON_MIN_ZIPF
        except Exception:
            return False
    if _LEXICON is None:
        return False
    if w in _LEXICON:
        return True
    for suf, rep in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 2 and (w[:-len(suf)] + rep) in _LEXICON:
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# TEXT PLAUSIBILITY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_PUNCT = ".,;:!?\"'()[]{}-–—*_`"

# Classic TrOCR filler on illegible input: "0", "000", "0.000", "1.000".
_JUNK_NUMERIC = re.compile(r"^(0+|0[.,]\d+|\d\.000)$")


def _norm_word(w: str) -> str:
    return w.strip(_PUNCT).lower()


def _token_stats(texts: List[str]) -> Dict[str, int]:
    """
    Token counts over `texts`:
      tokens — all whitespace tokens
      junk   — tokens with no letters/digits ("#", "—") or filler numerics
      words  — purely alphabetic tokens of 2+ letters
      hits   — words found in the lexicon
    Numbers, initials and mixed tokens are neutral (neither junk nor words).
    """
    tokens = words = hits = junk = 0
    for text in texts:
        for t in text.split():
            tokens += 1
            core = t.strip(_PUNCT)
            if not core or not any(c.isalnum() for c in core) or _JUNK_NUMERIC.match(core):
                junk += 1
                continue
            alpha = core.replace("'", "")
            if len(alpha) >= 2 and alpha.isalpha():
                words += 1
                if _is_word(alpha):
                    hits += 1
    return {"tokens": tokens, "words": words, "hits": hits, "junk": junk}


def _is_loop(text: str) -> bool:
    """True for decoder repetition loops ("for # for your own ... your own")."""
    toks = [_norm_word(t) or t for t in text.split()]
    if len(toks) < 4:
        return False
    bigrams = Counter(zip(toks, toks[1:]))
    if sum(1 for n in bigrams.values() if n >= 2) >= 2 or any(n >= 3 for n in bigrams.values()):
        return True
    top = Counter(toks).most_common(1)[0][1]
    return top >= 3 and top / len(toks) >= 0.4


def _is_garbage_text(raw: str) -> bool:
    """Content-only rejection of a raw TrOCR string (before cleaning)."""
    if not raw.strip() or not any(c.isalnum() for c in raw):
        return True
    toks = raw.split()
    if len(toks) >= 3:
        s = _token_stats([raw])
        if s["junk"] / s["tokens"] >= LINE_MAX_JUNK_RATE:
            return True
    return _is_loop(raw)


def _clean_text(text: str) -> str:
    """
    Remove standalone '#' tokens (TrOCR's usual placeholder for an illegible
    blob) and collapse an immediately doubled word ("THE THE" → "THE").
    Note: this also collapses a genuine "had had" — an accepted trade-off.
    """
    out: List[str] = []
    for t in text.split():
        if not t.strip("#"):
            continue
        core = _norm_word(t)
        if out and len(core) >= 2 and core == _norm_word(out[-1]):
            continue
        out.append(t)
    return " ".join(out)


def _boundary_fix(prev_words: List[str], next_words: List[str]) -> Tuple[List[str], List[str]]:
    """
    Resolve a word read twice across a boundary where two reads physically
    overlap (overlapping crop segments, or two overlapping elements):

      "Past"    | "Pastime."  → drop "Past"      (truncated read of the same word)
      "Pastime" | "time."     → drop "time."
      "the"     | "the"       → drop the second
      "Pasti"   | "stime."    → merge to "Pastime."  (≥3-char overlap)

    Only call this where an overlap is known to exist; at a clean word gap a
    prefix relation ("the" | "then") is real text, not a duplicate.
    """
    prev_words, next_words = list(prev_words), list(next_words)
    if not prev_words or not next_words:
        return prev_words, next_words
    a, b = _norm_word(prev_words[-1]), _norm_word(next_words[0])
    if not a or not b:
        return prev_words, next_words
    if a == b:
        return prev_words, next_words[1:]
    if len(a) >= 3 and b.startswith(a):
        return prev_words[:-1], next_words
    if len(b) >= 3 and a.endswith(b):
        return prev_words, next_words[1:]
    pa = prev_words[-1].rstrip(_PUNCT)
    nb = next_words[0].lstrip(_PUNCT)
    for k in range(min(len(pa), len(nb)) - 1, 2, -1):
        if pa.lower().endswith(nb[:k].lower()):
            return prev_words[:-1] + [pa + nb[k:]], next_words[1:]
    return prev_words, next_words


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE LEGIBILITY METRIC
# ══════════════════════════════════════════════════════════════════════════════

def _mean_run_length(mask: np.ndarray) -> Optional[float]:
    """Mean length of True runs along axis 1 (runs of 1 px are ignored as speckle)."""
    rows, cols = mask.shape
    padded = np.zeros((rows, cols + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    d = np.diff(padded, axis=1).ravel()
    starts = np.flatnonzero(d == 1)
    ends   = np.flatnonzero(d == -1)
    lengths = ends - starts
    lengths = lengths[lengths >= 2]
    return float(lengths.mean()) if lengths.size else None


def _run_anisotropy(gray: np.ndarray) -> Optional[float]:
    """
    Horizontal / vertical mean ink-run length.

    Legible Latin type is dominated by vertical stems: horizontal runs are
    about one stroke wide, vertical runs about an x-height long, so the ratio
    sits well below 1.  Type smeared sideways in scanning or microfilming
    (the dashed, streaky body columns on degraded pages) inverts this.
    Returns None when the crop is too small, too blank or too low-contrast
    to measure.
    """
    if gray.size < ANISO_MIN_PIXELS:
        return None
    p_ink   = float(np.percentile(gray, 2))
    p_paper = float(np.percentile(gray, 90))
    if p_paper - p_ink < 30:
        return None
    ink = gray < (p_ink + p_paper) / 2
    if ink.mean() < 0.01:
        return None
    h = _mean_run_length(ink)
    v = _mean_run_length(ink.T)
    if not h or not v:
        return None
    return h / v


def _gray_crop(img: Image.Image, box: List[float]) -> np.ndarray:
    iw, ih = img.size
    x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
    x1, y1 = min(iw, int(box[2])), min(ih, int(box[3]))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0), dtype=np.uint8)
    return np.asarray(img.crop((x0, y0, x1, y1)).convert("L"))


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_models():
    """Load Surya layout + detection and TrOCR recognition; load the lexicon."""
    logger.info("=" * 62)
    logger.info("  LOADING MODELS  (Surya Layout + TrOCR Recognition)")
    logger.info("=" * 62)

    os.environ.setdefault("LAYOUT_BATCH_SIZE",      "4")
    os.environ.setdefault("DETECTOR_BATCH_SIZE",    "4")
    os.environ.setdefault("RECOGNITION_BATCH_SIZE", "8")

    try:
        if hasattr(settings, "LAYOUT_BATCH_SIZE"):
            settings.LAYOUT_BATCH_SIZE   = int(os.environ["LAYOUT_BATCH_SIZE"])
        if hasattr(settings, "DETECTOR_BATCH_SIZE"):
            settings.DETECTOR_BATCH_SIZE = int(os.environ["DETECTOR_BATCH_SIZE"])
    except Exception as exc:
        logger.warning(f"  Could not patch Surya settings object: {exc}")

    if not setup_pdfa_resources():
        logger.warning("  PDF/A resources setup incomplete.")

    if VET_ENABLED:
        load_lexicon()

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
    "SectionHeader":     "Section-header",
    "PageHeader":        "Page-header",
    "PageFooter":        "Page-footer",
    "ListItem":          "List-item",
    "TableOfContents":   "Table-of-contents",
    "InlineMath":        "Text-inline-math",
    "TextInlineMath":    "Text-inline-math",
    "Header":            "Page-header",
    "Footer":            "Page-footer",
    "Heading":           "Section-header",
    "Title":             "Section-header",
    "Handwritten":       "Handwriting",
    "section_header":    "Section-header",
    "page_header":       "Page-header",
    "page_footer":       "Page-footer",
    "list_item":         "List-item",
    "table_of_contents": "Table-of-contents",
    "inline_math":       "Text-inline-math",
    "text_inline_math":  "Text-inline-math",
    "handwriting":       "Handwriting",
    "figure_caption":    "Caption",
    "FigureCaption":     "Caption",
    "table_caption":     "Caption",
    "TableCaption":      "Caption",
    "paragraph":         "Text",
    "Paragraph":         "Text",
    "body":              "Text",
    "Body":              "Text",
}

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
    """Same-label non-maximum suppression over layout regions (largest wins)."""
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
    """Single-linkage clustering of `regions` into horizontal y-bands."""
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
    """Correct Surya's `position` for masthead regions it mis-scores."""
    if len(layout_regions) < 2:
        return

    bands = _group_by_y_band(layout_regions)
    if len(bands) < 2:
        return

    body_y0, body_y1, body_members = max(bands, key=lambda b: len(b[2]))

    banner_members: List[Dict] = []
    for y0, y1, members in bands:
        if members is body_members:
            continue
        if y1 <= body_y0 and any(r["label"] in HEADER_LABELS for r in members):
            banner_members.extend(members)

    if not banner_members:
        return

    floor = min(r["position"] for r in body_members)
    correctly_placed = [r for r in banner_members if r["position"] < floor]
    misplaced        = [r for r in banner_members if r["position"] >= floor]
    if not misplaced:
        return

    if not correctly_placed:
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
    Normalise one LayoutPredictor page result, rescaling bboxes from Surya's
    internal model space (image_bbox) back to full-resolution pixel space.
    """
    regions: List[Dict] = []
    if result is None or not hasattr(result, "bboxes"):
        return regions, None

    image_bbox: Optional[List[float]] = (
        list(result.image_bbox)
        if hasattr(result, "image_bbox") and result.image_bbox
        else None
    )

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

    for box in result.bboxes:
        if hasattr(box, "bbox") and box.bbox:
            bbox = _scale_bbox([float(v) for v in box.bbox])
        elif hasattr(box, "polygon") and len(box.polygon) >= 4:
            xs   = [float(p[0]) for p in box.polygon]
            ys   = [float(p[1]) for p in box.polygon]
            bbox = _scale_bbox([min(xs), min(ys), max(xs), max(ys)])
        else:
            continue

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
    Run TrOCR on a single image crop.  Returns (text, mean max-token prob).
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


def _join_segments(texts: List[str], overlapped: Optional[List[bool]] = None) -> str:
    """
    Join OCR'd segment texts left-to-right.  At a boundary where the two
    segments physically overlapped (overlapped[k] is True for segment k),
    the doubly-read word is resolved with _boundary_fix().  At a boundary
    cut in a blank inter-word gap nothing is removed.
    """
    out: List[str] = []
    for k, t in enumerate(texts):
        words = t.split()
        if out and words and (overlapped is None or overlapped[k]):
            out, words = _boundary_fix(out, words)
        out.extend(words)
    return " ".join(out)


def _segment_bounds(crop: Image.Image, seg_w: int) -> List[Tuple[int, int, bool]]:
    """
    Split a wide crop into pieces of at most ~seg_w px, cutting in a blank
    column run (an inter-word gap) wherever one exists in the last
    HEADER_CUT_SEARCH_FRAC of each piece.  The widest blank run in the
    window is chosen, which favours word spaces over letter spacing.

    Returns [(x0, x1, overlapped), ...]; `overlapped` is True when no gap was
    found and the piece had to start inside the previous one.
    """
    gray = np.asarray(crop.convert("L"), dtype=np.float32)
    h, w = gray.shape
    p_ink, p_paper = np.percentile(gray, 5), np.percentile(gray, 90)
    ink_cols = (gray < (p_ink + p_paper) / 2).sum(axis=0)
    blank    = ink_cols <= max(1, int(0.01 * h))

    segs: List[Tuple[int, int, bool]] = []
    x, overlapped = 0, False
    while x < w:
        if w - x <= int(seg_w * 1.25):          # avoid a tiny trailing sliver
            segs.append((x, w, overlapped))
            break
        lo, hi = x + int(seg_w * (1 - HEADER_CUT_SEARCH_FRAC)), x + seg_w
        best: Optional[Tuple[int, int]] = None
        run_start: Optional[int] = None
        for c in range(lo, hi + 1):
            is_blank = c < hi and bool(blank[c])
            if is_blank and run_start is None:
                run_start = c
            elif not is_blank and run_start is not None:
                length = c - run_start
                if length >= HEADER_CUT_MIN_GAP_PX and (best is None or length >= best[1]):
                    best = (run_start, length)
                run_start = None
        if best:
            cut = best[0] + best[1] // 2
            segs.append((x, cut, overlapped))
            x, overlapped = cut, False
        else:
            segs.append((x, hi, overlapped))
            x, overlapped = hi - int(seg_w * HEADER_SEGMENT_OVERLAP), True
    return segs


def _trocr_read_wide_crop(
    crop: Image.Image,
    processor: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
    max_ar: float = MAX_HEADER_AR,
) -> Tuple[str, float]:
    """
    OCR a crop that may be much wider than it is tall (masthead, banner).

    Crops within `max_ar` are read in one call.  Wider crops are split with
    _segment_bounds() — preferably at blank gaps between words, so no word
    is cut in half and nothing needs de-duplicating — and joined with
    _join_segments().  Blank pieces are skipped rather than read (TrOCR
    invents text for blank input).
    """
    w, h = crop.size
    if h <= 0 or w / h <= max_ar:
        return _trocr_read(crop, processor, model)

    seg_w = max(1, int(h * max_ar))
    gray  = np.asarray(crop.convert("L"))
    p_ink, p_paper = np.percentile(gray, 5), np.percentile(gray, 90)
    ink   = gray < (p_ink + p_paper) / 2

    texts: List[str]   = []
    flags: List[bool]  = []
    confs: List[float] = []
    bounds = _segment_bounds(crop, seg_w)
    for x0, x1, overlapped in bounds:
        if ink[:, x0:x1].mean() < 0.005:
            continue
        text, conf = _trocr_read(crop.crop((x0, 0, x1, h)), processor, model)
        text = text.strip()
        if text:
            texts.append(text)
            flags.append(overlapped)
            confs.append(conf)

    joined     = _join_segments(texts, flags)
    confidence = sum(confs) / len(confs) if confs else 0.0
    logger.debug(
        f"      [WIDE-HEADER] AR={w/h:.1f} split into {len(bounds)} piece(s) "
        f"({sum(1 for b in bounds if not b[2])} at word gaps) → {len(joined)} char(s)"
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
    Heuristic noise filter for (cleaned) TrOCR output, judged against the
    physical size of its box.
    """
    if not text:
        return True
    if h < MIN_LINE_H or w < MIN_LINE_W:
        return True
    ar = w / h
    if ar < 0.1 or ar > 400:
        return True
    tc = text.strip()
    tl = len(tc)
    # A very short read off a long line is TrOCR's language prior talking
    # ("to", "0", "#" for a 700-px smeared line), not a reading.
    if tl <= SHORT_TEXT_MAX_CHARS and ar > SHORT_TEXT_MAX_AR:
        return True
    if tl == 1:
        return confidence < SINGLE_CHAR_CONFIDENCE_THRESHOLD
    if confidence < min_conf:
        return True
    if len(set(tc)) == 1 and tl > 2:
        return True
    noise_pats = [r"^[oOlI\.\|]+$", r"^[^a-zA-Z0-9\s]+$"]
    for pat in noise_pats:
        if re.match(pat, tc) and confidence < SINGLE_CHAR_CONFIDENCE_THRESHOLD:
            return True
    if (tl > 3 and tc.isalpha()
            and not any(c in "aeiouy" for c in tc.lower())
            and confidence < 0.7):
        return True
    if tl >= 4 and (w / (h * tl)) > SPARSE_LINE_WIDTH_RATIO:
        return True
    # More characters than can physically fit in the box: a hallucinated run-on.
    if tl >= 4 and (w / (h * tl)) < MIN_CHAR_W_RATIO:
        return True
    return False


def _accept_line(
    raw: str,
    confidence: float,
    h: int,
    w: int,
    min_conf: float = CONFIDENCE_THRESHOLD,
) -> Optional[str]:
    """
    Full line-level gate: content check on the raw read, then cleaning, then
    the geometric/confidence noise filter on the cleaned text.
    Returns the cleaned text, or None if the line is rejected.
    """
    if _is_garbage_text(raw):
        return None
    text = _clean_text(raw)
    if _is_noise(text, confidence, h, w, min_conf=min_conf):
        return None
    return text


def _surya_line_bboxes(
    crop: Image.Image,
    det_predictor: DetectionPredictor,
) -> List[List[float]]:
    """Surya line detection on a crop; crop-relative boxes sorted top-to-bottom."""
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
# PER-REGION OCR  (two-pass: detect → TrOCR per line, fallback whole-crop)
# ══════════════════════════════════════════════════════════════════════════════

def _new_region_stats() -> Dict:
    return {
        "n_detected":     0,     # lines found by the detector (above minimum size)
        "n_rejected":     0,     # of those, rejected by _accept_line
        "expected_chars": 0.0,   # characters that physically fit in the detected lines
        "accepted_chars": 0,     # characters actually read on accepted lines (pass 1)
        "raw_texts":      [],    # every raw TrOCR read, accepted or not
        "rejected_boxes": [],    # absolute boxes of rejected lines (sweep must not re-read)
        "pass":           None,
    }


def ocr_region(
    page_image: Image.Image,
    raw_image: Image.Image,
    region: Dict,
    det_predictor: DetectionPredictor,
    trocr_processor: TrOCRProcessor,
    trocr_model: VisionEncoderDecoderModel,
) -> Tuple[List[Dict], Dict]:
    """
    Two-pass OCR for a single layout region.  Returns (elements, stats); the
    stats feed vet_region().

    Pass 1 — line detection + TrOCR per line (every line through _accept_line).
    Pass 2 — whole-crop read, only for:
      - SINGLE_BLOCK_LABELS (headers, captions: often one block of type), or
      - other labels where the detector found NO lines and the region is
        line-shaped (w/h ≥ PASS2_MIN_AR).
    Pass 2 is no longer attempted for a body region whose lines were found
    but all rejected: that pattern means the region is illegible, and a
    whole multi-line column squashed into one read only yields a
    hallucinated sentence.
    """
    stats = _new_region_stats()
    x0, y0, x1, y1 = [int(c) for c in region["bbox"]]
    iw, ih          = page_image.size
    label           = region["label"]
    is_header       = label in HEADER_LABELS
    single_block    = label in SINGLE_BLOCK_LABELS

    x0 = max(0, x0);  y0 = max(0, y0)
    x1 = min(iw, x1); y1 = min(ih, y1)

    rw, rh = x1 - x0, y1 - y0
    if rw < MIN_REGION_W or rh < MIN_REGION_H:
        return [], stats

    if is_header:
        crop = preprocess_header_crop(raw_image.crop((x0, y0, x1, y1)))
        read = lambda img: _trocr_read_wide_crop(img, trocr_processor, trocr_model)
    else:
        crop = page_image.crop((x0, y0, x1, y1))
        read = lambda img: _trocr_read(img, trocr_processor, trocr_model)

    # ── Pass 1: line detection → TrOCR per line ───────────────────────────────
    line_bboxes = _surya_line_bboxes(crop, det_predictor)
    pass1_elems: List[Dict] = []

    for lb in line_bboxes:
        lx0, ly0, lx1, ly1 = [int(v) for v in lb]
        lh, lw = ly1 - ly0, lx1 - lx0
        if lh < MIN_LINE_H or lw < MIN_LINE_W:
            continue
        abs_bbox = [lx0 + x0, ly0 + y0, lx1 + x0, ly1 + y0]
        stats["n_detected"]     += 1
        stats["expected_chars"] += lw / (CHAR_W_FACTOR * lh)

        raw, confidence = read(crop.crop((lx0, ly0, lx1, ly1)))
        stats["raw_texts"].append(raw)
        text = _accept_line(raw, confidence, lh, lw)
        if text is None:
            stats["n_rejected"] += 1
            stats["rejected_boxes"].append([float(v) for v in abs_bbox])
            logger.debug(f"      [NOISE] conf={confidence:.2f} {lw}×{lh}px | {raw[:40]}")
            continue
        stats["accepted_chars"] += len(text)
        pass1_elems.append({
            "text":             text,
            "bbox":             abs_bbox,
            "confidence":       confidence,
            "font_size":        max(6.0, min(lh * 0.85, 72.0)),
            "source_label":     label,
            "reading_position": region["position"],
        })

    n_pass1 = len(pass1_elems)

    if n_pass1 > 0 and not single_block:
        logger.debug(f"      Pass 1 (det+TrOCR): {n_pass1} line(s)")
        stats["pass"] = 1
        return pass1_elems, stats

    # ── Pass 2 gate ───────────────────────────────────────────────────────────
    if not single_block:
        if stats["n_detected"] > 0:
            logger.debug(
                f"      All {stats['n_detected']} detected line(s) rejected for "
                f"label={label} — region treated as illegible; no whole-crop fallback."
            )
            return [], stats
        if rw / rh < PASS2_MIN_AR:
            logger.debug(
                f"      No lines detected in a tall {label} block ({rw}×{rh}px) "
                f"— whole-crop fallback skipped."
            )
            return [], stats

    # ── Pass 2: whole-crop TrOCR ──────────────────────────────────────────────
    raw_wb, conf_wb = read(crop)
    stats["raw_texts"].append(raw_wb)
    text_wb = _accept_line(raw_wb, conf_wb, rh, rw)
    pass2_elems: List[Dict] = []
    if text_wb is not None:
        pass2_elems.append({
            "text":             text_wb,
            "bbox":             [float(x0), float(y0), float(x1), float(y1)],
            "confidence":       conf_wb,
            "font_size":        max(6.0, min(rh * 0.85, 72.0)),
            "source_label":     label,
            "reading_position": region["position"],
        })

    if pass2_elems:
        chars1 = sum(len(e["text"]) for e in pass1_elems)
        chars2 = len(pass2_elems[0]["text"])
        logger.debug(
            f"      Pass 2 (whole-crop TrOCR): {chars2} char(s)  "
            f"[Pass 1 had {n_pass1} line(s), {chars1} char(s)]"
        )
        if chars2 >= chars1:
            stats["pass"] = 2
            return pass2_elems, stats

    if n_pass1 > 0:
        logger.debug(f"      Kept Pass 1 ({n_pass1} lines) over Pass 2 ({len(pass2_elems)})")
        stats["pass"] = 1
        return pass1_elems, stats

    logger.debug(f"      Both passes empty for label={label}")
    return [], stats


# ══════════════════════════════════════════════════════════════════════════════
# LEGIBILITY VETTING
# ══════════════════════════════════════════════════════════════════════════════

def vet_region(
    region: Dict,
    elems: List[Dict],
    stats: Dict,
    raw_image: Image.Image,
) -> Dict:
    """
    Second-stage quality gate for one OCR'd region.

    Each indicator that fails adds a "strike".  The region is dropped only at
    VET_STRIKES_TO_REJECT strikes, so a legible region survives any single
    unreliable signal (a names-heavy ad failing the dictionary test, a
    region with a few torn lines failing the reject-rate test, ...).

    Indicators (each only evaluated when there is enough evidence):
      line-reject  — share of detected lines rejected by _accept_line
      char-yield   — characters read vs. characters that fit in the lines
      junk         — share of junk tokens across all raw reads
      lexical      — share of dictionary words across all raw reads
      anisotropy   — horizontal/vertical ink-run ratio of the raw crop
    """
    v: Dict = {
        "reject_rate": None, "char_yield": None, "junk_rate": None,
        "lex_rate": None, "anisotropy": None, "strikes": [],
    }
    nd = stats["n_detected"]
    if nd >= VET_MIN_LINES:
        rr = stats["n_rejected"] / nd
        v["reject_rate"] = rr
        if rr > VET_MAX_LINE_REJECT_RATE:
            v["strikes"].append(f"line-reject {rr:.0%}")
        if stats["expected_chars"] > 0:
            cy = stats["accepted_chars"] / stats["expected_chars"]
            v["char_yield"] = cy
            if cy < VET_MIN_CHAR_YIELD:
                v["strikes"].append(f"char-yield {cy:.2f}")

    ts = _token_stats(stats["raw_texts"])
    if ts["tokens"] >= VET_MIN_TOKENS_FOR_JUNK:
        jr = ts["junk"] / ts["tokens"]
        v["junk_rate"] = jr
        if jr > VET_MAX_JUNK_RATE:
            v["strikes"].append(f"junk {jr:.0%}")
    if _lexicon_available() and ts["words"] >= VET_MIN_WORDS:
        lr = ts["hits"] / ts["words"]
        v["lex_rate"] = lr
        if lr < VET_MIN_LEXICAL_RATE:
            v["strikes"].append(f"lexical {lr:.0%}")

    an = _run_anisotropy(_gray_crop(raw_image, region["bbox"]))
    v["anisotropy"] = an
    if an is not None and an > VET_MAX_RUN_ANISOTROPY:
        v["strikes"].append(f"anisotropy {an:.2f}")

    exempt = region["label"] in VET_EXEMPT_LABELS
    v["keep"] = exempt or len(v["strikes"]) < VET_STRIKES_TO_REJECT
    # Keep the sweep out of regions judged illegible, and out of regions where
    # every detected line was already examined and rejected.
    v["exclude_from_sweep"] = (not v["keep"]) or (nd >= VET_MIN_LINES and not elems)
    return v


def _sweep_line_suspect(text: str, gray: np.ndarray) -> Optional[str]:
    """
    Per-line vetting for recall-sweep lines, which have no region context.
    A single failing indicator rejects the line.
    """
    if _lexicon_available():
        s = _token_stats([text])
        if s["words"] >= SWEEP_LEX_MIN_WORDS and s["hits"] / s["words"] < SWEEP_MIN_LEXICAL_RATE:
            return f"lexical {s['hits']}/{s['words']}"
    an = _run_anisotropy(gray)
    if an is not None and an > VET_MAX_RUN_ANISOTROPY:
        return f"anisotropy {an:.2f}"
    return None


def drop_repeated_short_lines(elements: List[Dict]) -> List[Dict]:
    """
    Drop short strings that recur REPEAT_SHORT_TEXT_MIN_COUNT+ times on one
    page and contain no real word of 4+ letters ("to", "0", "1907 08").
    Repeated short fillers are a hallucination signature; genuine repeated
    ad text ("Moscow", "Idaho") contains a 4+ letter word and is kept.
    """
    def norm(t: str) -> str:
        return re.sub(r"\s+", " ", t.strip().lower())

    counts = Counter(norm(e["text"]) for e in elements
                     if len(norm(e["text"])) <= REPEAT_SHORT_TEXT_MAX_LEN)
    bad = {
        t for t, n in counts.items()
        if n >= REPEAT_SHORT_TEXT_MIN_COUNT
        and not any(len(w) >= 4 for w in re.findall(r"[a-z]+", t))
    }
    if not bad:
        return elements
    kept = [e for e in elements if norm(e["text"]) not in bad]
    logger.info(
        f"  [REPEAT] dropped {len(elements) - len(kept)} repeated filler line(s): "
        + ", ".join(repr(b) for b in sorted(bad))
    )
    return kept


def resolve_row_overlaps(elements: List[Dict]) -> List[Dict]:
    """
    For two elements on the same visual row whose boxes overlap horizontally
    (the same stretch of print read twice by different passes), resolve the
    doubly-read boundary word with _boundary_fix() — e.g. "Go to The Past"
    + "Pastime." → "Go to The" + "Pastime.".  Elements left empty are dropped.
    """
    if len(elements) < 2:
        return elements
    order = sorted(range(len(elements)), key=lambda i: elements[i]["bbox"][1])
    fixed = 0
    for oi, i in enumerate(order):
        a = elements[i]
        for j in order[oi + 1:]:
            b = elements[j]
            if b["bbox"][1] > a["bbox"][3]:
                break
            if not a["text"].strip() or not b["text"].strip():
                continue
            y_ov = min(a["bbox"][3], b["bbox"][3]) - max(a["bbox"][1], b["bbox"][1])
            sh   = min(a["bbox"][3] - a["bbox"][1], b["bbox"][3] - b["bbox"][1])
            if sh <= 0 or y_ov / sh <= ROW_OVERLAP_FRACTION:
                continue
            left, right = (a, b) if a["bbox"][0] <= b["bbox"][0] else (b, a)
            if left["bbox"][2] - right["bbox"][0] <= 0:      # side by side, no overlap
                continue
            if right["bbox"][2] <= left["bbox"][2]:          # containment: dedupe's job
                continue
            lw, rw = left["text"].split(), right["text"].split()
            nl, nr = _boundary_fix(lw, rw)
            if nl != lw or nr != rw:
                logger.debug(
                    f"      [ROW-OVERLAP] {left['text'][-30:]!r} | {right['text'][:30]!r} "
                    f"→ {' '.join(nl)[-30:]!r} | {' '.join(nr)[:30]!r}"
                )
                left["text"], right["text"] = " ".join(nl), " ".join(nr)
                fixed += 1
    if fixed:
        logger.info(f"  [ROW-OVERLAP] resolved {fixed} doubly-read boundary word(s).")
    return [e for e in elements if e["text"].strip()]


# ══════════════════════════════════════════════════════════════════════════════
# RECALL SWEEP  –  recover text no layout region claimed
# ══════════════════════════════════════════════════════════════════════════════

def _coverage(box: List[float], elements: List[Dict], pad: float = 0.0) -> float:
    """Fraction of `box` already lying inside existing elements' boxes (0–1)."""
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
    n      = -(-(length - tile) // stride) + 1
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
    """x-ranges occupied by column-body layout regions, merged into bands."""
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
    """Relocate recovered-line clusters that form a whole missed column."""
    bands = _column_bands(layout_regions, page_w)
    if not bands or not recovered:
        return 0

    unclaimed = [
        e for e in recovered
        if not any(b[0] <= (e["bbox"][0] + e["bbox"][2]) / 2 <= b[1] for b in bands)
    ]
    if not unclaimed:
        return 0

    clusters: List[List] = []
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
        left   = [b[3] for b in bands if b[1] <= centre]
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
    """Give every element a "_row" index (mutates in place) for the final sort."""
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
    """Remove an element that is a duplicate re-read of another's physical area."""
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
            if b["bbox"][1] > ay1:
                break
            area_b  = max(1.0, (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]))
            overlap = _overlap_area(a["bbox"], b["bbox"])
            if overlap / min(area_a, area_b) < DEDUPE_CONTAINMENT:
                continue
            if max(area_a, area_b) / min(area_a, area_b) > DEDUPE_MAX_AREA_RATIO:
                continue

            a_furn = a.get("source_label") in FURNITURE_LABELS
            b_furn = b.get("source_label") in FURNITURE_LABELS
            if a_furn != b_furn:
                loser_key = j if a_furn else i
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
                break

    if n_dropped:
        logger.info(
            f"  [DEDUPE] removed {n_dropped} duplicate element(s) "
            f"(same content recognised twice)."
        )
    return [e for k, e in enumerate(elements) if not dropped[k]]


def _split_wide_sweep_line(box: List[float], bands: List[List[float]]) -> List[List[float]]:
    """Split a sweep line that straddles a known column-band edge."""
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
    elements: List[Dict],
    layout_regions: List[Dict],
    det_predictor: DetectionPredictor,
    trocr_processor: TrOCRProcessor,
    trocr_model: VisionEncoderDecoderModel,
    examined_boxes: Optional[List[List[float]]] = None,
    excluded_boxes: Optional[List[List[float]]] = None,
    raw_image: Optional[Image.Image] = None,
) -> List[Dict]:
    """
    Recall safety-net: tile the page, detect lines, OCR any line not already
    covered.

    New in this revision:
    - `examined_boxes` (lines the region pass read and rejected) count as
      covered, so the sweep no longer gives a rejected hallucination a
      second chance to pass.
    - Lines lying inside `excluded_boxes` (regions vetted illegible) are
      skipped entirely.
    - Every recovered line passes _accept_line and, with VET_ENABLED,
      _sweep_line_suspect (lexical + ink anisotropy, measured on the raw,
      un-sharpened render `raw_image`).
    """
    iw, ih = page_image.size
    m      = SWEEP_EDGE_MARGIN
    raw_image = raw_image or page_image
    recovered: List[Dict] = []
    known: List[Dict]     = list(elements) + [{"bbox": b} for b in (examined_boxes or [])]
    excluded: List[Dict]  = [{"bbox": b} for b in (excluded_boxes or [])]
    column_bands = _column_bands(layout_regions, iw)
    n_vetted = 0

    for ty in _tile_origins(ih, SWEEP_TILE, SWEEP_OVERLAP):
        for tx in _tile_origins(iw, SWEEP_TILE, SWEEP_OVERLAP):
            tx1, ty1 = min(tx + SWEEP_TILE, iw), min(ty + SWEEP_TILE, ih)
            tile = page_image.crop((tx, ty, tx1, ty1))
            for lx0, ly0, lx1, ly1 in _surya_line_bboxes(tile, det_predictor):
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
                    if excluded and _coverage(box, excluded) >= SWEEP_COVERED_FRAC:
                        continue
                    if _coverage(box, known, pad=SWEEP_COVERAGE_PAD) >= SWEEP_COVERED_FRAC:
                        continue
                    line = page_image.crop(tuple(int(v) for v in box))
                    raw, conf = _trocr_read(line, trocr_processor, trocr_model)
                    text = _accept_line(raw, conf, int(bh), int(bw),
                                        min_conf=SWEEP_MIN_CONFIDENCE)
                    if text is None:
                        logger.debug(
                            f"      [SWEEP-NOISE] conf={conf:.2f} "
                            f"{int(bw)}×{int(bh)}px | {raw[:40]}"
                        )
                        # Remember it so an overlapping tile doesn't re-read it.
                        known.append({"bbox": box})
                        continue
                    if VET_ENABLED:
                        reason = _sweep_line_suspect(text, _gray_crop(raw_image, box))
                        if reason:
                            n_vetted += 1
                            logger.debug(
                                f"      [SWEEP-VET{'-DRY' if VET_DRY_RUN else ''}] "
                                f"{reason} | {text[:50]}"
                            )
                            if not VET_DRY_RUN:
                                known.append({"bbox": box})
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

    if n_vetted:
        logger.info(
            f"  [SWEEP-VET] {n_vetted} recovered line(s) "
            f"{'would be' if VET_DRY_RUN else 'were'} rejected as illegible."
        )
    moved = _assign_recovered_positions(recovered, layout_regions, iw, ih)
    if moved:
        logger.info(f"  [SWEEP] {moved} recovered line(s) lie in a column the layout stage skipped.")
    return recovered


# ══════════════════════════════════════════════════════════════════════════════
# DEBUG VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _debug_path(stem: str, page_num: int, tag: str, ext: str) -> Path:
    """Path for one debug output file (rolling 'latest_*' names in overwrite mode)."""
    if DEBUG_OVERWRITE:
        return DEBUG_PATH / f"latest_{tag}.{ext}"
    return DEBUG_PATH / f"{stem}_p{page_num:03d}_{tag}.{ext}"


def _debug_img_path(stem: str, page_num: int, tag: str) -> Path:
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
    extra_covered: Optional[List[List[float]]] = None,
) -> float:
    """
    Diagnostic: how much printed ink still lies outside every OCR'd box?
    Picture/Figure regions and regions vetted illegible (`extra_covered`)
    count as intentionally skipped.
    """
    s      = AUDIT_SCALE
    small  = page_image.convert("L").reduce(s)
    sw, sh = small.size
    ink    = small.point(lambda v: 255 if v < AUDIT_INK_LEVEL else 0)
    ink    = ink.filter(ImageFilter.MaxFilter(3))

    covered = Image.new("L", (sw, sh), 0)
    cdraw   = ImageDraw.Draw(covered)
    pad     = 6
    boxes   = [e["bbox"] for e in elements]
    boxes  += [r["bbox"] for r in layout_regions if r["label"] in SKIP_LABELS]
    boxes  += list(extra_covered or [])
    for x0, y0, x1, y1 in boxes:
        cdraw.rectangle([(x0 - pad) / s, (y0 - pad) / s, (x1 + pad) / s, (y1 + pad) / s], fill=255)

    uncovered = ImageChops.subtract(ink, covered)
    total = ink.histogram()[255]
    miss  = uncovered.histogram()[255]
    frac  = miss / total if total else 0.0

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
    if extra_covered:
        msg += f" (excluding {len(extra_covered)} region(s) vetted illegible)"
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


def save_ocr_debug(
    image: Image.Image,
    elements: List[Dict],
    path: Path,
    rejected_boxes: Optional[List[List[float]]] = None,
) -> None:
    """OCR overlay; regions vetted illegible are marked with a red X."""
    img  = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    font = _pil_font(12)
    for x0, y0, x1, y1 in (rejected_boxes or []):
        draw.rectangle([x0, y0, x1, y1], outline=REJECTED_COLOUR + (230,),
                       fill=REJECTED_COLOUR + (28,), width=4)
        draw.line([x0, y0, x1, y1], fill=REJECTED_COLOUR + (160,), width=3)
        draw.line([x0, y1, x1, y0], fill=REJECTED_COLOUR + (160,), width=3)
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


def save_vet_report(results: List[Tuple[Dict, Dict, Dict, int]], path: Path,
                    filename: str, page_num: int) -> None:
    """
    One row per OCR'd region with every vetting indicator, for calibration.
    Compare the numbers for regions you can read against the smeared ones and
    move the VET_* thresholds into the gap between them.
    """
    def f(v, fmt):
        return format(v, fmt) if v is not None else "  —"

    lines = [
        f"FILE: {filename}   PAGE: {page_num}   "
        f"{'DRY RUN (nothing dropped)' if VET_DRY_RUN else 'LIVE'}   "
        f"lexicon: {_LEXICON_SOURCE}",
        f"Thresholds: reject>{VET_MAX_LINE_REJECT_RATE:.0%}  yield<{VET_MIN_CHAR_YIELD:.2f}  "
        f"junk>{VET_MAX_JUNK_RATE:.0%}  lexical<{VET_MIN_LEXICAL_RATE:.0%}  "
        f"aniso>{VET_MAX_RUN_ANISOTROPY:.2f}  strikes≥{VET_STRIKES_TO_REJECT}",
        "=" * 118,
        f"{'POS':>5}  {'LABEL':<16} {'LINES':>5} {'OUT':>4} {'REJ':>5} {'YIELD':>6} "
        f"{'JUNK':>5} {'LEX':>5} {'ANISO':>6}  VERDICT   STRIKES",
        "-" * 118,
    ]
    for region, stats, v, n_out in sorted(results, key=lambda t: t[0]["position"]):
        verdict = "keep" if v["keep"] else "REJECT"
        lines.append(
            f"{region['position']:>5}  {region['label']:<16} {stats['n_detected']:>5} {n_out:>4} "
            f"{f(v['reject_rate'], '>5.0%')} {f(v['char_yield'], '>6.2f')} "
            f"{f(v['junk_rate'], '>5.0%')} {f(v['lex_rate'], '>5.0%')} "
            f"{f(v['anisotropy'], '>6.2f')}  {verdict:<8}  {', '.join(v['strikes']) or '—'}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"    [DEBUG] Vetting report → {path.name}")


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

    1.  Preprocess
    2.  Layout (+ 2b masthead reorder)
    3.  OCR per region
    3a. Vet    — drop regions whose reads are judged illegible (vet_region)
    3b. Sweep  — recover unclaimed lines (skipping examined / illegible areas)
    3c. Dedupe, row-overlap fix, repeated-filler removal
    4.  Sort
    5.  Audit
    """
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
        _reorder_banner_regions(layout_regions)

    # ── Stage 3: Per-region DetectionPredictor + TrOCR ────────────────────────
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

    all_elements: List[Dict]             = []
    examined_boxes: List[List[float]]    = []   # rejected lines — sweep must not re-read
    illegible_boxes: List[List[float]]   = []   # regions the sweep must stay out of
    vetoed_boxes: List[List[float]]      = []   # regions whose output was dropped (debug)
    vet_results: List[Tuple[Dict, Dict, Dict, int]] = []
    n_vet_lines = 0

    for ri, region in enumerate(sorted(text_regions, key=lambda r: r["position"])):
        lbl  = region["label"]
        bbox = region["bbox"]
        logger.debug(
            f"    Region {ri+1}/{len(text_regions)}: {lbl} "
            f"pos={region['position']}  "
            f"bbox=({bbox[0]:.0f},{bbox[1]:.0f}→{bbox[2]:.0f},{bbox[3]:.0f})"
        )
        elems, stats = ocr_region(
            processed, pil_image, region,
            det_predictor, trocr_processor, trocr_model,
        )
        examined_boxes.extend(stats["rejected_boxes"])

        # ── Stage 3a: legibility vetting ──────────────────────────────────────
        if VET_ENABLED:
            v = vet_region(region, elems, stats, pil_image)
            vet_results.append((region, stats, v, len(elems)))
            if not v["keep"]:
                vetoed_boxes.append(list(bbox))
                n_vet_lines += len(elems)
                logger.info(
                    f"  [VET{'-DRY' if VET_DRY_RUN else ''}] "
                    f"{'would drop' if VET_DRY_RUN else 'dropped'} {lbl} pos={region['position']} "
                    f"({len(elems)} line(s)) — {', '.join(v['strikes'])}"
                )
                if not VET_DRY_RUN:
                    elems = []
            if v["exclude_from_sweep"] and not VET_DRY_RUN:
                illegible_boxes.append(list(bbox))

        logger.debug(f"      → {len(elems)} element(s) accepted.")
        all_elements.extend(elems)

    if VET_ENABLED:
        logger.info(
            f"  [VET] {len(vetoed_boxes)} region(s) judged illegible; "
            f"{n_vet_lines} line(s) {'would be' if VET_DRY_RUN else 'were'} dropped."
        )
        save_vet_report(vet_results,
                        _debug_path(stem, page_num, "04_vetting_report", "txt"),
                        filename, page_num)

    # ── Stage 3b: Recall sweep ────────────────────────────────────────────────
    if SWEEP_ENABLED:
        recovered = sweep_uncovered_text(
            processed, all_elements, layout_regions,
            det_predictor, trocr_processor, trocr_model,
            examined_boxes=examined_boxes,
            excluded_boxes=illegible_boxes,
            raw_image=pil_image,
        )
        logger.info(
            f"  [SWEEP] recovered {len(recovered)} line(s) the layout stage missed."
        )
        all_elements.extend(recovered)

    # ── Stage 3c: De-duplicate, fix row overlaps, drop repeated fillers ──────
    before_dedupe = len(all_elements)
    all_elements  = dedupe_overlapping_elements(all_elements)
    if len(all_elements) < before_dedupe:
        logger.info(
            f"  [DEDUPE] {before_dedupe - len(all_elements)} duplicate element(s) removed."
        )
    all_elements = resolve_row_overlaps(all_elements)
    all_elements = drop_repeated_short_lines(all_elements)

    # ── Stage 4: Final reading order sort ────────────────────────────────────
    _assign_visual_rows(all_elements)
    all_elements.sort(key=lambda e: (e["reading_position"], e["_row"], e["bbox"][0]))
    for e in all_elements:
        del e["_row"]

    logger.info(f"  [RESULT] {len(all_elements)} OCR element(s) on page {page_num}.")

    # ── Stage 5: Coverage audit (diagnostic only) ─────────────────────────────
    if AUDIT_ENABLED:
        try:
            audit_coverage(processed, all_elements, layout_regions, stem, page_num,
                           extra_covered=None if VET_DRY_RUN else vetoed_boxes)
        except Exception as exc:
            logger.warning(f"  [COVERAGE] audit failed: {exc}")

    if all_elements or vetoed_boxes:
        save_ocr_debug(pil_image, all_elements,
                       _debug_img_path(stem, page_num, "02_ocr_overlay"),
                       rejected_boxes=vetoed_boxes)
        save_ocr_report(all_elements,
                        _debug_path(stem, page_num, "02_ocr_report", "txt"),
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
    """Insert `elements` into `page` as invisible text (render mode 3)."""
    iw, ih = img_size
    pw, ph = page.rect.width, page.rect.height
    sx, sy = pw / iw, ph / ih

    logger.debug(f"    {iw}×{ih}px → {pw:.1f}×{ph:.1f}pt  (sx={sx:.4f}, sy={sy:.4f})")

    writer   = fitz.TextWriter(page.rect)
    inserted = 0

    for elem in elements:
        bx0, by0, bx1, by1 = elem["bbox"]
        try:
            text        = elem["text"] + ELEMENT_SEPARATOR
            x0_pt       = bx0 * sx
            x1_pt       = bx1 * sx
            baseline_pt = by1 * sy
            fontsize    = max(MIN_FONT_PT, elem["font_size"] * sy)

            available_w = min(x1_pt, pw) - x0_pt
            if available_w > 0:
                text_w = font.text_length(text, fontsize=fontsize)
                if text_w > available_w:
                    fontsize = max(MIN_CLAMPED_FONT_PT, fontsize * available_w / text_w)

            writer.append(
                fitz.Point(x0_pt, baseline_pt),
                text,
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
    """Add an invisible OCR text layer to `input_path` and write `output_path`."""
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

            apply_document_metadata(doc, filename)

            if using_freesans:
                try:
                    doc.subset_fonts()
                    logger.info("  Embedded font subset to the glyphs used.")
                except Exception as exc:
                    logger.warning(
                        f"  Font subsetting skipped ({exc}); the full font stays "
                        f"embedded.  `pip install fonttools` enables subsetting."
                    )

            doc.save(
                output_path,
                deflate=True,
                garbage=4,
                clean=True,
                deflate_images=False,
                encryption=fitz.PDF_ENCRYPT_KEEP,
            )
            logger.info(f"  Saved: {output_path}")

        setup_pdfa_compliance(output_path)

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
    """Keep the output within 15 % of the original using deflate only."""
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
        f"  Vetting  : {'off' if not VET_ENABLED else ('DRY RUN (report only)' if VET_DRY_RUN else 'on')}"
    )
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