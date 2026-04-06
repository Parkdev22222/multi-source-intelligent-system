"""
Generate system architecture diagram using PIL/Pillow.
Output: docs/architecture.png
"""
from PIL import Image, ImageDraw, ImageFont
import os

# ─── Canvas ───────────────────────────────────────────────────────────────────
W, H = 1800, 1280
bg   = (10, 14, 22)      # #0a0e14

img  = Image.new("RGB", (W, H), bg)
draw = ImageDraw.Draw(img)

# ─── Fonts ────────────────────────────────────────────────────────────────────
FONT_BASE = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

def fnt(size, bold=False):
    path = FONT_BOLD if bold else FONT_BASE
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

F_TITLE  = fnt(28, bold=True)
F_SECT   = fnt(17, bold=True)
F_BODY   = fnt(14)
F_SMALL  = fnt(12)
F_TINY   = fnt(11)

# ─── Colour palette ───────────────────────────────────────────────────────────
C_INPUT   = (34,  197, 94)   # green-500   — sensors / input
C_DETECT  = (16,  185, 129)  # emerald-500 — detection
C_DB      = (99,  102, 241)  # indigo-500  — databases
C_PAIR    = (6,   182, 212)  # cyan-500    — pairing
C_LLM     = (168,  85, 247)  # purple-500  — LLM / reporting
C_API     = (20,  184, 166)  # teal-500    — web API
C_UI      = (59,  130, 246)  # blue-500    — dashboard / UI
C_CFG     = (245, 158, 11)   # amber-500   — config / RAG
C_ARROW   = (100, 116, 139)  # slate-500
C_TXT     = (226, 232, 240)  # slate-200
C_DIM     = (100, 116, 139)  # slate-500
C_BORDER  = (30,  41,  59)   # slate-800
C_BG_BOX  = (15,  23,  42)   # slate-900   — box fill

# ─── Helpers ──────────────────────────────────────────────────────────────────
def rounded_rect(d, xy, r=10, fill=C_BG_BOX, outline=C_BORDER, width=2):
    x0,y0,x1,y1 = xy
    d.rounded_rectangle([x0,y0,x1,y1], radius=r, fill=fill, outline=outline, width=width)

def center_text(d, cx, cy, text, font, fill=C_TXT):
    bb = d.textbbox((0,0), text, font=font)
    tw = bb[2]-bb[0]; th = bb[3]-bb[1]
    d.text((cx-tw//2, cy-th//2), text, font=font, fill=fill)

def arrow_v(d, x, y0, y1, color=C_ARROW):
    """Vertical arrow from y0 to y1."""
    d.line([(x, y0), (x, y1)], fill=color, width=2)
    # arrowhead
    d.polygon([(x-6, y1-10), (x+6, y1-10), (x, y1)], fill=color)

def arrow_h(d, x0, x1, y, color=C_ARROW):
    """Horizontal arrow from x0 to x1."""
    d.line([(x0, y), (x1, y)], fill=color, width=2)
    d.polygon([(x1-10, y-6), (x1-10, y+6), (x1, y)], fill=color)

# ─── Title ────────────────────────────────────────────────────────────────────
center_text(draw, W//2, 34, "Multi-Source Intelligent System (MSIS)  —  Architecture", F_TITLE, C_TXT)
draw.line([(60, 60), (W-60, 60)], fill=C_BORDER, width=1)

# ─── Layout constants ─────────────────────────────────────────────────────────
# Column centres  (left pipeline / right side panels)
CX_MAIN   = 540
CX_DB_L   = 200          # Sensor DB
CX_DB_R   = 900          # Pairing DB / Reports DB
CX_CFG    = 1180         # Config / RAG
CX_CLIP   = 1400         # CLIP tracker
CX_API    = 1180         # Web API (lower)
CX_DASH   = 1400         # Dashboard (lower)

BW  = 320   # standard box width
SBW = 220   # side box width
BH  = 56    # standard box height

# Row Y centres  (main pipeline flows down)
ROW = {}
ROW["ingest"]   = 110
ROW["detect"]   = 220
ROW["sensordb"] = 330
ROW["pair"]     = 460
ROW["pairdb"]   = 570
ROW["report"]   = 700
ROW["reportdb"] = 810
ROW["webapi"]   = 920
ROW["dash"]     = 1050


# ─── SECTION LABELS ───────────────────────────────────────────────────────────
def section_label(d, x, y, txt, color):
    bb = d.textbbox((0,0), txt, font=F_SECT)
    tw = bb[2]-bb[0]
    d.text((x - tw//2, y), txt, font=F_SECT, fill=color)

# ─── 1. INPUT SOURCES  (row 0)  y~110 ─────────────────────────────────────────
input_boxes = [
    (80,   "Satellite\nImagery",     C_INPUT),
    (300,  "Drone\nImagery",         C_INPUT),
    (520,  "GPS /\nMetadata",        C_INPUT),
    (740,  "Manual\nUpload",         C_INPUT),
]
for bx, label, col in input_boxes:
    x0,y0 = bx, ROW["ingest"]-30
    x1,y1 = bx+170, ROW["ingest"]+30
    rounded_rect(draw, [x0,y0,x1,y1], fill=(20,28,44), outline=col, width=2)
    cy = (y0+y1)//2
    lines = label.split("\n")
    if len(lines)==2:
        bb0 = draw.textbbox((0,0), lines[0], font=F_BODY); h0 = bb0[3]-bb0[1]
        bb1 = draw.textbbox((0,0), lines[1], font=F_BODY); h1 = bb1[3]-bb1[1]
        center_text(draw, (x0+x1)//2, cy-h0//2-1, lines[0], F_BODY, col)
        center_text(draw, (x0+x1)//2, cy+h1//2+1, lines[1], F_BODY, col)
    else:
        center_text(draw, (x0+x1)//2, cy, label, F_BODY, col)

# merge arrow down
arrow_v(draw, CX_MAIN, ROW["ingest"]+32, ROW["detect"]-34)

# ─── 2. DETECTION LAYER  y~220 ────────────────────────────────────────────────
x0,y0 = CX_MAIN-BW//2, ROW["detect"]-34
x1,y1 = CX_MAIN+BW//2, ROW["detect"]+34
rounded_rect(draw, [x0,y0,x1,y1], fill=(14,30,22), outline=C_DETECT, width=2)
center_text(draw, CX_MAIN, ROW["detect"]-12, "SAM3  (facebook/sam3)", F_SECT, C_DETECT)
center_text(draw, CX_MAIN, ROW["detect"]+12, "Text-prompted concept segmentation", F_SMALL, C_DIM)

# ─── 3. SENSOR DB  y~330 ──────────────────────────────────────────────────────
arrow_v(draw, CX_MAIN, ROW["detect"]+36, ROW["sensordb"]-34)
x0,y0 = CX_MAIN-BW//2, ROW["sensordb"]-34
x1,y1 = CX_MAIN+BW//2, ROW["sensordb"]+34
rounded_rect(draw, [x0,y0,x1,y1], fill=(17,20,50), outline=C_DB, width=2)
center_text(draw, CX_MAIN, ROW["sensordb"]-12, "Sensor DB  (SQLite)", F_SECT, C_DB)
center_text(draw, CX_MAIN, ROW["sensordb"]+12, "image_records · detection_records", F_SMALL, C_DIM)

# ─── 4. TEMPORAL PAIRING  y~460 ───────────────────────────────────────────────
arrow_v(draw, CX_MAIN, ROW["sensordb"]+36, ROW["pair"]-44)
x0,y0 = CX_MAIN-BW//2, ROW["pair"]-44
x1,y1 = CX_MAIN+BW//2, ROW["pair"]+44
rounded_rect(draw, [x0,y0,x1,y1], fill=(8,34,40), outline=C_PAIR, width=2)
center_text(draw, CX_MAIN, ROW["pair"]-24, "Temporal Pairing Engine", F_SECT, C_PAIR)
center_text(draw, CX_MAIN, ROW["pair"],     "Mode A: SAM3 video predictor tracker", F_SMALL, C_DIM)
center_text(draw, CX_MAIN, ROW["pair"]+20,  "Mode B: CLIP cosine + geo proximity",  F_SMALL, C_DIM)

# ─── 5. PAIRING DB  y~570 ─────────────────────────────────────────────────────
arrow_v(draw, CX_MAIN, ROW["pair"]+46, ROW["pairdb"]-34)
x0,y0 = CX_MAIN-BW//2, ROW["pairdb"]-34
x1,y1 = CX_MAIN+BW//2, ROW["pairdb"]+34
rounded_rect(draw, [x0,y0,x1,y1], fill=(17,20,50), outline=C_DB, width=2)
center_text(draw, CX_MAIN, ROW["pairdb"]-12, "Pairing DB  (SQLite)", F_SECT, C_DB)
center_text(draw, CX_MAIN, ROW["pairdb"]+12, "new · matched · moved · disappeared", F_SMALL, C_DIM)

# ─── 6. REPORTING LAYER  y~700 ────────────────────────────────────────────────
arrow_v(draw, CX_MAIN, ROW["pairdb"]+36, ROW["report"]-44)
x0,y0 = CX_MAIN-BW//2, ROW["report"]-44
x1,y1 = CX_MAIN+BW//2, ROW["report"]+44
rounded_rect(draw, [x0,y0,x1,y1], fill=(26,14,46), outline=C_LLM, width=2)
center_text(draw, CX_MAIN, ROW["report"]-22, "Reporting Layer", F_SECT, C_LLM)
center_text(draw, CX_MAIN, ROW["report"]+2,   "EXAONE4-32b  (vLLM / Ollama)",     F_BODY, C_LLM)
center_text(draw, CX_MAIN, ROW["report"]+24,  "Intelligence report + KO translation", F_SMALL, C_DIM)

# ─── 7. REPORTS DB  y~810 ─────────────────────────────────────────────────────
arrow_v(draw, CX_MAIN, ROW["report"]+46, ROW["reportdb"]-34)
x0,y0 = CX_MAIN-BW//2, ROW["reportdb"]-34
x1,y1 = CX_MAIN+BW//2, ROW["reportdb"]+34
rounded_rect(draw, [x0,y0,x1,y1], fill=(17,20,50), outline=C_DB, width=2)
center_text(draw, CX_MAIN, ROW["reportdb"]-12, "Reports DB  (SQLite)", F_SECT, C_DB)
center_text(draw, CX_MAIN, ROW["reportdb"]+12, "report_records · analysis sessions", F_SMALL, C_DIM)

# ─── 8. WEB API  y~920 ────────────────────────────────────────────────────────
arrow_v(draw, CX_MAIN, ROW["reportdb"]+36, ROW["webapi"]-34)
x0,y0 = CX_MAIN-BW//2, ROW["webapi"]-34
x1,y1 = CX_MAIN+BW//2, ROW["webapi"]+34
rounded_rect(draw, [x0,y0,x1,y1], fill=(6,28,30), outline=C_API, width=2)
center_text(draw, CX_MAIN, ROW["webapi"]-12, "Web API  (Flask + aiohttp)", F_SECT, C_API)
center_text(draw, CX_MAIN, ROW["webapi"]+12, "REST /api/*  ·  /stream  ·  /ws", F_SMALL, C_DIM)

# ─── 9. DASHBOARD  y~1050 ─────────────────────────────────────────────────────
arrow_v(draw, CX_MAIN, ROW["webapi"]+36, ROW["dash"]-44)
x0,y0 = CX_MAIN-BW//2, ROW["dash"]-44
x1,y1 = CX_MAIN+BW//2, ROW["dash"]+44
rounded_rect(draw, [x0,y0,x1,y1], fill=(6,18,42), outline=C_UI, width=2)
center_text(draw, CX_MAIN, ROW["dash"]-22, "Dashboard  (HTML / JS)", F_SECT, C_UI)
center_text(draw, CX_MAIN, ROW["dash"]+2,   "Leaflet map  ·  Report viewer",   F_BODY, C_UI)
center_text(draw, CX_MAIN, ROW["dash"]+24,  "Matching visualizer  ·  Mobile UI", F_SMALL, C_DIM)

# ─── RIGHT SIDE PANELS ────────────────────────────────────────────────────────
# Panel helper
def side_box(cx, cy, w, h, title, lines, col, fill_col=None):
    fc = fill_col or (16,16,32)
    x0,y0 = cx-w//2, cy-h//2
    x1,y1 = cx+w//2, cy+h//2
    rounded_rect(draw, [x0,y0,x1,y1], fill=fc, outline=col, width=2)
    lh = (h-22)//max(len(lines),1)
    ty = y0+14
    bb = draw.textbbox((0,0), title, font=F_SECT)
    tw = bb[2]-bb[0]
    draw.text((cx-tw//2, ty), title, font=F_SECT, fill=col)
    ty += 24
    for l in lines:
        bb2 = draw.textbbox((0,0), l, font=F_TINY)
        tw2 = bb2[2]-bb2[0]
        draw.text((cx-tw2//2, ty), l, font=F_TINY, fill=C_DIM)
        ty += 18

# ─── Config panel (right, aligned with pairing) ──────────────────────────────
CX_R1 = 1200; CY_R1 = (ROW["detect"] + ROW["pair"])//2 - 20
side_box(CX_R1, CY_R1, 440, 250, "Config  (src/config.py)",
    ["SAM3_MODEL_NAME  ·  SAM3_DEVICE",
     "TILE_ENABLED · TILE_SIZE · TILE_OVERLAP",
     "MAX_BBOX_AREA_RATIO  (0.15 / 0.70)",
     "TRACKING_MODE  (sam3_tracker | similarity)",
     "LLM_BACKEND  (vllm | ollama)",
     "COORDINATE_MATCH_RADIUS · MOVE_THRESHOLD",
     "DOCTRINE_ENABLED  ·  DOCTRINE_TOP_K"],
    C_CFG, fill_col=(22,16,6))

# Arrow: Config → Detection
arrow_h(draw, CX_R1-220, CX_MAIN+BW//2+2, ROW["detect"], C_CFG)

# ─── Doctrine RAG panel ───────────────────────────────────────────────────────
CX_R2 = 1200; CY_R2 = ROW["report"]
side_box(CX_R2, CY_R2, 440, 170, "Doctrine RAG  (FAISS)",
    ["Military doctrine documents",
     "FAISS vector index  ·  SBERT embeddings",
     "Top-K chunks → LLM prompt context",
     "build_doctrine_vectordb.py"],
    C_CFG, fill_col=(22,16,6))

# Arrow: DocRAG → Reporting
arrow_h(draw, CX_R2-220, CX_MAIN+BW//2+2, ROW["report"], C_CFG)

# ─── SR panel (Super-Resolution) ─────────────────────────────────────────────
CX_SR = 1200; CY_SR = ROW["ingest"]
side_box(CX_SR, CY_SR, 440, 110, "Super-Resolution",
    ["Real-ESRGAN  (optional)  →  PIL LANCZOS fallback",
     f"Target: 8000×6000 px before SAM3"],
    C_DETECT, fill_col=(10,24,18))

# Arrow: SR → Detection (horizontal)
arrow_h(draw, CX_SR-220, CX_MAIN+BW//2+2, ROW["detect"]-18, C_DETECT)

# ─── CLIP / Similarity panel ─────────────────────────────────────────────────
CX_CL = 1200; CY_CL = ROW["pairdb"]
side_box(CX_CL, CY_CL, 440, 140, "Similarity Tracker  (Mode B)",
    ["CLIP  openai/clip-vit-base-patch16",
     "Mask-crop → image embedding",
     "cosine sim · geo proximity · size sim",
     "Static class geo pre-matching (Step 0)"],
    C_PAIR, fill_col=(8,26,34))

# Arrow: CLIP → Pairing
arrow_h(draw, CX_CL-220, CX_MAIN+BW//2+2, ROW["pair"]+10, C_PAIR)

# ─── Divider line between left pipeline and right panels ──────────────────────
draw.line([(930, 75), (930, H-40)], fill=C_BORDER, width=1)

# ─── Title labels for right section ─────────────────────────────────────────
section_label(draw, 1200, 76, "Support Components & Configuration", C_DIM)

# ─── LEGEND ──────────────────────────────────────────────────────────────────
legend_items = [
    ("Input / Sensors",      C_INPUT),
    ("Detection (SAM3)",     C_DETECT),
    ("Database (SQLite)",    C_DB),
    ("Pairing Engine",       C_PAIR),
    ("LLM / Reporting",      C_LLM),
    ("Web API",              C_API),
    ("Dashboard / UI",       C_UI),
    ("Config / RAG",         C_CFG),
]
lx = 70; ly = H - 55
draw.text((lx, ly-18), "Legend:", font=F_SMALL, fill=C_DIM)
for i, (label, color) in enumerate(legend_items):
    bx = lx + i * 198
    draw.rounded_rectangle([bx, ly, bx+14, ly+14], radius=3, fill=color)
    draw.text((bx+18, ly), label, font=F_TINY, fill=C_TXT)

# ─── Bottom line ─────────────────────────────────────────────────────────────
draw.line([(60, H-70), (W-60, H-70)], fill=C_BORDER, width=1)
draw.text((60, H-62), "github.com/Parkdev22222/multi-source-intelligent-system", font=F_TINY, fill=C_DIM)

# ─── Save ─────────────────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(__file__), "..", "docs")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "architecture.png")
img.save(out_path, "PNG", optimize=True)
print(f"Saved: {out_path}  ({W}×{H})")
