import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ObjectLens · AI Detection",
    page_icon="🔍",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0d0d;
    color: #f0f0f0;
}

h1, h2, h3 { font-family: 'Space Mono', monospace; }

.stApp { background-color: #0d0d0d; }

.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}
.hero h1 {
    font-size: 2.8rem;
    letter-spacing: -1px;
    color: #f0f0f0;
    margin-bottom: 0.3rem;
}
.hero h1 span { color: #00e5a0; }
.hero p {
    color: #888;
    font-size: 1rem;
    font-weight: 300;
    max-width: 480px;
    margin: 0 auto;
}

.divider {
    border: none;
    border-top: 1px solid #222;
    margin: 1.5rem 0;
}

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 3px;
    color: #00e5a0;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.stat-grid {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin: 1rem 0;
}
.stat-card {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 14px 20px;
    flex: 1;
    min-width: 110px;
    text-align: center;
}
.stat-card .num {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    color: #00e5a0;
    line-height: 1;
}
.stat-card .label {
    font-size: 0.72rem;
    color: #666;
    margin-top: 4px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.obj-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #161616;
    border: 1px solid #222;
    border-radius: 8px;
    padding: 10px 16px;
    margin-bottom: 6px;
}
.obj-name { font-weight: 600; font-size: 0.95rem; }
.obj-conf {
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    color: #00e5a0;
}
.obj-count {
    background: #00e5a020;
    color: #00e5a0;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
}

.stButton > button {
    background: #00e5a0 !important;
    color: #0d0d0d !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 1px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 2rem !important;
    transition: opacity 0.15s ease !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.85 !important; }

[data-testid="stCameraInput"] label,
[data-testid="stFileUploader"] label {
    color: #aaa !important;
    font-size: 0.85rem !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 1px;
    color: #666;
}
.stTabs [aria-selected="true"] { color: #00e5a0 !important; }
</style>
""", unsafe_allow_html=True)


# ── 1. load_model ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load YOLOv8n (CPU-only, cached across reruns)."""
    model = YOLO("yolov8n.pt")
    model.to("cpu")
    return model


# ── 2. preprocess_image ───────────────────────────────────────────────────────
def preprocess_image(pil_img: Image.Image, max_size: int = 640) -> np.ndarray:
    """Resize image to at most max_size on the long side, return BGR ndarray."""
    w, h = pil_img.size
    scale = min(max_size / max(w, h), 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ── 3. run_detection ──────────────────────────────────────────────────────────
def run_detection(model, bgr_img: np.ndarray, conf_threshold: float = 0.35):
    """Run YOLOv8 inference; return raw results."""
    results = model(bgr_img, conf=conf_threshold, device="cpu", verbose=False)
    return results


# ── 4. draw_boxes ─────────────────────────────────────────────────────────────
def draw_boxes(bgr_img: np.ndarray, results) -> np.ndarray:
    """Draw bounding boxes, labels and confidence scores on the image."""
    img = bgr_img.copy()
    accent = (0, 229, 160)   # #00e5a0 in BGR

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = r.names[cls_id]
            text = f"{label}  {conf:.0%}"

            # Box
            cv2.rectangle(img, (x1, y1), (x2, y2), accent, 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 8, y1), accent, -1)

            # Label text
            cv2.putText(
                img, text,
                (x1 + 4, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (13, 13, 13), 1, cv2.LINE_AA
            )
    return img


# ── 5. display_results ────────────────────────────────────────────────────────
def display_results(results, annotated_bgr: np.ndarray):
    """Show annotated image, stats, and per-object breakdown."""
    # Convert back to RGB for Streamlit
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    st.image(annotated_rgb, use_container_width=True)

    # Aggregate detections
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = r.names[cls_id]
            conf = float(box.conf[0])
            detections.append({"label": label, "conf": conf})

    if not detections:
        st.info("No objects detected above the confidence threshold.")
        return

    total = len(detections)
    avg_conf = sum(d["conf"] for d in detections) / total
    unique_classes = len({d["label"] for d in detections})

    # ── Stats grid ──
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card"><div class="num">{total}</div><div class="label">Objects</div></div>
        <div class="stat-card"><div class="num">{unique_classes}</div><div class="label">Classes</div></div>
        <div class="stat-card"><div class="num">{avg_conf:.0%}</div><div class="label">Avg Conf</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Per-class breakdown ──
    st.markdown('<p class="section-label">Detected Objects</p>', unsafe_allow_html=True)

    from collections import defaultdict
    class_data = defaultdict(lambda: {"count": 0, "max_conf": 0.0})
    for d in detections:
        class_data[d["label"]]["count"] += 1
        class_data[d["label"]]["max_conf"] = max(class_data[d["label"]]["max_conf"], d["conf"])

    for label, info in sorted(class_data.items(), key=lambda x: -x[1]["count"]):
        st.markdown(f"""
        <div class="obj-row">
            <span class="obj-name">{label}</span>
            <span class="obj-conf">{info['max_conf']:.0%}</span>
            <span class="obj-count">×{info['count']}</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Download button ──
    st.markdown("<br>", unsafe_allow_html=True)
    pil_out = Image.fromarray(annotated_rgb)
    buf = io.BytesIO()
    pil_out.save(buf, format="PNG")
    st.download_button(
        label="⬇  Download Result",
        data=buf.getvalue(),
        file_name="detection_result.png",
        mime="image/png",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN UI
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
    <h1>Object<span>Lens</span></h1>
    <p>Real-time object detection powered by YOLOv8 — snap a photo or upload an image to get started.</p>
</div>
<hr class="divider">
""", unsafe_allow_html=True)

# ── Sidebar: settings ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-label">Settings</p>', unsafe_allow_html=True)
    conf_threshold = st.slider("Confidence threshold", 0.1, 0.95, 0.35, 0.05)
    max_size = st.select_slider("Max image size (px)", options=[320, 480, 640, 800], value=640)
    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.75rem;color:#444;font-family:\'Space Mono\',monospace;">YOLOv8n · CPU · Ultralytics</p>',
        unsafe_allow_html=True,
    )

# ── Load model ─────────────────────────────────────────────────────────────────
with st.spinner("Loading YOLOv8 model…"):
    model = load_model()

# ── Input tabs ─────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Image Input</p>', unsafe_allow_html=True)
tab_cam, tab_upload = st.tabs(["📷  Camera", "📁  Upload"])

pil_image = None

with tab_cam:
    cam_img = st.camera_input("Take a photo")
    if cam_img:
        pil_image = Image.open(cam_img).convert("RGB")

with tab_upload:
    uploaded = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "webp", "bmp"])
    if uploaded:
        pil_image = Image.open(uploaded).convert("RGB")

# ── Detection ──────────────────────────────────────────────────────────────────
if pil_image:
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    col_btn, _ = st.columns([1, 2])
    with col_btn:
        detect_btn = st.button("RUN DETECTION →")

    if detect_btn:
        with st.spinner("Running inference…"):
            bgr = preprocess_image(pil_image, max_size=max_size)
            results = run_detection(model, bgr, conf_threshold=conf_threshold)
            annotated = draw_boxes(bgr, results)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<p class="section-label">Results</p>', unsafe_allow_html=True)
        display_results(results, annotated)
