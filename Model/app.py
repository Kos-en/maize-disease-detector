import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Maize Disease Detector",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & Root ── */
:root {
    --soil:    #1a1208;
    --bark:    #2d1f0e;
    --moss:    #2e4a1e;
    --leaf:    #4a7c2f;
    --lime:    #8aba3b;
    --streak:  #f0c040;
    --blight:  #c0522a;
    --healthy: #5ab552;
    --cream:   #f5efdf;
    --sand:    #e8dfc8;
    --mist:    rgba(245,239,223,0.07);
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--soil) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--cream);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Grain overlay ── */
body::before {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
    opacity: 0.35;
}

/* ── Hero header ── */
.hero {
    background: linear-gradient(160deg, var(--bark) 0%, var(--soil) 60%);
    border-bottom: 1px solid rgba(138,186,59,0.2);
    padding: 3.5rem 4rem 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 60% 80% at 90% 50%, rgba(74,124,47,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--lime);
    margin-bottom: 0.75rem;
    opacity: 0.9;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.4rem, 5vw, 4.2rem);
    font-weight: 900;
    line-height: 1.05;
    color: var(--cream);
    margin: 0 0 0.5rem;
}
.hero-title span { color: var(--lime); }
.hero-sub {
    font-size: 1rem;
    font-weight: 300;
    color: rgba(245,239,223,0.6);
    max-width: 520px;
    line-height: 1.6;
    margin-top: 0.6rem;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(138,186,59,0.12);
    border: 1px solid rgba(138,186,59,0.3);
    border-radius: 999px;
    padding: 0.3rem 0.85rem;
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--lime);
    margin-top: 1.2rem;
    letter-spacing: 0.04em;
}

/* ── Main layout ── */
.main-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
    min-height: calc(100vh - 200px);
}
.panel {
    padding: 3rem 4rem;
    border-right: 1px solid var(--mist);
}
.panel-right { border-right: none; }

/* ── Section labels ── */
.section-label {
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: rgba(245,239,223,0.35);
    margin-bottom: 1.2rem;
    display: flex; align-items: center; gap: 0.6rem;
}
.section-label::after {
    content: ''; flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(245,239,223,0.15), transparent);
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    border: 1.5px dashed rgba(138,186,59,0.35) !important;
    border-radius: 12px !important;
    background: rgba(46,74,30,0.1) !important;
    transition: border-color 0.2s, background 0.2s;
    padding: 1rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(138,186,59,0.65) !important;
    background: rgba(46,74,30,0.18) !important;
}
[data-testid="stFileUploaderDropzone"] { background: transparent !important; }
[data-testid="stFileUploaderDropzoneInstructions"] span { color: var(--sand) !important; }
[data-testid="stFileUploadDropzone"] p { color: rgba(245,239,223,0.5) !important; font-size: 0.9rem !important; }
[data-testid="stBaseButton-secondary"] {
    background: rgba(138,186,59,0.15) !important;
    border: 1px solid rgba(138,186,59,0.4) !important;
    color: var(--lime) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}

/* ── Uploaded image frame ── */
[data-testid="stImage"] img {
    border-radius: 10px !important;
    border: 1px solid rgba(245,239,223,0.1) !important;
}

/* ── Result card ── */
.result-card {
    background: linear-gradient(135deg, rgba(46,74,30,0.25) 0%, rgba(26,18,8,0.4) 100%);
    border: 1px solid rgba(138,186,59,0.2);
    border-radius: 14px;
    padding: 1.8rem 2rem;
    margin-top: 1.5rem;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--accent-color, var(--lime));
}
.result-disease {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent-color, var(--lime));
    line-height: 1.1;
    margin-bottom: 0.3rem;
}
.result-confidence {
    font-size: 0.82rem;
    font-weight: 500;
    color: rgba(245,239,223,0.45);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 1.4rem;
}
.confidence-bar-wrap {
    background: rgba(245,239,223,0.08);
    border-radius: 999px;
    height: 5px;
    margin-bottom: 1.6rem;
    overflow: hidden;
}
.confidence-bar {
    height: 100%;
    border-radius: 999px;
    background: var(--accent-color, var(--lime));
    transition: width 0.8s cubic-bezier(0.22,1,0.36,1);
}

/* ── Disease info blocks ── */
.info-block {
    background: rgba(245,239,223,0.04);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-top: 0.8rem;
    border-left: 3px solid var(--accent-color, var(--lime));
}
.info-block h4 {
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: rgba(245,239,223,0.4);
    margin: 0 0 0.5rem;
}
.info-block p {
    font-size: 0.88rem;
    font-weight: 300;
    color: rgba(245,239,223,0.75);
    line-height: 1.65;
    margin: 0;
}

/* ── Stat pills ── */
.stat-row {
    display: flex; gap: 0.7rem; flex-wrap: wrap;
    margin-top: 1rem;
}
.stat-pill {
    background: rgba(245,239,223,0.06);
    border: 1px solid rgba(245,239,223,0.1);
    border-radius: 999px;
    padding: 0.35rem 0.9rem;
    font-size: 0.75rem;
    font-weight: 400;
    color: rgba(245,239,223,0.55);
    display: flex; align-items: center; gap: 0.35rem;
}
.stat-pill strong { color: rgba(245,239,223,0.85); font-weight: 500; }

/* ── Empty state ── */
.empty-state {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    min-height: 320px;
    opacity: 0.4;
    text-align: center;
    gap: 1rem;
}
.empty-icon { font-size: 4rem; line-height: 1; }
.empty-text { font-size: 0.9rem; font-weight: 300; color: var(--sand); max-width: 240px; line-height: 1.5; }

/* ── Model info sidebar strip ── */
.model-strip {
    background: rgba(45,31,14,0.7);
    border-top: 1px solid var(--mist);
    padding: 1.2rem 4rem;
    display: flex; gap: 3rem; flex-wrap: wrap;
    align-items: center;
}
.model-stat { font-size: 0.75rem; color: rgba(245,239,223,0.4); }
.model-stat strong { display: block; font-size: 1.1rem; font-weight: 700; color: var(--lime); font-family: 'Playfair Display', serif; }

/* ── Spinner override ── */
[data-testid="stSpinner"] { color: var(--lime) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--soil); }
::-webkit-scrollbar-thumb { background: var(--moss); border-radius: 999px; }
</style>
""", unsafe_allow_html=True)

# ── Load model (cached) ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return YOLO("maize_disease_model_v2_final.pt")

# ── Disease metadata ───────────────────────────────────────────────────────────
DISEASE_META = {
    "maize_blight": {
        "label": "Northern Leaf Blight",
        "color": "#c0522a",
        "icon": "🍂",
        "cause": "Fungal — Exserohilum turcicum. Characterised by long cigar-shaped tan lesions running parallel to leaf margins.",
        "action": "Apply fungicide (e.g. azoxystrobin) early. Remove infected debris. Rotate crops next season.",
        "yield_risk": "15 – 70% yield loss in severe outbreaks",
    },
    "maize_streak_virus": {
        "label": "Maize Streak Virus",
        "color": "#f0c040",
        "icon": "⚡",
        "cause": "Viral — transmitted by the leafhopper Cicadulina mbila. Pale spots on youngest leaves merge into yellow streaks along veins.",
        "action": "No chemical cure. Control leafhopper vectors with imidacloprid. Plant resistant varieties (e.g. SEEDCO SC403). Rogue infected plants early.",
        "yield_risk": "Up to 100% yield loss in severe cases",
    },
    "healthy_maize": {
        "label": "Healthy Maize",
        "color": "#5ab552",
        "icon": "✅",
        "cause": "No disease detected. The leaf shows normal coloration and structure.",
        "action": "Continue current management. Monitor regularly — early detection is key to preventing spread.",
        "yield_risk": "No current risk",
    },
}

def get_meta(class_name: str) -> dict:
    key = class_name.lower().replace(" ", "_")
    for k, v in DISEASE_META.items():
        if k in key or key in k:
            return v
    # fallback
    return {"label": class_name, "color": "#8aba3b", "icon": "🔬",
            "cause": "—", "action": "Consult an agricultural extension officer.", "yield_risk": "Unknown"}

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Strathmore University · East African Field Research</div>
    <div class="hero-title">Maize Disease<br><span>Early Detection</span></div>
    <div class="hero-sub">Upload a maize leaf photograph for instant AI-powered diagnosis of Northern Leaf Blight and Maize Streak Virus.</div>
    <div class="hero-badge">🌽 &nbsp;YOLOv8n · 99.3% accuracy · 3 MB model · 0.9ms inference</div>
</div>
""", unsafe_allow_html=True)

# ── Two-column layout ──────────────────────────────────────────────────────────
col_left, col_right = st.columns(2, gap="small")

with col_left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">01 — Upload Image</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop a maize leaf photo here",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.markdown('<div class="section-label" style="margin-top:1.8rem">02 — Source Image</div>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)

        # stat pills
        w, h = img.size
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-pill">📐 <strong>{w} × {h}</strong> px</div>
            <div class="stat-pill">📁 <strong>{uploaded_file.name}</strong></div>
            <div class="stat-pill">🎨 <strong>RGB</strong></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📷</div>
            <div class="empty-text">Upload a clear photograph of a maize leaf to begin analysis</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="panel panel-right">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">03 — Detection Results</div>', unsafe_allow_html=True)

    if uploaded_file:
        with st.spinner("Analysing leaf..."):
            model = load_model()
            results = model(img)

        # ── Annotated image ──
        res_plotted = results[0].plot()
        st.image(res_plotted, caption="Model detections", use_container_width=True)

        # ── Parse top prediction ──
        result = results[0]
        if hasattr(result, 'probs') and result.probs is not None:
            # Classification model
            top_idx = int(result.probs.top1)
            confidence = float(result.probs.top1conf)
            class_name = result.names[top_idx]
        elif result.boxes is not None and len(result.boxes) > 0:
            # Detection model — pick highest-confidence box
            confs = result.boxes.conf.cpu().numpy()
            top_idx_box = int(np.argmax(confs))
            confidence = float(confs[top_idx_box])
            cls_idx = int(result.boxes.cls[top_idx_box].cpu().numpy())
            class_name = result.names[cls_idx]
        else:
            class_name = "healthy_maize"
            confidence = 1.0

        meta = get_meta(class_name)
        conf_pct = round(confidence * 100, 1)
        bar_width = int(confidence * 100)

        # ── Result card ──
        st.markdown(f"""
        <div class="result-card" style="--accent-color: {meta['color']}">
            <div class="result-disease">{meta['icon']}&nbsp; {meta['label']}</div>
            <div class="result-confidence">Confidence &nbsp;·&nbsp; {conf_pct}%</div>
            <div class="confidence-bar-wrap">
                <div class="confidence-bar" style="width:{bar_width}%"></div>
            </div>
            <div class="info-block">
                <h4>Pathology</h4>
                <p>{meta['cause']}</p>
            </div>
            <div class="info-block" style="margin-top:0.7rem">
                <h4>Recommended Action</h4>
                <p>{meta['action']}</p>
            </div>
            <div class="info-block" style="margin-top:0.7rem">
                <h4>Yield Risk</h4>
                <p>{meta['yield_risk']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🔬</div>
            <div class="empty-text">Detection results will appear here after you upload a leaf image</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer strip ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="model-strip">
    <div class="model-stat"><strong>YOLOv8n-cls</strong>Architecture</div>
    <div class="model-stat"><strong>99.3%</strong>Test Accuracy</div>
    <div class="model-stat"><strong>3.0 MB</strong>Model Size</div>
    <div class="model-stat"><strong>0.9 ms</strong>Inference Speed</div>
    <div class="model-stat"><strong>7,400</strong>Training Images</div>
    <div class="model-stat"><strong>East Africa</strong>Training Region</div>
</div>
""", unsafe_allow_html=True)