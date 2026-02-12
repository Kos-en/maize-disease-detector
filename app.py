import streamlit as st
from PIL import Image
import numpy as np
import os
import cv2
import json
import joblib
import pandas as pd
import onnxruntime as ort

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
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500&display=swap');
:root {
    --bg:       #0c0f08;
    --surface:  #141a0e;
    --border:   rgba(138,186,59,0.12);
    --lime:     #8aba3b;
    --lime-dim: rgba(138,186,59,0.15);
    --cream:    #e8e4d9;
    --muted:    rgba(232,228,217,0.45);
    --red:      #d94040;
    --amber:    #e6a817;
    --green:    #3daa5c;
}
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    font-family: 'Inter', sans-serif;
    color: var(--cream);
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Hero ── */
.hero {
    padding: 3rem 3.5rem 2.2rem;
    border-bottom: 1px solid var(--border);
}
.hero-row { display: flex; align-items: baseline; gap: 1rem; flex-wrap: wrap; }
.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem; font-weight: 700; margin: 0; color: var(--cream);
}
.hero h1 span { color: var(--lime); }
.hero-tag {
    font-size: 0.7rem; font-weight: 500; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--muted);
}
.hero p {
    font-size: 0.88rem; color: var(--muted); margin: 0.5rem 0 0;
    max-width: 640px; line-height: 1.6; font-weight: 300;
}

/* ── Panels ── */
.panel { padding: 2.5rem 3.5rem; }
.label {
    font-size: 0.62rem; font-weight: 500; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 1rem;
}

/* ── Uploader ── */
[data-testid="stFileUploader"] {
    border: 1.5px dashed rgba(138,186,59,0.3) !important;
    border-radius: 10px !important;
    background: var(--surface) !important;
    padding: 0.8rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(138,186,59,0.55) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span { color: var(--cream) !important; }
[data-testid="stBaseButton-secondary"] {
    background: var(--lime-dim) !important;
    border: 1px solid rgba(138,186,59,0.35) !important;
    color: var(--lime) !important; border-radius: 6px !important;
    font-size: 0.8rem !important;
}
[data-testid="stImage"] img {
    border-radius: 8px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}

/* ── Pipeline stages ── */
.stage {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.55rem 0.9rem; border-radius: 8px; margin-bottom: 0.5rem;
    font-size: 0.8rem; font-weight: 500;
    border-left: 3px solid transparent;
}
.stage-pass  { background: rgba(61,170,92,0.08); border-left-color: var(--green); color: var(--green); }
.stage-fail  { background: rgba(217,64,64,0.08); border-left-color: var(--red); color: var(--red); }
.stage-warn  { background: rgba(230,168,23,0.08); border-left-color: var(--amber); color: var(--amber); }
.stage-info  { background: rgba(138,186,59,0.06); border-left-color: var(--lime); color: var(--lime); }
.stage .tag  { font-size: 0.62rem; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.7; }

/* ── Result card ── */
.result-card {
    background: var(--surface);
    border: 1px solid var(--border); border-radius: 12px;
    padding: 1.6rem 1.8rem; margin-top: 1.2rem;
}
.result-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem; font-weight: 700; line-height: 1.1;
    color: var(--accent, var(--lime)); margin-bottom: 0.15rem;
}
.result-conf {
    font-size: 0.75rem; color: var(--muted); letter-spacing: 0.06em;
    text-transform: uppercase; margin-bottom: 1rem;
}
.bar-track {
    background: rgba(255,255,255,0.06); border-radius: 99px;
    height: 4px; margin-bottom: 1.4rem; overflow: hidden;
}
.bar-fill {
    height: 100%; border-radius: 99px;
    background: var(--accent, var(--lime));
    transition: width 0.6s ease;
}
.detail {
    background: rgba(255,255,255,0.025); border-radius: 8px;
    padding: 1rem 1.2rem; margin-top: 0.6rem;
    border-left: 2px solid var(--accent, var(--lime));
}
.detail h4 {
    font-size: 0.62rem; font-weight: 500; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--muted); margin: 0 0 0.35rem;
}
.detail p {
    font-size: 0.84rem; font-weight: 300; color: rgba(232,228,217,0.72);
    line-height: 1.6; margin: 0;
}

/* ── Empty state ── */
.empty {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; min-height: 300px; opacity: 0.35; text-align: center;
}
.empty-icon { font-size: 3rem; margin-bottom: 0.6rem; }
.empty-text { font-size: 0.85rem; color: var(--cream); max-width: 220px; line-height: 1.5; }

/* ── Footer ── */
.foot {
    border-top: 1px solid var(--border); padding: 1rem 3.5rem;
    display: flex; gap: 2.5rem; flex-wrap: wrap; align-items: center;
}
.foot-stat { font-size: 0.72rem; color: var(--muted); }
.foot-stat strong {
    display: block; font-size: 0.95rem; font-weight: 600;
    color: var(--lime); font-family: 'Space Grotesk', sans-serif;
}
</style>
""", unsafe_allow_html=True)


# ── Constants ──────────────────────────────────────────────────────────────────
IMGSZ = 320                    # must match training imgsz
MLND_OVERRIDE_THRESH  = 0.70   # ensemble must be >= this to override YOLO
MSV_RECHECK_THRESH    = 0.75   # higher bar when YOLO said MSV (it's a related class)

DISEASE_META = {
    "healthy_maize": {
        "label": "Healthy Maize", "color": "#3daa5c", "icon": "✅",
        "cause": "No disease detected. Normal green coloration and intact leaf structure.",
        "action": "Continue current management practices and scout weekly.",
        "risk": "No current risk",
    },
    "maize_blight": {
        "label": "Northern Leaf Blight", "color": "#c0522a", "icon": "🍂",
        "cause": "Fungal — Exserohilum turcicum. Long cigar-shaped tan lesions parallel to leaf margins.",
        "action": "Apply fungicide (azoxystrobin) at first sign. Remove debris. Rotate crops next season.",
        "risk": "15–70 % yield loss in severe outbreaks",
    },
    "maize_streak_virus": {
        "label": "Maize Streak Virus", "color": "#e6a817", "icon": "⚡",
        "cause": "Viral — transmitted by leafhopper Cicadulina mbila. Continuous pale-yellow streaks on youngest leaves.",
        "action": "No chemical cure. Control leafhoppers (imidacloprid). Plant resistant varieties.",
        "risk": "Up to 100 % yield loss in severe cases",
    },
    "mlnd": {
        "label": "Maize Lethal Necrosis", "color": "#d94040", "icon": "🚨",
        "cause": "Co-infection of MCMV + SCMV. Interrupted chlorotic stripes, rapid leaf-margin necrosis, and mottling.",
        "action": "Rogue and burn infected plants immediately. Quarantine zone. Enforce non-cereal rotation.",
        "risk": "Up to 100 % complete crop failure",
    },
}


def get_meta(class_name):
    key = class_name.lower().replace(" ", "_")
    for k, v in DISEASE_META.items():
        if k in key or key in k:
            return v
    return {"label": class_name, "color": "#8aba3b", "icon": "🔬",
            "cause": "—", "action": "Consult an agricultural extension officer.", "risk": "Unknown"}


# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    d = "weights"
    yolo = ort.InferenceSession(os.path.join(d, "maize_detector.onnx"))
    ens  = joblib.load(os.path.join(d, "mlnd_validation_ensemble.joblib"))
    sc   = joblib.load(os.path.join(d, "mlnd_scaler.joblib"))
    feats = pd.read_csv(os.path.join(d, "mlnd_feature_columns.csv"), header=None)[0].tolist()
    with open(os.path.join(d, "severity_config.json")) as f:
        sev = json.load(f)
    return yolo, ens, sc, feats, sev


# ── Crop to leaf ───────────────────────────────────────────────────────────────
def crop_to_leaf(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (30, 30, 30), (90, 255, 255))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img_bgr
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    px, py = int(w * 0.05), int(h * 0.05)
    crop = img_bgr[max(0, y - py):min(img_bgr.shape[0], y + h + py),
                   max(0, x - px):min(img_bgr.shape[1], x + w + px)]
    return crop if crop.size / img_bgr.size > 0.20 else img_bgr


# ── Feature extractor — 28 features, matches notebook v5 exactly ──────────────
def extract_leaf_features(img_bgr, expected_features):
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    total_px = h * w

    green_mask  = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    yellow_mask = cv2.inRange(hsv, (15, 50, 50), (35, 255, 255))
    necro_mask  = cv2.inRange(hsv, (0,  10, 10), (20, 200, 200))

    green_frac  = cv2.countNonZero(green_mask)  / total_px
    yellow_frac = cv2.countNonZero(yellow_mask) / total_px
    necro_frac  = cv2.countNonZero(necro_mask)  / total_px
    roughness   = cv2.Laplacian(gray, cv2.CV_64F).var()

    contours, _ = cv2.findContours(necro_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= 10]

    aspects, circs, centroids = [], [], []
    for cnt in valid:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        _, (cw, ch_), _ = cv2.minAreaRect(cnt)
        aspects.append(max(cw, ch_) / (min(cw, ch_) + 1e-6))
        circs.append((4 * np.pi * area) / (peri ** 2 + 1e-6))
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            centroids.append([M["m10"] / M["m00"] / w, M["m01"] / M["m00"] / h])

    elongated   = np.mean(aspects) if aspects else 0.0
    irregular   = 1.0 - np.mean(circs) if circs else 0.0
    boundary_sh = np.std(circs) if circs else 0.0

    patchiness = inf_distrib = 0.0
    if len(centroids) > 1:
        arr = np.array(centroids)
        patchiness  = np.std(arr, axis=0).mean()
        inf_distrib = np.mean(np.linalg.norm(arr - arr.mean(axis=0), axis=1))

    lesion_dist = len(valid) / (total_px / 1000 + 1e-6)
    transition  = 1.0 - boundary_sh

    # Clip to [0, 1]
    ch_v = np.clip(yellow_frac, 0, 1)
    gy_v = np.clip((green_frac / (yellow_frac + 1e-6)) / 10.0, 0, 1)
    ne_v = np.clip(necro_frac, 0, 1)
    ro_v = np.clip(roughness / 5000.0, 0, 1)
    pa_v = np.clip(patchiness, 0, 1)
    ld_v = np.clip(lesion_dist, 0, 1)
    el_v = np.clip(elongated / 10.0, 0, 1)
    ir_v = np.clip(irregular, 0, 1)
    id_v = np.clip(inf_distrib, 0, 1)
    bs_v = np.clip(boundary_sh, 0, 1)
    ts_v = np.clip(transition, 0, 1)

    raw = {
        # 11 base features
        "Chlorosis_Intensity":   ch_v,
        "Green_Yellow_Ratio":    gy_v,
        "Necrotic_Area":         ne_v,
        "Texture_Roughness":     ro_v,
        "Patchiness":            pa_v,
        "Lesion_Distribution":   ld_v,
        "Elongated_Streaks":     el_v,
        "Irregular_Lesions":     ir_v,
        "Infected_Distribution": id_v,
        "Boundary_Sharpness":    bs_v,
        "Transition_Smoothness": ts_v,
        # 13 engineered features
        "Chloro_x_Necrotic":     ch_v * ne_v,
        "Yellow_Severity":       ch_v * (1 - gy_v),
        "Damage_Spread":         ro_v * id_v,
        "Lesion_Complexity":     el_v * ir_v,
        "Visual_Distress":       (ne_v + ch_v + (1 - gy_v)) / 3.0,
        "Lesion_Spread":         ld_v * pa_v,
        "Boundary_Contrast":     bs_v - ts_v,
        "Necrotic_Green_Ratio":  np.clip(ne_v / (gy_v + 1e-6), 0, 5) / 5,
        "Infection_Density":     np.clip(id_v / (pa_v + 1e-6), 0, 5) / 5,
        "Chlorosis_sq":          ch_v ** 2,
        "Necrotic_sq":           ne_v ** 2,
        "Roughness_sq":          ro_v ** 2,
        # 4 stripe-discriminative features (MSV vs MLND)
        "Streak_Patch_Ratio":    np.clip(el_v / (pa_v + 1e-6), 0, 5) / 5,
        "Distrib_Uniformity":    id_v * (1 - pa_v),
        "Streak_Continuity":     el_v * ts_v,
        "Fragmentation_Index":   ir_v * pa_v,
    }

    available = {k: v for k, v in raw.items() if k in expected_features}
    return pd.DataFrame([available])[expected_features]


# ── Severity ───────────────────────────────────────────────────────────────────
def compute_severity(feature_row, cfg):
    score = 0.0
    for feat, weight in cfg["weights"].items():
        if feat not in feature_row.index:
            continue
        val = feature_row[feat]
        score += abs(weight) * (1.0 - val) if weight < 0 else weight * val
    score = float(np.clip(score, 0.0, 1.0))
    if score < cfg["thresh_mild"]:
        return score, "Mild"
    elif score < cfg["thresh_moderate"]:
        return score, "Moderate"
    else:
        return score, "Severe"


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-row">
        <h1>Maize <span>Diagnosis</span></h1>
        <span class="hero-tag">Cascaded Pipeline · Strathmore University</span>
    </div>
    <p>Upload a maize leaf photo. The system runs YOLO classification, then validates
       with a tabular ensemble — specifically catching MSV ↔ MLND confusion — and
       grades severity for confirmed MLND cases.</p>
</div>
""", unsafe_allow_html=True)

# ── Columns ────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2, gap="small")

with col_left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="label">Upload</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop a maize leaf photo",
                                type=["jpg", "jpeg", "png"],
                                label_visibility="collapsed")
    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
        st.markdown('<div class="label" style="margin-top:1.4rem">Source image</div>',
                    unsafe_allow_html=True)
        st.image(img_pil, use_container_width=True)
        w_img, h_img = img_pil.size
        st.caption(f"{w_img}×{h_img} px · {uploaded.name}")
    else:
        st.markdown("""
        <div class="empty">
            <div class="empty-icon">📷</div>
            <div class="empty-text">Upload a clear photo of a maize leaf to begin</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="label">Diagnosis</div>', unsafe_allow_html=True)

    if uploaded:
        with st.spinner("Running cascade..."):

            yolo_sess, ensemble, scaler, expected_feats, sev_cfg = load_pipeline()

            # ── Preprocess ─────────────────────────────────────────────────────
            img_cv  = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            img_cv  = crop_to_leaf(img_cv)
            img_320 = cv2.resize(img_cv, (IMGSZ, IMGSZ))

            # ── STAGE 1: YOLO ──────────────────────────────────────────────────
            blob = np.expand_dims(
                np.transpose(img_320[:, :, ::-1].astype(np.float32) / 255.0, (2, 0, 1)),
                axis=0,
            )
            probs = yolo_sess.run(None, {yolo_sess.get_inputs()[0].name: blob})[0][0]
            top1  = int(np.argmax(probs))
            conf  = float(probs[top1])

            meta_map = yolo_sess.get_modelmeta().custom_metadata_map
            if "names" in meta_map:
                import ast
                nd = ast.literal_eval(meta_map["names"])
                classes = [nd[i] for i in sorted(nd)]
            else:
                classes = ["Healthy_Maize", "Maize_Blight", "Maize_Streak_Virus", "MLND"]

            cls_name  = classes[top1]
            yolo_label = get_meta(cls_name)["label"]

            # ── STAGE 2: Ensemble (always runs) ────────────────────────────────
            features = extract_leaf_features(img_320, expected_feats)
            feat_sc  = scaler.transform(features)
            ens_pred = ensemble.predict(feat_sc)[0]
            ens_prob = ensemble.predict_proba(feat_sc)[0]
            mlnd_p   = float(ens_prob[1])

            stages_html = ""
            severity_html = ""
            final_cls  = cls_name
            final_conf = conf

            is_yolo_mlnd = cls_name.upper() == "MLND"
            is_yolo_msv  = "STREAK" in cls_name.upper()

            # ── Decision logic ─────────────────────────────────────────────────

            if is_yolo_mlnd:
                # YOLO said MLND → ensemble validates
                if ens_pred == 1:
                    stages_html += f"""
                    <div class="stage stage-pass">
                        <span class="tag">Stage 1</span> YOLO → MLND ({conf*100:.0f}%)
                    </div>
                    <div class="stage stage-pass">
                        <span class="tag">Stage 2</span> Ensemble confirmed MLND ({mlnd_p*100:.0f}%)
                    </div>"""
                else:
                    # False positive — YOLO said MLND but ensemble disagrees
                    # Check if it's likely MSV (the common confusion)
                    fallback = np.copy(probs)
                    fallback[top1] = 0.0
                    next_idx   = int(np.argmax(fallback))
                    next_cls   = classes[next_idx]
                    next_conf  = float(fallback[next_idx])
                    final_cls  = next_cls
                    final_conf = next_conf

                    is_msv_fallback = "STREAK" in next_cls.upper()
                    stages_html += f"""
                    <div class="stage stage-fail">
                        <span class="tag">Stage 1</span> YOLO → MLND ({conf*100:.0f}%) — rejected by ensemble
                    </div>
                    <div class="stage stage-warn">
                        <span class="tag">Stage 2</span> Ensemble says healthy ({(1-mlnd_p)*100:.0f}%)
                        {"— likely MSV misread as MLND (similar stripe patterns)" if is_msv_fallback else ""}
                    </div>
                    <div class="stage stage-info">
                        <span class="tag">Fallback</span> Reclassified → {get_meta(next_cls)['label']}
                    </div>"""

            elif is_yolo_msv:
                # YOLO said MSV → special recheck because MSV and MLND look similar
                if mlnd_p >= MSV_RECHECK_THRESH:
                    # Ensemble is very confident this is actually MLND, not MSV
                    final_cls = "MLND"
                    stages_html += f"""
                    <div class="stage stage-warn">
                        <span class="tag">Stage 1</span> YOLO → MSV ({conf*100:.0f}%)
                    </div>
                    <div class="stage stage-fail">
                        <span class="tag">Stage 2</span> Ensemble override → MLND ({mlnd_p*100:.0f}%)
                        — interrupted stripes suggest MLND, not continuous MSV streaks
                    </div>"""
                elif mlnd_p >= 0.40:
                    # Weak signal — keep MSV but warn
                    stages_html += f"""
                    <div class="stage stage-pass">
                        <span class="tag">Stage 1</span> YOLO → MSV ({conf*100:.0f}%)
                    </div>
                    <div class="stage stage-warn">
                        <span class="tag">Stage 2</span> Weak MLND signal ({mlnd_p*100:.0f}%)
                        — monitor closely, re-photograph in 3 days
                    </div>"""
                else:
                    stages_html += f"""
                    <div class="stage stage-pass">
                        <span class="tag">Stage 1</span> YOLO → MSV ({conf*100:.0f}%)
                    </div>
                    <div class="stage stage-pass">
                        <span class="tag">Stage 2</span> No MLND signal ({mlnd_p*100:.0f}%) — MSV confirmed
                    </div>"""

            else:
                # YOLO said Healthy or Blight
                if mlnd_p >= MLND_OVERRIDE_THRESH:
                    old_label = yolo_label
                    final_cls = "MLND"
                    stages_html += f"""
                    <div class="stage stage-info">
                        <span class="tag">Stage 1</span> YOLO → {old_label} ({conf*100:.0f}%)
                    </div>
                    <div class="stage stage-fail">
                        <span class="tag">Stage 2</span> Ensemble override → MLND ({mlnd_p*100:.0f}%)
                    </div>"""
                else:
                    stages_html += f"""
                    <div class="stage stage-pass">
                        <span class="tag">Stage 1</span> YOLO → {yolo_label} ({conf*100:.0f}%)
                    </div>
                    <div class="stage stage-pass">
                        <span class="tag">Stage 2</span> No MLND override needed ({mlnd_p*100:.0f}%)
                    </div>"""

            # ── STAGE 3: Severity (MLND only) ──────────────────────────────────
            if final_cls.upper() == "MLND":
                score, grade = compute_severity(features.iloc[0], sev_cfg)
                grade_color = {"Mild": "var(--green)", "Moderate": "var(--amber)", "Severe": "var(--red)"}.get(grade, "var(--lime)")
                stages_html += f"""
                <div class="stage" style="background:rgba(255,255,255,0.03);border-left-color:{grade_color};color:{grade_color}">
                    <span class="tag">Stage 3</span> Severity: {grade} (score {score:.2f})
                </div>"""

        # ── Render pipeline stages ─────────────────────────────────────────
        st.markdown(stages_html, unsafe_allow_html=True)

        # ── Result card ────────────────────────────────────────────────────
        meta = get_meta(final_cls)
        pct  = round(final_conf * 100, 1) if final_cls == cls_name else round(mlnd_p * 100, 1)
        bar  = int(np.clip(pct, 1, 100))

        st.markdown(f"""
        <div class="result-card" style="--accent:{meta['color']}">
            <div class="result-name">{meta['icon']}  {meta['label']}</div>
            <div class="result-conf">confidence · {pct}%</div>
            <div class="bar-track"><div class="bar-fill" style="width:{bar}%"></div></div>
            <div class="detail"><h4>Pathology</h4><p>{meta['cause']}</p></div>
            <div class="detail"><h4>Recommended action</h4><p>{meta['action']}</p></div>
            <div class="detail"><h4>Yield risk</h4><p>{meta['risk']}</p></div>
        </div>""", unsafe_allow_html=True)

        # ── Expandable diagnostics ─────────────────────────────────────────
        with st.expander("Diagnostic details"):

            st.markdown("**YOLO class probabilities**")
            yolo_df = pd.DataFrame({
                "Class": classes,
                "Probability": [f"{p*100:.2f}%" for p in probs],
            }).sort_values("Probability", ascending=False, key=lambda s: [float(x.strip('%')) for x in s]).reset_index(drop=True)
            st.dataframe(yolo_df, use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("**Ensemble MLND probability**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Healthy", f"{ens_prob[0]*100:.1f}%")
            c2.metric("MLND",    f"{mlnd_p*100:.1f}%",
                      delta="override active" if mlnd_p >= MLND_OVERRIDE_THRESH else "below threshold",
                      delta_color="normal" if mlnd_p >= MLND_OVERRIDE_THRESH else "inverse")
            c3.metric("Override at", f"{int(MLND_OVERRIDE_THRESH*100)}%")

            st.divider()
            st.markdown("**Extracted leaf features**")
            feat_vals = features.iloc[0]
            feat_df = pd.DataFrame({
                "Feature": expected_feats,
                "Value":   [round(float(feat_vals[f]), 4) for f in expected_feats],
                "Level":   ["🔴 High" if feat_vals[f] > 0.6
                            else "🟡 Mid" if feat_vals[f] > 0.3
                            else "🟢 Low" for f in expected_feats],
            })
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

    else:
        st.markdown("""
        <div class="empty">
            <div class="empty-icon">🔬</div>
            <div class="empty-text">Results appear here after upload</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="foot">
    <div class="foot-stat"><strong>3-Stage</strong>Cascade</div>
    <div class="foot-stat"><strong>27</strong>Features</div>
    <div class="foot-stat"><strong>LR+XGB+LGB+SVM</strong>Stack</div>
    <div class="foot-stat"><strong>320px</strong>Inference</div>
    <div class="foot-stat"><strong>MSV↔MLND</strong>Confusion Guard</div>
</div>
""", unsafe_allow_html=True)