import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

st.set_page_config(page_title="Oral Cancer Histopathology Classifier", page_icon="🔬", layout="wide")

# ---------------------------------------------------------------------------
# THEME / CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500&display=swap');

.stApp {
    background: radial-gradient(circle at 20% 0%, #14101c 0%, #0B0D12 45%, #0B0D12 100%);
    color: #EDEAE4;
}
[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] { display: none; }

.eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #D6547A;
    margin-bottom: 0.4rem;
}
.hero-title {
    font-family: 'Source Serif 4', serif;
    font-weight: 600;
    font-size: 2.6rem;
    line-height: 1.1;
    color: #F4F1EA;
    margin-bottom: 0.6rem;
}
.hero-sub {
    font-family: 'Inter', sans-serif;
    color: #9A9FA8;
    font-size: 0.95rem;
    max-width: 640px;
    margin-bottom: 1.2rem;
}
.stat-strip {
    display: flex;
    gap: 2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #8A8F98;
    border-top: 1px solid #262A33;
    border-bottom: 1px solid #262A33;
    padding: 0.7rem 0;
    margin-bottom: 2rem;
}
.stat-strip b { color: #C9A8E0; font-weight: 500; }

.upload-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6C4A9E;
    margin-bottom: 0.5rem;
}

/* Circular "eyepiece" frame */
.scope {
    position: relative;
    width: 100%;
    aspect-ratio: 1 / 1;
    border-radius: 50%;
    overflow: hidden;
    border: 2px solid #3A2C52;
    box-shadow: 0 0 0 6px #0B0D12, 0 0 0 7px #262A33, inset 0 0 60px rgba(0,0,0,0.6);
    background: #14101c;
}
.scope img { width: 100%; height: 100%; object-fit: cover; }
.scope::before {
    content: "";
    position: absolute; inset: 0;
    background:
        linear-gradient(to right, transparent 49.5%, rgba(214,84,122,0.35) 49.5%, rgba(214,84,122,0.35) 50.5%, transparent 50.5%),
        linear-gradient(to bottom, transparent 49.5%, rgba(214,84,122,0.35) 49.5%, rgba(214,84,122,0.35) 50.5%, transparent 50.5%);
    pointer-events: none;
}
.scope-caption {
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6C7180;
    margin-top: 0.6rem;
}

/* Full/flat frame (toggle alt state) */
.flat {
    width: 100%;
    aspect-ratio: 1 / 1;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #262A33;
    background: #14101c;
}
.flat img { width: 100%; height: 100%; object-fit: contain; }

/* View-mode toggle, styled like a lab switch */
div[data-testid="stRadio"] > label { display: none; }
div[data-testid="stRadio"] > div {
    flex-direction: row;
    gap: 0;
    border: 1px solid #262A33;
    border-radius: 6px;
    width: fit-content;
    overflow: hidden;
}
div[data-testid="stRadio"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.4rem 0.9rem;
    margin: 0 !important;
    color: #8A8F98;
}
div[data-testid="stRadio"] label:has(input:checked) {
    background: #1E212B;
    color: #D6547A;
}

/* Readout panel */
.readout {
    font-family: 'JetBrains Mono', monospace;
    border: 1px solid #262A33;
    border-left: 4px solid var(--accent, #6C4A9E);
    background: #12141B;
    border-radius: 6px;
    padding: 1.4rem 1.6rem;
    margin-top: 1.5rem;
}
.readout-label { font-size: 0.7rem; letter-spacing: 0.14em; color: #6C7180; text-transform: uppercase; }
.readout-value { font-size: 2.1rem; font-weight: 700; color: var(--accent, #6C4A9E); margin: 0.15rem 0 0.8rem 0; }
.readout-conf { font-size: 0.95rem; color: #C9CDD6; }

.gauge-track { width: 100%; height: 8px; background: #1E212B; border-radius: 4px; overflow: hidden; margin-top: 0.4rem; }
.gauge-fill { height: 100%; border-radius: 4px; background: var(--accent, #6C4A9E); }

.disclaimer {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    color: #6C7180;
    border-top: 1px solid #262A33;
    padding-top: 1rem;
    margin-top: 2rem;
}

/* OOD warning panel */
.ood-warning {
    font-family: 'JetBrains Mono', monospace;
    border: 1px solid #E8543E;
    border-left: 4px solid #E8543E;
    background: #1C1213;
    border-radius: 6px;
    padding: 1.2rem 1.6rem;
    margin-top: 1rem;
    color: #F4A79A;
}
.ood-warning b { color: #E8543E; }
.ood-detail {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    color: #8A8F98;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
class_names = ["Normal", "OSCC"]
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

@st.cache_resource
def load_model():
    m = efficientnet_b0(weights=None)
    num_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(num_features, 2))
    m.load_state_dict(torch.load("model.pth", map_location=device))
    m = m.to(device)
    m.eval()
    return m

model = load_model()

@st.cache_resource
def load_ood_stats():
    feature_mean = np.load("feature_mean.npy")
    feature_cov_inv = np.load("feature_cov_inv.npy")
    threshold = float(np.load("ood_threshold.npy")[0])
    return feature_mean, feature_cov_inv, threshold

feature_mean, feature_cov_inv, ood_threshold = load_ood_stats()
feature_extractor = nn.Sequential(model.features, model.avgpool, nn.Flatten())
feature_extractor.eval()

gradients, activations = [], []
def save_gradient(module, grad_input, grad_output): gradients.append(grad_output[0])
def save_activation(module, input, output): activations.append(output)

target_layer = model.features[-1]
target_layer.register_forward_hook(save_activation)
target_layer.register_full_backward_hook(save_gradient)

def unnormalize(img_tensor):
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(imagenet_std) + np.array(imagenet_mean)
    return np.clip(img, 0, 1)

def generate_gradcam(image_tensor, class_idx):
    gradients.clear(); activations.clear()
    torch.set_grad_enabled(True)
    inp = image_tensor.unsqueeze(0).to(device)
    inp.requires_grad_()
    output = model(inp)
    model.zero_grad()
    output[0, class_idx].backward()
    grads = gradients[0].squeeze(0)
    acts = activations[0].squeeze(0)
    weights = grads.mean(dim=(1, 2))
    cam = torch.zeros(acts.shape[1:], dtype=torch.float32).to(device)
    for i, w in enumerate(weights):
        cam += w * acts[i]
    cam = torch.relu(cam).detach().cpu().numpy()
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    torch.set_grad_enabled(False)
    return cam

def color_check(pil_img):
    """Cheap heuristic: does this image's color palette resemble H&E staining
    (purple/violet hematoxylin + pink/magenta eosin)? Returns (passed, fraction)."""
    img_np = np.array(pil_img.resize((128, 128)))
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    hue, sat = hsv[:, :, 0], hsv[:, :, 1]

    # OpenCV hue range 0-179. H&E purples/pinks/magentas sit roughly 120-179 and wrap to 0-10.
    in_he_range = ((hue >= 120) | (hue <= 10)) & (sat > 25)
    fraction = in_he_range.mean()
    return fraction > 0.15, fraction

def mahalanobis_check(img_tensor):
    """Feature-space distance from the training distribution. Returns (passed, distance)."""
    with torch.no_grad():
        feat = feature_extractor(img_tensor.unsqueeze(0).to(device)).cpu().numpy()[0]
    diff = feat - feature_mean
    distance = float(diff @ feature_cov_inv @ diff.T)
    return distance <= ood_threshold, distance

def to_b64(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def scope_html(pil_img, caption, view_mode):
    b64 = to_b64(pil_img)
    frame_class = "scope" if view_mode == "Eyepiece" else "flat"
    return f"""
    <div class="{frame_class}"><img src="data:image/png;base64,{b64}" /></div>
    <div class="scope-caption">{caption}</div>
    """

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown('<div class="eyebrow">Histopathology · Binary Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Oral Cancer Tissue Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Upload an H&E-stained oral tissue slide. The model classifies '
    'Normal vs. OSCC (Oral Squamous Cell Carcinoma) tissue and highlights the regions that '
    'drove its decision.</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="stat-strip">'
    '<span>ARCHITECTURE&nbsp; <b>EfficientNet-B0</b></span>'
    '<span>TEST ACCURACY&nbsp; <b>88.0%</b></span>'
    '<span>ROC-AUC&nbsp; <b>0.937</b></span>'
    '<span>CLASSES&nbsp; <b>Normal / OSCC</b></span>'
    '</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------
# UPLOAD + INFERENCE
# ---------------------------------------------------------------------------
st.markdown('<div class="upload-label">Load slide image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(" ", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")
    img_tensor = eval_transform(input_image).to(device)

    # --- OOD gate: cheap color check first, then feature-space distance ---
    color_ok, color_frac = color_check(input_image)
    dist_ok, distance = mahalanobis_check(img_tensor)

    if not color_ok or not dist_ok:
        reason = []
        if not color_ok:
            reason.append("color palette doesn't resemble H&E staining")
        if not dist_ok:
            reason.append("feature pattern falls outside the training distribution")
        st.markdown(f"""
        <div class="ood-warning">
            <b>⚠ Not recognized as an oral tissue histopathology slide</b>
            <div class="ood-detail">Reason: {" and ".join(reason)}.<br>
            Color match: {color_frac:.0%} · Feature distance: {distance:.0f} (threshold: {ood_threshold:.0f})<br>
            Classification was skipped rather than forcing a Normal/OSCC label on an out-of-distribution image.</div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    with st.spinner("Analyzing tissue sample..."):
        with torch.no_grad():
            output = model(img_tensor.unsqueeze(0))
            probs = torch.softmax(output, dim=1)[0]
            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()

        cam = generate_gradcam(img_tensor, pred_idx)
        orig_img = unnormalize(img_tensor)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = np.clip(0.5 * orig_img + 0.5 * heatmap, 0, 1)

        orig_pil = Image.fromarray((orig_img * 255).astype(np.uint8))
        overlay_pil = Image.fromarray((overlay * 255).astype(np.uint8))

    label = class_names[pred_idx]
    accent = "#3FB8AF" if label == "Normal" else "#E8543E"

    view_mode = st.radio(
        "View mode", ["Eyepiece", "Full"], horizontal=True, label_visibility="collapsed"
    )

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown(scope_html(orig_pil, "Specimen · original", view_mode), unsafe_allow_html=True)
    with col2:
        st.markdown(scope_html(overlay_pil, "Grad-CAM · model focus", view_mode), unsafe_allow_html=True)

    st.markdown(f"""
    <div class="readout" style="--accent: {accent};">
        <div class="readout-label">Classification</div>
        <div class="readout-value">{label}</div>
        <div class="readout-conf">Confidence: {confidence:.1%}</div>
        <div class="gauge-track"><div class="gauge-fill" style="width:{confidence*100:.1f}%; --accent:{accent};"></div></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(
    '<div class="disclaimer">⚠️ Research / portfolio demo only — not a diagnostic tool. '
    'Best results with high-resolution, minimally compressed slide images.</div>',
    unsafe_allow_html=True
)
