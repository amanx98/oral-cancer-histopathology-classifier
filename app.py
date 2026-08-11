import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

st.set_page_config(page_title="Oral Cancer Histopathology Classifier", layout="centered")

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
    model = efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(num_features, 2))
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model

model = load_model()

gradients = []
activations = []

def save_gradient(module, grad_input, grad_output):
    gradients.append(grad_output[0])

def save_activation(module, input, output):
    activations.append(output)

target_layer = model.features[-1]
target_layer.register_forward_hook(save_activation)
target_layer.register_full_backward_hook(save_gradient)

def unnormalize(img_tensor):
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(imagenet_std) + np.array(imagenet_mean)
    return np.clip(img, 0, 1)

def generate_gradcam(image_tensor, class_idx):
    gradients.clear()
    activations.clear()
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

st.title("Oral Cancer Histopathology Classifier")
st.markdown(
    "Upload an H&E-stained oral tissue image. Classifies **Normal vs OSCC** "
    "(Oral Squamous Cell Carcinoma) and shows which regions influenced the decision.\n\n"
    "**Test accuracy:** 88% | **ROC-AUC:** 0.937\n\n"
    "⚠️ Research/portfolio demo only — not a diagnostic tool. "
    "Best results with high-resolution, uncompressed images."
)

uploaded_file = st.file_uploader("Upload a slide image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")
    st.image(input_image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Analyzing..."):
        img_tensor = eval_transform(input_image).to(device)

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
        overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))

    label = class_names[pred_idx]
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prediction", label)
    with col2:
        st.metric("Confidence", f"{confidence:.1%}")

    st.image(overlay_img, caption="Grad-CAM: what the model focused on", use_container_width=True)