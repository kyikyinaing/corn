import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.transforms.functional import resize as tv_resize
from PIL import Image
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import io
import traceback

# -------------------------------------------------
# BASE DIRECTORY (this makes paths work everywhere)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------
# 1. Load class names
# -------------------------------------------------
class_names_path = os.path.join(BASE_DIR, "class_names.json")
with open(class_names_path, "r") as f:
    CLASS_NAMES = json.load(f)

NUM_CLASSES = len(CLASS_NAMES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# 2. Custom CNN (lecture-style Net)  (UNCHANGED)
# -------------------------------------------------
class Net(nn.Module):
    def __init__(self, num_classes=4):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.BN1 = nn.BatchNorm2d(10)

        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.BN2 = nn.BatchNorm2d(20)
        self.conv2_drop = nn.Dropout2d()

        # 224x224 -> 110x110 -> 53x53, channels=20
        self.fc1 = nn.Linear(20 * 53 * 53, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 20 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)  # log probabilities


# -------------------------------------------------
# 3. ResNet18 model definition (transfer learning)
# -------------------------------------------------
def create_resnet18(num_classes: int):
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# -------------------------------------------------
# 4. Load both models (cached)
# -------------------------------------------------
@st.cache_resource
def load_resnet_model():
    model = create_resnet18(NUM_CLASSES)
    model_path = os.path.join(BASE_DIR, "corn_resnet18_best.pth")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

@st.cache_resource
def load_customcnn_model():
    model = Net(num_classes=NUM_CLASSES)
    model_path = os.path.join(BASE_DIR, "corn_custom_cnn.pth")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# -------------------------------------------------
# 5. Preprocessing (MUST match training)
# -------------------------------------------------
IMG_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(image: Image.Image):
    img = image.convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0)  # (1, 3, H, W)
    return img


# -------------------------------------------------
# 6. Grad-CAM implementation (FIXED for ResNet18)
#    - No backward_hook
#    - Use autograd.grad
# -------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.activations = None

        def forward_hook(module, inp, out):
            self.activations = out  # keep tensor for autograd

        self.hook_handle = target_layer.register_forward_hook(forward_hook)

    def remove(self):
        self.hook_handle.remove()

    def generate(self, input_tensor, target_class=None):
        self.activations = None

        with torch.enable_grad():
            input_tensor = input_tensor.requires_grad_(True)
            output = self.model(input_tensor)

            if target_class is None:
                target_class = int(output.argmax(dim=1).item())

            score = output[0, target_class]

            if self.activations is None:
                raise RuntimeError("Grad-CAM failed: no activations captured. Check target layer.")

            grads = torch.autograd.grad(
                outputs=score,
                inputs=self.activations,
                retain_graph=True,
                allow_unused=True
            )[0]

            if grads is None:
                raise RuntimeError("Grad-CAM failed: gradients are None. Try different target layer.")

            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
            cam = torch.relu(cam)

            cam -= cam.min()
            if cam.max() != 0:
                cam /= cam.max()

            return cam  # (1,1,h,w)


def generate_gradcam_overlay(pil_image: Image.Image, model_type: str, target_class: int | None = None) -> Image.Image:
    """
    Returns a PIL image with Grad-CAM heatmap overlay for the selected model.
    """
    img_resized = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x = preprocess_image(img_resized).to(device)

    if model_type == "ResNet18 (Transfer Learning)":
        model = load_resnet_model()
        target_layer = model.layer4  # ✅ safest for ResNet18
    else:
        model = load_customcnn_model()
        target_layer = model.conv2

    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(x, target_class=target_class).squeeze().detach().cpu()
    gradcam.remove()

    cam_resized = tv_resize(
        cam.unsqueeze(0).unsqueeze(0),
        [IMG_SIZE, IMG_SIZE]
    ).squeeze().numpy()

    img_np = np.array(img_resized).astype(np.float32) / 255.0

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img_np)
    ax.imshow(cam_resized, cmap="jet", alpha=0.4)
    ax.axis("off")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    overlay = Image.open(buf)
    return overlay


# -------------------------------------------------
# 7. Model comparison (fill with your real test metrics)
# -------------------------------------------------
MODEL_METRICS = {
    "ResNet18 (Transfer Learning)": {
        "Test Accuracy (%)": 91.16,
        "Macro F1-score (%)": 88.09
    },
    "Custom CNN (Lecture-style)": {
        "Test Accuracy (%)": 82.81,
        "Macro F1-score (%)": 70.05
    },
}


# -------------------------------------------------
# 8. Prediction helper (UNCHANGED)
# -------------------------------------------------
def predict(image: Image.Image, model_type: str):
    x = preprocess_image(image).to(device)

    if model_type == "ResNet18 (Transfer Learning)":
        model = load_resnet_model()
        with torch.no_grad():
            outputs = model(x)  # logits
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    else:
        model = load_customcnn_model()
        with torch.no_grad():
            log_probs = model(x)  # log_softmax
            probs = torch.exp(log_probs).cpu().numpy()[0]

    top_idx = int(np.argmax(probs))
    top_label = CLASS_NAMES[top_idx]
    top_prob = float(probs[top_idx])
    return top_idx, top_label, top_prob, probs


# -------------------------------------------------
# 9. Streamlit UI (FIXED checkbox logic)
# -------------------------------------------------
st.set_page_config(page_title="Corn Disease Detector", layout="centered")
st.title("🌽 Corn Leaf Disease Detection")

st.sidebar.markdown("### **Choose Model**")

model_type = st.sidebar.selectbox(
    "",
    ("ResNet18 (Transfer Learning)", "Custom CNN (Lecture-style)")
)


with st.sidebar.expander("📊 Model Comparison (Test Set)", expanded=False):
    st.write("These values come from your Colab evaluation.")
    for name, metrics in MODEL_METRICS.items():
        st.markdown(f"**{name}**")
        for k, v in metrics.items():
            st.write(f"- {k}: {v}")
        st.write("---")

st.write(f"Current model: **{model_type}**")
st.write("Upload an image and the model will classify it as **Blight**, **Common Rust**, **Gray Leaf Spot**, or **Healthy**.")

# ---- Session state (so checkbox works after rerun) ----
if "pred_done" not in st.session_state:
    st.session_state.pred_done = False
    st.session_state.top_idx = None
    st.session_state.label = None
    st.session_state.confidence = None
    st.session_state.probs = None
    st.session_state.image = None
    st.session_state.model_type = None

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing..."):
            top_idx, label, confidence, probs = predict(image, model_type)

        st.session_state.pred_done = True
        st.session_state.top_idx = top_idx
        st.session_state.label = label
        st.session_state.confidence = confidence
        st.session_state.probs = probs
        st.session_state.image = image
        st.session_state.model_type = model_type

if st.session_state.pred_done:
    st.success(f"**Prediction:** {st.session_state.label}")
    st.write(f"Confidence: **{st.session_state.confidence:.2%}**")

    st.subheader("Class Probabilities")
    prob_table = {
        "Class": CLASS_NAMES,
        "Probability": [f"{p:.2%}" for p in st.session_state.probs]
    }
    st.table(prob_table)

    show_cam = st.checkbox("Show Grad-CAM heatmap for this model")

    if show_cam:
        try:
            with st.spinner("Generating Grad-CAM..."):
                overlay = generate_gradcam_overlay(
                    st.session_state.image,
                    st.session_state.model_type,
                    target_class=st.session_state.top_idx
                )
            st.image(
                overlay,
                caption=f"Grad-CAM ({st.session_state.model_type})",
                use_container_width=True
            )
        except Exception:
            st.error("Grad-CAM failed. Error log:")
            st.code(traceback.format_exc())
else:
    st.info("Please upload a corn leaf image to begin.")
