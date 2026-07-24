import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import random
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Astronomical Image Classifier", page_icon="🔭", layout="centered")

CLASSES = ['Elliptical', 'Nebula', 'Planet', 'Spiral', 'Star Cluster']
WEIGHTS_PATH = "best_model.pth"


# -----------------------------
# Space background
# -----------------------------
def _make_star_layer(n_stars, seed, max_x=100, max_y=100):
    """Generate a CSS box-shadow string that draws n_stars single-pixel dots
    scattered sparsely across the far edges of the page. Coordinates are in
    vw/vh (viewport-relative units), not raw pixels, so the pattern scales
    correctly to any screen width instead of being confined to one corner."""
    rng = random.Random(seed)
    points = [f"{rng.uniform(0, max_x):.2f}vw {rng.uniform(0, max_y):.2f}vh #FFF" for _ in range(n_stars)]
    return ", ".join(points)


def _band_xy(t, max_x, max_y, lean=0.32):
    """Position along the near-vertical galactic core band at parameter
    t in [0, 1], with a gentle diagonal lean (matching a real Milky Way
    band photo) rather than a perfectly straight vertical line."""
    bx = max_x * 0.5 + (0.5 - t) * max_x * lean
    by = t * max_y
    return bx, by


def _make_band_star_layer(n_stars, seed, color, t_min, t_max, max_x=100, max_y=100, spread=9):
    """Generate box-shadow star points clustered along a sub-range [t_min,
    t_max] of the core band, with Gaussian jitter perpendicular to the
    band's lean. Coordinates are in vw/vh so the band spans the full page
    width regardless of screen size. Used to build a color-graded star
    field: warm gold/copper density at the core, cooling to amethyst/
    violet/cobalt further out."""
    rng = random.Random(seed)
    points = []
    for _ in range(n_stars):
        t = rng.uniform(t_min, t_max)
        bx, by = _band_xy(t, max_x, max_y)
        x = max(0, min(max_x, bx + rng.gauss(0, spread)))
        y = max(0, min(max_y, by + rng.gauss(0, spread * 0.6)))
        points.append(f"{x:.2f}vw {y:.2f}vh {color}")
    return ", ".join(points)


def inject_space_background():
    # Sparse outer starfield, kept low-density so page edges stay clear for text
    edge_stars = _make_star_layer(160, seed=42)

    # Dense core band, color-graded along its length:
    # cobalt/violet at the top -> amethyst -> warm gold/copper core -> amethyst -> deep purple at the bottom
    band_cobalt_top     = _make_band_star_layer(110, seed=11, color="#9fc4ff", t_min=0.00, t_max=0.20, spread=13)
    band_violet_upper   = _make_band_star_layer(140, seed=12, color="#b79bff", t_min=0.12, t_max=0.35, spread=12)
    band_gold_core       = _make_band_star_layer(320, seed=13, color="#ffe3b0", t_min=0.30, t_max=0.68, spread=9)
    band_copper_core     = _make_band_star_layer(220, seed=14, color="#ffb27a", t_min=0.32, t_max=0.66, spread=7)
    band_white_core      = _make_band_star_layer(260, seed=15, color="#ffffff", t_min=0.28, t_max=0.70, spread=10)
    band_amethyst_lower = _make_band_star_layer(140, seed=16, color="#c79bff", t_min=0.62, t_max=0.85, spread=12)
    band_purple_bottom = _make_band_star_layer(110, seed=17, color="#7d6bd8", t_min=0.80, t_max=1.00, spread=13)

    st.html(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;500;600&display=swap');

        .stApp {{
            background: radial-gradient(ellipse at center, #0a0812 0%, #050308 55%, #000000 100%);
            overflow: hidden;
        }}

        /* Galactic core band: a dense, near-vertical convergence of glowing
           nebulosity, color-graded from cool cobalt/violet at the ends to
           warm gold/copper at the core, with dark dust lanes woven through.
           Built entirely from layered, blurred CSS gradients. */
        .core-layer {{
            position: fixed;
            inset: 0;
            z-index: 0;
            pointer-events: none;
            overflow: hidden;
        }}
        .core-glow {{
            position: absolute;
            top: -25%;
            left: -40%;
            width: 180%;
            height: 150%;
            transform: rotate(16deg);
            background:
                radial-gradient(ellipse 38% 14% at 50% 12%, rgba(159,196,255,0.28) 0%, rgba(159,196,255,0) 75%),
                radial-gradient(ellipse 44% 17% at 47% 28%, rgba(183,155,255,0.32) 0%, rgba(183,155,255,0) 75%),
                radial-gradient(ellipse 55% 22% at 50% 45%, rgba(255,227,176,0.48) 0%, rgba(255,227,176,0) 75%),
                radial-gradient(ellipse 50% 20% at 51% 55%, rgba(255,178,122,0.42) 0%, rgba(255,178,122,0) 75%),
                radial-gradient(ellipse 46% 18% at 48% 68%, rgba(199,155,255,0.32) 0%, rgba(199,155,255,0) 75%),
                radial-gradient(ellipse 40% 15% at 50% 85%, rgba(125,107,216,0.28) 0%, rgba(125,107,216,0) 75%);
            filter: blur(16px);
            animation: core-drift 110s ease-in-out infinite alternate;
        }}
        .core-brightcenter {{
            position: absolute;
            top: -25%;
            left: -40%;
            width: 180%;
            height: 150%;
            transform: rotate(16deg);
            background: radial-gradient(ellipse 24% 9% at 49% 48%, rgba(255,244,220,0.55) 0%, rgba(255,244,220,0) 78%);
            filter: blur(9px);
            animation: pulse-core 8s ease-in-out infinite;
        }}
        .dust-lanes {{
            position: absolute;
            top: -25%;
            left: -40%;
            width: 180%;
            height: 150%;
            transform: rotate(16deg);
            background:
                radial-gradient(ellipse 12% 24% at 47% 30%, rgba(0,0,0,0.5) 0%, rgba(0,0,0,0) 80%),
                radial-gradient(ellipse 11% 20% at 52% 47%, rgba(0,0,0,0.45) 0%, rgba(0,0,0,0) 80%),
                radial-gradient(ellipse 13% 22% at 46% 63%, rgba(0,0,0,0.48) 0%, rgba(0,0,0,0) 80%),
                radial-gradient(ellipse 10% 18% at 51% 78%, rgba(0,0,0,0.40) 0%, rgba(0,0,0,0) 80%);
            filter: blur(10px);
            mix-blend-mode: multiply;
        }}
        @keyframes core-drift {{
            0%   {{ transform: rotate(16deg) translate(0, 0); }}
            100% {{ transform: rotate(16deg) translate(-14px, 10px); }}
        }}
        @keyframes pulse-core {{
            0%, 100% {{ opacity: 0.7; }}
            50%      {{ opacity: 1; }}
        }}

        /* Star layers: sparse at the far edges, densening and color-shifting
           toward the core band, per _band_xy positioning above */
        .edge-stars, .band-cobalt, .band-violet-u, .band-gold, .band-copper,
        .band-white, .band-amethyst-l, .band-purple-b {{
            position: fixed;
            top: 0; left: 0;
            width: 1px; height: 1px;
            border-radius: 50%;
            z-index: 0;
            pointer-events: none;
        }}
        .edge-stars     {{ background: transparent; box-shadow: {edge_stars};          animation: twinkle 6.5s ease-in-out infinite; }}
        .band-cobalt     {{ background: transparent; box-shadow: {band_cobalt_top};     animation: twinkle 5s ease-in-out infinite; }}
        .band-violet-u   {{ background: transparent; box-shadow: {band_violet_upper};   animation: twinkle 5.5s ease-in-out infinite reverse; }}
        .band-gold       {{ background: transparent; box-shadow: {band_gold_core};      animation: twinkle 4s ease-in-out infinite; }}
        .band-copper     {{ background: transparent; box-shadow: {band_copper_core};    animation: twinkle 4.5s ease-in-out infinite reverse; }}
        .band-white      {{ background: transparent; box-shadow: {band_white_core};     animation: twinkle 3.5s ease-in-out infinite; }}
        .band-amethyst-l {{ background: transparent; box-shadow: {band_amethyst_lower}; animation: twinkle 5.5s ease-in-out infinite; }}
        .band-purple-b   {{ background: transparent; box-shadow: {band_purple_bottom};  animation: twinkle 6s ease-in-out infinite reverse; }}
        @keyframes twinkle {{
            0%, 100% {{ opacity: 0.9; }}
            50%      {{ opacity: 0.3; }}
        }}

        /* Keep actual app content above the background layers, in a
           slightly translucent panel so text stays fully readable while
           the galactic core glows through behind it */
        .block-container {{
            position: relative;
            z-index: 2;
            background: rgba(6, 6, 14, 0.50);
            border: 1px solid rgba(255, 225, 180, 0.14);
            border-radius: 18px;
            padding: 2rem 2.5rem !important;
            backdrop-filter: blur(6px);
        }}

        /* Futuristic typography */
        html, body, [class*="css"] {{
            font-family: 'Rajdhani', sans-serif;
        }}
        h1 {{
            font-family: 'Orbitron', sans-serif !important;
            font-weight: 900 !important;
            letter-spacing: 0.04em;
            background: linear-gradient(90deg, #ffe3b0, #ffb27a, #c79bff, #9fc4ff);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 30px rgba(255, 178, 122, 0.35);
        }}
        h2, h3 {{
            font-family: 'Orbitron', sans-serif !important;
            font-weight: 700 !important;
            letter-spacing: 0.03em;
            color: #ffe3b0 !important;
            text-shadow: 0 0 14px rgba(255, 210, 140, 0.25);
        }}
        p, label, .stMarkdown, span {{
            color: #d9dcf5 !important;
            font-weight: 500;
        }}

        @media (prefers-reduced-motion: reduce) {{
            .core-glow, .core-brightcenter, .edge-stars, .band-cobalt, .band-violet-u,
            .band-gold, .band-copper, .band-white, .band-amethyst-l, .band-purple-b {{
                animation: none !important;
            }}
        }}
        </style>

        <div class="core-layer">
            <div class="core-glow"></div>
            <div class="core-brightcenter"></div>
            <div class="dust-lanes"></div>
        </div>
        <div class="edge-stars"></div>
        <div class="band-cobalt"></div>
        <div class="band-violet-u"></div>
        <div class="band-gold"></div>
        <div class="band-copper"></div>
        <div class="band-white"></div>
        <div class="band-amethyst-l"></div>
        <div class="band-purple-b"></div>
        """
    )


inject_space_background()


# -----------------------------
# Model definition (must match training exactly)
# -----------------------------
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * 4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * 4),
            )

        layers.append(Block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels * 4

        for _ in range(num_residual_blocks - 1):
            layers.append(Block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


def ResNet50(img_channels=3, num_classes=5):
    return ResNet(Block, [3, 4, 6, 3], img_channels, num_classes)


# -----------------------------
# Preprocessing (must match training exactly)
# -----------------------------
class IncreaseContrast(object):
    def __call__(self, img):
        return TF.adjust_contrast(img, contrast_factor=1.8)


inference_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    IncreaseContrast(),
    transforms.ToTensor(),
])


# -----------------------------
# Cached model loader
# -----------------------------
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet50(img_channels=3, num_classes=len(CLASSES)).to(device)
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


# -----------------------------
# UI
# -----------------------------
st.title("Astronomical Image Classifier")
st.write(
    "Upload an  image and this model will classify it as "
    "**Elliptical Galaxy**, **Nebula**, **Planet**, **Spiral Galaxy**, or **Star Cluster**, "
    "along with a heatmap showing which regions of the image drove the prediction."
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

try:
    model, device = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(
        f"Could not load model weights from '{WEIGHTS_PATH}'. "
        f"Make sure the weights file is uploaded to this Space. Error: {e}"
    )

if uploaded_file is not None and model_loaded:
    raw_image = Image.open(uploaded_file).convert("RGB")

    input_tensor = inference_transform(raw_image).unsqueeze(0).to(device)

    with st.spinner("Classifying..."):
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            confidence, predicted = torch.max(probs, 0)
            pred_class_idx = predicted.item()
            pred_class_name = CLASSES[pred_class_idx]

        # Grad-CAM (needs gradients, so run outside no_grad)
        target_layers = [model.layer4[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(pred_class_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        display_img = raw_image.resize((256, 256))
        rgb_img = np.float32(display_img) / 255.0
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    st.success(f"**Prediction: {pred_class_name}** ({confidence.item() * 100:.1f}% confidence)")

    st.subheader("All class probabilities")
    for cls, p in sorted(zip(CLASSES, probs.tolist()), key=lambda x: -x[1]):
        st.write(f"{cls}: {p * 100:.2f}%")
        st.progress(min(max(p, 0.0), 1.0))

    st.subheader("Input vs. Heatmap")
    col1, col2 = st.columns(2)
    with col1:
        st.image(display_img, caption="Input Image", use_container_width=True)
    with col2:
        st.image(cam_image, caption=f"Heatmap: {pred_class_name}", use_container_width=True)

elif not model_loaded:
    st.info("Fix the model weights issue above before uploading an image.")
else:
    st.info("Upload an image above to get started.")
