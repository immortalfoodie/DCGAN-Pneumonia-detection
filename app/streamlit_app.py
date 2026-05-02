"""Streamlit web app for diagnosis, GAN generation, performance dashboard, and literature comparison."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root (parent of this file's directory) is on sys.path
# so that `config`, `models`, `utils`, etc. are importable regardless of
# the working directory from which Streamlit is launched.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import io
import json
import random
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from fpdf import FPDF
from PIL import Image

from config import Config
from models.dcgan import Generator
from models.full_model import PneumoniaDetectionModel
from utils.augmentation import get_val_transforms
from utils.grad_cam import GradCAM
from utils.visualization import draw_boxes, encode_png_bytes
INFER_IMAGE_SIZE = 224



try:
    import pydicom
except Exception:
    pydicom = None


st.set_page_config(layout="wide", page_title="PneumoAI — Detection System", page_icon="🫁")


def apply_theme() -> None:
    """Inject neo-brutalist UI styling."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700;800&family=Archivo+Black&display=swap');
        :root {
            --bg: #080c1a;
            --panel: #121a30;
            --panel-2: #0e1528;
            --ink: #edf2ff;
            --muted: #9aa7cb;
            --accent: #ff5a36;
            --accent-2: #57a5ff;
            --accent-3: #f9d423;
            --line: #e6ecff;
            --shadow: #000000;
        }
        html, body, [class*="css"] {
            font-family: 'Space Grotesk', sans-serif;
            color: var(--ink);
        }
        h1, h2, h3, h4 {
            font-family: 'Archivo Black', 'Space Grotesk', sans-serif;
            letter-spacing: 0.3px;
            color: var(--ink);
        }
        .stApp {
            background:
                radial-gradient(circle at 10% 0%, rgba(249, 212, 35, 0.15), transparent 44%),
                radial-gradient(circle at 90% 8%, rgba(87, 165, 255, 0.20), transparent 42%),
                repeating-linear-gradient(
                    45deg,
                    rgba(237, 242, 255, 0.03) 0px,
                    rgba(237, 242, 255, 0.03) 2px,
                    transparent 2px,
                    transparent 14px
                ),
                var(--bg);
            color: var(--ink);
        }
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li {
            color: var(--ink);
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
        }
        .banner {
            background: linear-gradient(140deg, #111a33 0%, #0d1530 100%);
            border: 3px solid var(--line);
            border-radius: 20px;
            padding: 18px 22px;
            margin-bottom: 16px;
            box-shadow: 8px 8px 0 var(--shadow);
            position: relative;
            overflow: hidden;
            animation: riseIn 0.55s ease-out both;
        }
        .banner::after {
            content: '';
            position: absolute;
            right: -28px;
            top: -32px;
            width: 124px;
            height: 124px;
            border-radius: 50%;
            background: var(--accent-2);
            border: 3px solid var(--line);
            transform: rotate(-14deg);
        }
        .banner h2 {
            margin: 0;
            max-width: 82%;
        }
        .banner p {
            margin: 8px 0 0 0;
            max-width: 82%;
            color: var(--muted);
            font-weight: 600;
        }
        .metric-card {
            background: var(--panel-2);
            padding: 14px;
            border-radius: 16px;
            border: 3px solid var(--line);
            box-shadow: 6px 6px 0 var(--shadow);
            animation: riseIn 0.55s ease-out both;
        }
        .glass {
            background: var(--panel-2);
            border: 3px solid var(--line);
            border-radius: 16px;
            padding: 14px;
            margin-bottom: 12px;
            box-shadow: 6px 6px 0 var(--shadow);
            animation: riseIn 0.55s ease-out both;
        }
        .diag-pill {
            display: inline-block;
            padding: 8px 14px;
            border-radius: 999px;
            font-weight: 800;
            letter-spacing: 0.4px;
            border: 3px solid var(--line);
            color: #ffffff;
            box-shadow: 4px 4px 0 var(--shadow);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1120 0%, #121a33 100%);
            border-right: 3px solid var(--line);
        }
        .sidebar-brand {
            background: #0f1830;
            border: 3px solid var(--line);
            border-radius: 14px;
            box-shadow: 4px 4px 0 var(--shadow);
            padding: 10px 12px;
            margin: 8px 0 14px 0;
        }
        .sidebar-brand .kicker {
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 1.1px;
            color: var(--accent-3);
        }
        .sidebar-brand .title {
            font-family: 'Archivo Black', 'Space Grotesk', sans-serif;
            font-size: 18px;
            line-height: 1.1;
            color: var(--ink);
        }
        .stButton > button {
            border-radius: 12px !important;
            border: 3px solid var(--line) !important;
            background: var(--accent-3) !important;
            color: #10131d !important;
            font-weight: 700 !important;
            box-shadow: 4px 4px 0 var(--shadow) !important;
            transition: transform 0.12s ease, box-shadow 0.12s ease, background 0.12s ease;
        }
        .stButton > button:hover {
            transform: translate(2px, 2px);
            box-shadow: 2px 2px 0 var(--shadow) !important;
            background: #ffd32e !important;
        }
        [data-testid="stDownloadButton"] button {
            border-radius: 12px !important;
            border: 3px solid var(--line) !important;
            background: var(--accent) !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            box-shadow: 4px 4px 0 var(--shadow) !important;
        }
        [data-testid="stFileUploader"],
        [data-testid="stExpander"],
        [data-testid="stAlert"],
        [data-testid="stImage"],
        [data-testid="stPlotlyChart"] {
            background: var(--panel-2);
            border: 2px solid var(--line);
            border-radius: 14px;
            box-shadow: 5px 5px 0 var(--shadow);
            padding: 8px;
            overflow: hidden;
        }
        [data-testid="stMetric"] {
            background: var(--panel-2);
            border: 2px solid var(--line);
            border-radius: 12px;
            box-shadow: 4px 4px 0 var(--shadow);
            padding: 8px 10px;
        }
        [data-testid="stRadio"] label p,
        [data-testid="stSelectbox"] label p,
        [data-testid="stSlider"] label p {
            font-weight: 700;
            color: var(--ink);
        }
        [data-testid="stRadio"] div[role="radiogroup"] > label {
            border: 2px solid var(--line);
            border-radius: 10px;
            padding: 6px 8px;
            margin-bottom: 6px;
            background: rgba(10, 15, 30, 0.92);
        }
        .stCaption {
            color: #9aa7cb !important;
        }
        @keyframes riseIn {
            from {
                opacity: 0;
                transform: translateY(12px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        @media (max-width: 768px) {
            .block-container {
                padding-top: 1rem;
                padding-left: 0.8rem;
                padding-right: 0.8rem;
            }
            .banner::after {
                display: none;
            }
            .banner h2,
            .banner p {
                max-width: 100%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def banner(title: str, subtitle: str) -> None:
    st.markdown(
        f"<div class='banner'><h2>{title}</h2><p>{subtitle}</p></div>",
        unsafe_allow_html=True,
    )


def style_plotly(fig: go.Figure, title: str | None = None, height: int | None = None) -> go.Figure:
    """Apply consistent neo-brutalist chart styling across Plotly figures."""
    layout_kwargs = {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "#10172b",
        "font": {"family": "Space Grotesk, sans-serif", "color": "#edf2ff", "size": 13},
        "colorway": ["#57a5ff", "#f9d423", "#ff5a36", "#69e2b0", "#d8a7ff"],
        "legend": {"orientation": "h", "x": 0, "y": 1.04, "xanchor": "left", "yanchor": "bottom"},
        "margin": {"l": 36, "r": 24, "t": 56, "b": 30},
    }
    if title is not None:
        layout_kwargs["title"] = title
    if height is not None:
        layout_kwargs["height"] = height
    fig.update_layout(**layout_kwargs)

    cartesian_types = {"scatter", "bar", "histogram", "box", "violin", "heatmap"}
    if any(getattr(trace, "type", "") in cartesian_types for trace in fig.data):
        fig.update_xaxes(
            showline=True,
            linewidth=2,
            linecolor="#d8e1ff",
            mirror=True,
            gridcolor="rgba(216,225,255,0.14)",
            zeroline=False,
        )
        fig.update_yaxes(
            showline=True,
            linewidth=2,
            linecolor="#d8e1ff",
            mirror=True,
            gridcolor="rgba(216,225,255,0.14)",
            zeroline=False,
        )
    return fig


def architecture_block() -> None:
    """Show implemented architecture details in UI."""
    st.markdown(
        """
        <div class='glass'>
        <h4 style='margin-top:0;'>Implemented Architecture in This Build</h4>
            <p style='margin-bottom:8px; color:#c7d3f2;'>
        This project implements a unified pipeline with a GAN augmentation branch and a DenseNet-FPN inference branch.
        </p>
            <ul style='margin-top:0; color:#eaf0ff;'>
          <li><b>GAN branch:</b> DCGAN-style generator (latent z=100), dual discriminators (Double SGAN pattern), spectral normalization, self-attention, hinge loss, and gradient penalty.</li>
          <li><b>Core model:</b> DenseNet-121 backbone with Feature Pyramid Network (P3/P4/P5), plus dual heads for classification and anchor-free box prediction.</li>
          <li><b>Explainability:</b> Grad-CAM and Grad-CAM++ utilities with heatmap overlay support.</li>
          <li><b>Training stack:</b> Focal loss (classification), CIoU loss (detection), mixed precision, scheduler, checkpointing, and metrics dashboard outputs.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _unwrap_checkpoint_state(checkpoint_payload: object, expected_key: str | None = None) -> dict[str, torch.Tensor]:
    """Extract tensor-only state dict and normalize DataParallel key prefixes."""
    if not isinstance(checkpoint_payload, dict):
        return {}

    if expected_key and expected_key in checkpoint_payload and isinstance(checkpoint_payload[expected_key], dict):
        raw_state = checkpoint_payload[expected_key]
    else:
        raw_state = checkpoint_payload

    normalized_state: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        if not torch.is_tensor(value):
            continue
        normalized_key = key[7:] if key.startswith("module.") else key
        normalized_state[normalized_key] = value
    return normalized_state


def _load_state_dict_compat(
    module: torch.nn.Module,
    checkpoint_payload: object,
    expected_key: str | None = None,
) -> tuple[int, int]:
    """Load checkpoint with backward compatibility for legacy spectral norm keys."""
    incoming_state = _unwrap_checkpoint_state(checkpoint_payload, expected_key)
    model_state = module.state_dict()

    merged_state: dict[str, torch.Tensor] = {}
    matched = 0

    for key, current_tensor in model_state.items():
        candidate_keys = [key]
        if key.endswith("weight_orig"):
            candidate_keys.append(key[: -len("weight_orig")] + "weight")

        loaded_tensor = None
        for candidate in candidate_keys:
            tensor = incoming_state.get(candidate)
            if tensor is not None and tensor.shape == current_tensor.shape:
                loaded_tensor = tensor
                break

        if loaded_tensor is None:
            merged_state[key] = current_tensor
        else:
            merged_state[key] = loaded_tensor
            matched += 1

    module.load_state_dict(merged_state, strict=True)
    return matched, len(model_state)


@st.cache_resource
def load_model() -> PneumoniaDetectionModel | None:
    """Load trained unified model if checkpoint exists."""
    ckpt_path = Config.path(Config.CHECKPOINT_DIR) / "best_model.pth"
    if not ckpt_path.exists():
        return None

    model = PneumoniaDetectionModel().to(Config.DEVICE)
    payload = torch.load(ckpt_path, map_location=Config.DEVICE)
    matched, total = _load_state_dict_compat(model, payload, expected_key="model")
    if matched == 0:
        return None
    if matched < total:
        st.info(f"Loaded checkpoint in compatibility mode ({matched}/{total} tensors matched).")
    model.eval()
    return model


def load_inference_threshold() -> float:
    """Load calibrated threshold from metrics or model checkpoint if available."""
    metrics_path = Config.path(Config.CHECKPOINT_DIR) / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
            if "threshold" in payload:
                threshold = float(payload["threshold"])
                return threshold if 0.2 <= threshold <= 0.9 else 0.5

    ckpt_path = Config.path(Config.CHECKPOINT_DIR) / "best_model.pth"
    if ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location="cpu")
        if isinstance(payload, dict) and "threshold" in payload:
            threshold = float(payload["threshold"])
            return threshold if 0.2 <= threshold <= 0.9 else 0.5

    return 0.5


@st.cache_resource
def load_generator() -> Generator | None:
    """Load generator checkpoint if available."""
    ckpt_dir = Config.path(Config.CHECKPOINT_DIR)
    candidate_paths = [
        ckpt_dir / "generator_best.pth",
        ckpt_dir / "gan_last.pth",
    ]
    ckpt_path = next((path for path in candidate_paths if path.exists()), None)
    if ckpt_path is None:
        return None

    generator = Generator(latent_dim=Config.LATENT_DIM).to(Config.DEVICE)
    payload = torch.load(ckpt_path, map_location=Config.DEVICE)
    if isinstance(payload, dict) and payload.get("initialized_only", False):
        return None
    matched, total = _load_state_dict_compat(generator, payload, expected_key="generator")
    if matched == 0:
        return None
    if matched < total:
        st.info(f"Loaded generator in compatibility mode ({matched}/{total} tensors matched).")
    generator.eval()
    return generator


def _collect_training_xray_paths() -> list[Path]:
    """Return pneumonia/normal training images for lightweight demo fallback."""
    train_root = Config.resolve_data_dir() / "train"
    patterns = ("*.png", "*.jpg", "*.jpeg")
    subfolders = ("PNEUMONIA", "NORMAL")
    image_paths: list[Path] = []
    for class_name in subfolders:
        class_dir = train_root / class_name
        if not class_dir.exists():
            continue
        for pattern in patterns:
            image_paths.extend(class_dir.glob(pattern))
    return image_paths


def _create_demo_synthetic_images(count: int) -> list[np.ndarray]:
    """Create demo synthetic-like samples if GAN checkpoint is unavailable."""
    source_paths = _collect_training_xray_paths()
    if not source_paths:
        return []

    images: list[np.ndarray] = []
    selected = random.choices(source_paths, k=count)
    for source in selected:
        img = np.array(Image.open(source).convert("L").resize((Config.GAN_IMAGE_SIZE, Config.GAN_IMAGE_SIZE)))
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
        if random.random() < 0.5:
            img = cv2.GaussianBlur(img, (3, 3), sigmaX=random.uniform(0.1, 1.0))
        alpha = random.uniform(0.9, 1.2)
        beta = random.uniform(-12, 12)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        images.append(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    return images


def _is_noise_like_image(image_rgb: np.ndarray) -> bool:
    """Heuristic check for untrained GAN outputs that look like static noise."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    # Natural X-rays are relatively smooth with anatomical structure;
    # random GAN outputs tend to have very high local variation.
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var > 2500


def _enhance_xray_image(image_rgb: np.ndarray) -> np.ndarray:
    """Apply light contrast/sharpness enhancement for display quality."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(enhanced, 1.35, blur, -0.35, 0)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)


def _is_low_diversity_batch(images: list[np.ndarray]) -> bool:
    """Detect mode-collapse-like batches where outputs are near-identical."""
    if len(images) < 3:
        return False
    small = [cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (64, 64)).astype(np.float32) for img in images]
    dists: list[float] = []
    for i in range(len(small)):
        for j in range(i + 1, len(small)):
            dists.append(float(np.mean(np.abs(small[i] - small[j]))))
    return bool(dists) and float(np.mean(dists)) < 7.0


def read_uploaded_image(uploaded_file) -> np.ndarray:
    """Read image upload including DICOM support."""
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".dcm":
        if pydicom is None:
            raise RuntimeError("pydicom is not installed. Install it to use DICOM uploads.")
        ds = pydicom.dcmread(io.BytesIO(uploaded_file.read()))
        arr = ds.pixel_array.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        arr = (arr * 255).astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)

    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


def preprocess_for_model(image: np.ndarray) -> torch.Tensor:
    """Apply eval preprocessing and convert to model input tensor."""
    transform = get_val_transforms(INFER_IMAGE_SIZE)
    tensor = transform(image=image)["image"].unsqueeze(0)
    return tensor.to(Config.DEVICE)


def create_pdf_report(original: np.ndarray, overlay: np.ndarray, label: str, confidence: float) -> bytes:
    """Create downloadable PDF report for diagnosis results."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        orig_path = Path(tmp_dir) / "original.png"
        overlay_path = Path(tmp_dir) / "overlay.png"
        Image.fromarray(original).save(orig_path)
        Image.fromarray(overlay).save(overlay_path)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Pneumonia Detection Report", ln=True)

        pdf.set_font("Helvetica", size=11)
        pdf.cell(0, 8, f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.cell(0, 8, f"Prediction: {label}", ln=True)
        pdf.cell(0, 8, f"Confidence: {confidence:.2%}", ln=True)

        pdf.image(str(orig_path), x=10, y=50, w=90)
        pdf.image(str(overlay_path), x=110, y=50, w=90)

        return bytes(pdf.output(dest="S"))


def diagnose_page() -> None:
    banner("PneumoAI Diagnostic System", "Upload chest X-ray and get classification, localization, and explainability.")
    with st.expander("Implemented Architecture (This Project)", expanded=False):
        architecture_block()

    model = load_model()
    cls_threshold = load_inference_threshold()
    uploaded = st.file_uploader("Upload PNG/JPG/DICOM", type=["png", "jpg", "jpeg", "dcm"])

    if uploaded is None:
        st.info("Upload an image to run inference.")
        return

    image = read_uploaded_image(uploaded)
    st.image(image, caption="Uploaded X-Ray", width="stretch")
    if not st.button("Run Diagnosis"):
        return

    if model is None:
        st.warning("No trained model found. Running demo mode with placeholder outputs.")
        prob = 0.82
        label = "PNEUMONIA"
        heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        cv2.circle(heatmap, (image.shape[1] // 2, image.shape[0] // 2), image.shape[0] // 4, 1.0, -1)
        boxes = np.array([[80, 80, image.shape[1] - 80, image.shape[0] - 80]], dtype=np.float32)
    else:
        tensor = preprocess_for_model(image)
        out = model(tensor, mode="inference")
        prob = float(out["classification"].view(-1)[0].item())
        label = "PNEUMONIA" if prob >= cls_threshold else "NORMAL"
        boxes = out["boxes"].detach().cpu().numpy() if out["boxes"].numel() else np.zeros((0, 4), dtype=np.float32)

        gradcam = GradCAM(model, model.backbone.backbone.denseblock4)
        heatmap = gradcam.generate(tensor, class_idx=0)

    boxed = draw_boxes(image, boxes.tolist())
    overlay = GradCAM.overlay(boxed, heatmap)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", width="stretch")
    with col2:
        st.image(overlay, caption="Grad-CAM Overlay + Boxes", width="stretch")

    color = "#16A34A" if label == "NORMAL" else "#DC2626"
    pill_text = "NORMAL" if label == "NORMAL" else "PNEUMONIA"
    st.markdown(
        f"<div class='diag-pill' style='background:{color};'>"
        f"{pill_text} • {prob:.2%}</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Calibrated decision threshold: {cls_threshold:.2f}")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Pneumonia Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ff5a36"},
                "steps": [
                    {"range": [0, 50], "color": "#1f2a4a"},
                    {"range": [50, 100], "color": "#4b2331"},
                ],
            },
        )
    )
    style_plotly(fig, height=310)
    st.plotly_chart(fig, use_container_width=True)

    pdf_data = create_pdf_report(image, overlay, label, prob)
    st.download_button("Download Report", data=pdf_data, file_name="pneumonia_report.pdf", mime="application/pdf")


def gan_page() -> None:
    banner("Synthetic X-Ray Generator — Powered by DCGAN", "Create synthetic pneumonia images and export as ZIP.")
    st.markdown(
        "<div class='glass'><b>Why synthetic images?</b> The training set is class-imbalanced, so synthetic minority samples can improve class balance for downstream training.</div>",
        unsafe_allow_html=True,
    )

    generator = load_generator()
    count = st.slider("Number of images", min_value=1, max_value=16, value=8)

    st.selectbox("Image type", ["Pneumonia X-Rays", "Normal X-Rays (coming soon)"], index=0)
    if st.button("Generate"):
        images: List[np.ndarray] = []
        fid_display = "N/A"

        if generator is None:
            st.warning(
                "No trained GAN checkpoint found. Run `python training/train_gan.py` once to create "
                "`checkpoints/generator_best.pth`. Loading sample/fallback demo images for now."
            )
            sample_paths = sorted(
                [
                    *Config.path(Config.GENERATED_DIR).glob("*.png"),
                    *Config.path(Config.GENERATED_DIR).glob("*.jpg"),
                    *Config.path(Config.GENERATED_DIR).glob("*.jpeg"),
                ]
            )[:count]
            for p in sample_paths:
                images.append(np.array(Image.open(p).convert("RGB")))
            if not images:
                images = _create_demo_synthetic_images(count)
        else:
            with torch.no_grad():
                z = torch.randn(count, Config.LATENT_DIM, device=Config.DEVICE)
                fake = generator(z)
                fake = ((fake + 1.0) / 2.0).clamp(0, 1)
                for i in range(count):
                    arr = (fake[i, 0].detach().cpu().numpy() * 255).astype(np.uint8)
                    images.append(cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB))
            fid_display = "See training logs/checkpoints for epoch-wise FID."

            noise_like = sum(1 for img in images if _is_noise_like_image(img))
            low_diversity = _is_low_diversity_batch(images)
            if noise_like >= max(1, int(0.6 * len(images))) or low_diversity:
                st.warning(
                    "GAN output quality is currently low (noise-like or low-diversity). "
                    "Showing enhanced demo synthetic samples instead. "
                    "Continue GAN training for more realistic and varied outputs."
                )
                images = _create_demo_synthetic_images(count)
                fid_display = "N/A (fallback demo samples)"

        if not images:
            st.info("No synthetic preview images are available yet. Train GAN once to generate model-based samples.")
            return
        images = [_enhance_xray_image(img) for img in images]

        cols = st.columns(4)
        for idx, img in enumerate(images):
            with cols[idx % 4]:
                st.image(img, width="stretch")

        st.caption(f"FID score: {fid_display}")

        mem_zip = io.BytesIO()
        with ZipFile(mem_zip, "w", compression=ZIP_DEFLATED) as zipf:
            for idx, img in enumerate(images):
                zipf.writestr(f"synthetic_{idx+1}.png", encode_png_bytes(img))

        st.download_button(
            "Download as ZIP",
            data=mem_zip.getvalue(),
            file_name="synthetic_xrays.zip",
            mime="application/zip",
        )


def dashboard_page() -> None:
    banner("Model Evaluation Dashboard", "Inspect validation metrics and learning curves.")
    with st.expander("Implemented Architecture (This Project)", expanded=False):
        architecture_block()

    metrics_path = Config.path(Config.CHECKPOINT_DIR) / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        metrics = {
            "accuracy": 0.96,
            "auc": 0.97,
            "f1": 0.96,
            "precision": 0.95,
            "recall": 0.97,
            "confusion_matrix": [[120, 10], [8, 140]],
            "roc": {"fpr": [0, 0.1, 1], "tpr": [0, 0.9, 1]},
            "history": {
                "train_loss": [0.8, 0.5, 0.3],
                "val_loss": [0.9, 0.6, 0.35],
                "train_acc": [0.7, 0.85, 0.94],
                "val_acc": [0.68, 0.83, 0.92],
            },
        }
        st.info("metrics.json not found. Showing placeholder dashboard data.")

    cards = st.columns(5)
    keys = ["accuracy", "auc", "f1", "precision", "recall"]
    for col, key in zip(cards, keys):
        with col:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric(key.upper(), f"{metrics[key]:.3f}")
            st.markdown("</div>", unsafe_allow_html=True)

    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=metrics["roc"]["fpr"], y=metrics["roc"]["tpr"], mode="lines", name="ROC"))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line={"dash": "dash"}))
    style_plotly(roc_fig, title=f"ROC Curve (AUC={metrics['auc']:.3f})")
    st.plotly_chart(roc_fig, use_container_width=True)

    cm = np.array(metrics["confusion_matrix"])
    cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale="YlOrRd", labels={"x": "Pred", "y": "True"})
    style_plotly(cm_fig, title="Confusion Matrix")
    st.plotly_chart(cm_fig, use_container_width=True)

    history = metrics["history"]
    line_df = pd.DataFrame(
        {
            "epoch": list(range(1, len(history["train_loss"]) + 1)),
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "train_acc": history["train_acc"],
            "val_acc": history["val_acc"],
        }
    )
    curve_fig = px.line(
        line_df,
        x="epoch",
        y=["train_loss", "val_loss", "train_acc", "val_acc"],
        markers=True,
        template="simple_white",
        title="Training Curves",
    )
    style_plotly(curve_fig)
    st.plotly_chart(curve_fig, use_container_width=True)

    pr_df = pd.DataFrame(
        {
            "class": ["Normal", "Pneumonia"],
            "precision": [metrics["precision"], metrics["precision"]],
            "recall": [metrics["recall"], metrics["recall"]],
        }
    )
    pr_fig = px.bar(pr_df, x="class", y=["precision", "recall"], barmode="group", template="simple_white")
    style_plotly(pr_fig)
    st.plotly_chart(pr_fig, use_container_width=True)


def literature_page() -> None:
    banner("Research Paper Comparison", "Comparative view of reviewed papers and architecture gap closure.")
    st.markdown(
        "<div class='glass'>This page compares prior work against the architecture implemented in this project: DenseNet-121 + FPN, dual heads, and GAN-based balancing support.</div>",
        unsafe_allow_html=True,
    )

    data = [
        {
            "Paper": "Wu et al. 2024",
            "Architecture": "ResNet-50 + FPN",
            "Task": "Anchor-Free Detection",
            "Metric": "AP 51.5",
            "Strength": "Strong localization",
            "Limitation": "No explicit balancing",
        },
        {
            "Paper": "Szepesi 2022",
            "Architecture": "5-block CNN + Dropout",
            "Task": "Classification",
            "Metric": "Acc 97.2%",
            "Strength": "High classification accuracy",
            "Limitation": "No localization",
        },
        {
            "Paper": "CheXNet",
            "Architecture": "DenseNet-121",
            "Task": "Classification + CAM",
            "Metric": "AUROC 0.768",
            "Strength": "Transfer learning",
            "Limitation": "Weak localization",
        },
        {
            "Paper": "Xu & Zhang 2026",
            "Architecture": "Double SGAN + ResNet18-SA",
            "Task": "GAN balancing + classification",
            "Metric": "Acc 95.83%",
            "Strength": "Imbalance handling",
            "Limitation": "Low resolution and no detection",
        },
    ]
    df = pd.DataFrame(data)

    def color_limitations(row: pd.Series) -> list[str]:
        colors = []
        for col in row.index:
            if col == "Strength":
                colors.append("background-color: rgba(34,197,94,0.25)")
            elif col == "Limitation":
                colors.append("background-color: rgba(239,68,68,0.25)")
            else:
                colors.append("")
        return colors

    st.dataframe(df.style.apply(color_limitations, axis=1), use_container_width=True)

    score_df = pd.DataFrame(
        {
            "paper": ["Paper 1", "Paper 2", "Paper 3", "Paper 4"],
            "score": [51.5, 97.2, 76.8, 95.83],
            "metric_type": ["AP", "Accuracy", "AUROC", "Accuracy"],
        }
    )
    fig = px.bar(score_df, x="paper", y="score", color="metric_type", template="simple_white")
    style_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        **How this project addresses all gaps**
        - Imbalance is handled by Double-SGAN based synthetic augmentation.
        - Scale sensitivity is handled by DenseNet-121 + FPN (P3/P4/P5).
        - Localization is delivered via anchor-free FCOS-style detection.
        - Explainability is provided with Grad-CAM/Grad-CAM++ overlays.
        """
    )


def main() -> None:
    apply_theme()
    st.sidebar.markdown(
        """
        <div class='sidebar-brand'>
            <div class='kicker'>PNEUMONIA DETECTION</div>
            <div class='title'>Neo Clinical Console</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio("Navigate", ["Diagnose", "GAN Gallery", "Performance Dashboard", "Paper Comparison"])

    if page == "Diagnose":
        diagnose_page()
    elif page == "GAN Gallery":
        gan_page()
    elif page == "Performance Dashboard":
        dashboard_page()
    else:
        literature_page()


if __name__ == "__main__":
    main()
