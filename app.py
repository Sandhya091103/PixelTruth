"""
PixelTruth - Streamlit demo (single file).

Upload a face image -> the model predicts Real vs Fake (AI-generated) with a
confidence score, plus an optional Grad-CAM heatmap of where it looked.

HOW TO RUN (from the project root):

    python app.py

Note: run it with plain `python app.py`, NOT `streamlit run app.py`.
On Windows + Python 3.13, Streamlit's server exits immediately under the
default event loop, so this file re-launches itself through Streamlit with the
correct (Selector) event loop applied first. When run the normal way, that fix
cannot be applied in time - so we bootstrap it here.

Uses the trained EfficientNetB0 at model/pixeltruth_efficientnet.pt.
Label convention: real = 0, fake = 1 (positive class = fake).
"""
import sys
import asyncio


# ----------------------------------------------------------------------
# The Streamlit UI (only runs inside the Streamlit runtime)
# ----------------------------------------------------------------------
def main():
    import numpy as np
    import torch
    import streamlit as st
    from PIL import Image
    from torchvision import transforms

    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    from src import config as cfg
    from src import models as models_mod

    # Guard against a known PyTorch + Streamlit watcher crash on torch.classes.
    try:
        torch.classes.__path__ = []
    except Exception:
        pass

    # --- Preprocessing (must match eval transforms in src/data.py) ---
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD)

    def preprocess(pil_img):
        """Return (rgb_float HxWx3 in [0,1], input_tensor 1x3xHxW normalized)."""
        img = pil_img.convert("RGB").resize((cfg.IMG_SIZE, cfg.IMG_SIZE))
        rgb_float = np.asarray(img, dtype=np.float32) / 255.0
        tensor = normalize(to_tensor(img)).unsqueeze(0).to(cfg.DEVICE)
        return rgb_float, tensor

    # A Grad-CAM target returning the single "fake" logit for one sample.
    class FakeLogitTarget:
        def __call__(self, model_output):
            return model_output[0]

    @st.cache_resource
    def load_model():
        model = models_mod.build_efficientnet(freeze_base=False)
        model.load_state_dict(
            torch.load(cfg.EFFICIENTNET_PATH, map_location=cfg.DEVICE))
        model.eval()
        return model

    @torch.no_grad()
    def predict(model, input_tensor):
        with torch.autocast(device_type=cfg.DEVICE.type, enabled=cfg.USE_AMP):
            logit = model(input_tensor)
        return torch.sigmoid(logit).item()

    def gradcam_overlay(model, rgb_float, input_tensor):
        cam = GradCAM(model=model, target_layers=[model.features[-1]])
        grayscale = cam(input_tensor=input_tensor, targets=[FakeLogitTarget()])[0]
        return show_cam_on_image(rgb_float, grayscale, use_rgb=True)

    # ------------------------------------------------------------------
    # Page
    # ------------------------------------------------------------------
    st.set_page_config(page_title="PixelTruth - Deepfake Detector",
                       page_icon="🕵️")
    st.title("🕵️ PixelTruth - Deepfake Face Detector")
    st.write(
        "Upload a face image. The model (EfficientNetB0, ~99% test accuracy) "
        "predicts whether the face is **Real** or **AI-generated (Fake)**.")

    if not cfg.EFFICIENTNET_PATH.exists():
        st.error(f"Model file not found: {cfg.EFFICIENTNET_PATH}\n\n"
                 "Train it first:  python -m src.train --model efficientnet")
        st.stop()

    model = load_model()

    with st.sidebar:
        st.header("Options")
        show_cam = st.checkbox("Show Grad-CAM heatmap", value=True,
                               help="Highlights the regions the model focused on.")
        st.caption(f"Device: {cfg.DEVICE}")

    uploaded = st.file_uploader("Choose a face image",
                                type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.info("Upload an image to get a prediction.")
        return

    pil_img = Image.open(uploaded)
    rgb_float, input_tensor = preprocess(pil_img)

    prob_fake = predict(model, input_tensor)
    is_fake = prob_fake >= 0.5
    label = "FAKE (AI-generated)" if is_fake else "REAL"
    confidence = prob_fake if is_fake else 1 - prob_fake

    col1, col2 = st.columns(2)
    with col1:
        st.image(rgb_float, caption="Input (resized to 224x224)",
                 use_container_width=True)
        if show_cam:
            overlay = gradcam_overlay(model, rgb_float, input_tensor)
            st.image(overlay, caption="Grad-CAM (red = high attention)",
                     use_container_width=True)
    with col2:
        if is_fake:
            st.error(f"### Prediction: {label}")
        else:
            st.success(f"### Prediction: {label}")
        st.metric("Confidence", f"{confidence * 100:.2f}%")
        st.progress(float(confidence))
        st.caption(f"Raw P(fake) = {prob_fake:.4f}")


# ----------------------------------------------------------------------
# Bootstrap: are we already inside Streamlit, or launched as a plain script?
# ----------------------------------------------------------------------
def _inside_streamlit_runtime():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _inside_streamlit_runtime():
    # Streamlit is running this file - render the UI.
    main()
else:
    # Plain `python app.py` - apply the Windows event-loop fix, then hand off
    # to Streamlit, which will re-run this file (now inside the runtime).
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", __file__, "--server.headless", "true"]
    raise SystemExit(stcli.main())
