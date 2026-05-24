import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(
    page_title="AccidentVision AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #0b1220 0%, #111f3a 45%, #192a55 100%);
        color: #f5f7fb;
    }
    .title-row {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .title-row h1 {
        margin: 0;
        line-height: 1.1;
    }
    .subtitle {
        color: #b8c7e0;
        margin-top: 8px;
    }
    .status-card,
    .metric-card,
    .info-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 24px 60px rgba(0,0,0,0.18);
    }
    .result-badge {
        border-radius: 18px;
        padding: 18px;
        color: white;
        font-weight: 600;
    }
    .danger {
        background: linear-gradient(135deg, #f63f5f 0%, #e66169 100%);
    }
    .success {
        background: linear-gradient(135deg, #2dd4bf 0%, #0ea5e9 100%);
    }
    .section-title {
        color: #e2e8f0;
        margin-bottom: 12px;
    }
    .section-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 20px;
    }
    .footer-text {
        color: #e2e8f0;
        font-size: 0.95rem;
        margin-top: 24px;
        text-align: center;
        font-weight: 600;
        letter-spacing: 0.02em;
        opacity: 0.92;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource

def load_accident_model():
    return tf.keras.models.load_model("best_accident_detection_model.keras")

model = load_accident_model()

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)

def predict_accident(image_array: np.ndarray) -> tuple[float, bool]:
    prediction = model.predict(image_array, verbose=0)
    score = float(prediction.ravel()[0])
    accident_detected = score < 0.5
    confidence = float((1.0 - score) if accident_detected else score) * 100.0
    return confidence, accident_detected

with st.sidebar:
    st.markdown("## 🚧 AccidentVision Dashboard")
    st.write(
        "Upload a traffic frame from CCTV or dashcam and get a real-time accident risk assessment."\
    )
    st.divider()
    st.markdown("### How to use")
    st.markdown(
        """
        - Upload a clear image with visible road and vehicles.
        - Wait for the model to evaluate the scene.
        - Review the risk status and confidence score.
        """
    )
    st.markdown("---")
    st.markdown("### Model Details")
    st.markdown(
        "- Architecture: Convolutional Neural Network\n"
        "- Input: 224×224 RGB image\n"
        "- Output: binary accident/non-accident probability"
    )
    st.markdown("---")
    st.markdown("### Notes")
    st.write(
        "The model works best with well-lit scenes and clear vehicle details. Avoid blurry or overly dark frames."
    )

st.markdown(
    "<div class='title-row'><h1>🚦 Accident Detection from CCTV Footage</h1></div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<p class='subtitle'>A production-ready AI utility for rapid traffic incident screening and visual risk analysis.</p>",
    unsafe_allow_html=True,
)

image_column, preview_column = st.columns([2, 1], gap="large")

with image_column:
    st.subheader("Image Upload")
    uploaded_file = st.file_uploader(
        "Upload JPEG or PNG image",
        type=["jpg", "jpeg", "png"],
        help="Choose a frame that clearly shows the roadway and vehicles.",
    )

with preview_column:
    st.subheader("Quick Info")
    st.metric(label="Model confidence threshold", value="50%")
    st.write("The system interprets the score using a binary accident risk threshold.")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded CCTV frame", use_container_width=True, clamp=True)

    try:
        processed_image = preprocess_image(image)
        with st.spinner("Analyzing the scene..."):
            confidence, accident_detected = predict_accident(processed_image)

        label = "Accident Detected" if accident_detected else "No Accident Detected"
        badge_class = "danger" if accident_detected else "success"
        badge_emoji = "⚠️" if accident_detected else "✅"

        st.markdown(
            f"<div class='status-card result-badge {badge_class}'>{badge_emoji} {label}</div>",
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Detection result", label)
        col2.metric("Confidence", f"{confidence:.2f}%")
        col3.metric(
            "Action recommendation",
            "Review immediately" if accident_detected else "Monitor the scene",
        )

        st.subheader("Prediction Summary")
        st.write(
            "This image was evaluated using the accident detection model. The confidence score reflects how strongly the model predicts the scene is an accident or not."
        )
        if accident_detected:
            st.write(
                "- The model has identified likely incident indicators such as vehicle displacement, smoke, or obstruction."
            )
        else:
            st.write(
                "- The scene appears stable with no strong accident indicators detected."
            )
    except Exception as error:
        st.error("Image processing or prediction failed.")
        st.write(f"Error details: {error}")
else:
    st.subheader("Ready when you are")
    st.write(
        "Upload a traffic image to start the accident detection workflow. The system will automatically preprocess your frame and return a confidence-backed prediction."
    )

st.markdown(
    "<div class='footer-text'>Built by Aatequa Ansari | Crafted with care for accurate accident detection.</div>",
    unsafe_allow_html=True,
)
