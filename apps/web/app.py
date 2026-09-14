from pathlib import Path
import sys
import torch
import streamlit as st
from PIL import Image, UnidentifiedImageError
import json

# PROJECT SETUP
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import (
    CLASS_NAMES,
    predict_image,
)

from src.config import (
    CONFIDENCE_THRESHOLD,
    IMAGE_SIZE,
)


# PAGE CONFIGURATION

st.set_page_config(
    page_title="SteelGuard AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# CUSTOM STYLING

st.markdown(
    """
    <style>

    /* ---------- Global spacing ---------- */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }


    /* ---------- Hero ---------- */

    .hero {
        padding: 1rem 0 2rem 0;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin-bottom: 0.25rem;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        opacity: 0.70;
        margin-top: 0;
        max-width: 760px;
    }


    /* ---------- Section labels ---------- */

    .section-label {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        opacity: 0.60;
        margin-bottom: 0.35rem;
    }


    /* ---------- Result ---------- */

    .prediction-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0;
    }

    .prediction-subtitle {
        opacity: 0.65;
        margin-top: 0.15rem;
    }


    /* ---------- Upload area ---------- */

    [data-testid="stFileUploader"] {
        border-radius: 14px;
    }


    /* ---------- Sidebar ---------- */

    .sidebar-note {
        font-size: 0.86rem;
        line-height: 1.5;
        opacity: 0.72;
    }


    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        opacity: 0.55;
        font-size: 0.82rem;
        padding-top: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# DEFECT INFORMATION

DEFECT_INFO = {
    "crazing": {
        "name": "Crazing",
        "description": (
            "Fine, irregular cracks or network-like patterns "
            "appearing across the steel surface."
        ),
    },
    "inclusion": {
        "name": "Inclusion",
        "description": (
            "Non-metallic material or embedded particles "
            "visible within the steel surface."
        ),
    },
    "patches": {
        "name": "Patches",
        "description": (
            "Localized areas showing surface irregularities "
            "or visually distinct regions."
        ),
    },
    "pitted_surface": {
        "name": "Pitted Surface",
        "description": (
            "Small cavities, pits, or depressions affecting "
            "the steel surface."
        ),
    },
    "rolled-in_scale": {
        "name": "Rolled-in Scale",
        "description": (
            "Scale or oxide material that has become embedded "
            "into the steel during rolling."
        ),
    },
    "scratches": {
        "name": "Scratches",
        "description": (
            "Linear surface marks or grooves produced by "
            "mechanical contact or handling."
        ),
    },
}


# HEADER

st.markdown("""
<div class="hero">
    <div class="hero-title">SteelGuard AI</div>
    <div class="hero-subtitle">
        AI-powered visual inspection for hot-rolled steel surfaces.
        Upload an image to classify the detected surface defect using a deep-learning computer vision model.
    </div>
</div>
""", unsafe_allow_html=True)


# SIDEBAR

with st.sidebar:

    st.markdown("## Inspection")

    st.markdown(
        """
        <div class="sidebar-note">
        Upload a steel surface image and SteelGuard AI will
        analyze it using a trained ResNet18 classification model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.metric(
        label="Supported defect classes",
        value="6",
    )

    st.metric(
        label="Decision threshold",
        value=f"{CONFIDENCE_THRESHOLD:.0%}",
    )

    st.metric(
        label="Model input",
        value=f"{IMAGE_SIZE[0]} × {IMAGE_SIZE[1]}",
    )

    st.divider()

    st.markdown("### Model")

    st.caption(
        "ResNet18 • Transfer Learning • PyTorch"
    )

    


# UPLOAD SECTION

st.markdown(
    '<div class="section-label">IMAGE INPUT</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a steel surface image",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Supported formats: JPG, JPEG, PNG and BMP.",
)


# EMPTY STATE

if uploaded_file is None:

    st.info(
        "Upload a steel surface image above to start an inspection."
    )

    st.markdown("### How it works")

    step1, step2, step3 = st.columns(3)

    with step1:
        st.markdown("#### Upload")
        st.caption(
            "Provide a clear image of the steel surface."
        )

    with step2:
        st.markdown("#### Analyze")
        st.caption(
            "The computer vision model processes the image."
        )

    with step3:
        st.markdown("#### Inspect")
        st.caption(
            "Review the predicted defect and confidence."
        )

    st.divider()

    st.markdown("### Supported defects")

    defect_columns = st.columns(3)

    for index, defect in enumerate(DEFECT_INFO.values()):

        with defect_columns[index % 3]:

            st.markdown(f"**{defect['name']}**")
            st.caption(defect["description"])


# IMAGE PROCESSING

else:

    try:

        image = Image.open(uploaded_file)

        # Force image loading so corrupted files are detected early.
        image.load()

        # Convert to RGB for consistent display/inference.
        display_image = image.convert("RGB")

    except (UnidentifiedImageError, OSError):

        st.error(
            "This file could not be read as a valid image. "
            "Please upload another image."
        )

        st.stop()


    # IMAGE METADATA

    image_width, image_height = display_image.size

    image_mode = display_image.mode

    file_size_mb = uploaded_file.size / (1024 * 1024)


    # MAIN LAYOUT

    image_column, result_column = st.columns(
        [1.05, 0.95],
        gap="large",
    )


    # IMAGE PANEL

    with image_column:

        st.markdown(
            '<div class="section-label">INSPECTION IMAGE</div>',
            unsafe_allow_html=True,
        )

        st.image(
            display_image,
            use_container_width=True,
        )

        st.caption(
            f"{uploaded_file.name}"
        )

        metadata_col1, metadata_col2, metadata_col3 = st.columns(3)

        with metadata_col1:
            st.metric(
                "Resolution",
                f"{image_width} × {image_height}",
            )

        with metadata_col2:
            st.metric(
                "Format",
                image.format or "Image",
            )

        with metadata_col3:
            st.metric(
                "Size",
                f"{file_size_mb:.2f} MB",
            )


    # AI RESULT PANEL

    with result_column:

        st.markdown(
            '<div class="section-label">AI INSPECTION RESULT</div>',
            unsafe_allow_html=True,
        )

        with st.spinner("Analyzing surface..."):

            try:

                result = predict_image(
                    display_image
                )

            except Exception:

                st.error(
                    "The image could not be analyzed. "
                    "Please try another image."
                )

                st.stop()

        if not result["is_valid_domain"]:
            st.error(
                "This image can't be inspected."
            )
            st.info(
                "SteelGuard is designed to inspect steel surface images"

            )
            st.stop()

        predicted_class = result["predicted_class"]

        confidence = result["confidence"]

        status = result["status"]

        probabilities = result["probabilities"]


        # Human-friendly class name
        defect = DEFECT_INFO.get(
            predicted_class,
            {
                "name": predicted_class.replace(
                    "_",
                    " ",
                ).title(),
                "description": (
                    "The model identified this surface defect "
                    "category."
                ),
            },
        )

        display_class = defect["name"]

        # PRIMARY PREDICTION

        st.markdown(
            f'<div class="prediction-title">{display_class}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="prediction-subtitle">
                Most likely surface defect category
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.metric(
            label="Model confidence",
            value=f"{confidence:.2%}",
        )

        # INSPECTION STATUS

        if confidence >= CONFIDENCE_THRESHOLD:

            st.success(
                "DEFECT CLASSIFIED — confidence exceeds "
                "the application threshold."
            )

        else:

            st.warning(
                "LOW CONFIDENCE — MANUAL INSPECTION RECOMMENDED"
            )


        # DEFECT DESCRIPTION

        st.markdown("### What was detected?")

        st.write(
            defect["description"]
        )


        # TOP PREDICTIONS

        st.markdown("### Prediction Breakdown")

        sorted_probabilities = sorted(
            probabilities.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        top_predictions = sorted_probabilities[:3]

        for rank, (class_name, probability) in enumerate(
            top_predictions,
            start=1,
        ):

            display_name = class_name.replace(
                "_",
                " ",
            ).title()

            probability_col, value_col = st.columns(
                [4, 1],
            )

            with probability_col:

                st.write(
                    f"**{rank}. {display_name}**"
                )

            with value_col:

                st.write(
                    f"**{probability:.2%}**"
                )

            st.progress(
                float(probability)
            )


        # CONFIDENCE INTERPRETATION

        if confidence < CONFIDENCE_THRESHOLD:

            st.info(
                "The model's leading predictions are relatively close. "
            )

        else:

            st.caption(
                "The displayed confidence represents the model's "
                "softmax output. It should not be interpreted as a "
                "calibrated probability of correctness."
            )


# FOOTER

st.divider()

st.markdown(
    """
    <div class="footer">
        SteelGuard · Industrial Surface Defect Inspection<br>
        Built by <strong>Priyanshu Sharma</strong>
    </div>
    """,
    unsafe_allow_html=True,
)

linkedin_col, spacer_col = st.columns([1, 5])

with linkedin_col:

    st.link_button(
        "Connect / Give Feedback",
        "https://www.linkedin.com/in/priyanshu-sharma-646233394/",
        help="Give feedback or connect with the developer.",
        use_container_width=True,
    )