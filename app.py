import os
import streamlit as st

from inference import detect_and_segment
from count_and_report import create_compliance_report, extract_prompt_from_planogram

st.set_page_config(page_title="Retail Shelf Compliance", layout="wide")
st.title("Retail Shelf Compliance Checker")
st.markdown("Upload a shelf photo + planogram → get instant AI report")

col1, col2 = st.columns(2)

with col1:
    uploaded_image = st.file_uploader("Upload Shelf Image", type=["jpg", "jpeg", "png"])

    manual_prompt = st.text_area(
        "Product prompt (separate with dots)",
        value=(
            "blueberry chex . chocolate chex . cinnamon chex . honey nut chex . "
            "wheat chex . rice chex . corn chex . crispix cereal . cracklin oat bran . "
            "wheaties . rice squares . corn squares . cheerios original . "
            "cheerios oat crunch . maple cheerios . multi grain cheerios . "
            "toasted oats cereal . life cereal original . life cereal giant size"
        ),
        height=140,
    )

with col2:
    uploaded_planogram = st.file_uploader("Upload Planogram JSON", type=["json"])

st.markdown("---")

use_auto_prompt = st.checkbox(
    "Use products from planogram (auto-prompt)",
    value=True,
)

st.subheader("Detection Settings")
box_threshold = st.slider("Box threshold", 0.05, 0.50, 0.15, 0.01)
text_threshold = st.slider("Text threshold", 0.05, 0.50, 0.12, 0.01)

st.markdown("---")

if uploaded_image and uploaded_planogram:
    image_path = "temp_image.jpg"
    planogram_path = "temp_planogram.json"

    with open(image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())
    with open(planogram_path, "wb") as f:
        f.write(uploaded_planogram.getbuffer())

    if st.button("Analyze Shelf Compliance", type="primary"):
        caption = manual_prompt

        if use_auto_prompt:
            auto_prompt = extract_prompt_from_planogram(planogram_path)
            if auto_prompt:
                caption = auto_prompt
                st.success("Auto-prompt generated from planogram.")
                with st.expander("Show auto-generated prompt", expanded=False):
                    st.write(caption)
            else:
                st.warning("Auto-prompt failed → using manual prompt.")

        with st.spinner("Running GroundingDINO on CPU…"):
            annotated_image, labels = detect_and_segment(
                image_path,
                caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

        if not labels:
            st.warning("No products detected! Lower thresholds or simplify prompt.")
            st.image(image_path, caption="Original image", use_container_width=True)
        else:
            report, counts = create_compliance_report(labels, planogram_path)

            left, right = st.columns(2)

            with left:
                st.image(annotated_image, caption="Detected Products", use_container_width=True)

            with right:
                st.success(f"Compliance Score: {report['compliance_score']:.1f}%")
                st.caption(
                    f"Presence: {report['presence_score']:.1f}% | "
                    f"Strict count: {report['strict_count_score']:.1f}%"
                )

                st.subheader("Product Counts")
                st.json({"Detected": counts})

                if report["missing"]:
                    st.error(
                        "Missing products:\n"
                        + "\n".join(report["missing"][:30])
                        + ("..." if len(report["missing"]) > 30 else "")
                    )
                if report["extra"]:
                    st.warning(
                        "Extra / unexpected:\n"
                        + "\n".join(report["extra"][:30])
                        + ("..." if len(report["extra"]) > 30 else "")
                    )

                if report["compliance_score"] >= 95:
                    st.balloons()
                    st.success("Excellent compliance!")
else:
    st.info("Please upload both a shelf image and a planogram JSON to begin.")
