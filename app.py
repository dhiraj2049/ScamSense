import streamlit as st
import pandas as pd
from detector import train_detector, predict_single, predict_batch

# --------------------------------------------------
# Page Setup
# --------------------------------------------------

st.set_page_config(
    page_title="Review Shield",
    page_icon="🛡️",
    layout="wide"
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return train_detector()

vectorizer, model = load_model()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("Review Shield")
st.sidebar.write(
    "An AI tool that helps detect suspicious or fake product reviews."
)

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit + Machine Learning")

# --------------------------------------------------
# Main Page
# --------------------------------------------------

st.title("Review Shield")
st.subheader("Fake Product Review Detector")

st.write(
    "Check whether a review looks genuine or suspicious "
    "based on writing patterns and spam signals."
)

tab1, tab2 = st.tabs([
    "Single Review Check",
    "Bulk CSV Analysis"
])

# --------------------------------------------------
# Single Review
# --------------------------------------------------

with tab1:

    st.markdown("### Analyze One Review")

    review_text = st.text_area(
        "Paste review text below:",
        height=160,
        placeholder="Example: Best product ever! Must buy now..."
    )

    if st.button("Analyze Review"):

        if review_text.strip():

            label, confidence = predict_single(
                review_text,
                vectorizer,
                model
            )

            if label == 1:
                st.error("Likely Fake / Spam Review")
            else:
                st.success("Likely Genuine Review")

            st.write(f"Confidence Score: {confidence:.2%}")

        else:
            st.warning("Please enter review text.")

# --------------------------------------------------
# Batch CSV Upload
# --------------------------------------------------

with tab2:

    st.markdown("### Upload CSV File")

    file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if file is not None:

        try:
            df = pd.read_csv(file)

            result_df = predict_batch(
                df,
                vectorizer,
                model
            )

            if result_df is None:
                st.error(
                    "CSV must contain a review column like: "
                    "review, text, content"
                )

            else:
                left, right = st.columns([1, 2])

                with left:
                    st.markdown("### Result Distribution")
                    st.bar_chart(
                        result_df["Prediction"].value_counts()
                    )

                with right:
                    st.markdown("### Reviewed Data")
                    st.dataframe(
                        result_df,
                        use_container_width=True
                    )

        except Exception as e:
            st.error(f"Error reading file: {e}")

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Disclaimer: Predictions are based on patterns and "
    "should be treated as guidance, not certainty."
)
