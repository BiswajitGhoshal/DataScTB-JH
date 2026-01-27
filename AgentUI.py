import streamlit as st
import pandas as pd
from io import StringIO

# -----------------------
# Page configuration
# -----------------------
st.set_page_config(
    page_title="Feedback Processor",
    layout="wide")

st.title("Feedback Processing Dashboard")

# -----------------------
# Mock data sources
# (replace with os.listdir or DB queries in real app)
# -----------------------
feedback_dirs_type_1 = [
    "data/feedback/type1/january",
    "data/feedback/type1/february",
    "data/feedback/type1/march"
]

feedback_dirs_type_2 = [
    "data/feedback/type2/internal",
    "data/feedback/type2/external"
]

# -----------------------
# Sidebar
# -----------------------
with st.sidebar:
    st.header("Configuration")

    st.subheader("Feedback Source Selection")

    # Feedback type 1 selector
    selected_dir_1 = st.selectbox(
        "Feedback Directory - Type 1",
        feedback_dirs_type_1
    )

    filter_1 = st.multiselect(
        "Filter (Type 1)",
        options=["Positive", "Neutral", "Negative"]
    )

    st.divider()

    # Feedback type 2 selector
    selected_dir_2 = st.selectbox(
        "Feedback Directory - Type 2",
        feedback_dirs_type_2
    )

    filter_2 = st.multiselect(
        "Filter (Type 2)",
        options=["Bug", "Feature Request", "General Comment"]
    )

    st.divider()

    # Output file name
    output_file_name = st.text_input(
        "Output File Name",
        value="processed_feedback.csv"
    )

    run_clicked = st.button("Go", type="primary")


# -----------------------
# Main panel layout
# -----------------------
col1, col2 = st.columns(2)

# -----------------------
# Panel 1 – Data Output
# -----------------------
with col1:
    st.subheader("Processed Feedback")
    panel_1 = st.container(height=420)

    with panel_1:
        if run_clicked:
            # Mock processed dataframe
            df = pd.DataFrame({
                "Feedback ID": range(1, 11),
                "Category": ["Positive", "Negative", "Neutral"] * 3 + ["Positive"],
                "Source": [selected_dir_1] * 10,
                "Comment": [f"Feedback comment {i}" for i in range(1, 11)]
            })

            if filter_1:
                df = df[df["Category"].isin(filter_1)]

            st.dataframe(df, use_container_width=True)
        else:
            st.info("Click **Go** to view processed feedback data.")


# -----------------------
# Panel 2 – Logs / Text Output
# -----------------------
with col2:
    st.subheader("Processing Logs & Summary")
    panel_2 = st.container(height=420)

    with panel_2:
        if run_clicked:
            log_text = f"""
Processing started...
Source (Type 1): {selected_dir_1}
Source (Type 2): {selected_dir_2}

Applied filters:
Type 1: {filter_1 if filter_1 else "None"}
Type 2: {filter_2 if filter_2 else "None"}

Steps completed:
- Loaded feedback files
- Applied filters
- Aggregated results
- Generated output file

Processing completed successfully.
"""
            st.text_area(
                label="Logs",
                value=log_text,
                height=340
            )
        else:
            st.info("Logs will appear here after processing.")

# -----------------------
# Download Output File
# -----------------------
if run_clicked:
    # Convert dataframe to CSV for download
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    st.download_button(
        label="Download Output File",
        data=csv_buffer.getvalue(),
        file_name=output_file_name,
        mime="text/csv"
    )

    st.success("Output file ready for download.")