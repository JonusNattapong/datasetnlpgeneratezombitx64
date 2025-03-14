import streamlit as st
import os
import tempfile
from typing import Optional, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt

from data_sources.text_source import TextDataSource, TextFormat
from augmentation.text_augmenter import TextAugmenter
from analysis.text_analyzer import TextAnalyzer

def initialize_session_state():
    """Initialize session state variables."""
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'augmented_data' not in st.session_state:
        st.session_state.augmented_data = None
    if 'data_source' not in st.session_state:
        st.session_state.data_source = None
    if 'augmenter' not in st.session_state:
        st.session_state.augmenter = None
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None

def render_data_upload() -> None:
    """Render the data upload section."""
    st.header("Data Upload")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['txt', 'csv', 'json'],
        help="Upload your text data file"
    )
    
    # Data format selection
    format_options = {
        "Plain Text": TextFormat.PLAIN,
        "CoNLL Format": TextFormat.CONLL,
        "JSON": TextFormat.JSON
    }
    selected_format = st.selectbox(
        "Select data format",
        options=list(format_options.keys())
    )
    
    if uploaded_file is not None:
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                
                # Initialize data source
                data_source = TextDataSource(format_options[selected_format])
                
                # Load data
                data = data_source.load_data(tmp_file.name)
                
                # Store in session state
                st.session_state.current_data = data
                st.session_state.data_source = data_source
                
                # Preview data
                st.subheader("Data Preview")
                st.write(f"Number of samples: {len(data)}")
                st.write("First few samples:")
                for i, sample in enumerate(data[:5]):
                    st.text(f"Sample {i + 1}:\n{sample}")
                
            # Clean up temporary file
            os.unlink(tmp_file.name)
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

def render_augmentation_options() -> None:
    """Render the augmentation options section."""
    st.header("Augmentation Options")
    
    if st.session_state.current_data is None:
        st.warning("Please upload data first.")
        return
    
    # Initialize augmenter if needed
    if st.session_state.augmenter is None:
        st.session_state.augmenter = TextAugmenter()
    
    augmenter = st.session_state.augmenter
    
    # Get parameter info
    param_info = augmenter.get_param_info()
    
    # Create input widgets for each parameter
    params = {}
    for param_name, info in param_info.items():
        if info['type'] == 'float':
            params[param_name] = st.slider(
                info['description'],
                min_value=float(info['range'][0]),
                max_value=float(info['range'][1]),
                value=augmenter.get_params().get(param_name, 0.0)
            )
        elif info['type'] == 'int':
            params[param_name] = st.slider(
                info['description'],
                min_value=int(info['range'][0]),
                max_value=int(info['range'][1]),
                value=augmenter.get_params().get(param_name, 1)
            )
    
    # Update augmenter parameters
    augmenter.set_params(params)
    
    # Preview button
    if st.button("Generate Preview"):
        with st.spinner("Generating preview..."):
            # Get first sample
            sample = st.session_state.current_data[0]
            
            # Generate preview
            preview_samples = augmenter.preview(sample, n_samples=3)
            
            st.subheader("Augmentation Preview")
            st.write("Original:")
            st.text(sample)
            st.write("Augmented samples:")
            for i, aug_sample in enumerate(preview_samples, 1):
                st.text(f"{i}. {aug_sample}")
    
    # Augment button
    if st.button("Augment Data"):
        with st.spinner("Augmenting data..."):
            augmented_samples = []
            for sample in st.session_state.current_data:
                augmented = augmenter.augment(sample)
                augmented_samples.extend(augmented)
            
            st.session_state.augmented_data = augmented_samples
            st.success(f"Generated {len(augmented_samples)} augmented samples!")

def render_analysis() -> None:
    """Render the analysis section."""
    st.header("Data Analysis")
    
    # Initialize analyzer if needed
    if st.session_state.analyzer is None:
        st.session_state.analyzer = TextAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Select data to analyze
    data_options = []
    if st.session_state.current_data is not None:
        data_options.append("Original Data")
    if st.session_state.augmented_data is not None:
        data_options.append("Augmented Data")
    if not data_options:
        st.warning("No data available for analysis.")
        return
    
    selected_data = st.selectbox(
        "Select data to analyze",
        options=data_options
    )
    
    data = (st.session_state.current_data if selected_data == "Original Data"
            else st.session_state.augmented_data)
    
    if st.button("Analyze"):
        with st.spinner("Analyzing data..."):
            # Calculate metrics
            metrics = analyzer.calculate_metrics(data)
            
            # Generate visualizations
            visualizations = analyzer.generate_visualizations(data)
            
            # Display metrics
            st.subheader("Metrics")
            metrics_df = pd.DataFrame([
                {"Metric": name, "Value": value, "Description": analyzer.get_metric_descriptions()[name]}
                for name, value in metrics.items()
            ])
            st.dataframe(metrics_df)
            
            # Display visualizations
            st.subheader("Visualizations")
            for name, fig in visualizations.items():
                st.write(f"**{name}**")
                st.pyplot(fig)

def render_download() -> None:
    """Render the download section."""
    st.header("Download")
    
    if st.session_state.augmented_data is None:
        st.warning("No augmented data available for download.")
        return
    
    # Create download buttons
    if st.download_button(
        "Download Augmented Data (TXT)",
        "\n".join(st.session_state.augmented_data),
        "augmented_data.txt",
        "text/plain"
    ):
        st.success("Download started!")

def main():
    """Main application."""
    st.title("Data Augmentation System")
    
    # Initialize session state
    initialize_session_state()
    
    # Create tabs
    tabs = st.tabs(["Upload", "Augment", "Analyze", "Download"])
    
    # Render each section in its tab
    with tabs[0]:
        render_data_upload()
    
    with tabs[1]:
        render_augmentation_options()
    
    with tabs[2]:
        render_analysis()
    
    with tabs[3]:
        render_download()

if __name__ == "__main__":
    main()
