"""
ASCII Art Generator - Local Web Interface
=========================================

A local web application for generating high-quality ASCII art using gradient descent.
Based on gradscii-art by Zellic (AGPL-3.0) with Y2K aesthetic from ascii.micr by Microck (MIT).

License: AGPL-3.0
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Attribution:
- gradscii-art: https://github.com/stong/gradscii-art (AGPL-3.0)
- ascii.micr: https://github.com/Microck/ascii.micr (MIT)
"""

import streamlit as st
import sys
import os
from pathlib import Path
import tempfile
from PIL import Image
import io
import base64

# Add gradscii-art to path
gradscii_path = Path(__file__).parent / "gradscii-art"
sys.path.insert(0, str(gradscii_path))

# Import gradscii-art modules
try:
    import torch
    import numpy as np
    from train import (
        train,
        create_char_bitmaps,
        load_target_image,
        save_result,
        CHARS,
        NUM_CHARS,
        DEVICE,
        ENCODING,
    )

    GRADSCII_AVAILABLE = True
except ImportError as e:
    GRADSCII_AVAILABLE = False
    st.error(f"Failed to import gradscii-art: {e}")

# Page configuration
st.set_page_config(
    page_title="ASCII.MICR - Local ASCII Art Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Y2K Aesthetic
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DotGothic16&family=Pixelify+Sans:wght@400;700&family=Tektur:wght@400;900&family=VT323&family=Space+Mono:wght@400;700&family=Silkscreen:wght@400;700&display=swap');

/* Root variables */
:root {
    --bg-color: #ffccff;
    --box-bg: #fff0f5;
    --content-bg: #ffe6f2;
    --text-main: #660066;
    --accent-main: #ff66b2;
    --accent-highlight: #00ffff;
    --border-light: #ffffff;
    --border-dark: #cc0066;
}

/* Main background with kaomoji pattern */
.stApp {
    background-color: #ffccff;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='130' height='30' viewBox='0 0 130 30'%3E%3Ctext x='50%25' y='50%25' fill='rgba(255,0,255,0.12)' font-family='monospace' font-size='20' text-anchor='middle' dominant-baseline='middle' transform='rotate(-15, 65, 15)'%3E%26%239516%3B%26%239524%3B%26%239516%3B%26%239524%3B%26%239508%3B%28%26%2365381%3B_%26%239500%3B%26%239516%3B%26%239524%3B%26%239516%3B%26%239524%3B%3C/text%3E%3C/svg%3E");
    background-size: 160px 36.92px;
}

/* Header styling */
.stApp header {
    background: linear-gradient(135deg, #ff66b2 0%, #ff99cc 100%);
}

/* Title styling */
h1 {
    font-family: 'Pixelify Sans', sans-serif !important;
    color: #00ffff !important;
    text-shadow: 2px 2px 0px #cc0066;
    letter-spacing: 2px;
}

/* Section headers */
h2, h3 {
    font-family: 'Silkscreen', cursive !important;
    color: #660066 !important;
}

/* Sidebar styling */
.css-1d391kg, .css-163ttbj {
    background-color: #fff0f5 !important;
    border-right: 2px solid #cc0066;
}

/* Widget labels */
.css-1v3fvcr, .stSlider label, .stCheckbox label, .stSelectbox label {
    font-family: 'Silkscreen', cursive !important;
    color: #660066 !important;
    font-size: 12px !important;
}

/* Slider styling */
stSlider > div > div > div {
    background-color: #ffe6f2 !important;
}

.stSlider > div > div > div > div {
    background-color: #ff66b2 !important;
}

/* Buttons */
.stButton > button {
    font-family: 'Silkscreen', cursive !important;
    background: #ff66b2 !important;
    color: white !important;
    border: 2px solid #ffffff !important;
    border-right-color: #cc0066 !important;
    border-bottom-color: #cc0066 !important;
    box-shadow: 4px 4px 0px rgba(0, 0, 0, 0.3) !important;
    font-weight: bold !important;
}

.stButton > button:hover {
    filter: brightness(1.1);
    transform: translate(1px, 1px);
    box-shadow: 3px 3px 0px rgba(0, 0, 0, 0.3) !important;
}

.stButton > button:active {
    transform: translate(2px, 2px);
    box-shadow: 1px 1px 0px rgba(0, 0, 0, 0.3) !important;
}

/* Primary button (Execute) */
.stButton > button[kind="primary"] {
    background: #00ffff !important;
    color: #660066 !important;
}

/* Terminal output styling */
.terminal-output {
    background-color: #2a002a;
    color: #ff66b2;
    font-family: 'Tektur', monospace !important;
    font-size: 10px;
    line-height: 1.1;
    padding: 15px;
    border: 3px inset #660066;
    overflow-x: auto;
    white-space: pre;
    max-height: 500px;
    overflow-y: auto;
}

/* Progress bar */
.stProgress > div > div {
    background-color: #ff66b2 !important;
}

/* Success/Info messages */
.stSuccess {
    background-color: #ccffcc !important;
    border: 2px solid #00cc00 !important;
    font-family: 'Silkscreen', cursive !important;
}

.stInfo {
    background-color: #ccffff !important;
    border: 2px solid #00cccc !important;
    font-family: 'Silkscreen', cursive !important;
}

/* Expander styling */
.streamlit-expanderHeader {
    font-family: 'Silkscreen', cursive !important;
    background-color: #ffe6f2 !important;
    color: #660066 !important;
}

/* File uploader */
.css-1x8cf1d {
    background-color: #ffe6f2 !important;
    border: 2px solid #cc0066 !important;
    font-family: 'Silkscreen', cursive !important;
}

/* Checkbox styling */
.stCheckbox > div > div > div {
    background-color: #ff66b2 !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background-color: #ffe6f2 !important;
    border: 2px solid #cc0066 !important;
}

/* Blink animation for dot */
@keyframes blink {
    50% { opacity: 0; }
}

.blink {
    animation: blink 1s infinite;
}

/* Custom window box */
.window-box {
    background: #fff0f5;
    border: 2px solid #ffffff;
    border-right-color: #cc0066;
    border-bottom-color: #cc0066;
    box-shadow: 8px 8px 0px rgba(0, 0, 0, 0.5);
    padding: 20px;
    margin: 10px 0;
}

/* Logo text */
.logo-text {
    font-family: 'Pixelify Sans', sans-serif;
    font-size: 28px;
    color: #00ffff;
    text-shadow: 3px 3px 0px #cc0066;
    letter-spacing: 3px;
}

/* About section */
.about-text {
    font-family: 'VT323', monospace;
    font-size: 16px;
    color: #660066;
    line-height: 1.4;
}
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "output_text" not in st.session_state:
    st.session_state.output_text = None
if "output_image" not in st.session_state:
    st.session_state.output_image = None
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "progress" not in st.session_state:
    st.session_state.progress = 0


def reset_output():
    """Reset output state"""
    st.session_state.output_text = None
    st.session_state.output_image = None
    st.session_state.progress = 0


def run_ascii_conversion(image_path, params):
    """
    Run ASCII art conversion using gradscii-art.

    Args:
        image_path: Path to input image
        params: Dictionary of parameters

    Returns:
        tuple: (output_png_path, output_txt_path)
    """
    import importlib
    import train as train_module

    importlib.reload(train_module)

    # Set global configuration variables
    train_module.CHAR_WIDTH = params["char_width"]
    train_module.CHAR_HEIGHT = params["char_height"]
    train_module.GRID_WIDTH = params["grid_width"]
    train_module.GRID_HEIGHT = params["grid_height"]
    train_module.ROW_GAP = params["row_gap"]
    train_module.IMAGE_WIDTH = params["char_width"] * params["grid_width"]
    train_module.IMAGE_HEIGHT = params["char_height"] * params["grid_height"] + params[
        "row_gap"
    ] * (params["grid_height"] - 1)
    train_module.ENCODING = params["encoding"]
    train_module.BANNED_CHARS = ["`", "\\"]

    # Initialize character set
    if params["encoding"] == "cp437":
        # CP437 character set (Code Page 437)
        chars = [chr(i) for i in range(32, 256)]
    else:
        # ASCII only
        chars = [chr(i) for i in range(32, 127)]

    train_module.CHARS = "".join(chars)
    train_module.NUM_CHARS = len(train_module.CHARS)

    # Precompute warp interpolation cache
    train_module.WARP_INTERP_CACHE = (
        train_module.precompute_warp_interpolation_structure(
            train_module.IMAGE_HEIGHT, train_module.IMAGE_WIDTH
        )
    )

    # Create character bitmaps
    char_bitmaps = train_module.create_char_bitmaps()

    # Load target image
    target_image = train_module.load_target_image(image_path)

    # Add batch dimension (B=1 for single image)
    target_image = target_image.unsqueeze(0)

    # Training parameters
    num_iterations = params["iterations"]
    lr = params["learning_rate"]
    diversity_weight = params["diversity_weight"]
    use_gumbel = params["use_gumbel"]
    temp_start = params["temp_start"]
    temp_end = params["temp_end"]
    optimize_alignment = params["optimize_alignment"]
    dark_mode = params["dark_mode"]

    # Progress bar placeholder
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Custom train function with progress tracking
    original_train = train_module.train

    def train_with_progress(*args, **kwargs):
        """Wrap train function to capture progress"""
        # Create steps directory
        import shutil

        if os.path.exists("steps"):
            shutil.rmtree("steps")
        os.makedirs("steps")

        # Run training
        result = original_train(*args, **kwargs)

        return result

    # Monkey patch for progress tracking
    train_module.train = train_with_progress

    try:
        # Run training
        status_text.text("Training... Please wait")

        logits, alignment_params = train_module.train(
            target_image=target_image,
            char_bitmaps=char_bitmaps,
            num_iterations=num_iterations,
            lr=lr,
            save_interval=0,  # Don't save intermediate results
            warmup_iterations=max(1, num_iterations // 10),
            diversity_weight=diversity_weight,
            use_gumbel=use_gumbel,
            temp_start=temp_start,
            temp_end=temp_end,
            protect_whitespace=True,
            multiscale_weight=0.0,
            optimize_alignment=optimize_alignment,
            alignment_lr=0.01,
            warp_reg_weight=0.01,
            dark_mode=dark_mode,
            temporal_weight=0.0,
        )

        # Save final result
        output_png = "output.png"
        output_txt = "output.txt"
        output_utf8 = "output.utf8.txt"

        train_module.save_result(
            logits=logits[0],  # First (and only) item in batch
            char_bitmaps=char_bitmaps,
            output_path=output_png,
            text_path=output_txt,
            utf8_path=output_utf8,
            temperature=0.01,
            target_image=target_image[0] if optimize_alignment else None,
            warp_params=None,
            dark_mode=dark_mode,
        )

        progress_bar.progress(100)
        status_text.text("Complete!")

        return output_png, output_txt

    except Exception as e:
        st.error(f"Error during training: {str(e)}")
        raise
    finally:
        # Restore original train function
        train_module.train = original_train


# Main UI
st.markdown(
    '<div class="logo-text">ASCII<span class="blink">.</span>MICR</div>',
    unsafe_allow_html=True,
)
st.markdown(
    "<h3 style='color: #660066; font-family: Silkscreen, cursive;'>Local Gradient Descent ASCII Art Generator</h3>",
    unsafe_allow_html=True,
)

# Sidebar controls
with st.sidebar:
    st.markdown("### CONTROLS")

    # File upload
    uploaded_file = st.file_uploader(
        "SOURCE FILE:",
        type=["png", "jpg", "jpeg", "gif", "bmp"],
        on_change=reset_output,
    )

    st.markdown("---")

    # Advanced controls expander
    with st.expander("ADVANCED SETTINGS", expanded=True):
        # Grid configuration
        st.markdown("**Grid Configuration**")
        col1, col2 = st.columns(2)
        with col1:
            char_width = st.slider("Char Width", 8, 16, 12, 1, key="char_width")
            grid_width = st.slider("Grid Width", 20, 80, 42, 1, key="grid_width")
        with col2:
            char_height = st.slider("Char Height", 16, 32, 24, 1, key="char_height")
            grid_height = st.slider("Grid Height", 10, 40, 21, 1, key="grid_height")

        row_gap = st.slider("Row Gap (px)", 0, 20, 6, 1, key="row_gap")

        st.markdown("---")

        # Training parameters
        st.markdown("**Training Parameters**")
        iterations = st.slider("Iterations", 100, 20000, 2000, 100, key="iterations")
        learning_rate = st.slider(
            "Learning Rate", 0.001, 0.1, 0.01, 0.001, key="learning_rate", format="%.3f"
        )
        diversity_weight = st.slider(
            "Diversity Weight",
            0.0,
            0.1,
            0.01,
            0.001,
            key="diversity_weight",
            format="%.3f",
        )

        st.markdown("---")

        # Temperature settings
        st.markdown("**Temperature Settings**")
        temp_start = st.slider("Temp Start", 0.1, 5.0, 1.0, 0.1, key="temp_start")
        temp_end = st.slider(
            "Temp End", 0.001, 1.0, 0.01, 0.001, key="temp_end", format="%.3f"
        )

        st.markdown("---")

        # Options
        st.markdown("**Options**")
        optimize_alignment = st.checkbox(
            "Optimize Alignment", value=False, key="optimize_alignment"
        )
        use_gumbel = st.checkbox("Use Gumbel Softmax", value=True, key="use_gumbel")
        dark_mode = st.checkbox("Dark Mode", value=True, key="dark_mode")

        st.markdown("---")

        # Encoding
        encoding = st.selectbox("Encoding", ["cp437", "ascii"], key="encoding")

    # Execute button
    execute_button = st.button("🚀 EXECUTE", type="primary", use_container_width=True)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### SOURCE PREVIEW")

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)

        # Save to temp file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            image.save(tmp_file.name)
            temp_image_path = tmp_file.name

        st.image(
            image,
            caption=f"Original ({image.size[0]}x{image.size[1]})",
            use_container_width=True,
        )
    else:
        st.info("Upload an image to get started")
        temp_image_path = None

with col2:
    st.markdown("### OUTPUT")

    if (
        st.session_state.output_text is not None
        and st.session_state.output_image is not None
    ):
        # Display output image
        st.image(
            st.session_state.output_image,
            caption="ASCII Art Result",
            use_container_width=True,
        )

        # Terminal-style ASCII output
        st.markdown("#### ASCII TEXT OUTPUT")
        st.markdown(
            f'<div class="terminal-output">{st.session_state.output_text}</div>',
            unsafe_allow_html=True,
        )

        # Download buttons
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="📋 COPY TEXT",
                data=st.session_state.output_text,
                file_name="ascii_art.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with col_dl2:
            with open(st.session_state.output_image, "rb") as img_file:
                st.download_button(
                    label="💾 SAVE PNG",
                    data=img_file.read(),
                    file_name="ascii_art.png",
                    mime="image/png",
                    use_container_width=True,
                )
    else:
        st.info("Click EXECUTE to generate ASCII art")

# Execute conversion
if execute_button:
    if uploaded_file is None:
        st.error("Please upload an image first!")
    elif not GRADSCII_AVAILABLE:
        st.error("gradscii-art engine not available. Please check installation.")
    else:
        try:
            # Reset output
            reset_output()

            # Prepare parameters
            params = {
                "char_width": char_width,
                "char_height": char_height,
                "grid_width": grid_width,
                "grid_height": grid_height,
                "row_gap": row_gap,
                "iterations": iterations,
                "learning_rate": learning_rate,
                "diversity_weight": diversity_weight,
                "temp_start": temp_start,
                "temp_end": temp_end,
                "optimize_alignment": optimize_alignment,
                "use_gumbel": use_gumbel,
                "dark_mode": dark_mode,
                "encoding": encoding,
            }

            # Run conversion
            with st.spinner("Generating ASCII art... This may take a while."):
                output_png, output_txt = run_ascii_conversion(temp_image_path, params)

            # Load outputs
            st.session_state.output_image = output_png
            with open(output_txt, "r", encoding=encoding) as f:
                st.session_state.output_text = f.read()

            # Clean up temp file
            if temp_image_path and os.path.exists(temp_image_path):
                os.unlink(temp_image_path)

            # Force rerun to display results
            st.rerun()

        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback

            st.error(traceback.format_exc())

# Footer with license info
st.markdown("---")
with st.expander("📜 LICENSE & ATTRIBUTION"):
    st.markdown(
        """
    <div class="about-text">
    <strong>ASCII Art Generator</strong><br><br>
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.<br><br>
    
    <strong>Attribution:</strong><br>
    • gradscii-art by Zellic - <a href="https://github.com/stong/gradscii-art">https://github.com/stong/gradscii-art</a> (AGPL-3.0)<br>
    • ascii.micr by Microck - <a href="https://github.com/Microck/ascii.micr">https://github.com/Microck/ascii.micr</a> (MIT)<br><br>
    
    POWERED BY: <a href="https://pytorch.org">PyTorch</a> & <a href="https://streamlit.io">Streamlit</a>
    </div>
    """,
        unsafe_allow_html=True,
    )
