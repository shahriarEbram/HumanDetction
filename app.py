# Python In-built packages
from pathlib import Path
import PIL
import threading
# External packages
import streamlit as st
from threading import Thread
# Local Modules
import settings
import helper



# Setting page layout
st.set_page_config(
    page_title="Balintech Security Cameras",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Balintech Security Cameras")

# Model Options
model_type ='Detection'

st.sidebar.header("Settings")

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 25)) / 100

# Selecting Detection Or Segmentation
model_path = Path(settings.DETECTION_MODEL)


# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)



# time_to_save_images = float(st.sidebar.slider(
#     "time to save images", 3, 15, 5)) / 100
#



col1, col2 = st.columns(2)
with col1:
    helper.play_rtsp_stream(confidence,
                            model,
                            "rtsp://admin:AMir98@192.168.252.22:554/live/2")
