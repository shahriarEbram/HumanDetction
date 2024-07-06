from ultralytics import YOLO
from datetime import datetime, timedelta
import cv2
import os
import streamlit as st
from khayyam import JalaliDatetime
from pathlib import Path
from playsound import playsound
from threading import Thread
import torch

detected_path = "F:\HD\yolov8-streamlit-detection-tracking-master\detected"

# Initialize the last saved time
last_saved_time = datetime.now() - timedelta(seconds=5)


# def play_sound(sound_file):
#     playsound(sound_file)


def save_detected_image(detected_path, res_plotted):
    global last_saved_time
    current_time = datetime.now()
    if current_time - last_saved_time >= timedelta(seconds=5):
        # Create the directory based on the current Jalali date
        jalali_date = JalaliDatetime(current_time)
        date_directory = jalali_date.strftime("%Y/%m/%d")
        full_path = os.path.join(detected_path, date_directory)
        os.makedirs(full_path, exist_ok=True)

        # Generate the file name based on the current time
        filename = os.path.join(full_path, current_time.strftime("%H.%M.%S") + ".jpg")

        # Save the detected image
        cv2.imwrite(filename, res_plotted)

        # Update the last saved time
        last_saved_time = current_time

        print(f"Image saved to {filename}")  # Debug statement


def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(model_path).to(device)
    model.classes = [0]  # Assuming class 0 is 'person'
    return model


def _display_detected_frames(conf, model, st_frame, image):
    image = cv2.resize(image, (720, int(720 * (9 / 16))))
    res = model.predict(image, conf=conf, classes=[0])
    res_plotted = res[0].plot()

    # Check if a person is detected
    if res[0].boxes:
        save_detected_image(detected_path, res_plotted)
        # if alert:
        #     play_sound_thread = Thread(target=play_sound, args=('alarm.mp3',))
        #     play_sound_thread.start()

    st_frame.image(res_plotted, caption='Detected Video', channels="BGR", use_column_width=True)


def play_rtsp_stream(conf, model, camera_ip):
    try:
        vid_cap = cv2.VideoCapture(camera_ip)
        st_frame = st.empty()
        while vid_cap.isOpened():
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(conf, model, st_frame, image)
            else:
                vid_cap.release()
                break
    except Exception as e:
        vid_cap.release()
        st.sidebar.error("Error loading RTSP stream: " + str(e))


# ******************************************************************************* #

# Setting page layout
st.set_page_config(
    page_title="Balintech Security Cameras",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Balintech Security Cameras")

# Model Options
model_type = 'Detection'

st.sidebar.header("Settings")

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 25)) / 100

# Selecting Detection Or Segmentation
model_path = Path("weights/yolov8n.pt")

# Load Pre-trained ML Model
try:
    model = load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

# time_to_save_images = float(st.sidebar.slider(
#     "time to save images", 3, 15, 5)) / 100
#

# alert = st.sidebar.checkbox("⚠️ Alert")

detection = st.sidebar.button("Open Images", type="primary")

if detection:
    path = os.path.realpath(detected_path)
    os.startfile(path)

col1, col2 = st.columns(2)
with col1:
    play_rtsp_stream(confidence,
                     model,
                     "rtsp://admin:Balintech2@192.168.252.15:554/live")

# ******************************************************************************* #
