from ultralytics import YOLO
from datetime import datetime, timedelta
import cv2
import os
import streamlit as st
from khayyam import JalaliDatetime

# Define the path where detected images will be saved
detected_path = "F:\HD\yolov8-streamlit-detection-tracking-master\detected"

# Initialize the last saved time
last_saved_time = datetime.now() - timedelta(seconds=5)

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
    model = YOLO(model_path)
    model.classes = [0]  # Assuming class 0 is 'person'
    return model

def _display_detected_frames(conf, model, st_frame, image):
    image = cv2.resize(image, (720, int(720 * (9 / 16))))
    res = model.predict(image, conf=conf, classes=[0])
    res_plotted = res[0].plot()

    # Check if a person is detected
    if res[0].boxes:
        save_detected_image(detected_path, res_plotted)

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

# Example usage (replace with actual parameters)
# conf = 0.5  # Confidence threshold
# model_path = 'path_to_model'  # Path to YOLO model
# camera_ip = 'rtsp://camera_ip_address'  # RTSP stream URL
# model = load_model(model_path)
# play_rtsp_stream(conf, model, camera_ip)
