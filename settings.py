from pathlib import Path
import sys

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
# Get the parent directory of the current file
ROOT = FILE.parent
# Add the root path to the sys.path list if it is not already there
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
# Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

# Sources
RTSP = 'RTSP'

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'office_4_detected.jpg'



# ML Model config
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'yolov8n.pt'


# Webcam
WEBCAM_PATH = 0
