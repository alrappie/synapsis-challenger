import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from flask import Flask, Response, render_template, request
import cv2
import numpy as np
from ultralytics import YOLO  # Use YOLOv8
import onnxruntime as ort

app = Flask(__name__)

# URL CCTV Streaming
# VIDEO_URL = "https://cctvjss.jogjakota.go.id/malioboro/NolKm_Utara.stream/playlist.m3u8"
VIDEO_URL = 'https://cctvjss.jogjakota.go.id/malioboro/Malioboro_30_Pasar_Beringharjo.stream/playlist.m3u8'
cap = cv2.VideoCapture(VIDEO_URL)

# Initial bounding box coordinates
bounding_boxes = [(400, 150, 600, 350)]  # (x1, y1, x2, y2)

# Left padding
PADDING_LEFT = 30

# Initial gridline settings
gridline_settings = {'isgrid': True, 'grid_step': 50, 'grid_thickness': 1}

# Load YOLOv8m Model
model = YOLO("yolov5nu.onnx")  # Load ONNX model
# Load ONNX model with ONNX Runtime
session = ort.InferenceSession("yolov5nu.onnx", providers=["CPUExecutionProvider"])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/update_bounding_box", methods=["POST"])
def update_bounding_box():
    global bounding_boxes
    x1 = int(request.form.get('x1'))
    y1 = int(request.form.get('y1'))
    x2 = int(request.form.get('x2'))
    y2 = int(request.form.get('y2'))
    
    bounding_boxes = [(x1, y1, x2, y2)]
    return "Bounding Box Updated", 200

@app.route("/update_grid_settings", methods=["POST"])
def update_grid_settings():
    global gridline_settings
    gridline_enabled = request.form.get('isgrid') == 'true'  # Convert to boolean
    gridline_settings['isgrid'] = gridline_enabled
    return "Gridline Settings Updated", 200


def run_onnx_inference(image):
    img = cv2.resize(image, (640, 640))  # Resize to model input size
    img = img.transpose(2, 0, 1)  # Convert HWC to CHW
    img = img.astype(np.float32) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Run inference
    outputs = session.run(None, {"images": img})

    # Extract detections (depends on the ONNX export format)
    return outputs

def generate_frames():
    global gridline_settings
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run YOLOv8 detection (only detect people)
        # results = model(frame, classes=[0])  # Only detect persons
        # detections = results[0].boxes.data.cpu().numpy()  # Get predictions
        # Run ONNX YOLO inference
        detections = run_onnx_inference(frame)

        # Draw bounding boxes around detected persons
        for det in detections:
            print(det)
            print(len(det))
            # x1, y1, x2, y2, conf, cls = det
            # Draw bounding box with confidence score
            # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # label = f"Person {conf:.2f}"
            # cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw the grid lines inside the frame if enabled
        height, width, _ = frame.shape
        grid_step = gridline_settings['grid_step']  # Distance between grid lines in pixels

        # Create a padded image with space for labels (add padding on the left)
        padded_frame = np.zeros((height + 60, width + PADDING_LEFT + 60, 3), dtype=np.uint8)
        padded_frame[30:30 + height, PADDING_LEFT + 30:PADDING_LEFT + 30 + width] = frame  # Copy the frame to the center

        if gridline_settings['isgrid']:  # Check if grid is enabled
            # Draw the grid lines (inside the padded area)
            for x in range(0, width, grid_step):
                # Vertical grid lines
                cv2.line(padded_frame, (PADDING_LEFT + 30 + x, 30), (PADDING_LEFT + 30 + x, 30 + height), (255, 255, 255), gridline_settings['grid_thickness'])
            for y in range(0, height, grid_step):
                # Horizontal grid lines
                cv2.line(padded_frame, (PADDING_LEFT + 30, 30 + y), (PADDING_LEFT + 30 + width, 30 + y), (255, 255, 255), gridline_settings['grid_thickness'])

        # Label the grid (outside the image)
        for x in range(0, width, grid_step):
            # Horizontal labels
            label = str(x)
            cv2.putText(padded_frame, label, (PADDING_LEFT + 30 + x, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for y in range(0, height, grid_step):
            # Vertical labels with extra space on the left
            label = str(y)
            cv2.putText(padded_frame, label, (5, 30 + y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw bounding boxes
        for (x1, y1, x2, y2) in bounding_boxes:
            cv2.rectangle(padded_frame, (PADDING_LEFT + 30 + x1, 30 + y1), (PADDING_LEFT + 30 + x2, 30 + y2), (0, 255, 0), 2)

        # Encode frame for streaming
        _, buffer = cv2.imencode(".jpg", padded_frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
