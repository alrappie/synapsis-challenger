import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from flask import Flask, Response, render_template
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# CCTV Streaming URL
VIDEO_URL = "https://cctvjss.jogjakota.go.id/malioboro/NolKm_Utara.stream/playlist.m3u8"
cap = cv2.VideoCapture(VIDEO_URL)

# Load YOLO Model
model = YOLO('./models/yolov5nu.onnx')

# Define the polygon (ROI)
polygon_points = np.array([(1200, 800), (600, 800), (600, 600), (1200, 600)], dtype=np.int32)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Use YOLO for tracking
        results = model.track(frame, persist=True, classes=[0],source=)  

        if results[0].boxes is not None:
            for det in results[0].boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls, track_id = det[:7]  # track_id is now provided by YOLO
                class_id = int(cls)

                if conf > 0.3:  # Filter out weak detections
                    # Draw a RED bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                    # Calculate center point
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    # Draw a tracking dot (WHITE)
                    cv2.circle(frame, (center_x, center_y), 5, (255, 255, 255), -1)

                    # Display Tracking ID
                    if track_id is not None:
                        cv2.putText(frame, f"ID {int(track_id)}", (int(x1) + 5, int(y1) - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw the polygon ROI
        cv2.polylines(frame, [polygon_points], isClosed=True, color=(255, 0, 0), thickness=2)

        # Encode frame for streaming
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
