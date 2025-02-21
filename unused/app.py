import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from flask import Flask, Response, render_template, request
import cv2
import numpy as np
from ultralytics import YOLO
# from ultralytics.utils import LOGGER
# LOGGER.setLevel(50)  # 50 = CRITICAL, hides all logs


app = Flask(__name__)

# URL CCTV Streaming
# VIDEO_URL = "https://cctvjss.jogjakota.go.id/malioboro/NolKm_Utara.stream/playlist.m3u8"
VIDEO_URL = "https://cctvjss.jogjakota.go.id/malioboro/Malioboro_10_Kepatihan.stream/playlist.m3u8"

cap = cv2.VideoCapture(VIDEO_URL)

# Bounding Box for tracking (x1, y1, x2, y2)
bounding_boxes = (400, 150, 600, 350)

# Load YOLO model (ONNX format)
model = YOLO('./models/yolov5nu.onnx')

object_status = {}  # Stores {id: "inside"/"outside"}
previous_centroids = {}  # Stores {id: (center_x, center_y)}
next_object_id = 0  # Counter for new objects

# Grid settings & padding
gridline_settings = {'isgrid': True, 'grid_step': 50, 'grid_thickness': 1}
PADDING_LEFT = 30
PADDING_TOP = 30

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/update_grid_settings", methods=["POST"])
def update_grid_settings():
    """API to enable/disable grid"""
    global gridline_settings
    gridline_enabled = request.form.get('isgrid') == 'true'
    gridline_settings['isgrid'] = gridline_enabled
    return "Gridline Settings Updated", 200

@app.route("/update_bounding_box", methods=["POST"])
def update_bounding_box():
    global bounding_boxes
    x1 = int(request.form.get('x1'))
    y1 = int(request.form.get('y1'))
    x2 = int(request.form.get('x2'))
    y2 = int(request.form.get('y2'))
    
    bounding_boxes = (x1, y1, x2, y2)
    return "Bounding Box Updated", 200

def is_inside_bbox(x1, y1, x2, y2, bbox):
    """Check if object is inside bounding box"""
    bx1, by1, bx2, by2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return bx1 <= center_x <= bx2 and by1 <= center_y <= by2

def assign_object_id(x1, y1, x2, y2):
    """Assigns a unique ID to detected objects based on proximity to previous detections."""
    global next_object_id
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    min_distance = float("inf")
    matched_id = None

    # Find closest previous object
    for obj_id, (prev_x, prev_y) in previous_centroids.items():
        distance = np.linalg.norm([center_x - prev_x, center_y - prev_y])
        if distance < 50:  # Threshold distance to consider the same object
            min_distance = distance
            matched_id = obj_id

    # Assign new ID if no match found
    if matched_id is None:
        matched_id = next_object_id
        next_object_id += 1

    # Update centroid tracking
    previous_centroids[matched_id] = (center_x, center_y)
    return matched_id

def generate_frames():
    global object_status
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run object detection
        results = model(frame, verbose=False)
        detections = results[0].boxes.data.cpu().numpy()

        # Filter for "person" class (COCO class 0)
        person_detections = [det for det in detections if int(det[5]) == 0]

        # Object tracking
        new_status = object_status.copy() 
        for det in person_detections:
            x1, y1, x2, y2, conf, cls = det[:6]
            obj_id = obj_id = assign_object_id(x1, y1, x2, y2)

            inside = is_inside_bbox(x1, y1, x2, y2, bounding_boxes)

            # Ensure new detections are tracked
            if obj_id not in object_status:
                object_status[obj_id] = "outside"  # Default to "outside"

            prev_status = object_status[obj_id]

            if prev_status == "outside" and inside:
                print(f"🔵 Person {obj_id} ENTERED the bounding box")
            elif prev_status == "inside" and not inside:
                print(f"🔴 Person {obj_id} EXITED the bounding box")

            # Update status
            object_status[obj_id] = "inside" if inside else "outside"

            # Draw bounding box
            color = (0, 255, 0) if inside else (0, 0, 255)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"Person {conf:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        object_status.update(new_status)

        # Grid settings
        height, width, _ = frame.shape
        grid_step = gridline_settings['grid_step']

        # Create padded frame (extra space for labels)
        padded_frame = np.zeros((height + PADDING_TOP, width + PADDING_LEFT, 3), dtype=np.uint8)
        padded_frame[PADDING_TOP:, PADDING_LEFT:] = frame

        # Draw grid if enabled
        if gridline_settings['isgrid']:
            for x in range(0, width, grid_step):
                cv2.line(padded_frame, (PADDING_LEFT + x, PADDING_TOP), (PADDING_LEFT + x, PADDING_TOP + height), (255, 255, 255), gridline_settings['grid_thickness'])
            for y in range(0, height, grid_step):
                cv2.line(padded_frame, (PADDING_LEFT, PADDING_TOP + y), (PADDING_LEFT + width, PADDING_TOP + y), (255, 255, 255), gridline_settings['grid_thickness'])

        # Add grid labels
        for x in range(0, width, grid_step):
            cv2.putText(padded_frame, str(x), (PADDING_LEFT + x, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for y in range(0, height, grid_step):
            cv2.putText(padded_frame, str(y), (5, PADDING_TOP + y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw bounding box
        x1, y1, x2, y2 = bounding_boxes
        cv2.rectangle(padded_frame, (PADDING_LEFT + x1, PADDING_TOP + y1), (PADDING_LEFT + x2, PADDING_TOP + y2), (255, 255, 255), 2)

        # Encode frame for streaming
        _, buffer = cv2.imencode(".jpg", padded_frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
