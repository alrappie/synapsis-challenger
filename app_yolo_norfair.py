import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import cv2
import numpy as np
import psycopg2
from flask import Flask, Response, render_template, jsonify, request
from ultralytics import YOLO
from norfair import Detection, Tracker
from dotenv import load_dotenv

# Flask app setup
app = Flask(__name__)


# Load environment variables from .env file
load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

# CCTV Streaming URL
# VIDEO_URL = "https://cctvjss.jogjakota.go.id/malioboro/NolKm_Utara.stream/playlist.m3u8" # for another link camera
VIDEO_URL = "https://cctvjss.jogjakota.go.id/malioboro/Malioboro_10_Kepatihan.stream/playlist.m3u8"
cap = cv2.VideoCapture(VIDEO_URL)

# Load YOLO Model
model = YOLO('./models/yolov5nu.onnx')

# Initialize Norfair Tracker
tracker = Tracker(distance_function="euclidean", distance_threshold=90, hit_counter_max=20, past_detections_length=15)

# Define the polygon (ROI)
# polygon_points = np.array([(1200, 800), (600, 800), (600, 600), (1200, 600)], dtype=np.int32) for another URL
polygon_points = np.array([(1200, 150), (700, 150), (700, 300), (1200, 300)], dtype=np.int32)
polygon_name = 'default'

# Dictionary to track person states
person_db = {}  # Mapping tracker ID -> Unique Database ID
last_person_id = 0  # Counter untuk ID unik

person_states = {}

frame_skip = 3  # Process every 5th frame


def connect_db():
    return psycopg2.connect(**DB_CONFIG)
        
def insert_event(person_id, event, x, y, area_name, polygon_points):
    try:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO person_tracking (person_id, event, x, y, area_name, polygon_coordinates, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            """,
            (int(person_id), str(event), float(x), float(y), str(area_name), str(polygon_points))
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print("Database error:", e)
        
def get_stats(live=False):
    conn = connect_db()
    cursor = conn.cursor()

    query = """
        SELECT event, COUNT(DISTINCT person_id) AS total, MAX(timestamp) AS timestamp
        FROM person_tracking
        WHERE 1=1
    """

    params = []

    if live:
        query += " AND timestamp >= NOW() - INTERVAL '10 seconds' "
        query += " GROUP BY event ORDER BY MAX(timestamp) DESC"
    else:
        start_date = request.args.get("start_date")  # Format: YYYY-MM-DD
        end_date = request.args.get("end_date")  # Format: YYYY-MM-DD
        limit = request.args.get("limit", default=10, type=int)
        offset = request.args.get("offset", default=0, type=int)
    
        if start_date:
            query += " AND timestamp >= %s "
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= %s "
            params.append(end_date)

        query += " GROUP BY event ORDER BY MAX(timestamp) DESC LIMIT %s OFFSET %s"
        params.append(limit)
        params.append(offset)

    cursor.execute(query, tuple(params))
    stats = cursor.fetchall()
    cursor.close()
    conn.close()

    return {"data": [{"event": row[0], "count": row[1], "latest_timestamp": row[2]} for row in stats]}


@app.route("/api/stats/")
def api_stats():
    return jsonify(get_stats())


@app.route("/api/stats/live")
def api_stats_live():
    return jsonify(get_stats(live=True))


@app.route("/api/config/area", methods=["POST"])
def update_area():
    global polygon_points
    global polygon_name
    data = request.json
    if "polygon" in data:
        polygon_points = np.array(data["polygon"], dtype=np.int32)
        polygon_name = data['name']
        return jsonify({"message": "Polygon updated successfully"}), 200
        
    return jsonify({"error": "Invalid data"}), 400


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

def is_inside_polygon(point, polygon):
    """
    Check if a point (x, y) is inside the given polygon using cv2.pointPolygonTest.
    Returns True if inside, False otherwise.
    """
    point = np.array(point, dtype=np.float32)  # Ensure correct data type
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def get_unique_person_id(tracker_id):
    global last_person_id
    if tracker_id not in person_db:
        last_person_id += 1
        person_db[tracker_id] = last_person_id
    return person_db[tracker_id]

def generate_frames():
    frame_count = 0
    grid_size = 100  # Adjust grid spacing
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:  # Skip frames to reduce load
            continue
        
        height, width, _ = frame.shape  # Get frame dimensions

        # 🔹 Draw the Grid
        for x in range(0, width, grid_size):
            cv2.line(frame, (x, 0), (x, height), (200, 200, 200), 1)  # Vertical lines
            cv2.putText(frame, str(x), (x + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        for y in range(0, height, grid_size):
            cv2.line(frame, (0, y), (width, y), (200, 200, 200), 1)  # Horizontal lines
            cv2.putText(frame, str(y), (5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        results = model(frame, verbose=False)  
        detections = []

        
        for det in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = det[:6]
            class_id = int(cls)
            if class_id == 0 and conf > 0.3:  # Only detect "person" class
                centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))  
                detections.append(Detection(points=np.array([[centroid[0], centroid[1]]]), scores=np.array([conf])))

        tracked_objects = tracker.update(detections)

        for obj in tracked_objects:
            x, y = obj.estimate.flatten().astype(int)
            inside = is_inside_polygon((x, y), polygon_points)
            prev_state = person_states.get(obj.id, "outside")
            
            if inside and prev_state == "outside":
                event = "entered"
            elif inside:
                event = "inside"
            elif not inside and prev_state in ["inside", "entered"]:
                event = "exited"
            else:
                event = prev_state  # No change
            
            insert_event(obj.id, event, x, y, polygon_name, polygon_points)
            person_states[obj.id] = event
            # 🆕 Dapatkan ID unik untuk orang ini
            # unique_id = get_unique_person_id(obj.id)
            
            color = (0, 255, 0) if inside else (0, 0, 255)
            cv2.circle(frame, (x, y), 5, color, -1)
            cv2.putText(frame, f"ID {obj.id} - {event}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw the polygon ROI
        cv2.polylines(frame, [polygon_points], isClosed=True, color=(255, 0, 0), thickness=2)

        # Encode frame for streaming
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
