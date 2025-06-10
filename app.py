from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
import cv2
import face_recognition
import os
import pickle
import numpy as np
import base64
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Restrict origins in production

# Directory for storing face encodings and images
KNOWN_DIR = "known_faces"
os.makedirs(KNOWN_DIR, exist_ok=True)

# Initialize video capture with error handling
try:
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        logging.error("Failed to open webcam")
        video_capture = None
except Exception as e:
    logging.error(f"Video capture initialization failed: {str(e)}")
    video_capture = None

def load_known_faces():
    """Load known face encodings and names from the known_faces directory."""
    known_encodings = []
    known_names = []
    try:
        for name in os.listdir(KNOWN_DIR):
            user_dir = os.path.join(KNOWN_DIR, name)
            encoding_path = os.path.join(user_dir, "encoding.pickle")
            if os.path.exists(encoding_path):
                with open(encoding_path, "rb") as f:
                    encodings = pickle.load(f)
                    if isinstance(encodings, list):
                        known_encodings.extend(encodings)
                        known_names.extend([name] * len(encodings))
                    else:
                        known_encodings.append(encodings)
                        known_names.append(name)
    except Exception as e:
        logging.error(f"Error loading known faces: {str(e)}")
    return known_names, known_encodings

def generate_frames():
    """Generate video frames for the live feed with face recognition."""
    if video_capture is None:
        logging.error("Video capture not initialized")
        return

    while True:
        try:
            success, frame = video_capture.read()
            if not success:
                logging.error("Failed to capture video frame")
                break

            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            faces = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, faces)

            known_names, known_encodings = load_known_faces()

            for (top, right, bottom, left), encoding in zip(faces, encodings):
                name = "Unknown"
                if known_encodings:
                    matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
                    face_distances = face_recognition.face_distance(known_encodings, encoding)
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_names[best_match_index]

                # Scale coordinates back to original frame size
                top, right, bottom, left = [v * 4 for v in (top, right, bottom, left)]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (left + 5, bottom - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            logging.error(f"Error in generate_frames: {str(e)}")
            break

@app.route('/')
def index():
    """Render the main page."""
    try:
        return render_template('index.html')
    except Exception as e:
        logging.error(f"Error rendering index.html: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Failed to load index page'}), 500

@app.route('/video_feed')
def video_feed():
    """Stream video feed with face recognition."""
    try:
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logging.error(f"Error in video_feed: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Failed to stream video'}), 500

@app.route('/register', methods=['POST'])
def register():
    """Register a new face from an uploaded image."""
    try:
        data = request.get_json()
        if not data or 'name' not in data or 'image' not in data:
            logging.error("Invalid input: missing name or image")
            return jsonify({'status': 'error', 'message': 'Name and image are required'}), 400

        name = data['name'].strip()
        if not name:
            logging.error("Invalid input: empty name")
            return jsonify({'status': 'error', 'message': 'Name cannot be empty'}), 400

        # Decode base64 image
        try:
            image_data = base64.b64decode(data['image'].split(',')[1])
            np_img = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        except Exception as e:
            logging.error(f"Image decode failed: {str(e)}")
            return jsonify({'status': 'error', 'message': 'Invalid image data'}), 400

        if frame is None:
            logging.error("Invalid image: decoded frame is None")
            return jsonify({'status': 'error', 'message': 'Uploaded file is not a valid image'}), 400

        # Resize image for faster processing
        if frame.shape[1] > 640:
            frame = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))

        # Detect face
        face_locations = face_recognition.face_locations(frame, model="hog")  # Use HOG for better performance
        if not face_locations:
            logging.error("No face detected in the image")
            return jsonify({'status': 'error', 'message': 'No face detected in the image'}), 400

        encoding = face_recognition.face_encodings(frame, [face_locations[0]])[0]

        # Save image and encoding
        user_dir = os.path.join(KNOWN_DIR, name)
        os.makedirs(user_dir, exist_ok=True)

        image_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        image_path = os.path.join(user_dir, image_filename)
        try:
            cv2.imwrite(image_path, frame)
            logging.info(f"Image saved at {image_path}")
        except Exception as e:
            logging.error(f"Failed to save image: {str(e)}")
            return jsonify({'status': 'error', 'message': 'Failed to save image'}), 500

        encoding_path = os.path.join(user_dir, "encoding.pickle")
        try:
            existing_encodings = []
            if os.path.exists(encoding_path):
                with open(encoding_path, "rb") as f:
                    existing_encodings = pickle.load(f)
                if not isinstance(existing_encodings, list):
                    existing_encodings = [existing_encodings]
            existing_encodings.append(encoding)
            with open(encoding_path, "wb") as f:
                pickle.dump(existing_encodings, f)
                logging.info(f"Encoding saved at {encoding_path}")
        except Exception as e:
            logging.error(f"Failed to save encoding: {str(e)}")
            return jsonify({'status': 'error', 'message': 'Failed to save encoding'}), 500

        return jsonify({'status': 'success', 'message': f'{name} registered successfully'})

    except Exception as e:
        logging.error(f"Server error in register: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Server error'}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'message': 'Server is running'})

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)
    except Exception as e:
        logging.error(f"Failed to start Flask server: {str(e)}")
    finally:
        if video_capture is not None:
            video_capture.release()