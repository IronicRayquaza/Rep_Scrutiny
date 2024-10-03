from flask import Flask, render_template, Response, jsonify
import pandas as pd
import numpy as np
import pickle
import mediapipe as mp
import cv2
from landmarks import landmarks  # Ensure landmarks are defined in a separate file or script

app = Flask(__name__)

# Load the trained machine learning model
with open('model/deadlift.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize MediaPipe for pose detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize global variables
current_stage = ''
counter = 0
bodylang_prob = np.array([0, 0])
bodylang_class = ''


# Video streaming generator function
def gen_frames():
    global current_stage, counter, bodylang_class, bodylang_prob

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Convert the image to RGB and process it with MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            
            # Draw pose landmarks on the frame
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(106, 13, 173), thickness=4, circle_radius=5),
                                      mp_drawing.DrawingSpec(color=(255, 102, 0), thickness=5, circle_radius=10))

            # Make predictions using the trained model
            try:
                if results.pose_landmarks:
                    # Extract pose landmarks into a row
                    row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
                    
                    # Prepare data as input for the model
                    X = pd.DataFrame([row], columns=landmarks)
                    
                    # Make predictions and update the status
                    bodylang_prob = model.predict_proba(X)[0]
                    bodylang_class = model.predict(X)[0]

                    if bodylang_class == "down" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
                        current_stage = "down"
                    elif current_stage == "down" and bodylang_class == "up" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
                        current_stage = "up"
                        counter += 1

            except Exception as e:
                print(f"Error: {e}")

            # Convert image back to BGR for OpenCV rendering
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame as part of the HTTP stream response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Route to render the main HTML page
@app.route('/')
def index():
    return render_template('index.html')


# Route to provide video streaming
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Route to provide the current status of the counter, stage, and probabilities
@app.route('/get_status')
def get_status():
    return jsonify({
        'counter': int(counter),
        'stage': current_stage,
        'probability': float(bodylang_prob[bodylang_prob.argmax()])  # Ensure the value is JSON-serializable
    })


# Add CSP header
@app.after_request
def add_security_headers(response):
    # Ensure external files are loaded while preventing inline JavaScript
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:;"
    return response


if __name__ == '__main__':
    app.run(debug=True)
