import cv2
import os
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Initialize the Flask application
app = Flask(__name__)

# Set up directories and allowed file types
UPLOAD_FOLDER = 'uploaded_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Model paths
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-4)', '(4-6)', '(6-13)', '(14-20)', '(21-35)', '(35-45)', '(45-65)', '(65-100)']
genderList = ['Male', 'Female']

# Load the models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to highlight faces in the image and return the face regions
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

# Route for uploading the image and detecting age and gender
@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the request contains a file
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    # Check if the file has a valid filename
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Check if the file is allowed
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Read the image for processing
            frame = cv2.imread(filepath)

            # Detect faces in the image
            resultImg, faceBoxes = highlightFace(faceNet, frame)

            # If no faces are detected, return an error
            if not faceBoxes:
                return jsonify({"error": "No face detected in the image"}), 400

            # Prepare for age and gender prediction
            predictions = []
            padding = 20

            # Process each face detected
            for faceBox in faceBoxes:
                # Validate that the face box is not empty and within image bounds
                if faceBox[2] - faceBox[0] <= 0 or faceBox[3] - faceBox[1] <= 0:
                    continue  # Skip invalid face boxes

                face = frame[max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
                            max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)]

                # Ensure the face region is not empty
                if face.size == 0:
                    continue  # Skip empty face regions

                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

                # Gender prediction
                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]

                # Age prediction
                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]

                # Store the predictions
                predictions.append({
                    "gender": gender,
                    "age": age[1:-1]  # Removing parentheses from age range
                })

            # Return the predictions as a JSON response
            return jsonify({"predictions": predictions}), 200

        finally:
            # Delete the file after processing and sending the response
            if os.path.exists(filepath):
                os.remove(filepath)

    else:
        return jsonify({"error": "Invalid file format"}), 400

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
