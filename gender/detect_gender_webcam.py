from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2

# Load the model
model_path = r'D:\Ml\gender_detection.h5'  # Absolute path to the model
model = load_model(model_path)

# Open webcam
webcam = cv2.VideoCapture(0)
classes = ['man', 'woman']

# Loop through frames
while webcam.isOpened():
    # Read frame from webcam
    status, frame = webcam.read()
    if not status:
        break

    # Convert frame to RGB (cv2 reads frames in BGR format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply face detection (replace this with your face detection method)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(rgb_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Crop and preprocess face
        face_crop = rgb_frame[y:y+h, x:x+w]
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # Predict gender
        conf = model.predict(face_crop)[0]
        idx = np.argmax(conf)
        label = classes[idx]
        label_text = "{}: {:.2f}%".format(label, conf[idx] * 100)

        # Draw label above rectangle
        y_text = y - 10 if y - 10 > 10 else y + 10
        cv2.putText(frame, label_text, (x, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display output
    cv2.imshow("Gender Detection", frame)

    # Press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
