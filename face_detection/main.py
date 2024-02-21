# main.py
import cv2
import dlib
import pandas as pd
from datetime import datetime
from torchvision.transforms import functional as F
from models.yolo_model import YOLOModel
import sys
print(sys.path)

# Define the paths to the YOLO configuration file and weights
yolo_config = "models\yolov5\models\yolov5s.yaml"
yolo_weights = "./yolov5s.pt"

# Load the YOLO model
model = YOLOModel(yolo_config, yolo_weights)
model.autoshape()  # Ensure the model is auto-shaped

# Load Dlib face detector
detector = dlib.get_frontal_face_detector()

# Initialize DataFrame for logging
columns = ['PersonID', 'ArrivalTime', 'LeftTime', 'TimeSpent']
df = pd.DataFrame(columns=columns)

# Open video capture
cap = cv2.VideoCapture("./video.mp4")

boxes_person = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO person detection
    results = model.detect(frame)

    # Process person detection results
    for pred in results.xyxy[0].cpu().numpy():
        if pred[5] == 0 and pred[4] > 0.5:  # Assuming person class ID is 0
            x_person, y_person, w_person, h_person = map(int, pred[:4])

            # Rectangle coordinates
            boxes_person.append([x_person, y_person, w_person, h_person])

            # Face detection using dlib
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            # Process face detection results
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()

                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Add face information to DataFrame
                person_id = len(df) + 1
                arrival_time = datetime.now()
                df = df.append({'PersonID': person_id, 'ArrivalTime': arrival_time, 'LeftTime': None, 'TimeSpent': None},
                               ignore_index=True)

                # Crop and save face image
                face_crop = frame[y:y + h, x:x + w]
                cv2.imwrite(f"face_images/person_{person_id}.jpg", face_crop)

    # Display the result
    cv2.imshow("Person and Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()

# Save DataFrame to CSV
df.to_csv("person_log.csv", index=False)
