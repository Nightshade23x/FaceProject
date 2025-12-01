from deepface import DeepFace
import cv2
import os
import numpy as np


# CUSTOM COSINE DISTANCE

def cosine_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# LOAD DATABASE

db_path = "known/"
print("Loading database...")

database = []

for file in os.listdir(db_path):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(db_path, file)
        emb = DeepFace.represent(img_path, model_name="Facenet", enforce_detection=False)
        name = "Samar"
        database.append((name, emb[0]["embedding"]))

print(f"Loaded {len(database)} known faces.")
print("Starting webcam...")


# REAL-TIME RECOGNITION

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        emb = DeepFace.represent(frame, model_name="Facenet", enforce_detection=False)
        best_name = "Unknown"
        best_distance = 999

        for name, db_emb in database:
            distance = cosine_distance(emb[0]["embedding"], db_emb)

            if distance < best_distance:
                best_distance = distance
                best_name = name

        if best_distance > 0.4:
            best_name = "Unknown"

    except:
        best_name = "Unknown"

    cv2.putText(frame, best_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
