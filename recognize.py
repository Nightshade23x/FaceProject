from deepface import DeepFace
import cv2
import os

db_path = "known/"
cap=cv2.VideoCapture(0)
print("Recognizing Face...")
while True:
    ret,frame=cap.read()
    if not ret:
        break
    try:
        result = DeepFace.find(frame, db_path=db_path, enforce_detection=False)
        if len(result)>0:
            identity=os.path.basename(result[0]['identity'][0])
            name=identity.split(".")[0]
        else:
            name="Unknown"
    except:
        name="Unknown"
    cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()