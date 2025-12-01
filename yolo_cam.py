from ultralytics import YOLO
import cv2

# Load YOLOv8 small model (recommended for fast real-time)
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO on frame
    results = model(frame, stream=True)

    # Draw boxes
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # bounding box
            cls = int(box.cls[0])        # class index
            conf = float(box.conf[0])    # confidence score
            label = model.names[cls]     # class name

            # Draw rectangle + label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

    cv2.imshow("YOLO Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
