from ultralytics import YOLO
import cv2
import numpy as np
import cvzone
model = YOLO("best922.pt")
class_names = model.names
image_paths = [
    "D:\\NLC Intern\\WhatsApp Image 2024-07-01 at 15.33.11_2b7027dc.jpg"
    #'I:/yolov8-conveyor-belt-damage-detection-main/punturehole.jpeg',
    #'I:/yolov8-conveyor-belt-damage-detection-main/Image3.jpeg',
    #'I:/yolov8-conveyor-belt-damage-detection-main/tear.jpeg'
]
for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image {img_path}")
        continue
    print(f"Processing image: {img_path}")
    img = cv2.resize(img, (1020, 500))
    h, w, _ = img.shape
    results = model.predict(img)
    if results is None or len(results) == 0:
        print(f"No results for image {img_path}")
        continue
    for r in results:
        boxes = r.boxes 
        masks = r.masks
        if masks is not None:
            masks = masks.data.cpu()
            for seg, box in zip(masks.data.cpu().numpy(), boxes):
                seg = cv2.resize(seg, (w, h))
                contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    d = int(box.cls)
                    c = class_names[d]
                    x, y, x1, y1 = cv2.boundingRect(contour)
                    cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
                    cv2.putText(img, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imshow('img', img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
