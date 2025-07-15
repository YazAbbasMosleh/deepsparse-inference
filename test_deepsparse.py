from deepsparse.yolov8 import YOLOv8Pipeline
import cv2
import numpy as np
import time



model_path = "./yolov8n.onnx"
print("model name read ...")

yolo_pipeline = YOLOv8Pipeline(model_path=model_path)
print("yolo pipeline created")

image_path = "./messi1.jpg"
image = cv2.imread(image_path)

# Run inference

pipeline_outputs= yolo_pipeline(images=[image_path])


t1 = time.time()
for _ in range(1000):
    _ = yolo_pipeline(images=[image_path])
print(f"----------------------Elpased: {(time.time() - t1)/1000}")

boxes = pipeline_outputs.boxes[0]  
scores = pipeline_outputs.scores[0]  
labels = pipeline_outputs.labels[0]  

print("Detections:")
for box, score, label in zip(boxes, scores, labels):
    x1, y1, x2, y2 = box
    print(f"Class: {label}, Confidence: {score:.2f}, Box: {x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}")

    cv2.rectangle(
        image,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        (0, 255, 0),  
        2,
    )

    cv2.putText(
        image,
        f"{label}: {score:.2f}",
        (int(x1), int(y1) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )

output_path = "./output.jpg"
cv2.imwrite(output_path, image)
print(f"Saved output to {output_path}")

cv2.imshow("Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()