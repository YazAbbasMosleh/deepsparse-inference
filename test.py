import cv2
import time
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
image = cv2.imread("messi1.jpg")

# Warm-up
_ = model(image)

n = 100
start = time.time()
for _ in range(n):
    _ = model(image)

average_time = (time.time() - start) / n
print(f"Average inference time: {average_time:.8f} seconds")
