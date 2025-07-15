from ultralytics import YOLO
import time


# Load the exported OpenVINO model
ov_model = YOLO("yolov8n_openvino_model/")


times = []
# Run inference

for _ in range(1000):
    t1 = time.time()
    results = ov_model("./messi1.jpg")
    times.append(time.time() - t1)


print(f"Average  time: {sum(times) / len(times)}")