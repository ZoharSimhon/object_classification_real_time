import fiftyone as fo # type: ignore
import fiftyone.zoo as foz # type: ignore
import torch # type: ignore
import cv2 # type: ignore
import numpy as np
import time  # Import time module


# Load the COCO dataset with FiftyOne
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections", "segmentations"],
    classes=["person", "car"],
    max_samples=50,
)

# Visualize the dataset in the FiftyOne App
session = fo.launch_app(dataset)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Capture video stream
cap = cv2.VideoCapture(0)

start_time = time.time()  # Record the start time

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_frame = cv2.resize(frame, (416, 416))
    input_frame = input_frame / 255.0
    input_frame = np.expand_dims(input_frame, axis=0)

    # Perform detection
    results = model(frame)
    
    # Process results and display bounding boxes on the frame
    results.render()  # Updates the frame with bounding boxes

    # Post-process the predictions
    # Implement the post-processing steps here
    # For example: Non-max suppression, mapping predictions to class labels

    # Display the frame with detected objects
    cv2.imshow('Frame', frame)
    # cv2.imshow('Frame', results.imgs[0])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Check if 10 seconds have passed
    elapsed_time = time.time() - start_time
    if elapsed_time > 10:
        break

cap.release()
cv2.destroyAllWindows()
