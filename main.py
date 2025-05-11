import torch
import cv2
import numpy as np
from ultralytics import YOLO
import time
from playsound import playsound
import pygame

MODEL_NAME = 'yolov8n.pt'
CAMERA_INDEX = 1 # Change this depending on camera using
CONFIDENCE_THRESHOLD = 0.5
ALERT_COOLDOWN_SECONDS = 30
last_alert_time = 0

try:
    model = YOLO(MODEL_NAME)
    print(f"Model '{MODEL_NAME}' loaded successfully")
    if torch.cuda.is_available():
        model.to('cuda')
        print("Using GPU for detection")
    else:
        model.to('cpu')
        print("Using CPU for detection")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Error: Could not open video stream from index {CAMERA_INDEX}")
    exit()
print(f"Video stream opened successfully from index {CAMERA_INDEX}")

def play_loud_sound():
    pygame.mixer.init()
    pygame.mixer.music.load("alert_sound.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def trigger_notifications(detection_details):
    global last_alert_time
    current_time = time.time()
    if (current_time - last_alert_time) > ALERT_COOLDOWN_SECONDS:
        print(f"HUMAN DETECTED: {detection_details}")
        play_loud_sound()
        last_alert_time = current_time
        print("Notifications triggered")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame. Trying again soon")
            time.sleep(5)
            cap.release()
            cap = cv2.VideoCapture(CAMERA_INDEX)
            if not cap.isOpened():
                print("Failed to reconnect to stream, exiting")
                break
            else:
                print("Reconnected to stream")
                continue

        results = model(frame, verbose=False)

        human_detected_this_frame = False
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id]
                confidence = float(box.conf[0])

                if label == 'person' and confidence >= CONFIDENCE_THRESHOLD:
                    human_detected_this_frame = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    details = f"Class: {label}, Confidence: {confidence:.2f} at ({x1},{y1})-({x2},{y2})"
                    trigger_notifications(details)
                    color = (0, 255, 0)
                    thickness = 2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
            if human_detected_this_frame:
                break

        cv2.imshow('Live Human Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

except KeyboardInterrupt:
    print("Stream stopped by user")
except Exception as e:
    print(f"An error occurred in the main loop: {e}")
finally:
    print("Releasing resources")
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()