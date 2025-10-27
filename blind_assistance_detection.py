"""
Blind Assistance Object Detection System
----------------------------------------
This project uses YOLOv8 for real-time object detection with voice alerts.
It is designed to assist visually impaired individuals by identifying nearby
objects and announcing them via speech.

Author: S. Percy Deborah
Model: YOLOv8 (Ultralytics)
Date: October 2025
"""
from ultralytics import YOLO
import cv2
from gtts import gTTS
import pygame
import tempfile
import os
import time

# Initialize YOLO model
model = YOLO("yolov8n.pt")
pygame.mixer.init()

# Define alert-worthy objects
warning_objects = ['person', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'chair', 'bench', 'table']

last_alert_time = 0

print("🚶 Blind Assistance Detection System Started...")
print("Press 'q' to exit.")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)
    detected_objects = [r.names[int(cls)] for r in results for cls in r.boxes.cls]

    if detected_objects:
        warnings = [obj for obj in detected_objects if obj in warning_objects]

        if warnings and time.time() - last_alert_time > 3:
            sentence = "Warning! " + ", ".join(warnings) + " detected ahead."
            print("🔊", sentence)

            # Save TTS audio safely
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                temp_path = fp.name
                tts = gTTS(sentence)
                tts.save(temp_path)

            # Wait briefly to ensure the file is ready
            time.sleep(0.3)

            # Verify file exists before loading
            if os.path.exists(temp_path):
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                last_alert_time = time.time()
            else:
                print("⚠️ Warning: Audio file not found at", temp_path)

    # Display annotated frame
    annotated = results[0].plot()
    cv2.imshow("Blind Assistance Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
print("🛑 System stopped.")
