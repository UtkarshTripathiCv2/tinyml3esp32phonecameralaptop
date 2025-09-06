import cv2
from ultralytics import YOLO
import requests
import time

# --- 1. SETUP FOR YOUR PHONE'S CAMERA ---
# IMPORTANT: Replace this with the URL from the IP Webcam app on your phone.
# Make sure to add "/video" at the end.
PHONE_CAMERA_URL = "http://[2402:8100:bc01]:8080/video" # <-- PASTE YOUR PHONE'S URL HERE

# --- 2. SETUP FOR YOUR ESP32 ---
# IMPORTANT: Replace this with the IP address of your ESP32.
ESP32_IP_ADDRESS = "1054"
blink_url = f"http://{ESP32_IP_ADDRESS}/blink"
# ----------------------------------------

# Load the YOLOv8n model
model = YOLO('yolov8n.pt')
class_names = model.names

# Initialize video capture from your phone's camera URL
video_capture = cv2.VideoCapture(PHONE_CAMERA_URL)
print("Connecting to phone's camera stream...")

# Check if the camera stream opened successfully
if not video_capture.isOpened():
    print("❌ Error: Could not open video stream from phone.")
    print("Please check the URL and ensure your phone and computer are on the same Wi-Fi.")
else:
    print("✅ Camera stream connected successfully!")
    print("Starting YOLOv8 detection... Press 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        # If a frame was not received correctly, the connection might be unstable
        if not ret or frame is None:
            print("Warning: Dropped or empty frame. Check Wi-Fi connection.")
            time.sleep(0.5) # Wait a bit before trying again
            continue

        # Run YOLOv8 inference on the frame from your phone
        results = model(frame)
        person_detected = False

        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                if class_names[class_id] == 'person':
                    person_detected = True
                    break
            if person_detected:
                break

        # If a person was detected, send a web request to the ESP32
        if person_detected:
            print("Person Detected!  *Sending command wirelessly to blink LED*")
            try:
                requests.get(blink_url, timeout=1)
            except requests.exceptions.RequestException as e:
                print(f"Could not connect to ESP32: {e}")

        # Display the resulting frame
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Live Detection (Phone Feed) - Press q to quit", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
print("Webcam turned off.")
