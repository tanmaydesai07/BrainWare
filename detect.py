import cv2
from ultralytics import YOLO
from utils import eeg_signal_queue
import pyttsx3
import threading
from queue import Queue
import socket
import time

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)

# Get available voices and set a female voice if available
voices = engine.getProperty('voices')
female_voice = None
engine.setProperty('voice', 1)
for voice in voices:
    if 'female' in voice.name.lower():
        female_voice = voice
        break

if female_voice:
    engine.setProperty('voice', female_voice.id)
else:
    print("No female voice found. Using default voice.")

# Queue for handling speech
speech_queue = Queue()

# Function to handle text-to-speech in a separate thread
def process_speech():
    while True:
        text = speech_queue.get()  # Wait until an item is available in the queue
        if text is None:  # Stop signal
            break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

# Start the TTS thread
speech_thread = threading.Thread(target=process_speech, daemon=True)
speech_thread.start()

# Setup UDP socket to send messages to ESP32
ESP32_IP = '192.168.111.40'  # Replace with your ESP32 IP address
ESP32_PORT = 12345
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Load the YOLOv8 model
model = YOLO("runs/detect/train7/weights/best.pt")  # Replace with your trained model path

# Initialize last detected object and timer
last_detected_object = None
last_action_time = time.time()

try:
    # Start video capture from ESP32 stream
    esp32_stream_url = "http://192.168.111.52"  # URL of the ESP32 camera
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError(f"Could not connect to ESP32 stream at {esp32_stream_url}")

    # Get frame dimensions for center calculation
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read initial frame from webcam.")
    frame_height, frame_width, _ = frame.shape
    camera_center_x, camera_center_y = frame_width // 2, frame_height // 2

    print("Camera opened successfully.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Run YOLO detection
        results = model(frame, conf=0.5)

        # Draw center dot
        cv2.circle(frame, (camera_center_x, camera_center_y), 5, (0, 0, 255), -1)

        selected_object = None
        highest_confidence = 0

        for result in results[0].boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            # Draw bounding box and label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = results[0].names[int(class_id)]
            confidence = f"{score:.2f}"
            cv2.putText(frame, f"{label} {confidence}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Check if the center dot is within the bounding box
            if (camera_center_x >= int(x1) and camera_center_x <= int(x2) and
                camera_center_y >= int(y1) and camera_center_y <= int(y2) and score > highest_confidence):
                highest_confidence = score
                selected_object = label

        # If an object is selected, process only after 3 seconds
        current_time = time.time()
        if selected_object and current_time - last_action_time >= 3:
            object_name = selected_object.split('_')[0]
            if object_name != last_detected_object:
                # Send UDP message to ESP32
                sock.sendto(object_name.encode(), (ESP32_IP, ESP32_PORT))
                print(f"Sent message: {object_name} to ESP32")

                # Add TTS to the queue
                if speech_queue.empty():  # Avoid overlapping speech
                    speech_queue.put(f"{object_name} Selected")
                last_detected_object = object_name
                last_action_time = current_time  # Reset the timer
        elif not selected_object and last_detected_object != "no object detected":
            if current_time - last_action_time >= 3:
                print("No object detected.")
                sock.sendto("no object detected".encode(), (ESP32_IP, ESP32_PORT))
                last_detected_object = "no object detected"
                last_action_time = current_time  # Reset the timer

        # Display frame
        frame = cv2.resize(frame, (800, 600))
        cv2.imshow('Live Object Detection', frame)

        # Exit on 'q' key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Closing window...")
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    print("Releasing camera and closing windows...")
    cap.release()
    cv2.destroyAllWindows()
    speech_queue.put(None)  # Signal TTS thread to stop
    speech_thread.join()  # Wait for TTS thread to finish
    sock.close()  # Close the UDP socket
    print("Camera released, windows closed, and UDP socket closed.")
