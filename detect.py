import time
import os
import cv2
import numpy as np
from torch.xpu import device
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import requests
import base64
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
from flask_socketio import SocketIO
import psycopg2
import threading
import queue
import difflib  # Add difflib for province correction

from sidewalk import get_sidewalk_coords

# Your API key for the Google Vision API
api_key = "AIzaSyBgqC-PH1Z-1vxJw3K_sRQFtHIctb3ndWM"

# Directory to serve images from
IMAGE_DIR = '../images'

# Set up Flask app and socket for real-time communication
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# OCR Queue
ocr_queue = queue.Queue()

# List of known correct provinces
known_provinces = [
    "กรุงเทพมหานคร", "เชียงใหม่", "เชียงราย", "ขอนแก่น", "นครราชสีมา",
    "ชลบุรี", "ภูเก็ต", "สงขลา", "สุราษฎร์ธานี", "นครศรีธรรมราช"
]

# Function to correct the province name based on similarity
def correct_province(detected_province):
    closest_match = difflib.get_close_matches(detected_province, known_provinces, n=1, cutoff=0.6)
    if closest_match:
        return closest_match[0]  # Return the closest match
    return detected_province  # If no close match is found, return the original

# Function to connect to PostgreSQL database
def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname="license-plate",
            user="postgres",
            password="danunai",
            host="localhost",
            port="5432"
        )
        return conn
    except psycopg2.Error as e:
        print(f"Database connection error: {e}")
        return None

# Insert plate information into PostgreSQL
def insert_plate(f_image, p_image, p_text, province, date=None):
    conn = get_db_connection()
    if conn is None:
        return False

    cursor = conn.cursor()
    try:
        # Insert into the PostgreSQL table (adjusted for your table's schema)
        cursor.execute(
            'INSERT INTO "plateDetection" (f_image, p_image, p_text, province, date) VALUES (%s, %s, %s, %s, to_char(NOW(), \'YYYY-MM-DD"T"HH24:MI:SS"Z"\'));',
            (f_image, p_image, p_text, province)
        )
        conn.commit()

        # Emit new data over WebSocket
        socketio.emit('new_plate', {
            "f_image": f_image,
            "p_image": p_image,
            "p_text": p_text,
            "province": province,
            "date": date
        })

        cursor.close()
        conn.close()
        return True
    except psycopg2.Error as e:
        print(f"Error inserting data: {e}")
        cursor.close()
        conn.close()
        return False

# Function to perform OCR using Google Vision API via HTTP request
def perform_ocr(image_path):
    try:
        with open(image_path, 'rb') as image_file:
            content = image_file.read()
    except IOError as e:
        print(f"Error opening image file: {e}")
        return []

    image_base64 = base64.b64encode(content).decode('utf-8')

    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    body = {
        "requests": [
            {
                "image": {"content": image_base64},
                "features": [{"type": "TEXT_DETECTION"}]
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()
        response_data = response.json()
    except requests.RequestException as e:
        print(f"HTTP request error: {e}")
        return []

    detected_texts = []
    if 'responses' in response_data and len(response_data['responses']) > 0:
        texts = response_data['responses'][0].get('textAnnotations', [])
        for text in texts:
            detected_texts.append(text['description'])

    return detected_texts

# OCR worker thread that continuously processes items from the queue
def ocr_worker():
    while True:
        try:
            plate_image_path, full_image_path = ocr_queue.get()  # Get item from queue
            if plate_image_path is None:
                break  # If None is sent, exit thread

            # List of Thai vowel characters (both standalone and combining)
            thai_vowels = ['ะ', 'า', 'ิ', 'ี', 'ึ', 'ื', 'ุ', 'ู', 'เ', 'แ', 'โ', 'ใ', 'ไ', '็', '่', '้', '๊', '๋']

            # Function to remove vowels from a given string
            def remove_thai_vowels(text):
                return ''.join([char for char in text if char not in thai_vowels])

            detected_texts = perform_ocr(plate_image_path)
            if detected_texts:
                lines = "\n".join(detected_texts).split('\n')
                if len(lines) >= 3:
                    license_text = f"{remove_thai_vowels(lines[0])}{lines[2]}"
                    province = correct_province(lines[1])  # Correct the detected province
                else:
                    license_text = "".join(detected_texts)
                    province = ""

                print(f"OCR detected: {license_text}, corrected province: {province}")

                # Insert into the database
                success = insert_plate(full_image_path, plate_image_path, license_text, province)
                if success:
                    print(f"Plate {license_text} successfully inserted.")
                else:
                    print(f"Failed to insert plate {license_text}.")
        finally:
            ocr_queue.task_done()  # Mark task as done

# Start OCR worker thread
ocr_thread = threading.Thread(target=ocr_worker)
ocr_thread.daemon = True  # This ensures the thread closes when the main program exits
ocr_thread.start()

# Load models only once
plate_model = YOLO("./model/platedetect.pt")
bike_model = YOLO("./model/yolov8m.pt")

# Open the video file
cap = cv2.VideoCapture('./video/2k3.mov')
window_title = 'Plate Detection CSPJ V.12 070924T1'

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

print('Success: Opened video.')
ret, frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
new_width = 720
new_height = int(new_width * original_height / original_width)

cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_title, new_width, new_height)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('video/playback/output.avi', fourcc, fps, (original_width, original_height))

line_y1 = 400
line_y2 = 1400
plate_zone_top_line = 700
plate_zone_bottom_line = 1100
thickness = 7

crop_frame = frame[line_y1:line_y2, :]
cv2.imwrite('captured_frame.png', crop_frame)
print("Frame captured and saved.")
start_coords, end_coords = get_sidewalk_coords('captured_frame.png', show_plot=True)

# Adjust the coordinates to match the original frame
start_coords = (start_coords[0], start_coords[1] + line_y1)
end_coords = (end_coords[0], end_coords[1] + line_y1)

desired_side = "left"
mid_x = (start_coords[0] + end_coords[0]) // 2
mid_y = (start_coords[1] + end_coords[1]) // 2

license_output_folder = "images/license_plates"
full_image_folder = "images/full_images"
os.makedirs(license_output_folder, exist_ok=True)
os.makedirs(full_image_folder, exist_ok=True)
plate_counter = 0
motorcycle_count = 0

font_path = "fonts/Kanit-Regular.ttf"
font_size = 80
font = ImageFont.truetype(font_path, font_size)

def draw_text_with_border(draw, text, position, font, border_color, text_color):
    x, y = position
    offsets = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
    for offset in offsets:
        draw.text((x + offset[0], y + offset[1]), text, font=font, fill=border_color)
    draw.text(position, text, font=font, fill=text_color)

def is_plate_between_lines(y1, y2, line_top, line_bottom):
    return y1 > line_top and y2 < line_bottom

capture_delay = 9  # Set the delay in seconds for plate image
full_capture_delay = 1  # Set the delay in seconds for full motorcycle image after plate detection
last_capture_time_motorcycle = 0
last_capture_time_plate = 0
frame_skip_rate = 2  # Change this value as needed
frame_counter = 0

# Main video loop with motorcycle and plate detection
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    if frame_counter % 2 != 0:
        continue  # Skip every 2nd frame for performance reasons

    # Detect motorcycles using the bike model
    bike_results = bike_model(frame)

    for bike_result in bike_results:
        for bike_box in bike_result.boxes:
            bike_conf = bike_box.conf[0]
            x1, y1, x2, y2 = map(int, bike_box.xyxy[0])
            if bike_conf > 0.7:
                bike_label = bike_model.names[int(bike_box.cls[0])]
                if bike_label == 'motorcycle':
                    bike_mid_x = (x1 + x2) // 2
                    bike_side = "left" if bike_mid_x < mid_x else "right"
                    if bike_side == desired_side:
                        color_bike = (255, 0, 0)

                        # Detect plates within the motorcycle bounding box
                        plate_results = plate_model(frame[y1:y2, x1:x2])
                        for plate_result in plate_results:
                            for plate_box in plate_result.boxes:
                                px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
                                plate_conf = plate_box.conf[0]
                                if plate_conf >= 0.9:
                                    current_time = time.time()
                                    # Save plate image if condition met
                                    if (current_time - last_capture_time_plate) >= capture_delay:
                                        plate_img = frame[y1 + py1:y1 + py2, x1 + px1:x1 + px2]
                                        if is_plate_between_lines(y1 + py1, y1 + py2, plate_zone_top_line, plate_zone_bottom_line):
                                            plate_counter += 1
                                            plate_image_path = f'{license_output_folder}/plate_{plate_counter}.png'
                                            cv2.imwrite(plate_image_path, plate_img)
                                            print(f"Captured plate {plate_counter}")
                                            last_capture_time_plate = current_time

                                            # Save full motorcycle image
                                            if (current_time - last_capture_time_motorcycle) >= full_capture_delay:
                                                motorcycle_count += 1
                                                full_image_path = f'{full_image_folder}/full_{motorcycle_count}.png'
                                                cv2.imwrite(full_image_path, frame)
                                                print(f"Full motorcycle image {motorcycle_count} captured.")
                                                last_capture_time_motorcycle = current_time

                                                # Queue the plate and full image for OCR processing
                                                ocr_queue.put((plate_image_path, full_image_path))

    # Draw lines and rectangles on the frame
    color_line = (0, 255, 0)
    cv2.line(frame, (0, plate_zone_top_line), (frame.shape[1], plate_zone_top_line), color_line, thickness)
    cv2.line(frame, (0, plate_zone_bottom_line), (frame.shape[1], plate_zone_bottom_line), color_line, thickness)

    out.write(frame)
    cv2.imshow(window_title, frame)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Send None to stop the OCR worker thread
ocr_queue.put(None)
ocr_thread.join()
