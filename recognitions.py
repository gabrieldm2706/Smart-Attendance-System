from pathlib import Path
import time

import cv2
import numpy as np
import requests

from app.database import get_connection

import unicodedata

def ascii_name(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()

from app.services.face_capture import (
    create_detector,
    detect_faces,
    normalize_face,
)
from thermal_reader import ThermalReader

# ---------------- Project settings ----------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"

LBPH_THRESHOLD = 70.0
CAMERA_INDEX = 0
ATTENDANCE_API_URL = "http://127.0.0.1:8000/attendance/check-in"

REQUIRED_STABLE_FRAMES = 5
CHECKIN_COOLDOWN_SECONDS = 86400
RETRY_SECONDS = 5                 # wait before retrying a failed API call

# ---------------- Thermal settings ----------------
ENABLE_THERMAL = True
THERMAL_PORT = "COM3"
REQUIRE_TEMPERATURE = True        # no check-in until a temperature was measured
MIN_TEMP_READINGS = 3             # readings collected before check-in
FEVER_THRESHOLD = 37.5            # C, after TEMP_OFFSET correction

THERMAL_MIRROR = False            # adjust until the box on the thermal panel
THERMAL_FLIP = False              # sits on the forehead of the same person
THERMAL_SCALE = 1.0               # webcam FOV / thermal FOV (e.g. 70/55 = 1.27)
THERMAL_DX = 0.0                  # shift left/right  (fraction of frame)
THERMAL_DY = 0.0                  # shift up/down
TEMP_OFFSET = 0.0                 # calibrate against a real thermometer

import os
from dotenv import load_dotenv

load_dotenv()

KIOSK_API_KEY = os.getenv("KIOSK_API_KEY", "")

# ---------------- Per-student state ----------------
stable_frames = {}                # student_number -> consecutive frames seen
temp_readings = {}                # student_number -> list of temperatures
last_checkin_time = {}            # only set after a SUCCESSFUL check-in
last_attempt_time = {}
attendance_messages = {}


def load_face_dataset():
    """Load enrolled face images and assign one label per student."""
    images, labels, label_to_student = [], [], {}

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

    student_folders = sorted(f for f in DATASET_DIR.iterdir() if f.is_dir())

    for label, student_folder in enumerate(student_folders):
        student_number = student_folder.name
        student_images = []

        for image_path in sorted(student_folder.glob("*.png")):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Skipping unreadable image: {image_path}")
                continue
            if image.shape != (200, 200):
                print(f"Skipping incorrect image size: {image_path}")
                continue
            student_images.append(image)

        if not student_images:
            print(f"No usable images for {student_number}")
            continue

        images.extend(student_images)
        labels.extend([label] * len(student_images))
        label_to_student[label] = student_number
        print(f"Loaded {len(student_images)} images for student {student_number}")

    if not images:
        raise RuntimeError("No valid face images found. "
                           "Check the dataset folders and PNG files.")

    return images, np.asarray(labels, dtype=np.int32), label_to_student

def load_student_names():
    """Returns {student_number: full name} straight from the database."""
    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT student_number,
                   CONCAT(first_name, ' ', last_name) AS student_name
            FROM students
            WHERE is_active = 1
        """)
        return {
            row["student_number"]: row["student_name"]
            for row in cursor.fetchall()
        }
    except Exception as error:
        print("Could not load student names:", repr(error))
        return {}
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

def submit_attendance(student_number, temperature=None):
    """Returns (message, success)."""
    payload = {"student_number": student_number}
    if temperature is not None:
        payload["temperature"] = temperature

    try:
        # response = requests.post(ATTENDANCE_API_URL, json=payload, timeout=5)
        response = requests.post(
            ATTENDANCE_API_URL,
            json=payload,
            headers={"X-API-Key": KIOSK_API_KEY},
            timeout=5,
        )
        try:
            result = response.json()
        except ValueError:
            return f"API error: HTTP {response.status_code}", False

        if response.status_code == 200:
            status = result.get("status", "RECORDED")
            return f"{status}: {result.get('message', 'Check-in successful')}", True

        detail = result.get("detail", "Check-in failed")
        return f"HTTP {response.status_code}: {detail}", False

    except requests.RequestException as error:
        print("Attendance API connection error:", error)
        return "API CONNECTION ERROR", False


def main():
    print("Loading enrolled face samples...")
    images, labels, label_to_student = load_face_dataset()

    print("\nTraining LBPH in memory...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, labels)
    print("LBPH model ready.")
    student_names = load_student_names()
    print(f"Loaded {len(student_names)} student name(s).")
    print("\nLoading YuNet face detector...")
    detector = create_detector()

    thermal = None
    if ENABLE_THERMAL:
        print(f"Starting thermal camera on {THERMAL_PORT}...")
        thermal = ThermalReader(
            THERMAL_PORT,
            mirror=THERMAL_MIRROR, flip=THERMAL_FLIP,
            scale=THERMAL_SCALE, dx=THERMAL_DX, dy=THERMAL_DY,
            temp_offset=TEMP_OFFSET,
        ).start()

    print("Opening webcam...")
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam. Check CAMERA_INDEX "
                           "or close other apps using the camera.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Recognition running. Press Q to quit.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read a frame from the webcam.")
                break

            fh, fw = frame.shape[:2]
            faces = detect_faces(detector, frame)
            recognized_this_frame = set()

            panel_boxes = []      # face boxes to draw on the thermal panel

            if faces is not None:
                for face in faces:
                    x, y, w, h = map(int, face[:4])
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, fw - x)
                    h = min(h, fh - y)

                    face_image = normalize_face(frame, face)
                    if face_image is None:
                        continue

                    label, distance = recognizer.predict(face_image)
                    student_number = None
                    info_text = ""

                    if label in label_to_student and distance <= LBPH_THRESHOLD:
                        # student_number = label_to_student[label]
                        # recognized_this_frame.add(student_number)
                        # box_color = (0, 200, 0)
                        # label_text = f"{student_number} (distance: {distance:.1f})"


                        student_number = label_to_student[label]
                        recognized_this_frame.add(student_number)
                        box_color = (0, 200, 0)
                        display_name = ascii_name(student_names.get(student_number, student_number))
                        label_text = f"{display_name} ({distance:.1f})"

                        stable_frames[student_number] = stable_frames.get(student_number, 0) + 1

                        # --- thermal reading for this face ---
                        if thermal:
                            temp = thermal.face_temperature((x, y, w, h), fw, fh)
                            if temp is not None:
                                temp_readings.setdefault(student_number, []).append(temp)
                                info_text = f"{temp:.1f} C"
                                if temp >= FEVER_THRESHOLD:
                                    info_text += "  HIGH TEMP"
                                    box_color = (0, 140, 255)
                            else:
                                info_text = "No thermal reading"

                        # --- check-in decision ---
                        now = time.monotonic()
                        last_ok = last_checkin_time.get(student_number)
                        cooled_down = last_ok is None or now - last_ok >= CHECKIN_COOLDOWN_SECONDS
                        last_try = last_attempt_time.get(student_number)
                        can_retry = last_try is None or now - last_try >= RETRY_SECONDS
                        readings = temp_readings.get(student_number, [])

                        if (stable_frames[student_number] >= REQUIRED_STABLE_FRAMES
                                and cooled_down and can_retry):
                            if thermal and REQUIRE_TEMPERATURE and len(readings) < MIN_TEMP_READINGS:
                                info_text = "Measuring temperature..."
                            else:
                                temperature = (round(float(np.median(readings)), 1)
                                               if readings else None)
                                message, ok = submit_attendance(student_number, temperature)
                                last_attempt_time[student_number] = now
                                if ok:
                                    last_checkin_time[student_number] = now
                                if temperature is not None:
                                    message += f" ({temperature} C)"
                                attendance_messages[student_number] = message
                                # print(f"{student_number}: {message}")
                                print(f"{student_names.get(student_number, student_number)}: {message}")
                    else:
                        label_text = f"UNKNOWN ({distance:.1f})"
                        box_color = (0, 0, 255)

                    panel_boxes.append(((x, y, w, h), box_color, info_text))

                    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                    cv2.putText(frame, label_text, (x, max(25, y - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                    if info_text:
                        cv2.putText(frame, info_text, (x, max(45, y - 32)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                    if student_number in attendance_messages:
                        cv2.putText(frame, attendance_messages[student_number][:55],
                                    (x, min(fh - 10, y + h + 25)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1)

            # Reset counters for students who left the frame
            for s in list(stable_frames):
                if s not in recognized_this_frame:
                    stable_frames.pop(s, None)
                    temp_readings.pop(s, None)

            # Webcam on the left, thermal camera on the right (same size)
            if thermal:
                panel = thermal.render_panel((fw, fh))
                for box, color, text in panel_boxes:
                    thermal.draw_face_box(panel, box, fw, fh, color, text)
                display = np.hstack([frame, panel])
            else:
                display = frame

            cv2.imshow("Smart Attendance - Recognition + Thermal", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        if thermal:
            thermal.stop()
        print("Camera closed.")


if __name__ == "__main__":
    main()
