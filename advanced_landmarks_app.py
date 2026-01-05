import cv2
import mediapipe as mp
import numpy as np
import time
import math

# Optional system control
######################### CURSOR #########################
try:
    import pyautogui
    HAS_PYAUTOGUI = True
except:
    HAS_PYAUTOGUI = False


######################### VOLUME #########################
HAS_PYCAW = False
try:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    devices = AudioUtilities.GetSpeakers()

    try:
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        vol_min, vol_max, _ = volume.GetVolumeRange()
        HAS_PYCAW = True
        print("Volume control enabled.")
    except Exception as e:
        print("Volume control not supported on this Python/Windows version.")
        print("Reason:", e)

except Exception as e:
    print("Pycaw not installed or unavailable:", e)
    HAS_PYCAW = False


######################### MEDIAPIPE MODELS #########################
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

face_model_path = "models/face_landmarker.task"
hand_model_path = "models/hand_landmarker.task"

face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=face_model_path),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=hand_model_path),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)

face_detector = vision.FaceLandmarker.create_from_options(face_options)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)


######################### HELPERS #########################
def get_emotion(landmarks, img):
    if len(landmarks) < 292:
        return "Neutral"

    h, w, _ = img.shape
    left = landmarks[61]
    right = landmarks[291]
    upper = landmarks[13]
    lower = landmarks[14]

    left_xy = np.array([left.x * w, left.y * h])
    right_xy = np.array([right.x * w, right.y * h])
    upper_xy = np.array([upper.x * w, upper.y * h])
    lower_xy = np.array([lower.x * w, lower.y * h])

    mouth_width = np.linalg.norm(right_xy - left_xy)
    mouth_open = np.linalg.norm(lower_xy - upper_xy)

    if mouth_width == 0:
        return "Neutral"

    ratio = mouth_open / mouth_width

    if ratio > 0.45:
        return "Surprised"
    elif ratio > 0.30:
        return "Happy"
    return "Neutral"


def classify_gesture(hand):
    tips = [4, 8, 12, 16, 20]
    pip = [3, 6, 10, 14, 18]

    fingers = []

    for t, p in zip(tips[1:], pip[1:]):
        fingers.append(hand[t].y < hand[p].y)

    thumb_open = abs(hand[4].x - hand[3].x) > 0.03

    count = sum(fingers) + (1 if thumb_open else 0)

    if count == 0:
        return "Fist"
    if count == 5:
        return "Open Palm"
    if fingers[1] and not fingers[0]:
        return "Thumbs Up"

    return "Unknown"


def control_volume(hand, img):
    if not HAS_PYCAW:
        return None

    h, w, _ = img.shape
    p1 = np.array([hand[4].x * w, hand[4].y * h])
    p2 = np.array([hand[8].x * w, hand[8].y * h])

    dist = np.linalg.norm(p1 - p2)

    min_dist = 20
    max_dist = 200
    dist = np.clip(dist, min_dist, max_dist)

    norm = (dist - min_dist) / (max_dist - min_dist)

    vol_level = vol_min + norm * (vol_max - vol_min)
    volume.SetMasterVolumeLevel(vol_level, None)

    return int(norm * 100)


def control_cursor(hand):
    if not HAS_PYAUTOGUI:
        return None

    screen_w, screen_h = pyautogui.size()
    x = hand[8].x * screen_w
    y = hand[8].y * screen_h
    pyautogui.moveTo(int(x), int(y), duration=0.01)


######################### MAIN #########################
cap = cv2.VideoCapture(0)

recording = False
writer = None

show_face = True
show_hands = True
show_emotion = True
show_gesture = True
enable_volume = False
enable_cursor = False


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    ts = int(time.time() * 1000)

    ########### FACE ###########
    face_result = face_detector.detect_for_video(mp_image, ts)

    if face_result.face_landmarks:
        landmarks = face_result.face_landmarks[0]

        if show_face:
            for p in landmarks:
                cv2.circle(frame, (int(p.x * w), int(p.y * h)), 1, (0, 255, 0), -1)

        if show_emotion:
            emotion = get_emotion(landmarks, frame)
            cv2.putText(frame, f"Emotion: {emotion}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    ########### HAND ###########
    hand_result = hand_detector.detect_for_video(mp_image, ts)

    if hand_result.hand_landmarks:
        for hand in hand_result.hand_landmarks:
            if show_hands:
                for p in hand:
                    cv2.circle(frame, (int(p.x * w), int(p.y * h)), 3, (0, 255, 255), -1)

            if show_gesture:
                g = classify_gesture(hand)
                cv2.putText(frame, f"Gesture: {g}", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if enable_volume and HAS_PYCAW:
                vol = control_volume(hand, frame)
                if vol:
                    cv2.putText(frame, f"Volume: {vol}%", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if enable_cursor:
                control_cursor(hand)

    ########### UI TEXT ###########
    cv2.putText(frame, "Face + Hand AI App | Press H for help", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("AI Landmark App (Python 3.12)", frame)

    ########### KEYS ###########
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("f"):
        show_face = not show_face
    elif key == ord("g"):
        show_hands = not show_hands
    elif key == ord("j"):
        show_emotion = not show_emotion
    elif key == ord("e"):
        show_gesture = not show_gesture
    elif key == ord("v"):
        if HAS_PYCAW:
            enable_volume = not enable_volume
            print("Volume control:", enable_volume)
        else:
            print("Volume Control NOT available on this setup.")
    elif key == ord("c"):
        enable_cursor = not enable_cursor
        print("Cursor:", enable_cursor)
    elif key == ord("h"):
        print("""
Controls:
Q Quit
F Toggle Face
G Toggle Hands
J Toggle Emotion
E Toggle Gesture Labels
V Volume Control (if supported)
C Cursor Control
        """)

cap.release()
cv2.destroyAllWindows()
