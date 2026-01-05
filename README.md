🧠 Face + Hand Landmark AI — Real Time MediaPipe + OpenCV Project

This project is an advanced real time Computer Vision system built using the latest MediaPipe Tasks API and OpenCV.
It detects Face Mesh + Hand Landmarks, recognizes gestures, estimates emotion, supports cursor control, and works perfectly on Python 3.12.

This project goes beyond basic tracking and demonstrates interactive AI + Human Input Systems, making it highly valuable for learning, portfolio, and recruiters.

🚀 Features
🧑‍🤝‍🧑 Face AI
✔ Face Mesh Landmark Detection
✔ Real-time tracking
✔ Emotion Estimation
(Happy / Surprised / Neutral)

✋ Hand AI
✔ Hand Landmark Detection
✔ Gesture Recognition
Fist
Open Palm
Thumbs Up
✔ Cursor Control (Move mouse using hand)

🖥 System Interaction
✔ Cursor Control via Hand (PyAutoGUI)
✔ Volume Control (auto disabled if unsupported — no crash)

⚙ Technical
✔ Works on Python 3.12
✔ Uses MediaPipe Tasks API (replaces removed mp.solutions)
✔ Lightweight and Fast
✔ Real-time webcam input
✔ Works even if some features fail gracefully

📸 Demo Expectations

When you run the app:

🟢 You should see:
Face mesh dots on your face
Yellow dots on your hand
Emotion text on top-left
Gesture label on screen

🟡 Optional:

Cursor starts moving with your hand if enabled
Volume control may work depending on OS + Python

▶️ How To Run The Project
1️⃣ Install Dependencies
pip install mediapipe==0.10.31 opencv-python numpy pyautogui pycaw comtypes

2️⃣ Download Required Models

Create folder:

models/
Download and place these inside:

Face Model
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
Save as:
models/face_landmarker.task

Hand Model
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
Save as:
models/hand_landmarker.task

3️⃣ Run
python advanced_landmarks_app.py

🎮 Controls During Demo
Key	Action
F	Toggle Face Mesh
G	Toggle Hand Tracking
J	Toggle Emotion Detection
E	Toggle Gesture Labels
C	Enable Cursor Control
V	Enable Volume Control (if supported)
H	Show help in console
Q	Quit

🧪 What to Show in Demo
1️⃣ Face Demo
Look at camera
✔ Face mesh appears
✔ Emotion text updates when smiling or opening mouth

2️⃣ Hand Demo
Show your right or left hand
✔ Yellow landmark dots appear
✔ Gesture text appears
Try:
Closed fist → Fist
Open hand → Open Palm
Thumbs up → Thumbs Up

3️⃣ Cursor Demo

Press:
C
Move your index finger slowly
✔ Mouse will follow hand

4️⃣ Volume Demo (If Supported)

Press:
V
Pinch thumb + index
Volume changes

If not supported, terminal prints:

Volume Control NOT available on this setup
No crash 👍

❗️ Notes

MediaPipe removed mp.solutions in new versions
This project uses MediaPipe Tasks API
Works with Python 3.12+
If PyCAW fails, volume automatically disables without crashing
Works on Windows, macOS, Linux (cursor only where supported)

🛠 Tech Stack
Python 3.12
MediaPipe 0.10.31
OpenCV
NumPy
PyAutoGUI
PyCAW (optional)

⭐ Why This Project Is Valuable

This is not just a basic demo. It demonstrates:

Realtime AI Processing
Human Gesture Interaction
Modern MediaPipe Tasks API
System Interaction via AI
Practical CV + AI Integration


📬 Contribution & Support

If you like this project:
⭐ Star the repo
🖊 Improve features
🐛 Report issues

Happy Building 👨‍💻🚀
