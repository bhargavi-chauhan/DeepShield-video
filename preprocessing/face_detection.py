import cv2
import numpy as np
# from moviepy.editor import mp

IMG_SIZE = 224

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ---------------- MULTI-FACE EXTRACTION ----------------
def extract_faces(video_path):
    cap = cv2.VideoCapture(video_path)
    faces_per_frame = []
    raw_frames = []
    
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 5 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = face_cascade.detectMultiScale(gray, 1.3, 5)

            frame_faces = []

            for (x, y, w, h) in detected:
                face = frame[y:y+h, x:x+w]
                face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                face = face / 255.0

                frame_faces.append(face)

            if len(frame_faces) > 0:
                faces_per_frame.append(frame_faces)
                raw_frames.append(frame)

        frame_count += 1

    cap.release()
    return faces_per_frame, raw_frames


# ---------------- MAIN FACE SELECTION ----------------
def select_main_face(faces_per_frame):
    selected_faces = []

    for faces in faces_per_frame:
        face = max(faces, key=lambda f: f.shape[0] * f.shape[1])
        selected_faces.append(face)

    return selected_faces


# ---------------- AUDIO FEATURE ----------------
def extract_audio_score(video_path):
    # try:
    #     clip = mp.VideoFileClip(video_path)

    #     if clip.audio is None:
    #         return None

    #     audio = clip.audio.to_soundarray()

    #     energy = np.mean(np.abs(audio))

    #     return min(energy, 1.0)

    # except:
        return None