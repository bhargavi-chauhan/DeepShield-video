import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F

from models.cnn_lstm_model import CNN_LSTM
# from models.cnn_model import DeepfakeCNN  # your old CNN model

import argparse

# ---------------- CONFIG ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, required=True)

args = parser.parse_args()

VIDEO_PATH = args.video

# VIDEO_PATH = "test.mp4"
SEQ_LEN = 10
IMG_SIZE = 224
MODEL_PATH = "models/deepshield_video_lstm.pth"
# CNN_MODEL_PATH = "models/deepshield_image_cnn.pth"

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- LOAD MODELS ----------------
lstm_model = CNN_LSTM().to(device)
lstm_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
lstm_model.eval()

# cnn_model = DeepfakeCNN().to(device)
# cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
# cnn_model.eval()

# ---------------- FACE DETECTOR ----------------
# face_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# )

from preprocessing.face_detection import (
    extract_faces,
    select_main_face,
    extract_audio_score
)

# ---------------- FRAME EXTRACTION ----------------
# def extract_faces(video_path):
#     cap = cv2.VideoCapture(video_path)
#     faces = []
#     raw_frames = []
    
#     frame_count = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if frame_count % 5 == 0:  # smart sampling
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             detected = face_cascade.detectMultiScale(gray, 1.3, 5)

#             for (x, y, w, h) in detected:
#                 face = frame[y:y+h, x:x+w]
#                 face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
#                 face = face / 255.0

#                 faces.append(face)
#                 raw_frames.append(face)

#         frame_count += 1

#     cap.release()
#     return faces, raw_frames

# ---------------- TEMPORAL VARIATION ----------------
def temporal_variation(frames):
    diffs = []

    for i in range(len(frames) - 1):
        f1 = frames[i] / 255.0
        f2 = frames[i+1] / 255.0

        diff = np.mean(np.abs(f1 - f2))
        diffs.append(diff)

    return np.mean(diffs) if len(diffs) > 0 else 0

# ---------------- CREATE SEQUENCE ----------------
# def create_sequence(frames):
#     if len(frames) == 0:
#         frames = [np.zeros((IMG_SIZE, IMG_SIZE, 3))]

#     if len(frames) < SEQ_LEN:
#         frames += [frames[-1]] * (SEQ_LEN - len(frames))

#     seq = frames[:SEQ_LEN]

#     seq = torch.from_numpy(np.array(seq)).float()
#     seq = seq.permute(0, 3, 1, 2)  # (T, C, H, W)

#     return seq.unsqueeze(0)  # (1, T, C, H, W)

# ---------------- MAIN PIPELINE ----------------
# faces, raw_frames = extract_faces(VIDEO_PATH)
faces_per_frame, raw_frames = extract_faces(VIDEO_PATH)
faces = select_main_face(faces_per_frame)

if len(faces) == 0:
    print("No faces detected!")
    exit()

# -------- LSTM Prediction --------
frame_scores = []

for i in range(len(faces) - SEQ_LEN + 1):
    seq = faces[i:i + SEQ_LEN]

    seq = torch.from_numpy(np.array(seq)).float()
    seq = seq.permute(0, 3, 1, 2)
    seq = seq.unsqueeze(0).to(device)

    with torch.no_grad():
        output = lstm_model(seq)
        prob = F.softmax(output, dim=1)[0][1].item()

    frame_scores.append(prob)

# fallback if very few frames
if len(frame_scores) == 0:
    frame_scores = [0.0]

frame_scores = np.convolve(frame_scores, np.ones(3)/3, mode='same')  
# -------- Frame-level CNN Prediction --------
# frame_scores = []

# for face in faces:
#     img = torch.tensor(face, dtype=torch.float32).permute(2, 0, 1)
#     img = img.unsqueeze(0).to(device)

#     with torch.no_grad():
#         output = cnn_model(img)
#         prob = F.softmax(output, dim=1)[0][1].item()

#     frame_scores.append(prob)

# frame_avg = np.mean(frame_scores)

# -------- Temporal Variation --------
variation_score = temporal_variation(raw_frames)

# -------- FINAL HYBRID SCORE --------
# final_score = (
#     0.6 * lstm_prob +
#     0.2 * frame_avg +
#     0.2 * variation_score
# )

# Final score = average of all sequences
# final_score = np.mean(frame_scores)
video_score = np.mean(frame_scores)

audio_score = extract_audio_score(VIDEO_PATH)

if audio_score is not None:
    final_score = 0.8 * video_score + 0.2 * audio_score
else:
    # final_score = video_score
    final_score = 0.85 * video_score + 0.15 * variation_score

# -------- DECISION --------
label = "FAKE" if final_score > 0.5 else "REAL"

# -------- TIMELINE GRAPH --------
os.makedirs("outputs", exist_ok=True)



plt.figure()
plt.plot(frame_scores)
plt.title("Frame-wise Fake Probability")
plt.xlabel("Frame")
plt.ylabel("Confidence")
plt.savefig("outputs/timeline.png")

# -------- OUTPUT --------
# print("\n===== DeepShield Video Result =====")
# print(f"LSTM Score        : {final_score:.4f}")
# # print(f"Frame Avg Score   : {frame_avg:.4f}")
# print(f"Temporal Variation: {variation_score:.4f}")
# print(f"Final Score       : {final_score:.4f}")
# print(f"Prediction        : {label}")
# print("Timeline saved at : outputs/timeline.png")
print("\n===== DeepShield Video Result =====")
print(f"Video Score       : {video_score:.4f}")

if audio_score is not None:
    print(f"Audio Score       : {audio_score:.4f}")
else:
    print("Audio Score       : Not Available")

print(f"Temporal Variation: {variation_score:.4f}")
print(f"Final Score       : {final_score:.4f}")
print(f"Prediction        : {'FAKE' if final_score > 0.5 else 'REAL'}")