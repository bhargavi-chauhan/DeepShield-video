import cv2
import torch
import numpy as np
import torch.nn.functional as F

# from models.cnn_lstm_model import CNN_LSTM
from models.cnn_transformer_model import CNN_Transformer

# ---------------- CONFIG ----------------
SEQ_LEN = 10
IMG_SIZE = 224
# MODEL_PATH = "models/deepshield_video_lstm.pth"
MODEL_PATH = "models/best_model.pth"

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ---------------- LOAD MODEL ----------------
model = CNN_Transformer().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 🔥 GLOBAL INFERENCE OPTIMIZATION
torch.set_grad_enabled(False)

# ---------------- FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ---------------- VIDEO STREAM ----------------
cap = cv2.VideoCapture(0)

buffers = {}      # per-face sequence buffer
histories = {}    # per-face smoothing history

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------- TITLE --------
    cv2.putText(frame, "DeepShield Live Detection",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for i, (x, y, w, h) in enumerate(faces):
        face_id = str(i)  # simple face ID

        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face / 255.0

        # -------- INIT BUFFERS --------
        if face_id not in buffers:
            buffers[face_id] = []
            histories[face_id] = []

        buffers[face_id].append(face)

        if len(buffers[face_id]) > SEQ_LEN:
            buffers[face_id].pop(0)

        # -------- ANALYZING STATE --------
        if len(buffers[face_id]) < SEQ_LEN:
            cv2.putText(frame, "Analyzing...",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)

            cv2.rectangle(frame, (x, y), (x+w, y+h),
                          (0, 255, 255), 2)
            continue

        # -------- PREDICTION --------
        seq = torch.from_numpy(np.array(buffers[face_id])).float()
        seq = seq.permute(0, 3, 1, 2)
        seq = seq.unsqueeze(0).to(device)

        output = model(seq)
        prob = F.softmax(output, dim=1)[0][1].item()

        # -------- SMOOTHING --------
        histories[face_id].append(prob)
        if len(histories[face_id]) > 5:
            histories[face_id].pop(0)

        prob = sum(histories[face_id]) / len(histories[face_id])

        # -------- TUNED THRESHOLDS --------
        if prob > 0.65:
            label = "FAKE"
            color = (0, 0, 255)

        elif prob > 0.45:
            label = "SUSPICIOUS"
            color = (0, 165, 255)

        else:
            label = "REAL"
            color = (0, 255, 0)

        # -------- DISPLAY --------
        cv2.putText(frame, f"{label} ({prob:.2f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    # -------- SHOW WINDOW --------
    cv2.imshow("DeepShield Real-Time", frame)

    # press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()

# buffer = []          # sequence buffer
# pred_history = []    # smoothing buffer

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # -------- TITLE --------
#     cv2.putText(frame, "DeepShield Live Detection",
#                 (20, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7, (255, 255, 255), 2)

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     # -------- MAIN FACE SELECTION --------
#     if len(faces) > 0:
#         (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])

#         face = frame[y:y+h, x:x+w]
#         face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
#         face = face / 255.0

#         buffer.append(face)

#         # keep only last SEQ_LEN frames
#         if len(buffer) > SEQ_LEN:
#             buffer.pop(0)

#         if len(buffer) < SEQ_LEN:
#             cv2.putText(frame, "Analyzing...",
#                         (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.8, (0, 255, 255), 2)

#         # -------- PREDICTION --------
#         if len(buffer) == SEQ_LEN:
#             seq = torch.from_numpy(np.array(buffer)).float()
#             seq = seq.permute(0, 3, 1, 2)
#             seq = seq.unsqueeze(0).to(device)

#             with torch.no_grad():
#                 output = model(seq)
#                 prob = F.softmax(output, dim=1)[0][1].item()

#             # -------- SMOOTHING --------
#             pred_history.append(prob)
#             if len(pred_history) > 5:
#                 pred_history.pop(0)

#             prob = sum(pred_history) / len(pred_history)

#             # label = "FAKE" if prob > 0.5 else "REAL"
#             if prob > 0.7:
#                 label = "FAKE (High Confidence)"
#             elif prob > 0.6:
#                 cv2.putText(frame, "⚠ Possible Manipulation",
#                             (20, 60),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             0.7, (0, 0, 255), 2)
#             elif prob > 0.5:
#                 label = "FAKE (Suspicious)"
#             else:
#                 label = "REAL"

#             # -------- DISPLAY --------
#             color = (0, 0, 255) if label == "FAKE" else (0, 255, 0)

#             cv2.putText(frame, f"{label} ({prob:.2f})",
#                         (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.8, color, 2)

#             cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

#     # print("Running...")

#     # -------- SHOW --------
#     cv2.imshow("DeepShield Real-Time", frame)

#     # press 'q' to quit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

