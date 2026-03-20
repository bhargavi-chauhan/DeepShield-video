import cv2
import os

def extract_frames(video_path, output_folder, frame_rate=5):
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % frame_rate == 0:
            cv2.imwrite(f"{output_folder}/frame_{saved}.jpg", frame)
            saved += 1
        
        count += 1
    
    cap.release()
    print(f"Saved {saved} frames")

# Example
extract_frames("sample.mp4", "frames/")