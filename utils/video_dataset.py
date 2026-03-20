import os
import cv2
import torch
from torch.utils.data import Dataset
import random
import numpy as np

SEQ_LEN = 6
IMG_SIZE = 224

class VideoDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        
        for label, category in enumerate(["real", "fake"]):
            folder = os.path.join(root_dir, category)
            
            for video in os.listdir(folder):
                self.samples.append((os.path.join(folder, video), label))
    
    def __len__(self):
        return len(self.samples)
    
    def load_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % 10 == 0:
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                frame = frame / 255.0
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def get_sequence(self, frames):
        # handle empty case
        if len(frames) == 0:
            frames = [np.zeros((IMG_SIZE, IMG_SIZE, 3))]

        if len(frames) < SEQ_LEN:
            frames += [frames[-1]] * (SEQ_LEN - len(frames))
        
        start = random.randint(0, len(frames) - SEQ_LEN)
        seq = frames[start:start + SEQ_LEN]

        seq = np.array(seq)
        seq = torch.from_numpy(seq).float()

        return seq.permute(0, 3, 1, 2)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self.load_frames(video_path)
        
        sequences = []
        
        for _ in range(3):
            seq = self.get_sequence(frames)
            sequences.append(seq)
        
        sequences = torch.stack(sequences)
        
        return sequences, torch.tensor(label)