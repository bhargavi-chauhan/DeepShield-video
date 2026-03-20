import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

from utils.video_dataset import VideoDataset
# from models.cnn_lstm_model import CNN_LSTM
from models.cnn_transformer_model import CNN_Transformer

def main():

    # =========================================
    # ARGPARSE
    # =========================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    DATASET_PATH = args.data_dir

    # =========================================
    # DEVICE CONFIGURATION
    # =========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n===== Device Configuration =====")
    print("Using device:", device)

    if device.type == "cuda":
        BATCH_SIZE = 32
        NUM_WORKERS = 2
        PIN_MEMORY = True
        AMP = True
        torch.backends.cudnn.benchmark = True
    else:
        BATCH_SIZE = 2
        NUM_WORKERS = 0
        PIN_MEMORY = False
        AMP = False

    print(f"Batch Size  : {BATCH_SIZE}")
    print(f"Workers     : {NUM_WORKERS}")
    print(f"Pin Memory  : {PIN_MEMORY}")
    print(f"AMP Enabled : {AMP}")
    print("================================\n")

    torch.set_num_threads(8)

    # =========================================
    # TRAIN CONFIG
    # =========================================
    EPOCHS = 30
    LR = 1e-4
    MODEL_SAVE_PATH = "models/best_model.pth"
    PATIENCE = 3  # early stopping

    # =========================================
    # DATASET + SPLIT
    # =========================================
    dataset = VideoDataset(DATASET_PATH)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Train size: {train_size}")
    print(f"Val size  : {val_size}\n")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=PIN_MEMORY)

    # =========================================
    # MODEL
    # =========================================
    # model = CNN_LSTM().to(device)
    model = CNN_Transformer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)

    best_val_acc = 0
    early_counter = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # =========================================
    # TRAINING LOOP
    # =========================================
    for epoch in range(EPOCHS):

        # -------- TRAIN --------
        model.train()
        total_loss = 0

        all_preds, all_labels = [], []

        print(f"\n🚀 Epoch {epoch+1}/{EPOCHS}")
        loop = tqdm(train_loader)

        for sequences, labels in loop:

            B, N, T, C, H, W = sequences.shape

            sequences = sequences.view(B*N, T, C, H, W).to(device)
            labels = labels.repeat_interleave(N).to(device)

            # with torch.cuda.amp.autocast(enabled=AMP):
            with torch.amp.autocast(device_type='cuda', enabled=AMP):
                outputs = model(sequences)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            loop.set_postfix(loss=loss.item())

        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        train_prec = precision_score(all_labels, all_preds)
        train_rec = recall_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds)

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0

        val_preds, val_labels = [], []
        val_probs = []

        with torch.no_grad():
            for sequences, labels in val_loader:

                B, N, T, C, H, W = sequences.shape

                sequences = sequences.view(B*N, T, C, H, W).to(device)
                labels = labels.repeat_interleave(N).to(device)

                outputs = model(sequences)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)[:, 1]

                preds = torch.argmax(outputs, dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())

        val_loss /= len(val_loader)

        val_acc = accuracy_score(val_labels, val_preds)
        val_prec = precision_score(val_labels, val_preds)
        val_rec = recall_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)

        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"\n📊 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | F1: {val_f1:.4f}")

        # -------- BEST MODEL --------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_counter = 0

            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("💾 Best model saved!")

            best_labels = val_labels
            best_preds = val_preds
            best_probs = val_probs

        else:
            early_counter += 1
            print(f"⏳ Early stopping counter: {early_counter}/{PATIENCE}")

            if early_counter >= PATIENCE:
                print("🛑 Early stopping triggered!")
                break

    # =========================================
    # PLOTS
    # =========================================
    os.makedirs("outputs", exist_ok=True)

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig("outputs/loss_curve.png")

    plt.figure()
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig("outputs/accuracy_curve.png")

    # =========================================
    # CONFUSION MATRIX (ONLY ON BEST)
    # =========================================
    cm = confusion_matrix(best_labels, best_preds)

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.savefig("outputs/confusion_matrix.png")

    # =========================================
    # ROC CURVE
    # =========================================
    fpr, tpr, _ = roc_curve(best_labels, best_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig("outputs/roc_curve.png")

    print("\n🎉 Training complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()



# ########################################################################################

# NEW

# import os
# import argparse
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# from utils.video_dataset import VideoDataset
# from models.cnn_lstm_model import CNN_LSTM


# def main():

#     # =========================================
#     # ARGPARSE
#     # =========================================
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_dir", type=str, required=True)
#     args = parser.parse_args()

#     DATASET_PATH = args.data_dir

#     # =========================================
#     # DEVICE CONFIGURATION
#     # =========================================
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     print("\n===== Device Configuration =====")
#     print("Using device:", device)

#     if device.type == "cuda":
#         BATCH_SIZE = 8
#         NUM_WORKERS = 2
#         PIN_MEMORY = True
#         AMP = True

#         # GPU optimization
#         torch.backends.cudnn.benchmark = True

#     else:
#         BATCH_SIZE = 2
#         NUM_WORKERS = 0
#         PIN_MEMORY = False
#         AMP = False

#     print(f"Batch Size  : {BATCH_SIZE}")
#     print(f"Workers     : {NUM_WORKERS}")
#     print(f"Pin Memory  : {PIN_MEMORY}")
#     print(f"AMP Enabled : {AMP}")
#     print("================================\n")

#     # CPU optimization
#     torch.set_num_threads(8)

#     # =========================================
#     # TRAINING CONFIG
#     # =========================================
#     EPOCHS = 10
#     LR = 1e-4
#     MODEL_SAVE_PATH = "models/deepshield_video_lstm.pth"

#     # =========================================
#     # DATASET
#     # =========================================
#     dataset = VideoDataset(DATASET_PATH)

#     real_count = sum(1 for _, label in dataset.samples if label == 0)
#     fake_count = sum(1 for _, label in dataset.samples if label == 1)

#     print("===== Dataset Statistics =====")
#     print(f"Total videos : {len(dataset)}")
#     print(f"Real videos  : {real_count}")
#     print(f"Fake videos  : {fake_count}")
#     print("=============================\n")

#     # =========================================
#     # DATALOADER
#     # =========================================
#     dataloader = DataLoader(
#         dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         num_workers=NUM_WORKERS,
#         pin_memory=PIN_MEMORY
#     )

#     # =========================================
#     # MODEL
#     # =========================================
#     model = CNN_LSTM().to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=LR)

#     # AMP scaler (⚠ must be AFTER AMP defined)
#     scaler = torch.cuda.amp.GradScaler(enabled=AMP)

#     # =========================================
#     # LOAD CHECKPOINT
#     # =========================================
#     if os.path.exists(MODEL_SAVE_PATH):
#         model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
#         print("✅ Loaded existing model checkpoint!\n")

#     # =========================================
#     # TRAINING LOOP
#     # =========================================
#     for epoch in range(EPOCHS):

#         model.train()
#         total_loss = 0

#         print(f"\n🚀 Epoch {epoch+1}/{EPOCHS}")

#         loop = tqdm(enumerate(dataloader), total=len(dataloader))

#         for i, (sequences, labels) in loop:

#             # (B, N, T, C, H, W)
#             B, N, T, C, H, W = sequences.shape

#             sequences = sequences.view(B * N, T, C, H, W).to(device, non_blocking=True)
#             labels = labels.repeat_interleave(N).to(device, non_blocking=True)

#             # ---------- FORWARD (AMP) ----------
#             with torch.cuda.amp.autocast(enabled=AMP):
#                 outputs = model(sequences)
#                 loss = criterion(outputs, labels)

#             # ---------- BACKWARD ----------
#             optimizer.zero_grad()
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             total_loss += loss.item()

#             # tqdm update
#             loop.set_postfix(loss=loss.item())

#         avg_loss = total_loss / len(dataloader)

#         print(f"✅ Epoch [{epoch+1}/{EPOCHS}] - Avg Loss: {avg_loss:.4f}")

#         # ---------- SAVE ----------
#         os.makedirs("models", exist_ok=True)
#         torch.save(model.state_dict(), MODEL_SAVE_PATH)
#         print("💾 Model checkpoint saved!")

#     print("\n🎉 Training complete. Model saved!")


# # =========================================
# # ENTRY POINT (VERY IMPORTANT FOR WINDOWS)
# # =========================================
# if __name__ == "__main__":
#     main()







# ########################################################################################

#  O L D




# torch.backends.cudnn.benchmark = True
# scaler = torch.cuda.amp.GradScaler(enabled=AMP)

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import os
# import argparse
# from tqdm import tqdm

# from utils.video_dataset import VideoDataset
# from models.cnn_lstm_model import CNN_LSTM

# def main():

#     # ---------------- ARGPARSE ----------------
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_dir", type=str, required=True)
#     args = parser.parse_args()

#     DATASET_PATH = args.data_dir

#     # # ---------------- CONFIG ----------------
#     # BATCH_SIZE = 2
#     # EPOCHS = 10
#     # LR = 1e-4
#     # MODEL_SAVE_PATH = "models/deepshield_video_lstm.pth"

#     # # ---------------- DEVICE ----------------
#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # print(f"Using device: {device}")

#     # # ---------------- DATA ----------------
#     # dataset = VideoDataset(DATASET_PATH)

#     # # 👉 Dataset stats
#     # real_count = sum(1 for _, label in dataset.samples if label == 0)
#     # fake_count = sum(1 for _, label in dataset.samples if label == 1)

#     # print("\n===== Dataset Statistics =====")
#     # print(f"Total videos : {len(dataset)}")
#     # print(f"Real videos  : {real_count}")
#     # print(f"Fake videos  : {fake_count}")
#     # print("=============================\n")

#     # dataloader = DataLoader(
#     #     dataset,
#     #     batch_size=BATCH_SIZE,
#     #     shuffle=True,
#     #     num_workers=0,
#     #     pin_memory=True
#     # )

#     # =========================================
#     # Device Configuration
#     # =========================================

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("\n===== Device Configuration =====")
#     print("Using device:", device)

#     if device.type == "cuda":
#         BATCH_SIZE = 8          # ⚠ keep small (video model = heavy)
#         NUM_WORKERS = 0         # safe for Windows
#         PIN_MEMORY = True
#         AMP = True
#     else:
#         BATCH_SIZE = 2
#         NUM_WORKERS = 0
#         PIN_MEMORY = False
#         AMP = False

#     print(f"Batch Size  : {BATCH_SIZE}")
#     print(f"Workers     : {NUM_WORKERS}")
#     print(f"Pin Memory  : {PIN_MEMORY}")
#     print(f"AMP Enabled : {AMP}")
#     print("================================\n")

#     # Optional CPU optimization
#     torch.set_num_threads(8)

#     # =========================================
#     # Training Config
#     # =========================================

#     EPOCHS = 10
#     LR = 1e-4
#     MODEL_SAVE_PATH = "models/deepshield_video_lstm.pth"

#     # =========================================
#     # Dataset
#     # =========================================

#     dataset = VideoDataset(DATASET_PATH)

#     # 👉 Dataset stats
#     real_count = sum(1 for _, label in dataset.samples if label == 0)
#     fake_count = sum(1 for _, label in dataset.samples if label == 1)

#     print("===== Dataset Statistics =====")
#     print(f"Total videos : {len(dataset)}")
#     print(f"Real videos  : {real_count}")
#     print(f"Fake videos  : {fake_count}")
#     print("=============================\n")

#     # =========================================
#     # DataLoader
#     # =========================================

#     dataloader = DataLoader(
#         dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         num_workers=NUM_WORKERS,
#         pin_memory=PIN_MEMORY
#     )

#     # ---------------- MODEL ----------------
#     model = CNN_LSTM().to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=LR)

#     # ---------------- LOAD CHECKPOINT ----------------
#     if os.path.exists(MODEL_SAVE_PATH):
#         model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
#         print("✅ Loaded existing model checkpoint!\n")

#     # ---------------- TRAINING ----------------
#     for epoch in range(EPOCHS):
#         model.train()
#         total_loss = 0

#         print(f"\n🚀 Epoch {epoch+1}/{EPOCHS}")

#         # tqdm progress bar
#         loop = tqdm(enumerate(dataloader), total=len(dataloader))

#         for i, (sequences, labels) in loop:

#             # sequences shape: (B, N, T, C, H, W)
#             B, N, T, C, H, W = sequences.shape

#             # reshape for model
#             sequences = sequences.view(B * N, T, C, H, W).to(device, non_blocking=True)

#             # repeat labels
#             labels = labels.repeat_interleave(N).to(device, non_blocking=True)

#             # forward
#             with torch.cuda.amp.autocast(enabled=AMP):
#                 outputs = model(sequences)
#                 loss = criterion(outputs, labels)
#             # outputs = model(sequences)
#             # loss = criterion(outputs, labels)

#             # backward
#             optimizer.zero_grad()
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#             # optimizer.zero_grad()
#             # loss.backward()
#             # optimizer.step()

#             total_loss += loss.item()

#             # 👉 Update tqdm bar
#             loop.set_postfix(loss=loss.item())

#         avg_loss = total_loss / len(dataloader)

#         print(f"✅ Epoch [{epoch+1}/{EPOCHS}] - Avg Loss: {avg_loss:.4f}")

#         # ---------------- SAVE MODEL ----------------
#         torch.save(model.state_dict(), MODEL_SAVE_PATH)
#         print("💾 Model checkpoint saved!")

#     print("\n🎉 Training complete. Model saved!")

# if __name__ == "__main__":
#     main()