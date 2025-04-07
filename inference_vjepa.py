import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# === Inference Parameters ===
INFER_DATASET_PATH = "/home/sascha/KubeFocus/video/final_attacks_dataset/image_dataset.pt"
MODEL_DIR = "/home/sascha/KubeFocus/artifacts/models"
MASKING_MODE = "random"
SEQ_LEN = 5
MASK_RATIO = 0.3
LATENT_DIM = 256
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Auto-load the model file ===
def find_model_file():
    pattern = f"vjepa_model_{MASKING_MODE}_len_{SEQ_LEN}_ratio_{MASK_RATIO}"
    for f in os.listdir(MODEL_DIR):
        if pattern in f and f.endswith(".pt"):
            print(f"📦 Found model: {f}")
            return os.path.join(MODEL_DIR, f)
    raise FileNotFoundError(f"No model found matching {pattern} in {MODEL_DIR}")

MODEL_PATH = find_model_file()

# === Dataset ===
class SequenceDatasetWithLabels(Dataset):
    def __init__(self, file_path, seq_len):
        raw_data = torch.load(file_path)
        self.seq_len = seq_len
        self.data = [d for d in raw_data if isinstance(d, dict) and "image" in d and "label" in d]

    def __len__(self):
        return max(0, len(self.data) - self.seq_len + 1)

    def __getitem__(self, idx):
        seq = self.data[idx:idx + self.seq_len]
        images = torch.stack([s["image"].float() / 255.0 for s in seq]).permute(0, 3, 1, 2)
        label = seq[-1]["label"]
        return images, label

# === Models ===
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8 * 128, latent_dim)
        )

    def forward(self, x):
        return self.net(x)

class TransformerPredictor(nn.Module):
    def __init__(self, latent_dim, seq_len, n_heads=4, num_layers=2):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=n_heads, dim_feedforward=latent_dim * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = x + self.pos_embedding[:, :x.size(1), :]
        return self.encoder(x)

# === Masking logic ===
def apply_mask(latents, mode, ratio):
    B, T, _ = latents.shape
    if mode == "random":
        return torch.rand(B, T, device=latents.device) < ratio
    elif mode == "causal":
        mask = torch.zeros(B, T, dtype=torch.bool, device=latents.device)
        mask[:, -int(ratio * T):] = True
        return mask
    elif mode == "block":
        mask = torch.zeros(B, T, dtype=torch.bool, device=latents.device)
        block_len = max(1, int(ratio * T))
        starts = torch.randint(0, T - block_len + 1, (B,), device=latents.device)
        for i in range(B):
            mask[i, starts[i]:starts[i] + block_len] = True
        return mask
    elif mode == "center":
        mask = torch.zeros(B, T, dtype=torch.bool, device=latents.device)
        mask[:, T // 2] = True
        return mask
    else:
        raise ValueError(f"Unknown masking mode: {mode}")

# === Inference ===
def inference(threshold=0.5):
    dataset = SequenceDatasetWithLabels(INFER_DATASET_PATH, SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    encoder = Encoder(LATENT_DIM).to(DEVICE)
    predictor = TransformerPredictor(LATENT_DIM, seq_len=SEQ_LEN).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH)
    encoder.load_state_dict(checkpoint["encoder"])
    predictor.load_state_dict(checkpoint["predictor"])
    mask_token = checkpoint["mask_token"].to(DEVICE)

    encoder.eval()
    predictor.eval()

    y_true, y_pred, surprise_scores = [], [], []

    with torch.no_grad():
        for images, label in loader:
            B, T, C, H, W = images.shape
            images = images.to(DEVICE)
            latents = encoder(images.view(B * T, C, H, W)).view(B, T, -1)

            mask = apply_mask(latents, MASKING_MODE, MASK_RATIO)
            if mask.sum() == 0 or (~mask).sum() == 0:
                continue

            masked_latents = latents.clone()
            masked_latents[mask] = mask_token

            predicted_latents = predictor(masked_latents)
            surprise = F.mse_loss(predicted_latents[mask], latents[mask]).item()

            predicted_label = 1 if surprise > threshold else 0
            y_true.append(label.item())
            y_pred.append(predicted_label)
            surprise_scores.append(surprise)

    print(f"\n🎯 Model: {os.path.basename(MODEL_PATH)}")
    print("🧪 Threshold:", threshold)
    print("🧠 Masking:", MASKING_MODE, "| Ratio:", MASK_RATIO, "| Seq len:", SEQ_LEN)



    # 🎨 Plot surprise score distribution by label
    surprise_0 = [s for s, y in zip(surprise_scores, y_true) if y == 0]
    surprise_1 = [s for s, y in zip(surprise_scores, y_true) if y == 1]
    print(f"Label 0 - mean: {torch.tensor(surprise_0).mean():.6f}, std: {torch.tensor(surprise_0).std():.6f}")
    print(f"Label 1 - mean: {torch.tensor(surprise_1).mean():.6f}, std: {torch.tensor(surprise_1).std():.6f}")
    plt.figure(figsize=(8, 5))
    plt.hist(surprise_0, bins=50, alpha=0.6, label="Label 0 (Normal)")
    plt.hist(surprise_1, bins=50, alpha=0.6, label="Label 1 (Attack)")
    plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold = {threshold}")
    plt.title(f"Surprise Score Distribution ({MASKING_MODE} masking)")
    plt.xlabel("Surprise Score (MSE)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plot_path = f"{MODEL_DIR}/surprise_plot_{MASKING_MODE}_len_{SEQ_LEN}_ratio_{MASK_RATIO}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"📈 Surprise plot saved to: {plot_path}")

    # 🧾 Print classification stats
    print("\n📊 Classification Report:\n", classification_report(y_true, y_pred))
    print("🧩 Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    inference(threshold=0.5)
