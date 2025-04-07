import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm



# CUDA boost
torch.backends.cudnn.benchmark = True

# Config
DATA_DIR = "/home/sascha/KubeFocus/video/final_benign_dataset/image_dataset.pt"
SEQ_LEN = 5
LATENT_DIM = 256
BATCH_SIZE = 256
EPOCHS = 10
MASK_RATIO = 0.3
MASKING_MODE = "random"  # or: "random", "block", "center","causal"
SAVE_PATH = "/home/sascha/KubeFocus/artifacts/models"
os.makedirs(SAVE_PATH, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SequenceDataset(Dataset):
    def __init__(self, file_path, seq_len):
        self.seq_len = seq_len

        print(f"📂 Loading data from: {file_path}")
        raw_data = torch.load(file_path)

        if not isinstance(raw_data, list):
            raise ValueError("The loaded .pt file must be a list of dicts with keys 'image' and 'label'.")

        self.data = [d for d in raw_data if isinstance(d, dict) and "image" in d and d["image"].ndim == 3]
        skipped = len(raw_data) - len(self.data)

        print(f"✅ Valid frames loaded: {len(self.data)}")
        print(f"❌ Skipped invalid frames: {skipped}")
        print(f"🧮 Usable sequences (len): {max(0, len(self.data) - self.seq_len + 1)}")

    def __len__(self):
        return max(0, len(self.data) - self.seq_len + 1)

    def __getitem__(self, idx):
        seq = self.data[idx:idx + self.seq_len]
        images = torch.stack([s["image"].float() / 255.0 for s in seq])  # shape (T, H, W, C)
        images = images.permute(0, 3, 1, 2)  # ➜ shape (T, C, H, W)
        return images


# Encoder
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 8x8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8 * 128, latent_dim)
        )

    def forward(self, x):
        return self.net(x)


class TransformerPredictor(nn.Module):
    def __init__(self, latent_dim, seq_len, n_heads=4, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=n_heads, dim_feedforward=latent_dim * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, latent_dim))

    def forward(self, x):
        # x: (B, T, D)
        x = x + self.pos_embedding[:, :x.size(1), :]
        return self.encoder(x)  # (B, T, D)

# # Add this globally in your script — define mask token once
mask_token = nn.Parameter(torch.zeros(LATENT_DIM), requires_grad=True).to(DEVICE)

def train():
    dataset = SequenceDataset(DATA_DIR, SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    encoder = Encoder(LATENT_DIM).to(DEVICE)
    predictor = TransformerPredictor(LATENT_DIM, seq_len=SEQ_LEN).to(DEVICE)

    mask_token = nn.Parameter(torch.zeros(LATENT_DIM, device=DEVICE))

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()) + [mask_token],
        lr=1e-4
    )

    epoch_losses = []

    for epoch in range(EPOCHS):
        encoder.train()
        predictor.train()
        total_loss = 0.0

        loop = tqdm(loader, desc=f"📚 Epoch {epoch+1}/{EPOCHS}", leave=False)
        for images in loop:
            B, T, C, H, W = images.shape
            images = images.to(DEVICE)

            latents = encoder(images.view(B * T, C, H, W)).view(B, T, -1)

            # Apply selected masking strategy
            if MASKING_MODE == "random":
                mask = torch.rand(B, T, device=DEVICE) < MASK_RATIO
            elif MASKING_MODE == "causal":
                num_masked = int(MASK_RATIO * T)
                mask = torch.zeros(B, T, dtype=torch.bool, device=DEVICE)
                mask[:, -num_masked:] = True
            elif MASKING_MODE == "block":
                mask = torch.zeros(B, T, dtype=torch.bool, device=DEVICE)
                block_len = max(1, int(MASK_RATIO * T))
                start = torch.randint(0, T - block_len + 1, (B,), device=DEVICE)
                for i in range(B):
                    mask[i, start[i]:start[i] + block_len] = True
            elif MASKING_MODE == "center":
                mask = torch.zeros(B, T, dtype=torch.bool, device=DEVICE)
                mask[:, T // 2] = True
            else:
                raise ValueError(f"Unknown masking mode: {MASKING_MODE}")

            if mask.sum() == 0 or (~mask).sum() == 0:
                continue

            masked_latents = latents.clone()
            masked_latents[mask] = mask_token

            predicted_latents = predictor(masked_latents)
            target = latents[mask]
            predicted = predicted_latents[mask]
            loss = F.mse_loss(predicted, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        epoch_loss = total_loss / len(loader)
        epoch_losses.append(epoch_loss)
        print(f"📉 Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.6f}")

    # Save model
    model_path = f"{SAVE_PATH}/vjepa_model__{MASKING_MODE}_len_{SEQ_LEN}_ratio_{MASK_RATIO}.pt"
    torch.save({
        "encoder": encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "mask_token": mask_token.detach().cpu()
    }, model_path)
    print(f"✅ Model saved to {model_path}")

    # Save loss plot
    plt.figure()
    plt.plot(range(1, EPOCHS + 1), epoch_losses, marker='o')
    plt.title(f"Training Loss - Masking: {MASKING_MODE} seq length: {SEQ_LEN} masking ratio: {MASK_RATIO}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    loss_plot_path = f"{SAVE_PATH}/loss_plot_{MASKING_MODE}_len_{SEQ_LEN}_ratio_{MASK_RATIO}.png"
    plt.savefig(loss_plot_path)
    print(f"📈 Loss plot saved to {loss_plot_path}")


if __name__ == "__main__":
    train()

