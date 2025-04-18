import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
import os
from datetime import datetime, timedelta
import time
from torch import Tensor

def future_row_masks(batch_size, max_length, mask_ratio):
    """
    Generate random masks for rows.
    
    Args:
        batch_size (int): Number of samples in the batch.
        max_length (int): Maximum sequence length.
        mask_ratio (float): Fraction of rows to mask (0 to 1).
    
    Returns:
        masks_enc (torch.Tensor): Boolean mask for encoder, True for visible rows.
        masks_pred (torch.Tensor): Boolean mask for predictor, True for masked rows.
    """
    num_visible = int(max_length * (1 - mask_ratio))
    masks_enc = torch.zeros(batch_size, max_length, dtype=torch.bool)
    masks_pred = torch.zeros(batch_size, max_length, dtype=torch.bool)
    for i in range(batch_size):
        perm = torch.randperm(max_length)
        visible_idx = perm[:num_visible]
        pred_idx = perm[num_visible:]
        masks_enc[i, visible_idx] = True
        masks_pred[i, pred_idx] = True
    return masks_enc, masks_pred

def generate_row_masks(batch_size, max_length, mask_ratio):
    """
    Generate random masks for rows.
    
    Args:
        batch_size (int): Number of samples in the batch.
        max_length (int): Maximum sequence length.
        mask_ratio (float): Fraction of rows to mask (0 to 1).
    
    Returns:
        masks_enc (torch.Tensor): Boolean mask for encoder, True for visible rows.
        masks_pred (torch.Tensor): Boolean mask for predictor, True for masked rows.
    """
    num_visible = int(max_length * (1 - mask_ratio))
    masks_enc = torch.zeros(batch_size, max_length, dtype=torch.bool)
    masks_pred = torch.zeros(batch_size, max_length, dtype=torch.bool)

    # future patches indices
    num_masked = int(max_length * mask_ratio)
    indices = torch.arange(0, max_length, dtype=torch.long)
    future_idx = indices[-num_masked:] # Indices of masked patches
    visible_idx = indices[:-num_masked]  # Indices of visible patches
    masks_enc[:, visible_idx] = True
    masks_pred[:, future_idx] = True
    return masks_enc, masks_pred

# Random mask generation function
def generate_masks(batch_size, total_patches, mask_ratio) -> tuple[Tensor, Tensor]:
    """
    Generate random masks for encoder (visible patches) and predictor (masked patches).
    Args:
        batch_size (int): Number of samples in the batch.
        total_patches (int): Total number of patches per video.
        mask_ratio (float): Fraction of patches to mask (0 to 1).
    Returns:
        masks_enc (torch.Tensor): Boolean mask for encoder, True for visible patches.
        masks_pred (torch.Tensor): Boolean mask for predictor, True for masked patches.
    """
    num_visible = int(total_patches * (1 - mask_ratio))  # e.g., 20 if total=80, mask_ratio=0.75
    masks_enc = torch.zeros(batch_size, total_patches, dtype=torch.bool)
    masks_pred = torch.zeros(batch_size, total_patches, dtype=torch.bool)
    for b in range(batch_size):
        # Randomly permute patch indices
        perm = torch.randperm(total_patches)
        visible_idx = perm[:num_visible]  # Indices of visible patches
        pred_idx = perm[num_visible:]     # Indices of masked patches
        masks_enc[b, visible_idx] = True
        masks_pred[b, pred_idx] = True
    return masks_enc, masks_pred

def mask_future(batch_size: int, total_patches: int, mask_ratio: float, fpv: int) -> tuple[Tensor, Tensor]:
    """
    Generate random masks for encoder (visible patches) and predictor (masked patches).
    Args:
        batch_size (int): Number of samples in the batch.
        total_patches (int): Total number of patches per video.
        mask_ratio (float): Fraction of patches to mask (0 to 1).
        fpv (int): frames per video
    Returns:
        masks_enc (torch.Tensor): Boolean mask for encoder, True for visible patches.
        masks_pred (torch.Tensor): Boolean mask for predictor, True for masked patches.
    """
    num_visible = int(total_patches * (1 - mask_ratio))  # e.g., 20 if total=80, mask_ratio=0.75
    masks_enc = torch.zeros(batch_size, total_patches, dtype=torch.bool)
    masks_pred = torch.zeros(batch_size, total_patches, dtype=torch.bool)
    
    # future patches indices
    num_frames_masked = int(fpv * mask_ratio)
    patch_per_frame = int(total_patches / fpv)
    indices = torch.arange(0, total_patches, dtype=torch.long)
    future_indices = indices[-num_frames_masked*patch_per_frame:] # Indices of masked patches
    visible_indices = indices[:-num_frames_masked*patch_per_frame]  # Indices of visible patches
    masks_enc[:, visible_indices] = True
    masks_pred[:, future_indices] = True
    return masks_enc, masks_pred

def scale_to_255(df, columns_to_exclude=None):
    df_scaled = df.copy()
    if not columns_to_exclude==None:
        for col in df.columns:
            if col not in columns_to_exclude:
                min_val = df[col].min()
                max_val = df[col].max()
                df_scaled[col] = 255 * (df[col] - min_val) / (max_val - min_val)
    else:
        for col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            df_scaled[col] = 255 * (df[col] - min_val) / (max_val - min_val)

    return df_scaled

def compute_tsne_grid(pairwise_attention_matrix, grid_size=32, perplexity=30):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    feature_embeddings_2d = tsne.fit_transform(pairwise_attention_matrix)  # Shape: [n_features, 2]
    
    tsne_min = np.min(feature_embeddings_2d, axis=0)
    tsne_max = np.max(feature_embeddings_2d, axis=0)
    
    tsne_scaled = (feature_embeddings_2d - tsne_min) / (tsne_max - tsne_min)
    tsne_grid = np.floor(tsne_scaled * (grid_size - 1)).astype(int)  # Shape: [n_features, 2]
    
    return tsne_grid

def train_autoencoder(model, data_tensor, epochs=100, lr=0.001, verbose_every=10):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    final_pairwise_attention = None
    final_per_feature_attention = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        reconstructed, pairwise_attention_matrix, per_feature_attention_coeffs = model(data_tensor)
        loss = criterion(reconstructed, data_tensor)

        loss.backward()
        optimizer.step()

        if epoch % verbose_every == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

        if epoch == epochs - 1:
            final_pairwise_attention = pairwise_attention_matrix.detach().cpu().numpy()
            final_per_feature_attention = per_feature_attention_coeffs.detach().cpu().numpy()

    return final_pairwise_attention, final_per_feature_attention


class DualAttentionAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DualAttentionAutoencoder, self).__init__()
        
        # Pairwise Attention mechanism: Learnable pairwise weights between features
        self.pairwise_attention_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))  # [n_features, n_features]
        nn.init.xavier_uniform_(self.pairwise_attention_weights)  # Initialize
        
        # Per-Sample Attention mechanism: Attention per feature for each sample
        self.per_feature_attention_weights = nn.Parameter(torch.Tensor(input_dim))  # [n_features]
        nn.init.uniform_(self.per_feature_attention_weights, a=0.0, b=1.0)  # Initialize
        
        # Encoder with additional layer
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), 
            nn.ReLU()
        )
        
        # Decoder with additional layer
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Pairwise attention mechanism
        pairwise_attention_scores = torch.matmul(x, self.pairwise_attention_weights) 
        pairwise_attention_coeffs = torch.softmax(pairwise_attention_scores, dim=-1) 
        pairwise_attention_matrix = torch.matmul(pairwise_attention_coeffs.T, pairwise_attention_coeffs)
        
        # Per-sample attention mechanism
        per_feature_attention_coeffs = torch.softmax(self.per_feature_attention_weights, dim=0) 
        attended_input = x * per_feature_attention_coeffs
        
        # Encoding and decoding with added layers
        encoded = self.encoder(attended_input)
        reconstructed = self.decoder(encoded)
        
        return reconstructed, pairwise_attention_matrix, per_feature_attention_coeffs