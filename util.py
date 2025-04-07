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
from tqdm import tqdm


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