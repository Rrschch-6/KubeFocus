import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List

class SpatiotemporalCNN(nn.Module):
    def __init__(self):
        super(SpatiotemporalCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 2)  # Binary classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (B, 16, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 32, 8, 8)
        x = self.dropout(x)
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class IntrusionDetector:
    def __init__(self, artifact_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.artifact_dir = artifact_dir
        self.grid_size = 32

        # Load scalers
        self.traffic_scaler = joblib.load(os.path.join(artifact_dir, "network_scaler.pkl"))
        self.metrics_scaler = joblib.load(os.path.join(artifact_dir, "metrics_scaler.pkl"))

        # Load TF-IDF vectorizer
        self.tfidf_vectorizer = joblib.load(os.path.join(artifact_dir, "tfidf_vectorizer.pkl"))

        # Load attention
        self.traffic_attention = torch.load(os.path.join(artifact_dir, "traffic_attention.pt")).numpy()
        self.metrics_attention = torch.load(os.path.join(artifact_dir, "metrics_attention.pt")).numpy()

        # Load t-SNE grids
        self.tsne_grid_network = np.load(os.path.join(artifact_dir, "tsne_grid_network.npy"))
        self.tsne_grid_metrics = np.load(os.path.join(artifact_dir, "tsne_grid_metrics.npy"))

        # Load model
        self.model = SpatiotemporalCNN().to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(artifact_dir, "model.pt"), map_location=self.device))
        self.model.eval()

        # Pod label mapping
        self.pod_mapping = {'node_k8s-master-1': 1, 'node_k8s-worker-1': 2, 'node_k8s-worker-2': 3}
        self.min_pod, self.max_pod = 1, 3

    def _scale_pod_label(self, value):
        return 255 * (value - self.min_pod) / (self.max_pod - self.min_pod)

    def predict(self, traffic_row: List[float], metrics_row: List[float], log_text: str, pod_name: str) -> int:
        # Scale input
        traffic = self.traffic_scaler.transform([traffic_row])[0]
        metrics = self.metrics_scaler.transform([metrics_row])[0]
        log_vector = self.tfidf_vectorizer.transform([log_text]).toarray()[0]

        # Scale pod label
        pod_label = self.pod_mapping.get(pod_name, 1)
        scaled_pod_label = self._scale_pod_label(pod_label)

        # Create RGB image
        image = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)

        for i, (x, y) in enumerate(self.tsne_grid_network):
            image[x, y, 0] = np.clip(traffic[i] * self.traffic_attention[i], 0, 255)

        for i, (x, y) in enumerate(self.tsne_grid_metrics):
            image[x, y, 1] = np.clip(metrics[i] * self.metrics_attention[i], 0, 255)

        for x in range(4):
            for y in range(4):
                image[x, y, 2] = np.clip(scaled_pod_label, 0, 255)

        for i, val in enumerate(log_vector):
            x = 4 + i // self.grid_size
            y = i % self.grid_size
            if x < self.grid_size:
                image[x, y, 2] = np.clip(val * 255, 0, 255)

        # Convert to tensor and run inference
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        with torch.no_grad():
            output = self.model(image_tensor.to(self.device))
            prediction = torch.argmax(output, dim=1).item()
        return prediction
