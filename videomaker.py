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
import joblib
from tqdm import tqdm
from util import scale_to_255,compute_tsne_grid,train_autoencoder,DualAttentionAutoencoder


#parameters
root_dir = '/home/sascha/KubeFocus'
epochs=150
lr=0.001
grid_size=32

#file locations
attention_csv_file = f'{root_dir}/datasets/final_benign_dataset.csv'
output_dir=os.path.join(root_dir,'video')
os.makedirs(output_dir, exist_ok=True)

METRIC_COLS = np.r_[0:25, 30:187]
NETWORK_COLS = np.r_[26:30]


# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


data = pd.read_csv(attention_csv_file)
data = data[data['attack'] == 0]
if 'timestamp' in data.columns:
    data = data.drop(columns=['timestamp'])

print('----------loading Metrics Data----------')

data_metrics = data.iloc[:, METRIC_COLS]
data_metrics = data_metrics.select_dtypes(include=[np.number])
data_metrics = data_metrics.fillna(0)
metrics_scaler = MinMaxScaler()
scaled_data_metrics = metrics_scaler.fit_transform(data_metrics)
X_scaled_metric = torch.tensor(scaled_data_metrics , dtype=torch.float32)
X_tensor_metric=X_scaled_metric

print('----------loading Network Data----------')
data_network = data.iloc[:, NETWORK_COLS]
network_scaler = MinMaxScaler()
scaled_data_network = network_scaler.fit_transform(data_network)
X_scaled_traffic = torch.tensor(scaled_data_network, dtype=torch.float32)
X_tensor_traffic=X_scaled_traffic

print('----------loading Log Data----------')
data['Pod_Logs'] = data['Pod_Logs'].fillna(0).astype('str')
pod_mapping = {'node_k8s-master-1': 1, 'node_k8s-worker-1': 2, 'node_k8s-worker-2': 3}
data['pod_label'] = data[['node_k8s-master-1', 'node_k8s-worker-1', 'node_k8s-worker-2']].idxmax(axis=1).map(pod_mapping)

print('----------training metrics camera----------')
input_dim = X_tensor_metric.shape[1]
model = DualAttentionAutoencoder(input_dim=input_dim, hidden_dim=64)
metrics_pairwise_attention_matrix, metrices_per_feature_attention = train_autoencoder(model, X_tensor_metric,epochs=epochs, lr=lr, verbose_every=10)
print("Final Pairwise Attention Matrix Shape:", metrics_pairwise_attention_matrix.shape)
print("Final Per-feature Attention Matrix Shape:", metrices_per_feature_attention.shape)
tsne_grid_metrics = compute_tsne_grid(metrics_pairwise_attention_matrix, grid_size=grid_size, perplexity=30)

print('----------training Network camera----------')

input_dim = X_tensor_traffic.shape[1]
model = DualAttentionAutoencoder(input_dim=input_dim, hidden_dim=64)
network_pairwise_attention_matrix, network_per_feature_attention = train_autoencoder(model, X_tensor_traffic,epochs=epochs, lr=lr, verbose_every=10)
print("Final Pairwise Attention Matrix Shape:", network_pairwise_attention_matrix.shape)
print("Final Per-feature Attention Matrix Shape:", network_per_feature_attention.shape)
tsne_grid_network = compute_tsne_grid(network_pairwise_attention_matrix, grid_size=grid_size, perplexity=2)


print('----------training Logs camera----------')
tfidf = TfidfVectorizer(norm='l2')
tfidf_matrix = tfidf.fit_transform(data['Pod_Logs'])
tfidf_array = tfidf_matrix.toarray()  # Convert to array
tfidf_array = np.nan_to_num(tfidf_array, nan=0, posinf=1, neginf=0)
data_list = []
for i in range(len(tfidf_array)):
    pod_label = data.iloc[i]['pod_label']
    embedding = torch.tensor(tfidf_array[i], dtype=torch.float32)

    if torch.isnan(embedding).any():
        print(f"Warning: NaN found in embedding at index {i}, replacing with zero tensor")
        embedding = torch.zeros_like(embedding)

    data_list.append((pod_label, embedding))
    if torch.isnan(embedding).any():
        print(f"Warning: NaN found in embedding at index {i}, replacing with zero tensor")
        embedding = torch.zeros_like(embedding)
    
    data_list.append((pod_label, embedding))

logs_embeddings=data_list

print('log list created. length of log list is:',len(data_list))

metrics_attention = metrices_per_feature_attention / metrices_per_feature_attention.max()
traffic_attention = network_per_feature_attention/ network_per_feature_attention.max()


print('----------creating video sequences for all CSVs in folder----------')

csv_folder = f'{root_dir}/datasets/'
csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

for csv_filename in csv_files:
    print(f'\nProcessing file: {csv_filename}')
    csv_path = os.path.join(csv_folder, csv_filename)
    video_name = os.path.splitext(csv_filename)[0]
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    data = pd.read_csv(csv_path)

    data_metrics = data.iloc[:, METRIC_COLS]
    data_metrics = data_metrics.select_dtypes(include=[np.number])

    data_network = data.iloc[:, NETWORK_COLS]
    data_network['attack'] = data['attack'].apply(lambda x: 0 if x == 0 else 1)

    traffic_df_scaled = scale_to_255(data_network, columns_to_exclude=['attack'])
    metrics_df_scaled = scale_to_255(data_metrics)

    pod_labels = [entry[0] for entry in logs_embeddings]
    min_pod_label, max_pod_label = min(pod_labels), max(pod_labels)

    def scale_pod_label(value):
        return 255 * (value - min_pod_label) / (max_pod_label - min_pod_label)

    logs_embeddings_scaled = []
    for entry in logs_embeddings:
        pod_label, embedding = entry
        scaled_pod_label = scale_pod_label(pod_label)
        embedding_min, embedding_max = embedding.min(), embedding.max()
        scaled_embedding = 255 * (embedding - embedding_min) / (embedding_max - embedding_min)
        scaled_embedding = torch.nan_to_num(scaled_embedding, nan=0.0, posinf=255.0, neginf=0.0)
        logs_embeddings_scaled.append((scaled_pod_label, scaled_embedding))

    print(f'Generating frames for: {csv_filename}')
    min_len = min(len(traffic_df_scaled), len(metrics_df_scaled), len(logs_embeddings_scaled))

    all_data = []

    for idx in tqdm(range(min_len), desc=f"Processing {video_name}"):
        traffic_sample = traffic_df_scaled.iloc[idx].drop(['attack']).values
        metrics_sample = metrics_df_scaled.iloc[idx].values
        logs_embedding = logs_embeddings_scaled[idx][1].numpy()
        scaled_pod_label = logs_embeddings_scaled[idx][0]
        label = int(traffic_df_scaled.iloc[idx]['attack'])

        image = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)

        # Red channel: traffic
        for feature_idx, (x, y) in enumerate(tsne_grid_network):
            #image[x, y, 0] = np.clip(traffic_sample[feature_idx] * traffic_attention[feature_idx], 0, 255)
            value = traffic_sample[feature_idx] * traffic_attention[feature_idx]
            if not np.isfinite(value):
                value = 0
            image[x, y, 0] = np.clip(value, 0, 255)

        # Green channel: metrics
        for feature_idx, (x, y) in enumerate(tsne_grid_metrics):
            #image[x, y, 1] = np.clip(metrics_sample[feature_idx] * metrics_attention[feature_idx], 0, 255)
            value = metrics_sample[feature_idx] * metrics_attention[feature_idx]
            if not np.isfinite(value):
                value = 0  # or use np.nan_to_num(value)
            image[x, y, 1] = np.clip(value, 0, 255)

        # Blue channel: logs
        for x in range(4):
            for y in range(4):
                image[x, y, 2] = np.clip(scaled_pod_label, 0, 255)

        feature_count = 0
        for x in range(4, grid_size):
            for y in range(grid_size):
                if feature_count < len(logs_embedding):
                    value = logs_embedding[feature_count]
                    image[x, y, 2] = np.clip(value, 0, 255)
                    feature_count += 1
                else:
                    break
            if feature_count >= len(logs_embedding):
                break

        image_tensor = torch.tensor(image, dtype=torch.uint8)
        all_data.append({'image': image_tensor, 'label': label})

    torch.save(all_data, os.path.join(video_output_dir, 'image_dataset.pt'))
    print(f"Saved {len(all_data)} image-label pairs to {video_output_dir}/image_dataset.pt")



artifacts_dir = os.path.join(root_dir, 'artifacts')
os.makedirs(artifacts_dir, exist_ok=True)

joblib.dump(metrics_scaler, os.path.join(artifacts_dir, 'metrics_scaler.pkl'))
joblib.dump(network_scaler, os.path.join(artifacts_dir, 'network_scaler.pkl'))
torch.save(torch.tensor(traffic_attention), os.path.join(artifacts_dir, 'traffic_attention.pt'))
torch.save(torch.tensor(metrics_attention), os.path.join(artifacts_dir, 'metrics_attention.pt'))

# Save t-SNE grids
np.save(os.path.join(artifacts_dir, 'tsne_grid_network.npy'), tsne_grid_network)
np.save(os.path.join(artifacts_dir, 'tsne_grid_metrics.npy'), tsne_grid_metrics)

# Save TF-IDF vectorizer for logs (if needed during live inference)
joblib.dump(tfidf, os.path.join(artifacts_dir, 'tfidf_vectorizer.pkl'))
print(f"Saved {len(all_data)} image-label pairs to {os.path.join(output_dir, 'image_dataset.pt')}")
