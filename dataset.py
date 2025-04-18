import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

class VideoDataset(Dataset):
    def __init__(self,
                 video_paths,
                 frames_per_video=5,
                 train=True):
        self.video_paths = video_paths
        self.frames_per_video = frames_per_video
        self.all_frames = torch.load(self.video_paths)
        self.num_all_videos = len(self.all_frames) // frames_per_video
        num_videos = int(self.num_all_videos * 0.9)
        if train:
            self.all_frames = self.all_frames[:num_videos * frames_per_video]
            self.num_videos = num_videos
        else:
            self.all_frames = self.all_frames[num_videos * frames_per_video:]
            self.num_videos = self.num_all_videos - num_videos



    def __len__(self):
        return self.num_videos

    def __getitem__(self, idx):
        start_idx = idx * self.frames_per_video
        end_idx = start_idx + self.frames_per_video
        video = torch.stack([self.all_frames[idx]['image'].to(torch.float32) for idx in range(start_idx, end_idx)], dim=0)
        return video.permute(3,0,1,2).to(torch.float32)
    
class DetectionDataset(Dataset):
    def __init__(self, 
                 mix_dataset_path,
                 frames_per_video=5,
                 train=True):
        super().__init__()
        self.video_paths = mix_dataset_path
        self.frames_per_video = frames_per_video
        self.all_frames = torch.load(self.video_paths)
        self.num_all_videos = len(self.all_frames) // frames_per_video
        # TODO: Find a way to create train/val sets
        
    def __len__(self) -> int:
        return self.num_all_videos

    def __getitem__(self, idx) -> dict:
        start_idx = idx * self.frames_per_video
        end_idx = start_idx + self.frames_per_video
        video = torch.stack([self.all_frames[idx]['image'].to(torch.float32) for idx in range(start_idx, end_idx)], dim=0)
        # if even one frame is malicious, the video is malicious, i.e label 1
        # else benign, i.e label 0
        labels = [self.all_frames[idx]['label'] for idx in range(start_idx, end_idx)]
        label = 1 if any(labels) else 0

        return {"video": video.permute(3,0,1,2).to(torch.float32),
                "label": torch.tensor(label).to(torch.long),
                "all_labels": labels,}
    
class TabularDataset(Dataset):
    def __init__(self,
                 path: str,
                 max_len: int,
                 label_col_name: str='attack') -> None:
        super().__init__()
        self.df = pd.read_csv(path)
        self.label = self.df[label_col_name].values
        self.df = self.df.drop(columns=[label_col_name])
        self.max_len = max_len
        self.num_features = self.df.shape[1]
        # Normalization
        scaler = StandardScaler()
        self.df = scaler.fit_transform(self.df)

        # Create sequences of length max_len from the rows
        self.sequences = []
        self.labels = []
        for i in range(len(self.df) - max_len + 1):
            seq = self.df[i:i + max_len]
            self.sequences.append(seq)
            self.labels.append(self.label[i:i + max_len])
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]
        label = self.labels[idx]
        # Convert to tensors
        seq_tensor = torch.tensor(seq, dtype=torch.float32)
        # label = 1 if any(label) else 0
        label_tensor = torch.tensor(label, dtype=torch.long)
        return {"sequence": seq_tensor, "label": label_tensor}

    # Pad sequences to max length L
    def pad_sequence(seq, max_len):
        pad_len = max_len - seq.shape[0]
        if pad_len > 0:
            padding = torch.zeros(pad_len, seq.shape[1])
            return torch.cat([torch.tensor(seq), padding], dim=0)
        return torch.tensor(seq[:max_len])