import torch
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self,
                 video_paths,
                 frames_per_video=5):
        self.video_paths = video_paths
        self.frames_per_video = frames_per_video
        self.all_frames = torch.load(self.video_paths)
        self.num_videos = len(self.all_frames) // frames_per_video

    def __len__(self):
        return self.num_videos

    def __getitem__(self, idx):
        start_idx = idx * self.frames_per_video
        end_idx = start_idx + self.frames_per_video
        video = torch.stack([self.all_frames[idx]['image'].to(torch.float32) for idx in range(start_idx, end_idx)], dim=0)
        return video.permute(3,0,1,2).to(torch.float32)