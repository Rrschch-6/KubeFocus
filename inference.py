import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import copy
import tqdm
import numpy as np
import matplotlib.pyplot as plt

from model import SimplifiedVisionTransformer, SimplePredictor
from dataset import DetectionDataset
from util import mask_future

from dataclasses import dataclass
from typing import Optional, Union

class SurpriseScoreEstimator(object):
    """ Do inference to predict future frames"""
    @dataclass
    class cfg:
        """ dataclass for model configutations"""
        num_frames: int = 5
        embed_dim: int = 64
        predictor_embed_dim: int = 32
        patch_size: int = 8
        depth: int = 1
        num_heads: int = 2
        mlp_ratio: float = 4.0
        mask_ratio: float = 0.5
        img_size : int = 32
        total_patches: int = num_frames * (img_size//patch_size)**2
        
    def __init__(self, pretrained_model_path: str, device: str = 'cuda') -> None:
        self.device = device
        self.model_cfg = self.cfg()
        # Check the fpv
        match = re.search(r'checkpoint_(\d+)_fpv', pretrained_model_path)
        if match:
            fpv = int(match.group(1))
            self.model_cfg.num_frames = fpv
            self.model_cfg.total_patches = fpv * (self.model_cfg.img_size // self.model_cfg.patch_size) ** 2
        (self.encoder, 
         self.target_encoder, 
         self.predictor ) = SurpriseScoreEstimator.prepare_models(pretrained_model_path, self.model_cfg)
        self.encoder.to(self.device)
        self.predictor.to(self.device)
        self.target_encoder.to(self.device)
        dataset = DetectionDataset("/home/kamyar/vjepa_data/dataset_a/image_dataset.pt", frames_per_video=self.model_cfg.num_frames)
        self.loader = DataLoader(dataset=dataset, batch_size=8, shuffle=False)

    @torch.no_grad()
    def inference(self, video: torch.Tensor) -> None:
        """
        Masks future video frames patches and calculates the distance between the predicted
        and actual frames using the encoder and predictor.
        Args:
            video (torch.Tensor): Input video tensor of shape (B, C, T, H, W).
        """
        x = video.to(self.device)  # Shape: (B, 3, num_frames, 32, 32)
        B = x.size(0)

        # Generate masks
        masks_enc, masks_pred = mask_future(B, self.model_cfg.total_patches,
                                            self.model_cfg.mask_ratio, fpv=self.model_cfg.num_frames)
        masks_enc = masks_enc.to(self.device)
        masks_pred = masks_pred.to(self.device)

        # Forward target encoder (full input)
        with torch.no_grad():
            h_full = self.target_encoder(x)  # (B, total_patches, embed_dim)
            # Select features for masked patches
            num_pred = int(self.model_cfg.num_frames * self.model_cfg.mask_ratio) * int(self.model_cfg.total_patches / self.model_cfg.num_frames)
            h = h_full[masks_pred].view(B, num_pred, self.model_cfg.embed_dim)  # (B, num_pred, embed_dim)

        # Forward encoder (visible patches)
        z = self.encoder(x, masks_enc)  # (B, num_visible, embed_dim)

        # Forward predictor (predict masked patches)
        pred = self.predictor(z, h, masks_enc, masks_pred)  # (B, num_pred, embed_dim)

        # Compute loss
        return F.mse_loss(pred.detach(), h.detach())

    @staticmethod
    def prepare_models(model_path: str, cfg) -> tuple[nn.Module, nn.Module, nn.Module]:
        """
        Load the encoder and predictor models from the specified path.
        Args:
            model_path (str): Path to the directory containing the model files.
        Returns:
            tuple[nn.Module, nn.Module, nn.Module]: Loaded encoder target_encoder and predictor models.
        """
        predictor = SimplePredictor(
            embed_dim=cfg.embed_dim,
            predictor_embed_dim=cfg.predictor_embed_dim,
            num_frames=cfg.num_frames,
            patch_size=cfg.patch_size,
            img_size=cfg.img_size,
            use_mask_tokens=True
        )
        encoder = SimplifiedVisionTransformer(
            num_frames=cfg.num_frames,
            embed_dim=cfg.embed_dim,
            patch_size=cfg.patch_size,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio
        )
        target_encoder = copy.deepcopy(encoder)

        SurpriseScoreEstimator.load_weights(path=model_path,
                                            encoder=encoder,
                                            target_encoder=target_encoder,
                                            predictor=predictor)
        encoder.eval()
        target_encoder.eval()
        predictor.eval()
        encoder.requires_grad_(False)
        predictor.requires_grad_(False)
        target_encoder.requires_grad_(False)

        return encoder, target_encoder, predictor

    @staticmethod
    def load_weights(path: str, encoder: nn.Module, 
                     target_encoder: nn.Module, predictor: nn.Module) -> None:
        """Saved Checkpoints have the following format
        
        checkpoint = {
                'encoder': encoder.state_dict(),
                'predictor': predictor.state_dict(),
                'target_encoder': target_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss
            }
        """
        if os.path.exists(path):
            checkpoint = torch.load(path)
            encoder.load_state_dict(checkpoint['encoder'])
            predictor.load_state_dict(checkpoint['predictor'])
            target_encoder.load_state_dict(checkpoint['target_encoder'])
        else:
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        print(f"Loaded weights from {path}")

    
    def plot_mse_histogram(self):
        """
        Plot the histogram of MSE values for the dataset.
        Blue is for Benign samples (label=0)
        Orange for Attach samples (label=1)
        """
        mse_values = []
        labels = []
        for data in tqdm.tqdm(self.loader):
            video, label = data['video'], data['label']
            mse = self.inference(video)
            mse_values += [mse.item() for b in range(video.size(0))]
            labels += list(label.numpy())

        # Convert to numpy arrays for plotting
        mse_values = np.array(mse_values)
        labels = np.array(labels)

        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(mse_values[labels == 0], bins=50, alpha=0.5, label='Benign', color='blue')
        plt.hist(mse_values[labels == 1], bins=50, alpha=0.5, label='Attack', color='orange')
        plt.xlabel('MSE Value')
        plt.ylabel('Frequency')
        plt.title('MSE Histogram for Benign and Attack Samples')
        plt.legend()
        plt.savefig(f"mse_histogram_{self.model_cfg.num_frames}fpv.png")



if __name__ == "__main__":
    import argparse
    import re
    parser = argparse.ArgumentParser(description="Inference with SurpriseScoreEstimator")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model weights")
    parser.add_argument("--device", type=str, default='cuda', help="Device to run the model on (cpu or cuda)")
    args = parser.parse_args()

    estimator = SurpriseScoreEstimator(pretrained_model_path=args.model_path, device=args.device)
    estimator.plot_mse_histogram()
        
        