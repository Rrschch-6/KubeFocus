import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import copy
import tqdm
import numpy as np
import matplotlib.pyplot as plt

from model import TabularPredictor, TabularTransformerEncoder
from dataset import TabularDataset
from util import future_row_masks

from dataclasses import dataclass

class TabularSurpriseScoreEstimator(object):
    @dataclass
    class cfg:
        """ dataclass for model configutations"""
        max_len: int = 5
        embed_dim: int = 64
        predictor_embed_dim: int = 32
        depth: int = 1
        num_heads: int = 2
        mlp_ratio: float = 4.0
        mask_ratio: float = 0.4
        num_features: int = 357
    def __init__(self, pretrained_model_path: str, data_path: str, label_col_name: str, device: str = 'cuda') -> None:
        self.device = device
        self.model_cfg = self.cfg()
        # Check the length
        import re
        match = re.search(r'checkpoint_tabular_(\d+)len_epoch_(\d+)', pretrained_model_path)
        if match:
            length = int(match.group(1))
            self.model_cfg.max_len = length

        test_data = TabularDataset(data_path, self.model_cfg.max_len, label_col_name=label_col_name)
        self.loader = DataLoader(dataset=test_data, batch_size=32, shuffle=False)
        self.model_cfg.num_features = test_data.num_features

        (self.encoder,
         self.target_encoder,
         self.predictor ) = TabularSurpriseScoreEstimator.prepare_models(pretrained_model_path, self.model_cfg)
        self.encoder.to(self.device).eval()
        self.predictor.to(self.device).eval()
        self.target_encoder.to(self.device).eval()
        
    @staticmethod
    def prepare_models(pretrained_model_path: str, model_cfg: cfg) -> tuple:
        """ Prepare the models for inference"""
        encoder = TabularTransformerEncoder(
            num_features=model_cfg.num_features,
            max_length=model_cfg.max_len,
            embed_dim=model_cfg.embed_dim,
            num_heads=model_cfg.num_heads,
            depth=model_cfg.depth
        )
        predictor = TabularPredictor(
            embed_dim=model_cfg.embed_dim,
            max_len=model_cfg.max_len,
            predictor_embed_dim=model_cfg.predictor_embed_dim
        )
        target_encoder = copy.deepcopy(encoder)
        for param in target_encoder.parameters():
            param.requires_grad = False

        checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        encoder.load_state_dict(checkpoint['encoder'])
        predictor.load_state_dict(checkpoint['predictor'])
        target_encoder.load_state_dict(checkpoint['target_encoder'])
        
        return encoder, target_encoder, predictor
    
    @torch.no_grad()
    def inference(self, sequence: torch.Tensor, return_masks: bool=False) -> None:
        """
        Masks rows in tabular sequences and calculates the MSE between predicted and actual representations.
        Args:
            sequence (torch.Tensor): Input sequence tensor of shape (B, max_len, num_features).
        Returns:
            torch.Tensor: MSE loss value.
        """
        x = sequence.to(self.device)  # Shape: (B, max_len, num_features)
        B = x.size(0)

        # Generate masks
        masks_enc, masks_pred = future_row_masks(B, self.model_cfg.max_len,
                                            self.model_cfg.mask_ratio)
        masks_enc = masks_enc.to(self.device)
        masks_pred = masks_pred.to(self.device)

        # Forward target encoder (full input)
        with torch.no_grad():
            h_full = self.target_encoder(x)  # (B, total_patches, embed_dim)
            # Select features for masked patches
            num_pred = max(1, self.model_cfg.max_len - int(self.model_cfg.max_len * (1 - self.model_cfg.mask_ratio)))
            h = h_full[masks_pred].view(B, num_pred, self.model_cfg.embed_dim)  # (B, num_pred, embed_dim)

        # Forward encoder (visible patches)
        z = self.encoder(x, masks_enc)  # (B, num_visible, embed_dim)

        # Forward predictor (predict masked patches)
        pred = self.predictor(z, h, masks_enc, masks_pred)  # (B, num_pred, embed_dim)

        # Compute loss
        if return_masks:
            return F.mse_loss(pred, h, reduction='none').mean(dim=(2)), masks_enc, masks_pred
        else:
            return F.mse_loss(pred, h, reduction='none').mean(dim=(2))

    def plot_mse_histogram(self):
        """
        Plot the histogram of MSE values for the dataset.
        Blue is for Benign samples (label=0)
        Orange for Attach samples (label=1)
        """
        mse_values = []
        labels = []
        for data in tqdm.tqdm(self.loader):
            sequence, label = data['sequence'], data['label'].to(self.device)
            mse, mask_enc, mask_pred = self.inference(sequence, return_masks=True)
            mse_values += mse.flatten().tolist() # B x num_pred
            labels += list(label[mask_pred].cpu().numpy()) # B x num_pred
            # print(mask_pred.shape, label.shape, labels[0].shape)

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
        plt.savefig(f"mse_histogram_{self.model_cfg.max_len}len_benchmark_train.png")

    def classifier(self, train_loader: str, alpha: float = 1.0):
        mse_values = []
        labels = []
        for data in tqdm.tqdm(train_loader):
            sequence = data['sequence']
            mse = self.inference(sequence)
            mse_values += mse.flatten().tolist()

        mean_mse = np.mean(mse_values)
        std_mse = np.std(mse_values)
        # Classification threshold
        self.threshold = mean_mse + alpha * std_mse

        for alpha in np.arange(-0.5, 0.5, 0.1):
            self.threshold = mean_mse + alpha * std_mse
            # classify test examples from self.loader and calculate the accuracy, and the confusion matrix
            y_pred = []
            y_true = []
            for data in tqdm.tqdm(self.loader):
                sequence, label = data['sequence'], data['label'].to(self.device)
                mse, mask_enc, mask_pred = self.inference(sequence, return_masks=True)
                mse = mse.flatten()
                y_pred += [1 if m > self.threshold else 0 for m in mse]
                y_true += list(label[mask_pred].flatten().cpu().numpy())
            y_pred = np.array(y_pred)
            y_true = np.array(y_true)
            accuracy = np.mean(y_pred == y_true)
            print("################################################")
            print(f"Accuracy: {accuracy * 100:.2f}%")
            # Print the percentage of benign and attack samples in the test set
            benign_percentage = np.mean(y_true == 0) * 100
            attack_percentage = np.mean(y_true == 1) * 100
            print(f"True Benign samples: {benign_percentage:.2f}%")
            print(f"True Attack samples: {attack_percentage:.2f}%")
            print(f"Threshold: {self.threshold:.4f}")
            benign_percentage = np.mean(y_pred == 0) * 100
            attack_percentage = np.mean(y_pred == 1) * 100
            print(f"Predicted Benign samples: {benign_percentage:.2f}%")
            print(f"Predicted Attack samples: {attack_percentage:.2f}%")
            print("################################################")

            # confusion matrix
            from sklearn.metrics import confusion_matrix
            from sklearn.metrics import ConfusionMatrixDisplay
            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f"Thr. {self.threshold:.4f} mean {mean_mse:.4f} std {std_mse:.4f}")
            plt.savefig(f"confusion_matrix_{self.model_cfg.max_len}len_benchmark_test_thr"+ (str(self.threshold)).replace('.','-')[:8]+".png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference with SurpriseScoreEstimator")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model weights")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the test data")
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training data")
    parser.add_argument("--label_col_name", type=str, default='attack', help="Column name for labels in the dataset")
    parser.add_argument("--device", type=str, default='cuda', help="Device to run the model on (cpu or cuda)")
    args = parser.parse_args()

    paths = [
        "logs_tabular_benchmark/checkpoint_tabular_10len_epoch_91.pth",
        "logs_tabular_benchmark/checkpoint_tabular_5len_epoch_91.pth",
        "logs_tabular_benchmark/checkpoint_tabular_3len_epoch_91.pth",
    ]
    for path in paths:
        print("### Path -> ", path, ":")
        args.model_path = path
        estimator = TabularSurpriseScoreEstimator(pretrained_model_path=args.model_path, 
                                                  data_path=args.data_path,
                                                  label_col_name=args.label_col_name, 
                                                  device=args.device)
        estimator.plot_mse_histogram()

        # Classification
        train_data = TabularDataset(args.train_path, estimator.model_cfg.max_len, label_col_name=args.label_col_name)
        train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=False)
        estimator.classifier(train_loader, alpha=0.0)
        