import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import copy
import tqdm

# Assuming these are defined in separate modules
from model import SimplifiedVisionTransformer, SimplePredictor
from dataset import VideoDataset
#TODO: Add tensorboard
# from tensorboardX import SummaryWriter

def main(args, device):
    # Initialize models
    encoder = SimplifiedVisionTransformer(
        num_frames=args.num_frames,
        embed_dim=args.embed_dim,
        patch_size=args.patch_size,
        depth=1,
        num_heads=2,
        mlp_ratio=4.0
    ).to(device)

    predictor = SimplePredictor(
        embed_dim=args.embed_dim,
        predictor_embed_dim=32,
        num_frames=args.num_frames,
        patch_size=args.patch_size,
        img_size=32,
        use_mask_tokens=True
    ).to(device)

    target_encoder = copy.deepcopy(encoder).to(device)
    for param in target_encoder.parameters():
        param.requires_grad = False

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=args.learning_rate
    )

    # Dataset and DataLoader
    # Replace with actual video file paths
    dataset = VideoDataset(args.video_paths, frames_per_video=args.num_frames)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Random mask generation function
    def generate_masks(batch_size, total_patches, mask_ratio):
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

    # Training loop
    pbar = tqdm.trange(0, args.num_epochs, desc=f'{args.num_frames} frames per video')
    for epoch in pbar:
        total_loss = 0
        for itr, x in enumerate(loader):
            x = x.to(device)  # Shape: (B, 3, num_frames, 32, 32)
            B = x.size(0)

            # Generate masks
            masks_enc, masks_pred = generate_masks(B, args.total_patches, args.mask_ratio)
            masks_enc = masks_enc.to(device)
            masks_pred = masks_pred.to(device)

            # Forward target encoder (full input)
            with torch.no_grad():
                h_full = target_encoder(x)  # (B, total_patches, embed_dim)
                # Select features for masked patches
                num_pred = args.total_patches - int(args.total_patches * (1 - args.mask_ratio))
                h = h_full[masks_pred].view(B, num_pred, args.embed_dim)  # (B, num_pred, embed_dim)

            # Forward encoder (visible patches)
            z = encoder(x, masks_enc)  # (B, num_visible, embed_dim)

            # Forward predictor (predict masked patches)
            pred = predictor(z, h, masks_enc, masks_pred)  # (B, num_pred, embed_dim)

            # Compute loss
            loss = F.mse_loss(pred, h)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update target encoder with EMA
            with torch.no_grad():
                for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                    param_k.data = args.momentum * param_k.data + (1 - args.momentum) * param_q.data

            total_loss += loss.item()

            # Logging every 10 iterations
            if itr % 10 == 0:
                pbar.set_postfix({"Iter": f"{itr}/{len(loader)}", "Loss": f"{loss.item():.4f}"})

        # Average loss per epoch
        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Average Loss: {avg_loss:.4f}")

        if epoch % 10 == 0:
            # Save checkpoint
            checkpoint = {
                'encoder': encoder.state_dict(),
                'predictor': predictor.state_dict(),
                'target_encoder': target_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss
            }
            torch.save(checkpoint, f'logs/checkpoint_{args.num_frames}_fpv_epoch_{epoch+1}.pth')

    print("Training completed!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train VJEPa model")
    parser.add_argument("--video_paths", type=str, required=True, help="Path to video dataset")
    parser.add_argument("--num_frames", type=int, default=5, help="Number of frames per video")
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--patch_size", type=int, default=8, help="Patch size")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--mask_ratio", type=float, default=0.75, help="Mask ratio for patches")
    parser.add_argument("--momentum", type=float, default=0.999, help="Momentum for target encoder update")
    parser.add_argument("--total_patches", type=int, default=80, help="Total patches per video")
    parser.add_argument("--img_size", type=int, default=32, help="Image size (height and width)")

    args = parser.parse_args()
    
    # Training
    args.total_patches = args.num_frames * (args.img_size // args.patch_size) ** 2
    print(f"Total patches: {args.total_patches}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args, device=device)
